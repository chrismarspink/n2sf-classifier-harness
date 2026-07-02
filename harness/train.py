"""train.py — mdeberta-n2sf 파인튜닝: mDeBERTa-v3 에 3-class(O/S/C) 헤드를 얹어 우리 데이터로 학습.

1차 테스트가 만든 라벨 데이터(weekend/results.db 의 detection.text + corpus.grade)로 학습한다.
제로샷(NLI) → **직접 3-class 분류기**로 전환되어 L3(위장) 케이스를 예시로 학습 → 돌파를 노린다.

- 베이스: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (로컬 캐시) → num_labels=3 헤드 교체
- 라벨: 0=OPEN, 1=SENSITIVE, 2=CONFIDENTIAL (data_classifier 내부 등급명과 일치)
- 가속: Apple MPS (가능 시), 없으면 CPU. 시간예산·에폭·서브샘플로 무인 실행에 안전.
- 산출: models/mdeberta-n2sf/ (save_pretrained + tokenizer) + training_log.json
- 누수 방지: 학습은 weekend 데이터, 평가는 하네스가 생성하는 **신선 test2 코퍼스**로 별도 수행.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path

BASE = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
GRADE2ID = {"O": 0, "S": 1, "C": 2}
ID2LABEL = {0: "OPEN", 1: "SENSITIVE", 2: "CONFIDENTIAL"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def load_data(db_path: str, per_class_cap: int = 12000, seed: int = 0, jsonl: str = None):
    import random, json
    pairs = []  # (text, grade_char)
    if db_path and os.path.exists(db_path):
        c = sqlite3.connect(db_path); c.row_factory = sqlite3.Row
        for r in c.execute("""SELECT d.text, cp.grade FROM detection d
                              JOIN corpus cp ON cp.doc_id=d.doc_id AND cp.fmt=d.fmt"""):
            pairs.append((r["text"] or "", r["grade"]))
    if jsonl and os.path.exists(jsonl):
        for line in open(jsonl, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line); pairs.append((o.get("text", ""), o.get("grade")))
    seen, data = set(), []
    for t, g in pairs:
        if not t or len(t) < 8 or g not in GRADE2ID:
            continue
        h = hashlib.md5(_norm(t).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        data.append((t[:2000], GRADE2ID[g]))
    # 클래스별 캡(균형)
    rng = random.Random(seed); rng.shuffle(data)
    per = {0: [], 1: [], 2: []}
    for t, y in data:
        if len(per[y]) < per_class_cap:
            per[y].append((t, y))
    bal = per[0] + per[1] + per[2]
    rng.shuffle(bal)
    return bal


def main(argv=None):
    ap = argparse.ArgumentParser(description="mdeberta-n2sf 파인튜닝")
    ap.add_argument("--db", default="weekend/results.db")
    ap.add_argument("--jsonl", default=None, help="추가 학습쌍 JSONL({text,grade}) — LLM 생성 데이터")
    ap.add_argument("--out", default="models/mdeberta-n2sf")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8, help="마이크로배치(MPS OOM 방지 위해 작게)")
    ap.add_argument("--accum", type=int, default=4, help="그래디언트 누적(유효배치 = batch×accum)")
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--save-every", type=int, default=300, help="N step마다 체크포인트 저장(무인 안전)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--per-class-cap", type=int, default=12000)
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--c-weight", type=float, default=1.5, help="기밀(C) 클래스 손실 가중(누락 억제)")
    args = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={dev}")

    data = load_data(args.db, args.per_class_cap, jsonl=args.jsonl)
    n_val = max(200, int(len(data) * 0.1))
    val, train = data[:n_val], data[n_val:]
    print(f"[train] 학습 {len(train)} / 검증 {len(val)}")

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=3, ignore_mismatched_sizes=True,
        id2label=ID2LABEL, label2id=LABEL2ID, torch_dtype=torch.float32)
    model = model.float().to(dev)   # 체크포인트가 fp16 → 학습은 fp32로 (MPS 안정)

    class DS(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return self.rows[i]

    def collate(batch):
        texts = [b[0] for b in batch]; ys = [b[1] for b in batch]
        enc = tok(texts, padding=True, truncation=True, max_length=args.maxlen, return_tensors="pt")
        enc["labels"] = torch.tensor(ys, dtype=torch.long)
        return enc

    tl = DataLoader(DS(train), batch_size=args.batch, shuffle=True, collate_fn=collate)
    vl = DataLoader(DS(val), batch_size=args.batch, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cw = torch.tensor([1.0, 1.0, args.c_weight], device=dev)
    lossfn = torch.nn.CrossEntropyLoss(weight=cw)

    deadline = time.time() + args.max_hours * 3600
    log = {"device": dev, "n_train": len(train), "n_val": len(val), "epochs": []}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    best_acc = -1.0

    def empty_cache():
        try:
            if dev == "mps":
                torch.mps.empty_cache()
            elif dev == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0; oom = 0
        opt.zero_grad()
        for step, batch in enumerate(tl):
            if time.time() > deadline:
                print("[train] 시간예산 소진 — 조기 종료"); break
            try:
                batch = {k: v.to(dev) for k, v in batch.items()}
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = lossfn(out.logits, batch["labels"]) / args.accum
                loss.backward()
                if (step + 1) % args.accum == 0:
                    opt.step(); opt.zero_grad()
                tot += float(loss) * args.accum; nb += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom += 1; opt.zero_grad(); empty_cache()
                    if oom <= 3:
                        print(f"[train] OOM step{step} 건너뜀(누적 {oom})", flush=True)
                    continue
                raise
            if step % 50 == 0:
                print(f"[train] ep{ep} step{step}/{len(tl)} loss={float(loss)*args.accum:.3f} "
                      f"({(time.time()-t0):.0f}s)", flush=True)
            if args.save_every and step > 0 and step % args.save_every == 0:
                model.save_pretrained(args.out); tok.save_pretrained(args.out)
                print(f"[train] 중간 체크포인트 저장 step{step} → {args.out}", flush=True)
        # 검증
        model.eval(); correct = 0; n = 0; cm = [[0]*3 for _ in range(3)]
        with torch.no_grad():
            for batch in vl:
                y = batch["labels"]
                b = {k: v.to(dev) for k, v in batch.items() if k != "labels"}
                pred = model(**b).logits.argmax(-1).cpu()
                for yi, pi in zip(y.tolist(), pred.tolist()):
                    cm[yi][pi] += 1; n += 1; correct += (yi == pi)
        acc = correct / max(n, 1)
        c_rec = cm[2][2] / max(sum(cm[2]), 1)   # 기밀 재현율
        print(f"[train] ep{ep} 검증 acc={acc:.4f} C재현율={c_rec:.4f} loss_avg={tot/max(nb,1):.3f}")
        log["epochs"].append({"epoch": ep, "val_acc": round(acc, 4), "c_recall": round(c_rec, 4),
                              "loss": round(tot/max(nb, 1), 4)})
        if acc >= best_acc:
            best_acc = acc
            model.save_pretrained(args.out); tok.save_pretrained(args.out)
            print(f"[train] 체크포인트 저장(acc={acc:.4f}) → {args.out}")
        if time.time() > deadline:
            break

    log["best_val_acc"] = round(best_acc, 4)
    Path(args.out, "training_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] 완료. best_val_acc={best_acc:.4f} → {args.out}")


if __name__ == "__main__":
    main()
