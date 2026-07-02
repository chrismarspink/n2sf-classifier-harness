"""train_soft.py — soft-label 지식증류(KD)로 학생(mdeberta-n2sf) 학습.

교사(GPT-4o)가 매긴 O/S/C **확률분포**를 목표로 학생이 log-softmax를 맞춘다(KD loss).
우리 주입 라벨이 아니라 **교사의 결정경계**를 모방 → 다양한 문서 일반화(held-out) 향상을 노림.

입력: teacher_labels.jsonl  각 줄 {"text":..., "probs":[pO,pS,pC]}
산출: models/mdeberta-n2sf/ (학생 가중치)
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, time
from pathlib import Path

BASE = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
ID2LABEL = {0: "OPEN", 1: "SENSITIVE", 2: "CONFIDENTIAL"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def _norm(t): return re.sub(r"\s+", " ", t).strip()


def load_soft(jsonl: str, cap: int = 0, seed: int = 0):
    import random
    seen, data = set(), []
    for line in open(jsonl, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        t = o.get("text", ""); p = o.get("probs")
        if not t or len(t) < 8 or not p or len(p) != 3:
            continue
        s = sum(p)
        if s <= 0:
            continue
        p = [x / s for x in p]
        h = hashlib.md5(_norm(t).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h); data.append((t[:2000], p))
    random.Random(seed).shuffle(data)
    return data[:cap] if cap else data


def main(argv=None):
    ap = argparse.ArgumentParser(description="soft-label KD 학생 학습")
    ap.add_argument("--soft-jsonl", required=True)
    ap.add_argument("--base", default=BASE, help="백본 인코더(HF id) — 라인업용")
    ap.add_argument("--train-top", type=int, default=0, help="상위 N개 인코더 레이어만 학습(0=전체). 대형모델 OOM 방지")
    ap.add_argument("--out", default="models/mdeberta-n2sf")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--save-every", type=int, default=300)
    args = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    data = load_soft(args.soft_jsonl, args.cap)
    if len(data) < 20:
        print(f"[train_soft] 데이터 부족({len(data)}) — 중단"); return
    print(f"[train_soft] device={dev} KD 학습쌍={len(data)}")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=3, ignore_mismatched_sizes=True,
        id2label=ID2LABEL, label2id=LABEL2ID, torch_dtype=torch.float32)
    model = model.float().to(dev)

    # 대형모델: 상위 N개 인코더 레이어 + 분류헤드만 학습(하위 동결) → MPS 메모리·시간 절감
    if args.train_top > 0:
        idxs = [int(m.group(1)) for n, _ in model.named_parameters()
                for m in [re.search(r"\.layer\.(\d+)\.", n)] if m]
        mx = max(idxs) if idxs else 0
        keep = set(range(mx - args.train_top + 1, mx + 1))
        ntrain = 0
        for n, p in model.named_parameters():
            m = re.search(r"\.layer\.(\d+)\.", n)
            if ("classifier" in n) or ("pooler" in n) or (m and int(m.group(1)) in keep):
                p.requires_grad = True; ntrain += p.numel()
            else:
                p.requires_grad = False
        print(f"[train_soft] 레이어 동결: 상위 {args.train_top}개+헤드만 학습(학습 파라미터 {ntrain/1e6:.0f}M)")

    class DS(Dataset):
        def __init__(s, r): s.r = r
        def __len__(s): return len(s.r)
        def __getitem__(s, i): return s.r[i]

    def collate(b):
        enc = tok([x[0] for x in b], padding=True, truncation=True,
                  max_length=args.maxlen, return_tensors="pt")
        enc["target"] = torch.tensor([x[1] for x in b], dtype=torch.float32)
        return enc

    dl = DataLoader(DS(data), batch_size=args.batch, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    logsm = torch.nn.LogSoftmax(dim=-1)
    deadline = time.time() + args.max_hours * 3600
    Path(args.out).mkdir(parents=True, exist_ok=True)

    def empty():
        try: torch.mps.empty_cache() if dev == "mps" else None
        except Exception: pass

    for ep in range(args.epochs):
        model.train(); t0 = time.time(); opt.zero_grad(); tot = 0; nb = 0
        for step, batch in enumerate(dl):
            if time.time() > deadline:
                print("[train_soft] 시간예산 소진"); break
            try:
                tgt = batch.pop("target").to(dev)
                b = {k: v.to(dev) for k, v in batch.items()}
                out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                # soft-CE (KD): -sum(teacher_p * log_softmax(student))
                loss = -(tgt * logsm(out.logits)).sum(dim=1).mean() / args.accum
                loss.backward()
                if (step + 1) % args.accum == 0:
                    opt.step(); opt.zero_grad()
                tot += float(loss) * args.accum; nb += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    opt.zero_grad(); empty(); continue
                raise
            if step % 50 == 0:
                print(f"[train_soft] ep{ep} step{step}/{len(dl)} loss={float(loss)*args.accum:.3f} ({time.time()-t0:.0f}s)", flush=True)
            if args.save_every and step > 0 and step % args.save_every == 0:
                model.save_pretrained(args.out); tok.save_pretrained(args.out)
        model.save_pretrained(args.out); tok.save_pretrained(args.out)
        print(f"[train_soft] ep{ep} 저장 loss_avg={tot/max(nb,1):.3f}", flush=True)
        if time.time() > deadline:
            break
    print(f"[train_soft] 완료 → {args.out}")


if __name__ == "__main__":
    main()
