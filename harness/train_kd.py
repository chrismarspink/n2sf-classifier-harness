"""train_kd.py — 개선 지식증류(KD) 학습기: base 0.8 돌파용.

train_soft.py 대비 추가:
  ① 클래스 균형 샘플링(--balance) : S 기아 해소(WeightedRandomSampler, argmax 클래스 역빈도)
  ② KD 온도(--temperature T)       : 교사 near-one-hot 라벨을 부드럽게(진짜 soft-label 복원, Hinton KD)
  ③ 하드 CE 혼합(--alpha)          : L=(1-α)·softCE(T²) + α·CE(hard) 안정화
  ④ maxlen 기본 256                : 평가와 정합(문서 뒷부분 단서 보존)
  ⑤ LR warmup+cosine               : 안정 수렴
  ⑥ held-out best 저장(--heldout)  : 매 에폭 정직 평가 → 최고(macro_f1, C재현율 우선) 체크포인트만 저장

입력: teacher_labels.jsonl  각 줄 {"text":..., "probs":[pO,pS,pC]}
"""
from __future__ import annotations

import argparse, json, math, re, time
from pathlib import Path

from .train_soft import load_soft, ID2LABEL, LABEL2ID, BASE
from . import metrics as M

SHORT = {"OPEN": "O", "SENSITIVE": "S", "CONFIDENTIAL": "C"}


def _heldout_eval(model, tok, heldout, dev, maxlen):
    """held-out 정직 평가 → (macro_f1, c_recall, over_rate)."""
    import torch
    model.eval()
    id2 = model.config.id2label
    def gidx(i):
        lab = id2.get(i, id2.get(str(i), ["OPEN", "SENSITIVE", "CONFIDENTIAL"][i]))
        return SHORT.get(lab, lab)
    rows = []
    with torch.no_grad():
        for h in heldout:
            enc = tok([h["text"]], truncation=True, max_length=maxlen, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            logits = model(**enc).logits[0]
            p = torch.softmax(logits, -1).tolist()
            by = {gidx(i): p[i] for i in range(len(p))}
            vec = [by.get("O", 0), by.get("S", 0), by.get("C", 0)]
            pred = ["O", "S", "C"][int(max(range(3), key=lambda k: vec[k]))]
            rows.append({"true_grade": h["grade"], "pred_grade": pred})
    m = M.compute(rows)
    return m["macro_f1"], m["c_recall"], m["over_rate"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="개선 KD 학습기(base 0.8 돌파)")
    ap.add_argument("--soft-jsonl", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default="models/n2sf-base")
    ap.add_argument("--heldout", default="distill_soft/heldout.json")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.1, help="warmup 비율")
    ap.add_argument("--temperature", type=float, default=2.0, help="KD 온도 T(>1=부드럽게)")
    ap.add_argument("--alpha", type=float, default=0.1, help="하드 CE 혼합 비율")
    ap.add_argument("--balance", action="store_true", help="클래스 균형 샘플링(S 기아 해소)")
    ap.add_argument("--train-top", type=int, default=0)
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=3.0)
    args = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_cosine_schedule_with_warmup)
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    data = load_soft(args.soft_jsonl, args.cap)
    if len(data) < 20:
        print(f"[train_kd] 데이터 부족({len(data)}) — 중단"); return
    heldout = json.loads(Path(args.heldout).read_text(encoding="utf-8")) if Path(args.heldout).exists() else []
    T, alpha = args.temperature, args.alpha
    print(f"[train_kd] device={dev} 학습쌍={len(data)} T={T} alpha={alpha} balance={args.balance} "
          f"maxlen={args.maxlen} epochs={args.epochs} held-out={len(heldout)}")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=3, ignore_mismatched_sizes=True,
        id2label=ID2LABEL, label2id=LABEL2ID, torch_dtype=torch.float32)
    model = model.float().to(dev)

    if args.train_top > 0:
        keep = None
        idxs = [int(m.group(1)) for n, _ in model.named_parameters()
                for m in [re.search(r"\.layer\.(\d+)\.", n)] if m]
        mx = max(idxs) if idxs else 0
        keep = set(range(mx - args.train_top + 1, mx + 1))
        for n, p in model.named_parameters():
            m = re.search(r"\.layer\.(\d+)\.", n)
            p.requires_grad = bool(("classifier" in n) or ("pooler" in n)
                                   or (m is not None and int(m.group(1)) in keep))

    # 클래스(argmax) 분포 & 균형 샘플러 가중치
    cls = [int(max(range(3), key=lambda k: p[k])) for _, p in data]
    cnt = [cls.count(0), cls.count(1), cls.count(2)]
    print(f"[train_kd] 학습 클래스분포 O/S/C = {cnt}")

    class DS(Dataset):
        def __init__(s, r): s.r = r
        def __len__(s): return len(s.r)
        def __getitem__(s, i): return s.r[i]

    def collate(b):
        enc = tok([x[0] for x in b], padding=True, truncation=True,
                  max_length=args.maxlen, return_tensors="pt")
        enc["target"] = torch.tensor([x[1] for x in b], dtype=torch.float32)
        return enc

    if args.balance:
        w = [1.0 / max(cnt[c], 1) for c in cls]
        sampler = WeightedRandomSampler(w, num_samples=len(data), replacement=True)
        dl = DataLoader(DS(data), batch_size=args.batch, sampler=sampler, collate_fn=collate)
    else:
        dl = DataLoader(DS(data), batch_size=args.batch, shuffle=True, collate_fn=collate)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps_total = (len(dl) // args.accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(steps_total * args.warmup), steps_total)
    logsm = torch.nn.LogSoftmax(dim=-1)
    eps = 1e-6
    deadline = time.time() + args.max_hours * 3600
    Path(args.out).mkdir(parents=True, exist_ok=True)
    best = (-1.0, -1.0)  # (c_recall>=1.0?, macro_f1) 튜플 비교용 → 여기선 (score,)

    def empty():
        try: torch.mps.empty_cache() if dev == "mps" else None
        except Exception: pass

    def save_if_best(ep):
        nonlocal best
        if not heldout:
            model.save_pretrained(args.out); tok.save_pretrained(args.out); return
        f1, crec, over = _heldout_eval(model, tok, heldout, dev, args.maxlen)
        # 안전 우선: C재현율 1.0 만족 시 +1.0 보너스로 우선, 그다음 macro_f1
        score = f1 + (1.0 if crec >= 1.0 else 0.0)
        tag = "★best" if score > best[0] else ""
        print(f"[train_kd] ep{ep} held-out F1={f1:.4f} C재현율={crec:.3f} 과대={over:.3f} {tag}", flush=True)
        if score > best[0]:
            best = (score, f1)
            model.save_pretrained(args.out); tok.save_pretrained(args.out)
        model.train()

    for ep in range(args.epochs):
        model.train(); t0 = time.time(); opt.zero_grad(); tot = 0.0; nb = 0; nan_skips = 0
        for step, batch in enumerate(dl):
            if time.time() > deadline:
                print("[train_kd] 시간예산 소진"); break
            try:
                tgt = batch.pop("target").to(dev)                       # 교사 확률 [B,3]
                b = {k: v.to(dev) for k, v in batch.items()}
                out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                logits = out.logits
                # ① 온도 소프트닝(교사 타깃만): softmax(log p / T). 학생은 T=1 → T² 증폭 없이 수치 안정
                soft_tgt = torch.softmax(torch.log(tgt.clamp(eps, 1.0)) / T, dim=-1)
                soft_ce = -(soft_tgt * logsm(logits)).sum(dim=1).mean()
                # ② 하드 CE(교사 argmax) 소량 혼합
                hard_ce = torch.nn.functional.cross_entropy(logits, tgt.argmax(dim=1))
                loss = (1 - alpha) * soft_ce + alpha * hard_ce
                if not torch.isfinite(loss):          # NaN/Inf 가드: 가중치 오염 방지 위해 배치 건너뜀
                    opt.zero_grad(); empty(); continue
                loss = loss / args.accum
                loss.backward()
                if (step + 1) % args.accum == 0:
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    if torch.isfinite(gnorm):          # grad NaN/Inf 가드: 오염 step 차단
                        opt.step(); sched.step()
                    else:
                        nan_skips += 1
                    opt.zero_grad()
                tot += float(loss.detach()) * args.accum; nb += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    opt.zero_grad(); empty(); continue
                raise
            if step % 100 == 0:
                print(f"[train_kd] ep{ep} step{step}/{len(dl)} loss={float(loss.detach())*args.accum:.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[train_kd] ep{ep} loss_avg={tot/max(nb,1):.3f} nan_skips={nan_skips}", flush=True)
        save_if_best(ep)
        if time.time() > deadline:
            break
    print(f"[train_kd] 완료 → {args.out} (best macro_f1={best[1]:.4f})")


if __name__ == "__main__":
    main()
