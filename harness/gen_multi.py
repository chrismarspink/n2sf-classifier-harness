"""gen_multi.py — 다국어 학습 라벨 증강(영/중/일). 소형 다국어 모델 증류용.

배경: 다국어 백본(xlmr 등)도 교사라벨이 한국어뿐이면 다국어가 안 살아남(실측 L5 0.556).
해법: 영/중/일 문서를 LLM으로 생성(문체 다양) + 유효 PII 주입 + 교사 soft-라벨 → 증류셋에 추가.

시작셋: distill_o4/teacher_labels_gpt5.jsonl (한국어 균형증강) → 여기에 en/zh/ja 추가.
산출: distill_o4/teacher_labels_multi.jsonl  (재개형: 이미 있으면 이어붙임)
"""
from __future__ import annotations

import argparse, hashlib, json, os, time
from pathlib import Path

from .gen_llm import generate, LLMFiller, _load_env
from . import distill


def _norm(t): return " ".join(t.split())
def _h(t): return hashlib.md5(_norm(t).encode()).hexdigest()
def _doc_text(doc): return "" if doc is None else (doc.title + "\n" + "\n".join(doc.paragraphs)).strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description="다국어(영/중/일) 학습 라벨 증강")
    ap.add_argument("--base-jsonl", default="distill_o4/teacher_labels_gpt5.jsonl")
    ap.add_argument("--out", default="distill_o4/teacher_labels_multi.jsonl")
    ap.add_argument("--langs", default="en,zh,ja")
    ap.add_argument("--per-cell", type=int, default=60, help="언어×등급 셀당 생성 수(총 langs×3×per_cell)")
    ap.add_argument("--gen", default="azure:gpt-4o")
    ap.add_argument("--teacher", default="o4-mini", choices=["gpt-4o", "o4-mini", "gpt-5"])
    ap.add_argument("--max-hours", type=float, default=4.0)
    args = ap.parse_args(argv)
    _load_env(); distill.TEACHER = args.teacher
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    # 재개형: 기존 out 있으면 그걸 이어씀, 없으면 base 복사
    seen = set(); n0 = 0
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    src = args.out if outp.exists() else args.base_jsonl
    tmp = str(outp) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as w:
        for line in open(src, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            if t:
                seen.add(_h(t)); w.write(line + "\n"); n0 += 1
    os.replace(tmp, outp)
    already_multi = sum(1 for l in open(outp, encoding="utf-8") if '"multi": true' in l)
    print(f"[gen_multi] 시작셋 {n0}건(그중 다국어 {already_multi}) 교사={args.teacher} langs={langs} 목표+={len(langs)*3*args.per_cell}", flush=True)

    plan = [(lg, g, d) for lg in langs for g in ("O", "S", "C")
            for d, _ in zip([2, 3, 4] * args.per_cell, range(args.per_cell))]
    filler = LLMFiller(seed=202)
    deadline = time.time() + args.max_hours * 3600
    added = 0
    with open(outp, "a", encoding="utf-8") as w:
        for i, (lg, g, d) in enumerate(plan):
            if time.time() > deadline:
                print("[gen_multi] 시간예산 소진"); break
            doc = generate(args.gen, g, d, f"multi-{lg}-{g}-{i}", filler, lang=lg)
            text = _doc_text(doc)
            if not text or len(text) < 20 or _h(text) in seen:
                continue
            probs = distill.teacher_soft(text)
            if not probs or sum(probs) <= 0:
                continue
            s = sum(probs); probs = [x / s for x in probs]
            seen.add(_h(text))
            w.write(json.dumps({"text": text[:2000], "probs": probs, "gen": args.gen,
                                "grade": g, "lang": lg, "multi": True}, ensure_ascii=False) + "\n")
            w.flush(); added += 1
            if added % 30 == 0:
                print(f"[gen_multi] +{added} (i={i}/{len(plan)})", flush=True)
    print(f"[gen_multi] 완료 +{added} → {outp} (총 {n0+added})")


if __name__ == "__main__":
    main()
