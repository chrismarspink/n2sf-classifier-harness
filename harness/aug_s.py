"""aug_s.py — S(민감)·경계 문서 증강: base 0.8 돌파(누락0 유지)용.

진단: 교사셋의 S가 9%뿐 → S 일반화 부족. 균형 샘플링은 반복만 늘릴 뿐 '다양성'은 못 늘림.
해법: 교사 LLM으로 **새로운 S·경계 문서를 다수 생성**하고 교사 soft-라벨을 붙여 증류셋에 추가.

절차(문서당 2콜): ① gpt-4o로 문서 생성(빠름·다양) + 유효 PII 주입(라벨 신뢰)
                  ② o4-mini로 O/S/C 확률 라벨(원본 distill_o4 세트와 일관)
산출: distill_o4/teacher_labels_aug.jsonl (= 원본 + 증강). 이후 train_kd로 재증류.
"""
from __future__ import annotations

import argparse, hashlib, json, os, time
from pathlib import Path

from .gen_llm import generate, LLMFiller, _load_env
from . import distill


def _norm(t): return " ".join(t.split())
def _h(t): return hashlib.md5(_norm(t).encode()).hexdigest()


def _doc_text(doc):
    if doc is None:
        return ""
    return (doc.title + "\n" + "\n".join(doc.paragraphs)).strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description="S·경계 문서 증강(교사 생성+라벨)")
    ap.add_argument("--base-jsonl", default="distill_o4/teacher_labels.jsonl")
    ap.add_argument("--out", default="distill_o4/teacher_labels_aug.jsonl")
    ap.add_argument("--n-s", type=int, default=300, help="생성할 S 문서 수")
    ap.add_argument("--n-bound", type=int, default=100, help="경계(O/C, S 근처) 문서 수")
    ap.add_argument("--gen", default="azure:gpt-4o", help="문서 생성기")
    ap.add_argument("--teacher", default="o4-mini", choices=["gpt-4o", "o4-mini"], help="라벨 교사")
    ap.add_argument("--max-hours", type=float, default=1.5)
    args = ap.parse_args(argv)
    _load_env()
    distill.TEACHER = args.teacher            # teacher_soft 가 이 전역을 참조

    # 원본 복사(누적) + 중복셋
    seen = set(); n_base = 0
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as w:
        for line in open(args.base_jsonl, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            t = o.get("text", "")
            if not t:
                continue
            seen.add(_h(t)); w.write(line + "\n"); n_base += 1
    print(f"[aug_s] 원본 {n_base}건 복사 → {outp}. 증강 시작(S={args.n_s}, 경계={args.n_bound}) 교사={args.teacher}")

    # 생성 계획: S 다수 + 경계 O/C(S 근처, 난이도 높음)
    plan = ([("S", d) for d in _spread(args.n_s, [2, 3, 4])] +
            [("O", d) for d in _spread(args.n_bound // 2, [2, 3])] +
            [("C", d) for d in _spread(args.n_bound - args.n_bound // 2, [3, 4])])
    filler = LLMFiller(seed=101)
    deadline = time.time() + args.max_hours * 3600
    added = 0; tried = 0
    with open(outp, "a", encoding="utf-8") as w:
        for i, (grade, diff) in enumerate(plan):
            if time.time() > deadline:
                print("[aug_s] 시간예산 소진"); break
            tried += 1
            doc = generate(args.gen, grade, diff, f"aug-{grade}-{i}", filler)
            text = _doc_text(doc)
            if not text or len(text) < 20 or _h(text) in seen:
                continue
            probs = distill.teacher_soft(text)
            if not probs or sum(probs) <= 0:
                continue
            s = sum(probs); probs = [x / s for x in probs]
            seen.add(_h(text))
            w.write(json.dumps({"text": text[:2000], "probs": probs,
                                "gen": args.gen, "grade": grade, "aug": True}, ensure_ascii=False) + "\n")
            w.flush()
            added += 1
            if added % 25 == 0:
                print(f"[aug_s] 진행 {added}건 추가(시도 {tried}) 경과 {int(time.time()-deadline+args.max_hours*3600)}s", flush=True)
    print(f"[aug_s] 완료 — 증강 {added}건(시도 {tried}). 총 {n_base+added}건 → {outp}")


def _spread(n, diffs):
    """n개를 난이도 목록에 고르게 분배."""
    return [diffs[i % len(diffs)] for i in range(max(n, 0))]


if __name__ == "__main__":
    main()
