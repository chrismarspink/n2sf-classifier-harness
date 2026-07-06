"""gen_cleaneval.py — 청정 미지분포 평가셋(학습에 안 쓴 생성기). 일반화 정직 측정용.

학습 생성기: gpt-4o + 템플릿(corpus) + Gemma. → 평가는 **o4-mini 생성**(4번째 스타일)로.
등급 라벨은 synth PII 주입으로 정책 고정(생성기와 무관). 출력: {text, grade}.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
from .gen_llm import generate, LLMFiller, _load_env

def _norm(t): return re.sub(r"\s+"," ",t).strip()

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="distill_soft/heldout_clean.json")
    ap.add_argument("--gen", default="azure:o4-mini", help="학습에 안 쓴 생성기")
    ap.add_argument("--per-grade", type=int, default=40)
    ap.add_argument("--langs", default="ko,en,zh,ja")
    args = ap.parse_args(argv)
    _load_env()
    langs=[x.strip() for x in args.langs.split(",") if x.strip()]
    filler=LLMFiller(seed=777); seen=set(); out=[]
    i=0
    for lang in langs:
        for grade in ("O","S","C"):
            made=0
            while made < args.per_grade // len(langs) + 1:
                i+=1
                doc=generate(args.gen, grade, (i%4)+1, f"clean-{lang}-{grade}-{i}", filler, lang=lang)
                t="" if doc is None else (doc.title+"\n"+"\n".join(doc.paragraphs)).strip()
                if not t or len(t)<20:
                    if i%3==0: break
                    continue
                h=hashlib.md5(_norm(t).encode()).hexdigest()
                if h in seen: continue
                seen.add(h); out.append({"text":t[:2000],"grade":grade,"gen":args.gen,"lang":lang}); made+=1
    Path(args.out).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    from collections import Counter
    print(f"[clean] {len(out)}건 → {args.out} · 등급 {dict(Counter(x['grade'] for x in out))} · gen={args.gen}")

if __name__=="__main__":
    main()
