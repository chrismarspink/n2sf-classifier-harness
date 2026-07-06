"""gen_robust.py — 다중 생성기 학습셋(근본책 #1). 분포 다양화로 OOD 강건성↑.

기존 다국어 교사셋(LLM 생성) + **비-LLM 템플릿(corpus, OOD와 다른 seed)** + **Gemma(다른 family)** 를
추가하고 o4-mini로 soft-라벨 → 한 생성기 과적합 완화. (재개형: out 있으면 이어붙임)
※ OOD 평가셋(corpus seed 1000~1003)과 겹치지 않도록 학습 템플릿은 seed 2000+ 사용.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, time
from pathlib import Path
from .gen_llm import generate, LLMFiller, _load_env
from .corpus import CorpusGen, _flat_text
from . import distill

def _norm(t): return re.sub(r"\s+"," ",t).strip()
def _h(t): return hashlib.md5(_norm(t).encode()).hexdigest()

def main(argv=None):
    ap = argparse.ArgumentParser(description="다중 생성기 강건 학습셋")
    ap.add_argument("--base-jsonl", default="distill_o4/teacher_labels_multi.jsonl")
    ap.add_argument("--out", default="distill_o4/teacher_labels_robust.jsonl")
    ap.add_argument("--n-template", type=int, default=250, help="비-LLM 템플릿 문서 수(seed 2000+)")
    ap.add_argument("--n-gemma", type=int, default=150, help="Gemma 생성 문서 수")
    ap.add_argument("--teacher", default="o4-mini")
    ap.add_argument("--max-hours", type=float, default=3.0)
    args = ap.parse_args(argv)
    _load_env(); distill.TEACHER = args.teacher
    deadline = time.time() + args.max_hours*3600

    seen=set(); n0=0
    outp=Path(args.out); outp.parent.mkdir(parents=True,exist_ok=True)
    src=args.out if outp.exists() else args.base_jsonl
    tmp=str(outp)+".tmp"
    with open(tmp,"w",encoding="utf-8") as w:
        for line in open(src,encoding="utf-8"):
            line=line.strip()
            if not line: continue
            try: t=json.loads(line).get("text","")
            except: continue
            if t: seen.add(_h(t)); w.write(line+"\n"); n0+=1
    os.replace(tmp,outp)
    print(f"[robust] 시작셋 {n0} 교사={args.teacher} +템플릿{args.n_template} +gemma{args.n_gemma}",flush=True)

    added=0
    w=open(outp,"a",encoding="utf-8")
    # 1) 비-LLM 템플릿(corpus, seed 2000+; OOD seed 1000~1003과 분리)
    made=0; seed=2000
    while made < args.n_template and time.time()<deadline:
        for d in CorpusGen(seed=seed).build(per_cell=8, difficulty=(seed%4)+1):
            if made>=args.n_template: break
            t=_flat_text(d)
            if not t or len(t)<20 or _h(t) in seen: continue
            p=distill.teacher_soft(t)
            if not p or sum(p)<=0: continue
            s=sum(p); p=[x/s for x in p]; seen.add(_h(t))
            w.write(json.dumps({"text":t[:2000],"probs":p,"gen":"template","grade":d.grade,"robust":True},ensure_ascii=False)+"\n")
            w.flush(); made+=1; added+=1
            if added%30==0: print(f"[robust] +{added} (template {made})",flush=True)
        seed+=1
    # 2) Gemma(다른 family)
    filler=LLMFiller(seed=303); gi=0
    langs=["ko","ko","ko","en","zh","ja"]
    while gi < args.n_gemma and time.time()<deadline:
        grade=["O","S","C"][gi%3]; lang=langs[gi%len(langs)]
        doc=generate("gemma",grade,(gi%4)+1,f"gem-{gi}",filler,lang=lang)
        gi+=1
        t="" if doc is None else (doc.title+"\n"+"\n".join(doc.paragraphs)).strip()
        if not t or len(t)<20 or _h(t) in seen: continue
        p=distill.teacher_soft(t)
        if not p or sum(p)<=0: continue
        s=sum(p); p=[x/s for x in p]; seen.add(_h(t))
        w.write(json.dumps({"text":t[:2000],"probs":p,"gen":"gemma","grade":grade,"robust":True},ensure_ascii=False)+"\n")
        w.flush(); added+=1
        if added%30==0: print(f"[robust] +{added} (gemma {gi})",flush=True)
    w.close()
    print(f"[robust] 완료 +{added} → {outp} (총 {n0+added})")

if __name__=="__main__":
    main()
