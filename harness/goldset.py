"""goldset.py — 사람 라벨링용 골드셋 후보 500건 추출(층화 + 경계 우선).

여러 생성기(gpt-4o/o4-mini/gemma/template)·언어·등급으로 층화 표본 + **불확실(교사확률 엔트로피↑) 우선**.
사람이 라벨할 CSV 출력(intended grade는 숨김 — 편향 방지). model_hint는 교사 argmax(참고용).
"""
from __future__ import annotations
import argparse, csv, json, math, random, re
from pathlib import Path
from collections import defaultdict

def _fam(gen):
    g=(gen or "").lower()
    if "gpt-4o" in g: return "gpt4o"
    if "o4-mini" in g: return "o4mini"
    if "gemma" in g: return "gemma"
    if "template" in g: return "template"
    if "azure" in g: return "azure"
    return "other"

def _ent(p):
    return -sum(x*math.log(x+1e-9) for x in p) if p else 0.0

def load_pool():
    rows=[]
    # 학습셋(생성기·언어 다양) — text/probs/gen/lang
    for f in ["distill_o4/teacher_labels_robust.jsonl","distill_o4/teacher_labels_multi.jsonl"]:
        if not Path(f).exists(): continue
        for line in open(f,encoding="utf-8"):
            line=line.strip()
            if not line: continue
            try: o=json.loads(line)
            except: continue
            t=o.get("text","");p=o.get("probs")
            if not t or len(t)<25: continue
            rows.append({"text":t,"gen":o.get("gen","?"),"lang":o.get("lang","ko"),
                         "probs":p,"ent":_ent(p) if p else 0.0})
        break  # robust면 충분
    # 평가셋 계열(다른 스타일)도 일부 편입
    for f,gen in [("distill_soft/heldout_clean.json","azure:o4-mini"),
                  ("distill_soft/heldout_ood.json","template")]:
        if not Path(f).exists(): continue
        for o in json.loads(Path(f).read_text(encoding="utf-8")):
            t=o.get("text","")
            if t and len(t)>=25:
                rows.append({"text":t,"gen":o.get("gen",gen),"lang":o.get("lang","ko"),"probs":None,"ent":0.0})
    return rows

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=500)
    ap.add_argument("--out",default="goldset_candidates.csv")
    ap.add_argument("--seed",type=int,default=7)
    args=ap.parse_args(argv)
    rng=random.Random(args.seed)
    pool=load_pool()
    # 중복 제거
    seen=set(); uniq=[]
    for r in pool:
        h=re.sub(r"\s+"," ",r["text"]).strip()[:200]
        if h in seen: continue
        seen.add(h); uniq.append(r)
    rng.shuffle(uniq)
    # 층화: (family) 균등 + 각 family 내 엔트로피 상위(경계) 우선 절반 + 랜덤 절반
    byfam=defaultdict(list)
    for r in uniq: byfam[_fam(r["gen"])].append(r)
    fams=[f for f in byfam if byfam[f]]
    per=max(args.n//max(len(fams),1),1)
    picked=[]
    for f in fams:
        lst=byfam[f]
        lst_sorted=sorted(lst,key=lambda r:-r["ent"])
        half=per//2
        cand=lst_sorted[:half]+rng.sample(lst,min(len(lst),per-half))
        # 중복 제거 후 per개
        s=set(); out=[]
        for r in cand:
            k=r["text"][:80]
            if k in s: continue
            s.add(k); out.append(r)
            if len(out)>=per: break
        picked+=out
    rng.shuffle(picked); picked=picked[:args.n]
    # CSV
    def hint(p):
        return ["O","S","C"][max(range(3),key=lambda k:p[k])] if p else ""
    with open(args.out,"w",encoding="utf-8-sig",newline="") as w:
        wr=csv.writer(w)
        wr.writerow(["id","source","lang","chars","model_hint","uncertainty","text","human_label","note"])
        for i,r in enumerate(picked,1):
            unc="high" if r["ent"]>0.6 else ("mid" if r["ent"]>0.2 else "low")
            wr.writerow([f"G{i:04d}",_fam(r["gen"]),r["lang"],len(r["text"]),hint(r["probs"]),unc,
                         r["text"][:1500],"",""])
    from collections import Counter
    print(f"[goldset] {len(picked)}건 → {args.out}")
    print(f"  생성기: {dict(Counter(_fam(r['gen']) for r in picked))}")
    print(f"  언어: {dict(Counter(r['lang'] for r in picked))}")
    print(f"  불확실(경계) high 비율: {sum(1 for r in picked if r['ent']>0.6)}/{len(picked)}")

if __name__=="__main__":
    main()
