"""gen_n2sf.py — N²SF 공식(§9 영향기반) 데이터 생성·라벨. C=국가안보, S=개인/내부, O=공개.

기존 데이터는 '형태 기반'(개인정보=C)이라 §9와 어긋남 → §9 기준 문서를 새로 생성하고
갱신된 티처 POLICY(§9)로 soft-라벨. 학습/held-out 분리 출력.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
from .gen_llm import _gpt5_client, _azure_client, _load_env
from . import distill
import os

# §9 등급별 생성 프롬프트(영향 기반)
G9 = {
 "C": "정부·공공기관의 **'비밀' 또는 '대외비'로 지정된 일반 행정 문서**. 실제 군사/첩보 내용이 아니라 "
      "문서의 **분류 지정·취급 표시와 행정 형식** 위주로 작성(제목에 [비밀]/[대외비]/[극비] 표시, "
      "'본 문서는 비밀 제○호로 지정', '무단 열람·배포 금지', 취급자 지정 등). 소재는 대외정책 검토 회의 안건, "
      "부처 간 협의, 예산·감사 중 비밀 지정된 건 등 **행정적 소재**로. 창작이 아닌 서식 흉내.",
 "S": "유출 시 개인·기관 이익을 침해할 수 있는 비공개 문서(국가안보 맥락 아님). 예: 개인정보(성명·주민등록번호·"
      "연락처·주소)가 든 명부/신청서, 내부 행정·인사·예산·정책검토, 법인 영업비밀, 시스템 로그·백업.",
 "O": "공개 가능한 문서. 예: 공식 발표된 보도자료, 공개 통계, 일반 공지·안내. 개인정보·기밀 없음.",
}
LANGS = {"ko": "한국어", "en": "영어", "zh": "중국어(简体)", "ja": "일본어"}

def _prompt(grade, lang, i):
    return (f"너는 공공기관 문서를 흉내 내는 합성 데이터 생성기다. **{LANGS[lang]}**로 다음 등급의 문서 1건을 "
            f"자연스럽게 작성하라(제목 + 본문 4~8문장, 다양한 부처·상황·문체).\n등급 설명: {G9[grade]}\n"
            f"실제 개인식별번호는 넣지 말고 필요하면 '○○○'로. 머리말/해설 없이 문서 본문만 출력.")

def _gen(grade, lang, i, deployment, apiver, gpt5=False):
    try:
        c = _gpt5_client() if gpt5 else _azure_client(apiver)
        kw={"model":deployment,"messages":[{"role":"user","content":_prompt(grade,lang,i)}]}
        if gpt5 or deployment.startswith(("o1","o3","o4")): kw["max_completion_tokens"]=1500
        else: kw["max_tokens"]=700; kw["temperature"]=1.0
        return (c.chat.completions.create(**kw).choices[0].message.content or "").strip()
    except Exception as e:
        print("  gen err",str(e)[:80]); return ""

def _norm(t): return re.sub(r"\s+"," ",t).strip()

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=35, help="등급×언어 셀당 문서수")
    ap.add_argument("--langs", default="ko,ko,en,zh,ja")   # ko 가중
    ap.add_argument("--gen", default="gpt-4o")
    ap.add_argument("--teacher", default="o4-mini")
    ap.add_argument("--out-train", default="distill_o4/teacher_labels_n2sf.jsonl")
    ap.add_argument("--out-heldout", default="distill_soft/heldout_n2sf.json")
    ap.add_argument("--heldout-frac", type=float, default=0.2)
    ap.add_argument("--max-hours", type=float, default=3.0)
    args=ap.parse_args(argv)
    _load_env(); distill.TEACHER=args.teacher
    dep=os.environ.get("AZURE_GPT4O_DEPLOYMENT","gpt-4o"); ver=os.environ.get("AZURE_GPT4O_APIVER","2025-01-01-preview")
    langs=[x.strip() for x in args.langs.split(",") if x.strip()]
    import time; deadline=time.time()+args.max_hours*3600
    import random; rng=random.Random(9)
    seen=set(); train=[]; held=[]
    tr=open(args.out_train,"w",encoding="utf-8")
    n=0
    for grade in ("C","S","O"):
        for lang in langs:
            for i in range(args.per_cell):
                if time.time()>deadline: break
                raw=_gen(grade,lang,i,dep,ver)
                raw=re.sub(r"```[a-zA-Z]*","",raw).replace("```","")
                raw=re.sub(r"^#{1,6}\s*","",raw,flags=re.M).strip()
                if not raw or len(raw)<30 or _norm(raw)[:120] in seen: continue
                seen.add(_norm(raw)[:120])
                probs=distill.teacher_soft(raw)     # §9 POLICY로 라벨
                if not probs or sum(probs)<=0: continue
                s=sum(probs); probs=[x/s for x in probs]
                teach=["O","S","C"][max(range(3),key=lambda k:probs[k])]
                rec={"text":raw[:2000],"probs":probs,"gen":f"azure:{args.gen}","lang":lang,
                     "grade":teach,"intended":grade,"n2sf":True}
                n+=1
                if rng.random()<args.heldout_frac:
                    held.append({"text":rec["text"],"grade":teach,"gen":rec["gen"],"lang":lang,"intended":grade})
                else:
                    tr.write(json.dumps(rec,ensure_ascii=False)+"\n"); tr.flush(); train.append(rec)
                if n%30==0: print(f"[gen_n2sf] {n}건 (train {len(train)} / held {len(held)})",flush=True)
    tr.close()
    Path(args.out_heldout).write_text(json.dumps(held,ensure_ascii=False,indent=2),encoding="utf-8")
    from collections import Counter
    print(f"[gen_n2sf] 완료 train={len(train)} held={len(held)}")
    print(f"  train 교사등급: {dict(Counter(r['grade'] for r in train))}")
    print(f"  held  교사등급: {dict(Counter(h['grade'] for h in held))}")
    # 교사-의도 일치율(라벨 품질)
    agree=sum(1 for r in train if r['grade']==r['intended'])/max(len(train),1)
    print(f"  교사=의도 일치율: {agree:.2f} (낮으면 §9 기준 모호/생성부정확)")

if __name__=="__main__":
    main()
