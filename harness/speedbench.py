"""speedbench.py — 등급분류 '속도' 벤치마크(정확도 아님). 용량 10~100MB × 등급 O/S/C.

측정:
  - 티어별 소요시간: T1(규칙: 정규식+키워드), T2(NER: Presidio+spaCy), T3(뉴럴: 배치 추론)
  - 어느 티어에서 등급이 확정되는가(early-exit): C 기밀표지는 T1에서 확정 → 이후 티어 불필요
  - 외부 LLM(o4-mini) 대비 속도: 청크당 지연 실측 → 전수검사 소요 추정(직렬/동시)
전수검사(누락 0): 텍스트를 청크로 쪼개 **모든 티어가 모든 청크를 검사**(특히 100MB). 뉴럴은 배치.
설계: 단일 대용량 문서는 NER/뉴럴이 truncate → 청킹으로 전 텍스트 커버(정확한 속도 측정).
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import data_classifier as dc

# 등급별 단위 콘텐츠(반복해 용량 생성). C는 기밀표지 → T1 early-exit 대상.
UNIT = {
 "O": "본 안내문은 일반 공개 자료입니다. 이번 분기 지역사회 행사와 공개 통계, 채용 공지 등을 안내합니다. "
      "자세한 내용은 공식 홈페이지에서 누구나 확인할 수 있습니다. 많은 관심 바랍니다.\n",
 "S": "내부 검토용 문서. 담당자 홍길동, 연락처 010-1234-5678, 이메일 hong@corp.com, 주소 서울시 강남구. "
      "인사·예산 관련 협의 사항과 시스템 로그 요약을 포함합니다.\n",
 "C": "[대외비] 본 문서는 대외비로 지정되었습니다. 무단 열람·배포 금지. 취급자 외 접근 불가. "
      "대외 정책 검토 회의 안건 및 부처 간 협의 진행 사항 보고.\n",
}

def make_text(grade, mb):
    unit = UNIT[grade]; target = mb * 1024 * 1024
    reps = target // len(unit.encode()) + 1
    return (unit * reps)[: ]  # 대략 target 바이트

def chunks(text, size=4000):
    return [text[i:i+size] for i in range(0, len(text), size)]

def _load_neural(path):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path).eval().to("cpu")
    return tok, mdl

def neural_batch(tok, mdl, texts, bs=64, maxlen=128):
    import torch
    grades=[]; id2=mdl.config.id2label; SHORT={"OPEN":"O","SENSITIVE":"S","CONFIDENTIAL":"C"}
    for i in range(0, len(texts), bs):
        b=texts[i:i+bs]
        enc=tok(b, truncation=True, max_length=maxlen, padding=True, return_tensors="pt")
        with torch.no_grad(): logits=mdl(**enc).logits
        idx=logits.argmax(-1).tolist()
        for k in idx:
            lab=id2.get(k, id2.get(str(k), ["OPEN","SENSITIVE","CONFIDENTIAL"][k]))
            grades.append(SHORT.get(lab, lab))
    return grades

def llm_chunk_latency(sample_chunks, n=4):
    """o4-mini(클라우드) 청크당 분류 지연 실측. 실패 시 None."""
    try:
        from harness import distill
        distill.TEACHER="o4-mini"
        ts=[]
        for c in sample_chunks[:n]:
            t0=time.perf_counter(); p=distill.teacher_soft(c[:6000])
            if p: ts.append(time.perf_counter()-t0)
        return sum(ts)/len(ts) if ts else None
    except Exception:
        return None

def gemma_chunk_latency(sample_chunks, n=4, url="http://localhost:11434/api/chat"):
    """로컬 Gemma(동일 하드웨어) 청크당 분류 지연 실측 — 공정 비교(같은 맥). 실패 시 None."""
    import json as _j, urllib.request
    ts=[]
    for c in sample_chunks[:n]:
        try:
            payload={"model":"gemma2:9b","stream":False,"options":{"temperature":0,"num_predict":8},
                     "messages":[{"role":"user","content":"문서를 O/S/C 중 하나로만 답:\n"+c[:4000]}]}
            req=urllib.request.Request(url, data=_j.dumps(payload).encode(), headers={"Content-Type":"application/json"})
            t0=time.perf_counter(); urllib.request.urlopen(req, timeout=120).read(); ts.append(time.perf_counter()-t0)
        except Exception: pass
    return sum(ts)/len(ts) if ts else None

def bench_size(grade, mb, model_path, llm_lat=None, gemma_lat=None, chunk=4000):
    text=make_text(grade, mb); real_mb=len(text.encode())/1024/1024
    cks=chunks(text, chunk); n=len(cks)
    # T1 규칙(키워드+정규식). 첫 청크에 기밀표지 있으면 early-exit 티어=T1
    t0=time.perf_counter()
    t1_hit=False
    for c in cks:
        kw=dc._scan_keywords(c, [], False)
        if any(f.get("secretFloor") for f in kw): t1_hit=True
    t1=time.perf_counter()-t0
    # T2 NER(Presidio+spaCy) 전수
    t0=time.perf_counter()
    for c in cks: dc.analyze(c, "ko")
    t2=time.perf_counter()-t0
    # T3 뉴럴 전수(배치)
    tok,mdl=_load_neural(model_path)
    t0=time.perf_counter(); neural_batch(tok, mdl, cks); t3=time.perf_counter()-t0
    del tok,mdl
    decide = "T1(규칙·기밀표지)" if t1_hit else "T3(뉴럴)"
    # early-exit 소요: C는 T1만, 그 외 T1+T2+T3(전수)
    ee = t1 if t1_hit else (t1+t2+t3)
    full = t1+t2+t3
    llm_full = (llm_lat*n) if llm_lat else None
    gem_full = (gemma_lat*n) if gemma_lat else None
    return {"grade":grade,"mb":round(real_mb,1),"chunks":n,
            "T1_s":round(t1,3),"T2_s":round(t2,2),"T3_s":round(t3,2),
            "full_s":round(full,2),"earlyexit_s":round(ee,3),"decide":decide,
            "MBps_full":round(real_mb/full,2) if full else 0,
            "gemma_full_s":round(gem_full,1) if gem_full else None,
            "speedup_vs_gemma":round(gem_full/full,1) if gem_full else None,
            "llm_full_s":round(llm_full,1) if llm_full else None,
            "speedup_vs_llm":round(llm_full/full,1) if llm_full else None}

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--grades", default="O,S,C")
    ap.add_argument("--model", default="models/n2sf-small", help="뉴럴(속도: 소형 권장)")
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--llm", action="store_true", help="o4-mini(클라우드) 청크지연 실측 포함")
    ap.add_argument("--gemma", action="store_true", help="로컬 Gemma(동일 HW) 청크지연 실측 포함")
    ap.add_argument("--out", default="final/speed_benchmark.json")
    a=ap.parse_args(argv)
    sizes=[int(x) for x in a.sizes.split(",")]; grades=a.grades.split(",")
    sample=[UNIT["S"]*200]
    llm_lat = llm_chunk_latency(sample) if a.llm else None
    gemma_lat = gemma_chunk_latency(sample) if a.gemma else None
    print(f"[speedbench] model={a.model} chunk={a.chunk} LLM청크={llm_lat} Gemma청크={gemma_lat}")
    rows=[]
    for g in grades:
        for mb in sizes:
            r=bench_size(g, mb, a.model, llm_lat, gemma_lat, a.chunk)
            rows.append(r)
            print(f"  {g} {r['mb']}MB chunks={r['chunks']} T1={r['T1_s']} T2={r['T2_s']} T3={r['T3_s']} "
                  f"full={r['full_s']}s ee={r['earlyexit_s']}s decide={r['decide']} "
                  f"vsGemma={r['speedup_vs_gemma']} vsLLM={r['speedup_vs_llm']}", flush=True)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump({"llm_chunk_latency_s":llm_lat,"gemma_chunk_latency_s":gemma_lat,"rows":rows},
                      open(a.out,"w"), ensure_ascii=False, indent=2)
    print(f"[speedbench] 완료 → {a.out}")

if __name__=="__main__":
    main()
