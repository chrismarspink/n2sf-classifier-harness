"""llm_baseline.py — 비교군: Claude(LLM)로 동일 텍스트를 등급 분류.

목적: GPU 없는 3-tier(정규식+NER+BERT) 모델이 LLM 수준 성능에 얼마나 근접하는지 정량 비교.
- 모델: claude-opus-4-8 (구조화 출력으로 O/S/C 강제, N²SF 정책을 시스템 프롬프트로 캐싱)
- 동일한 **추출 텍스트**를 입력 → 내용 기준 공정 비교(텍스트가 같은 포맷은 1회만 호출해 비용 절감)
- 지연(latency)·토큰 사용량 측정 → 속도/비용 비교 데이터 생성
- API 키/인증이 없으면 우아하게 건너뛴다(하네스 나머지는 정상 동작).

주의: LLM 은 깨끗한 텍스트를 받지만 3-tier 모델은 추출까지 책임진다 — 이 비대칭은 리포트에 명시.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import time
from typing import Dict, List, Optional

from .db import DB

LLM_MODEL = "claude-opus-4-8"

POLICY = """당신은 한국 N²SF(국가 네트워크 보안 프레임워크) 문서 등급 분류기다.
문서 내용을 읽고 다음 3등급 중 하나로 분류한다.

- C (CONFIDENTIAL / 기밀): 강한 개인식별자(주민등록번호, 신용카드번호, 여권번호, 계좌번호,
  API/시크릿 키, AWS 키)가 하나라도 있거나; 개인식별 정보(이름·연락처·이메일·주소 등)가
  대량(약 10건 이상, 명부/내보내기 수준)으로 있거나; '극비/대외비/기밀/Confidential' 같은
  기밀 등급 라벨이 실제 그 문서를 기밀로 지정하는 의미로 쓰인 경우.
- S (SENSITIVE / 민감): 제한적인 개인정보(소수의 이름·연락처·이메일·주소·사업자등록번호)가
  있으나 강식별자도, 대량 PII도, 기밀 라벨도 없는 경우.
- O (OPEN / 공개): 개인식별 정보도 기밀 라벨도 없는 일반 공개 문서. '기밀이 아니다'처럼
  기밀이라는 단어가 부정·일반 문맥으로만 등장하면 공개로 본다.

보안 관점에서 과소분류(실제 기밀을 낮게)는 치명적이므로 강식별자나 대량 PII가 보이면 C로 한다.
반드시 grade 는 'O','S','C' 중 하나로만 답한다."""

SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": "string", "enum": ["O", "S", "C"]},
        "rationale": {"type": "string"},
    },
    "required": ["grade", "rationale"],
    "additionalProperties": False,
}


def get_client():
    try:
        import anthropic
        client = anthropic.Anthropic()
        return client
    except Exception:
        return None


def classify_llm(client, text: str, model: str = LLM_MODEL) -> dict:
    """단일 텍스트 → {grade, rationale, latency_ms, input_tokens, output_tokens, error?}."""
    t0 = time.perf_counter()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            system=[{"type": "text", "text": POLICY, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user",
                       "content": f"다음 문서를 N²SF 등급으로 분류하라.\n\n<문서>\n{text[:6000]}\n</문서>"}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        latency = (time.perf_counter() - t0) * 1000
        txt = next((b.text for b in resp.content if b.type == "text"), "{}")
        data = json.loads(txt)
        return {"grade": data.get("grade", "O"), "rationale": data.get("rationale", ""),
                "latency_ms": round(latency, 1),
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0) or 0}
    except Exception as exc:
        return {"grade": None, "rationale": "", "latency_ms": (time.perf_counter() - t0) * 1000,
                "input_tokens": 0, "output_tokens": 0, "error": str(exc)}


def run_llm_baseline(db: DB, meta: Dict[tuple, dict], detections: Dict[tuple, dict],
                     split: str = "test", model: str = LLM_MODEL,
                     max_files: int = 200, concurrency: int = 8, log=print) -> dict:
    """split 의 파일을 LLM 으로 분류(동일 텍스트는 1회). llm_predictions 저장 + 요약 반환."""
    client = get_client()
    if client is None:
        log("  LLM 비교 건너뜀: anthropic SDK/클라이언트 초기화 실패.")
        return {"available": False, "reason": "no anthropic client"}

    # 대상 파일 수집 + 동일 텍스트 dedup
    targets = []
    for (doc_id, fmt), m in meta.items():
        if m["split"] != split:
            continue
        d = detections.get((doc_id, fmt))
        if not d:
            continue
        targets.append((doc_id, fmt, m["grade"], d["text"]))
    targets = targets[:max_files]
    by_text: Dict[str, List[tuple]] = {}
    for doc_id, fmt, grade, text in targets:
        by_text.setdefault(hashlib.md5(text.encode("utf-8", "ignore")).hexdigest(),
                            []).append((doc_id, fmt, grade, text))

    uniq = [v[0] for v in by_text.values()]   # 텍스트당 대표 1건
    log(f"  LLM 분류 대상: 파일 {len(targets)} → 고유 텍스트 {len(uniq)} (model={model}, conc={concurrency})")

    # 헬스체크 1건(인증/모델 확인)
    probe = classify_llm(client, uniq[0][3], model) if uniq else {"error": "empty"}
    if probe.get("error"):
        log(f"  LLM 비교 건너뜀: 첫 호출 오류 — {probe['error'][:160]}")
        return {"available": False, "reason": probe["error"]}

    results: Dict[str, dict] = {}

    def work(rep):
        doc_id, fmt, grade, text = rep
        th = hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()
        r = classify_llm(client, text, model)
        return th, r

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        for th, r in ex.map(work, uniq):
            results[th] = r
    wall = time.perf_counter() - t0

    # 대표 결과를 동일 텍스트의 모든 파일로 전파 → llm_predictions 저장
    rows = []
    errs = 0
    lat, intok, outtok, cache = [], 0, 0, 0
    for th, members in by_text.items():
        r = results.get(th, {"grade": None})
        if r.get("error") or r.get("grade") is None:
            errs += 1
            continue
        lat.append(r["latency_ms"]); intok += r["input_tokens"]; outtok += r["output_tokens"]
        cache += r.get("cache_read", 0)
        for doc_id, fmt, grade, text in members:
            rows.append({"doc_id": doc_id, "fmt": fmt, "model": model,
                         "true_grade": grade, "llm_grade": r["grade"],
                         "latency_ms": r["latency_ms"],
                         "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                         "raw": {"rationale": r.get("rationale", "")}})
    db.upsert_many("llm_predictions", rows)

    lat_sorted = sorted(lat)
    p50 = lat_sorted[len(lat_sorted)//2] if lat_sorted else 0.0
    p95 = lat_sorted[min(len(lat_sorted)-1, int(0.95*(len(lat_sorted)-1)+0.5))] if lat_sorted else 0.0
    summary = {"available": True, "model": model, "files_scored": len(rows),
               "unique_texts": len(uniq), "errors": errs,
               "latency_p50_ms": round(p50, 1), "latency_p95_ms": round(p95, 1),
               "wall_s": round(wall, 1), "input_tokens": intok, "output_tokens": outtok,
               "cache_read_tokens": cache}
    log(f"  LLM 완료: 파일 {len(rows)} 저장, 오류 {errs}, "
        f"지연 p50={summary['latency_p50_ms']}ms p95={summary['latency_p95_ms']}ms, "
        f"토큰 in={intok} out={outtok} cacheRead={cache}")
    return summary
