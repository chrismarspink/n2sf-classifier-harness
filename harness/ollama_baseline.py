"""ollama_baseline.py — 비교군: 로컬 LLM(ollama, 예: gemma2:9b)로 동일 텍스트를 등급 분류.

Claude API(키·과금 필요) 대신 **로컬 ollama** 로 LLM 비교군을 구성한다. API 키 불필요, 오프라인.
llm_baseline 과 동일한 N²SF 정책 프롬프트·스키마를 써서 공정 비교하고, 지연·토큰을 측정해
llm_predictions 테이블에 저장(model=ollama 모델명) → 리포트의 LLM_vs_BERT 가 자동 채워진다.

ollama 가 떠 있어야 한다:  `ollama serve` (기본 http://localhost:11434), `ollama pull gemma2:9b`.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import re
import time
import urllib.request
from typing import Dict, List

from .db import DB
from .llm_baseline import POLICY, SCHEMA

OLLAMA_URL = "http://localhost:11434/api/chat"


def _norm_hash(t: str) -> str:
    # 공백 정규화 후 해시 → 같은 문서의 포맷 변이를 1회 호출로 통합(의미 동일).
    return hashlib.md5(re.sub(r"\s+", " ", t).strip().encode("utf-8", "ignore")).hexdigest()


def _post(url: str, payload: dict, timeout: float = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "ignore"))


def classify_ollama(text: str, model: str, url: str = OLLAMA_URL) -> dict:
    payload = {
        "model": model, "stream": False, "format": SCHEMA,   # 구조화 출력(JSON 스키마)
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": POLICY},
            {"role": "user", "content": f"다음 문서를 N²SF 등급으로 분류하라.\n\n<문서>\n{text[:6000]}\n</문서>"},
        ],
    }
    t0 = time.perf_counter()
    try:
        r = _post(url, payload)
        latency = (time.perf_counter() - t0) * 1000
        content = (r.get("message") or {}).get("content", "{}")
        try:
            data = json.loads(content)
            grade = data.get("grade")
        except Exception:
            grade = None
        if grade not in ("O", "S", "C"):     # 관대 파싱(스키마 미준수 대비)
            up = (content or "").upper()
            grade = next((g for g in ("C", "S", "O") if f'"{g}"' in up or f"GRADE: {g}" in up), None)
        return {"grade": grade, "rationale": data.get("rationale", "") if grade else "",
                "latency_ms": round(latency, 1),
                "input_tokens": r.get("prompt_eval_count", 0) or 0,
                "output_tokens": r.get("eval_count", 0) or 0}
    except Exception as exc:
        return {"grade": None, "latency_ms": (time.perf_counter() - t0) * 1000,
                "input_tokens": 0, "output_tokens": 0, "error": str(exc)}


def run_ollama_baseline(db: DB, meta: Dict[tuple, dict], detections: Dict[tuple, dict],
                        split: str = "test", model: str = "gemma2:9b",
                        max_files: int = 200, concurrency: int = 2,
                        url: str = OLLAMA_URL, log=print) -> dict:
    """split 의 파일을 ollama 모델로 분류(동일 텍스트 1회). llm_predictions 저장 + 요약."""
    targets = []
    for (doc_id, fmt), m in meta.items():
        if m["split"] != split:
            continue
        d = detections.get((doc_id, fmt))
        if d:
            targets.append((doc_id, fmt, m["grade"], d["text"]))
    targets = targets[:max_files]
    by_text: Dict[str, List[tuple]] = {}
    for t in targets:
        by_text.setdefault(_norm_hash(t[3]), []).append(t)
    uniq = [v[0] for v in by_text.values()]
    log(f"  ollama({model}) 분류 대상: 파일 {len(targets)} → 고유 텍스트 {len(uniq)} (conc={concurrency})")

    probe = classify_ollama(uniq[0][3], model, url) if uniq else {"error": "empty"}
    if probe.get("error"):
        log(f"  ollama 비교 건너뜀: {probe['error'][:160]}")
        return {"available": False, "reason": probe["error"]}

    results: Dict[str, dict] = {}

    def work(rep):
        th = _norm_hash(rep[3])
        return th, classify_ollama(rep[3], model, url)

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        for th, r in ex.map(work, uniq):
            results[th] = r
    wall = time.perf_counter() - t0

    rows, errs, lat, intok, outtok = [], 0, [], 0, 0
    for th, members in by_text.items():
        r = results.get(th, {"grade": None})
        if r.get("error") or r.get("grade") is None:
            errs += 1
            continue
        lat.append(r["latency_ms"]); intok += r["input_tokens"]; outtok += r["output_tokens"]
        for doc_id, fmt, grade, text in members:
            rows.append({"doc_id": doc_id, "fmt": fmt, "model": model,
                         "true_grade": grade, "llm_grade": r["grade"],
                         "latency_ms": r["latency_ms"], "input_tokens": r["input_tokens"],
                         "output_tokens": r["output_tokens"],
                         "raw": {"rationale": r.get("rationale", "")}})
    db.upsert_many("llm_predictions", rows)
    ls = sorted(lat)
    p50 = ls[len(ls)//2] if ls else 0.0
    p95 = ls[min(len(ls)-1, int(0.95*(len(ls)-1)+0.5))] if ls else 0.0
    summary = {"available": True, "model": model, "files_scored": len(rows),
               "unique_texts": len(uniq), "errors": errs,
               "latency_p50_ms": round(p50, 1), "latency_p95_ms": round(p95, 1),
               "wall_s": round(wall, 1), "input_tokens": intok, "output_tokens": outtok,
               "cache_read_tokens": 0}
    log(f"  ollama 완료: 파일 {len(rows)} 저장, 오류 {errs}, "
        f"지연 p50={summary['latency_p50_ms']}ms p95={summary['latency_p95_ms']}ms")
    return summary
