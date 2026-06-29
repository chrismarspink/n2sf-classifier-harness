"""detect.py — 비싼 탐지 단계(추출 + Presidio NER + 뉴럴)를 1회 실행·캐시.

이 단계만 무겁다(spaCy 로드, 뉴럴 모델 추론). 결과(원시 presidio findings + 모델별 뉴럴 결과)를
detection 테이블에 캐시하면, 이후 점수·앙상블 스윕은 재추론 없이 수천 조합을 돌 수 있다.

성능(속도) 측정: 추출 시간·탐지 시간·뉴럴 시간을 기록한다. 점수 단계 지연은 score.py 에서 잰다.
뉴럴 호출은 동일 텍스트(해시) 간 중복 제거한다(포맷이 달라도 추출 텍스트가 같으면 1회만 추론).
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import data_classifier as dc
from .db import DB

# CPU 친화 뉴럴 백엔드(한국어+다국어, exemplar/zeroshot 다양성). 빈 리스트면 규칙 전용.
DEFAULT_MODELS = ["minilm", "ko-sroberta", "mdeberta"]


def _text_hash(t: str) -> str:
    # 공백 정규화 후 해시 → 같은 문서의 포맷별 추출 변이를 한 뉴럴 호출로 통합(의미는 동일).
    norm = re.sub(r"\s+", " ", t).strip()
    return hashlib.md5(norm.encode("utf-8", "ignore")).hexdigest()


def detect_one(path: str, locale: str, models: List[str],
               neural_cache: Dict[str, dict]) -> dict:
    """단일 파일 → 추출 + presidio + 뉴럴(텍스트해시 캐시). 결과 dict."""
    t0 = time.perf_counter()
    text, ftype, warnings = dc.extract_text(path)
    extract_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    presidio = dc.analyze(text, locale) if text else []
    detect_ms = (time.perf_counter() - t1) * 1000

    th = _text_hash(text)
    if th not in neural_cache:
        neural_cache[th] = {}
    nc = neural_cache[th]
    neural: Dict[str, dict] = {}
    for m in models:
        if m not in nc:
            r = dc.neural_infer(text, locale, m)
            # 추론 시간 측정(모델 로드 후 첫 호출엔 로드시간 포함될 수 있음)
            nc[m] = r
        neural[m] = nc[m]

    return {"text": text, "extract_ms": round(extract_ms, 2),
            "detect_ms": round(detect_ms, 2), "presidio": presidio,
            "neural": neural, "warnings": warnings}


def run_detection(db: DB, corpus_rows: List[dict], locale: str = "ko",
                  models: Optional[List[str]] = None, det_config: str = "base",
                  log=print) -> int:
    """코퍼스 전체 탐지 실행·캐시. 이미 캐시된 (doc,fmt,det_config) 는 건너뜀. 처리 건수 반환."""
    models = DEFAULT_MODELS if models is None else models
    neural_cache: Dict[str, dict] = {}
    done = 0
    n = len(corpus_rows)
    for i, row in enumerate(corpus_rows):
        if row.get("render_error"):
            continue
        cached = db.query(
            "SELECT 1 FROM detection WHERE doc_id=? AND fmt=? AND det_config=?",
            (row["doc_id"], row["fmt"], det_config))
        if cached:
            continue
        try:
            res = detect_one(row["path"], locale, models, neural_cache)
        except Exception as exc:
            log(f"  detect FAIL {row['doc_id']}.{row['fmt']}: {exc}")
            continue
        db.upsert("detection", {
            "doc_id": row["doc_id"], "fmt": row["fmt"], "det_config": det_config,
            "locale": locale, "text": res["text"], "extract_ms": res["extract_ms"],
            "detect_ms": res["detect_ms"], "presidio_json": res["presidio"],
            "neural_json": res["neural"], "warnings": res["warnings"]})
        done += 1
        if done % 25 == 0:
            db.conn.commit()
            log(f"  detected {done} (at {i+1}/{n})")
    db.conn.commit()
    return done


def load_detection(db: DB, det_config: str = "base",
                   doc_prefix: Optional[str] = None) -> Dict[tuple, dict]:
    """(doc_id, fmt) → {text, presidio, neural, extract_ms, detect_ms} 캐시 로드.
    doc_prefix 지정 시 해당 접두 doc_id 만 로드(반복별 스코프)."""
    out = {}
    if doc_prefix:
        rows = db.query("SELECT * FROM detection WHERE det_config=? AND doc_id LIKE ?",
                        (det_config, doc_prefix + "%"))
    else:
        rows = db.query("SELECT * FROM detection WHERE det_config=?", (det_config,))
    for r in rows:
        out[(r["doc_id"], r["fmt"])] = {
            "text": r["text"] or "",
            "presidio": DB.jload(r["presidio_json"], []),
            "neural": DB.jload(r["neural_json"], {}),
            "extract_ms": r["extract_ms"], "detect_ms": r["detect_ms"],
            "warnings": DB.jload(r["warnings"], []),
        }
    return out
