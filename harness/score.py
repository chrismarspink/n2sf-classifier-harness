"""score.py — 캐시된 탐지 결과 위에서 점수·앙상블을 재생(싼 단계, 스윕 대상).

data_classifier 의 **실제 함수**(_aggregate_findings/_scan_keywords/_score/_combine)를 그대로
재사용해, classify_text 와 동일한 결과를 내되 추출·NER·뉴럴 재실행 없이 점수 설정만 바꿔 평가한다.

점수 설정(config) 키 (모두 선택):
    entity        {타입: 가중치}             — 엔티티 점수 가중치 override (예: KR_ACCOUNT 하향)
    keyword       [{keyword,weight,label}]    — 추가 등급 키워드
    thresholds    {confidential, sensitive}   — 등급 임계값
    tier          {rules,ner,neural}          — 앙상블 tier 가중치
    ensemble      escalate|vote|weighted|max-rank|soft
    model         뉴럴 백엔드 키 (llm_mode 일 때)
    llm_mode      bool — 뉴럴 tier 사용
    bulk_threshold int — 대량 PII escalation 임계 (기본 10)
    supersede     {loser_type: [winner_types]} — 겹치는 winner 가 있으면 loser finding 제거
                  (예: {"KR_ACCOUNT": ["KR_PHONE"]} → 전화번호와 겹친 계좌 오탐 제거)
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import data_classifier as dc

_GRADES_FULL = ["OPEN", "SENSITIVE", "CONFIDENTIAL"]


def _apply_supersede(raws: List[dict], rules: Optional[Dict[str, List[str]]]) -> List[dict]:
    if not rules:
        return raws
    out = []
    for r in raws:
        winners = rules.get(r["entity_type"])
        if winners and any(
            (o["entity_type"] in winners and r["start"] < o["end"] and r["end"] > o["start"])
            for o in raws):
            continue
        out.append(r)
    return out


def predict(text: str, presidio: List[dict], neural: Dict[str, dict],
            cfg: dict, locale: str = "ko") -> Tuple[str, float, float, float]:
    """(grade_full, score, confidence, score_ms) — classify_text 의 점수·앙상블 단계 재생."""
    t0 = time.perf_counter()
    entity_overrides = {k: float(v) for k, v in (cfg.get("entity") or {}).items()}
    tier_weights = {**dc.DEFAULT_TIER_WEIGHTS, **(cfg.get("tier") or {})}
    thr = cfg.get("thresholds") or {}
    c_thr = float(thr.get("confidential", dc.C_THRESHOLD))
    s_thr = float(thr.get("sensitive", dc.S_THRESHOLD))
    extra_kw = [(k.get("keyword", ""), float(k.get("weight", 0) or 0), k.get("label", ""))
                for k in (cfg.get("keyword") or []) if k.get("keyword")]
    bulk_thr = int(cfg.get("bulk_threshold", dc.BULK_PII_THRESHOLD))

    raws = _apply_supersede(list(presidio), cfg.get("supersede"))
    pii_findings = dc._aggregate_findings(raws, text, locale, verbose=False)
    kw_findings = dc._scan_keywords(text, extra_kw, verbose=False)
    findings = pii_findings + kw_findings

    score, grade, confidence = dc._score(findings, entity_overrides, c_thr, s_thr)

    bulk_count = sum(f["count"] for f in findings if f["type"] in dc.BULK_PII_TYPES)
    bulk_pii = bulk_count >= bulk_thr
    if bulk_pii and dc.GRADE_RANK[grade] < dc.GRADE_RANK["CONFIDENTIAL"]:
        grade, confidence = "CONFIDENTIAL", max(confidence, 0.9)

    rules_grade = grade if any(f["source"] != "ner" for f in findings) else "OPEN"
    ner_grade = grade if any(f["source"] in ("presidio", "ner") for f in findings) else "OPEN"
    ner_conf = min(1.0, max((f["confidence"] for f in pii_findings), default=0.0))

    neural_result = None
    if cfg.get("llm_mode"):
        model = cfg.get("model", "minilm")
        neural_result = neural.get(model)

    method = cfg.get("ensemble", "escalate")
    if method not in dc.ENSEMBLE_METHODS:
        method = "escalate"
    ensemble = dc._combine(base_grade=grade, base_conf=confidence,
                           tier_grades={"rules": rules_grade, "ner": ner_grade},
                           tier_confs={"rules": confidence, "ner": ner_conf},
                           neural_result=neural_result, method=method, weights=tier_weights)
    grade, confidence = ensemble["grade"], ensemble["confidence"]

    if bulk_pii:                         # 컴플라이언스 floor — 앙상블이 하향 못 함
        grade = "CONFIDENTIAL"

    score_ms = (time.perf_counter() - t0) * 1000
    return grade, round(score, 3), round(confidence, 3), round(score_ms, 4)


def predict_corpus(detections: Dict[tuple, dict], cfg: dict,
                   locale: str = "ko") -> Dict[tuple, dict]:
    """캐시된 탐지 전체에 config 적용 → {(doc_id,fmt): {grade_full, grade, score, conf, ms,...}}."""
    out = {}
    for (doc_id, fmt), d in detections.items():
        g_full, score, conf, ms = predict(d["text"], d["presidio"], d["neural"], cfg, locale)
        out[(doc_id, fmt)] = {
            "grade_full": g_full, "grade": dc.SHORT[g_full],
            "score": score, "confidence": conf, "score_ms": ms,
            # 참고용 누적 지연: 추출 + 탐지(+ 점수). 뉴럴 로드시간은 별도 벤치에서 측정.
            "extract_ms": d.get("extract_ms") or 0.0, "detect_ms": d.get("detect_ms") or 0.0,
        }
    return out
