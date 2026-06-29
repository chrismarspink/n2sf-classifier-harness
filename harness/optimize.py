"""optimize.py — 2계층 탐색으로 최적 설정 자동 탐색.

비용 구조에 맞춘 단계적 탐색(맹목적 그리드 X):
  Stage 0  기본 설정(baseline) 평가 — 규칙전용 / 규칙+뉴럴(문서 기본) 비교 기준
  Stage 1  규칙+NER 점수설정 스윕 — supersede(전화↔계좌 오탐 제거)·임계값·KR_ACCOUNT 가중치
  Stage 2  Stage1 최적 위에 뉴럴 백엔드 × 앙상블 방식 스윕
  Stage 3  최적 뉴럴 조합의 tier 가중치 미세조정

valid 스플릿에서 목적함수로 튜닝하고, test 스플릿에서 최종 보고(과적합 방지).
모든 설정의 지표를 DB(configs/metrics)에 기록해 리더보드를 만든다.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import time
from typing import Dict, List, Optional, Tuple

import data_classifier as dc
from . import metrics as M
from . import score as S
from .db import DB


def assign_splits(doc_ids: List[str], seed: int = 7) -> Dict[str, str]:
    """doc_id 단위로 valid/test 50:50 분할(같은 문서의 모든 포맷은 동일 스플릿)."""
    import random
    ids = sorted(set(doc_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    half = len(ids) // 2
    split = {d: "valid" for d in ids[:half]}
    split.update({d: "test" for d in ids[half:]})
    return split


def _cfg_id(run_id: str, cfg: dict) -> str:
    raw = run_id + "|" + json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return "cfg_" + hashlib.md5(raw.encode()).hexdigest()[:14]


def evaluate(detections: Dict[tuple, dict], meta: Dict[tuple, dict], cfg: dict,
             split: str, locale: str, c_target: float) -> Tuple[dict, List[dict]]:
    """config 를 split 에 적용 → (metrics, prediction rows)."""
    preds = S.predict_corpus(detections, cfg, locale)
    rows = []
    for (doc_id, fmt), p in preds.items():
        m = meta.get((doc_id, fmt))
        if not m or m["split"] != split:
            continue
        rows.append({
            "doc_id": doc_id, "fmt": fmt, "category": m["category"],
            "true_grade": m["grade"], "pred_grade": p["grade"],
            "score": p["score"], "confidence": p["confidence"],
            "latency_ms": (p["extract_ms"] + p["detect_ms"] + p["score_ms"]),
            "score_ms": p["score_ms"],
        })
    return M.compute(rows, c_target=c_target), rows


def _record(db: DB, run_id: str, cfg: dict, kind: str, label: str,
            detections, meta, locale, c_target,
            store_preds: bool = False) -> Tuple[str, dict, dict]:
    """config 를 valid·test 평가 후 DB 기록. (config_id, valid_metrics, test_metrics)."""
    cid = _cfg_id(run_id, cfg)
    vm, vrows = evaluate(detections, meta, cfg, "valid", locale, c_target)
    tm, trows = evaluate(detections, meta, cfg, "test", locale, c_target)
    db.upsert("configs", {"config_id": cid, "run_id": run_id, "kind": kind,
                          "label": label, "params": cfg})
    for split, mm in (("valid", vm), ("test", tm)):
        db.upsert("metrics", {
            "run_id": run_id, "config_id": cid, "split": split,
            "objective": mm["objective"], "macro_f1": mm["macro_f1"],
            "accuracy": mm["accuracy"], "c_recall": mm["c_recall"],
            "c_precision": mm["c_precision"], "under_rate": mm["under_rate"],
            "over_rate": mm["over_rate"], "p50_ms": mm["p50_ms"], "p95_ms": mm["p95_ms"],
            "detail": mm})
    if store_preds:
        for split, rows in (("valid", vrows), ("test", trows)):
            db.upsert_many("predictions", [{
                "run_id": run_id, "config_id": cid, "doc_id": r["doc_id"], "fmt": r["fmt"],
                "true_grade": r["true_grade"], "pred_grade": r["pred_grade"],
                "score": r["score"], "confidence": r["confidence"],
                "elapsed_ms": r["latency_ms"], "split": split} for r in rows])
    db.conn.commit()
    return cid, vm, tm


def run_optimize(db: DB, run_id: str, detections: Dict[tuple, dict], meta: Dict[tuple, dict],
                 locale: str = "ko", c_target: float = 0.98,
                 models: Optional[List[str]] = None, log=print) -> dict:
    """2계층 탐색 실행. 최적 설정·기준선 메타 반환."""
    models = models or ["minilm", "ko-sroberta", "mdeberta"]
    # 캐시에 실제로 존재하는 뉴럴 모델만 사용
    avail = set()
    for d in detections.values():
        avail |= set((d.get("neural") or {}).keys())
    models = [m for m in models if m in avail]

    results = []   # (label, cfg, valid_metrics, test_metrics, config_id)

    def consider(cfg, kind, label, store=False):
        cid, vm, tm = _record(db, run_id, cfg, kind, label, detections, meta, locale, c_target, store)
        results.append((label, cfg, vm, tm, cid))
        log(f"  [{kind:8}] {label:42} valid: obj={vm['objective']:.3f} "
            f"macroF1={vm['macro_f1']:.3f} Crec={vm['c_recall']:.3f} "
            f"under={vm['under_rate']:.3f} over={vm['over_rate']:.3f}")
        return vm, tm

    # ── Stage 0: 기준선 ──
    log("Stage 0 — 기준선")
    consider({}, "baseline", "default(rules+NER, no neural)", store=True)
    if models:
        consider({"llm_mode": True, "model": models[0], "ensemble": "escalate"},
                 "baseline", f"default+neural({models[0]},escalate)", store=True)

    # ── Stage 1: 규칙+NER 점수설정 스윕 ──
    log("Stage 1 — 규칙+NER 점수설정 스윕")
    supersede_opts = [None, {"KR_ACCOUNT": ["KR_PHONE"]}]
    acct_opts = [6.0, 2.5, 1.0]                 # KR_ACCOUNT 가중치
    cthr_opts = [5.5, 6.5, 8.0]
    sthr_opts = [0.75, 1.0, 1.5]
    best_s1 = None
    for sup, acct, cthr, sthr in itertools.product(supersede_opts, acct_opts, cthr_opts, sthr_opts):
        cfg = {"thresholds": {"confidential": cthr, "sensitive": sthr},
               "entity": {"KR_ACCOUNT": acct}}
        if sup:
            cfg["supersede"] = sup
        label = f"sup={'Y' if sup else 'N'} acct={acct} C>={cthr} S>={sthr}"
        vm, tm = consider(cfg, "sweep", label)
        if best_s1 is None or vm["objective"] > best_s1[1]["objective"]:
            best_s1 = (cfg, vm, tm)
    log(f"  → Stage1 최적: {best_s1[0]}  obj={best_s1[1]['objective']:.3f}")

    # ── Stage 2: 뉴럴 백엔드 × 앙상블 방식 ──
    best_overall = best_s1
    if models:
        log("Stage 2 — 뉴럴 백엔드 × 앙상블")
        base = dict(best_s1[0])
        for model in models:
            for method in dc.ENSEMBLE_METHODS:
                cfg = dict(base); cfg.update({"llm_mode": True, "model": model, "ensemble": method})
                vm, tm = consider(cfg, "sweep", f"neural={model} ens={method}")
                if vm["objective"] > best_overall[1]["objective"]:
                    best_overall = (cfg, vm, tm)
        log(f"  → Stage2 최적: {best_overall[0]}  obj={best_overall[1]['objective']:.3f}")

        # ── Stage 3: tier 가중치 미세조정 ──
        if best_overall[0].get("llm_mode"):
            log("Stage 3 — tier 가중치 미세조정")
            base = dict(best_overall[0])
            for nw in [0.5, 1.0, 1.5, 2.0]:
                cfg = dict(base); cfg["tier"] = {"rules": 1.0, "ner": 1.0, "neural": nw}
                vm, tm = consider(cfg, "sweep", f"{base.get('model')}/{base.get('ensemble')} neuralW={nw}")
                if vm["objective"] > best_overall[1]["objective"]:
                    best_overall = (cfg, vm, tm)
            log(f"  → Stage3 최적: {best_overall[0]}  obj={best_overall[1]['objective']:.3f}")

    # ── 최적 설정 확정: test 평가 + 예측 저장 ──
    best_cfg, best_vm, best_tm = best_overall
    best_cid, _, best_tm2 = _record(db, run_id, best_cfg, "best", "BEST (optimized)",
                                    detections, meta, locale, c_target, store_preds=True)
    log(f"\n★ 최적 설정: {best_cfg}")
    log(f"   valid: obj={best_vm['objective']:.3f} macroF1={best_vm['macro_f1']:.3f} "
        f"Crec={best_vm['c_recall']:.3f}")
    log(f"   test : macroF1={best_tm2['macro_f1']:.3f} acc={best_tm2['accuracy']:.3f} "
        f"Crec={best_tm2['c_recall']:.3f} under={best_tm2['under_rate']:.3f} over={best_tm2['over_rate']:.3f}")

    return {"best_config": best_cfg, "best_config_id": best_cid,
            "best_valid": best_vm, "best_test": best_tm2,
            "models_available": models, "n_configs": len(results)}


def bench_latency(detections: Dict[tuple, dict], best_cfg: dict, locale: str,
                  sample: int = 40, log=print) -> dict:
    """최적 설정으로 end-to-end 분류 지연을 측정(웜 모델). LLM 대비 속도 비교용."""
    # 모델 워밍업
    items = list(detections.items())[:sample]
    cfg = best_cfg
    # 점수 단계만 따로(추출/탐지는 캐시 시간 사용)
    score_times, total_times = [], []
    # 뉴럴 추론 시간 측정(웜): config 가 뉴럴을 쓰면 모델별 1회 측정 후 평균
    neural_ms = 0.0
    if cfg.get("llm_mode") and items:
        m = cfg.get("model", "minilm")
        txt = items[0][1]["text"]
        dc.neural_infer(txt, locale, m)        # 워밍업
        t = time.perf_counter()
        for _, d in items[:min(10, len(items))]:
            dc.neural_infer(d["text"], locale, m)
        neural_ms = (time.perf_counter() - t) / min(10, len(items)) * 1000
    for (doc_id, fmt), d in items:
        _, _, _, sms = S.predict(d["text"], d["presidio"], d["neural"], cfg, locale)
        score_times.append(sms)
        total_times.append((d.get("extract_ms") or 0) + (d.get("detect_ms") or 0) + sms + neural_ms)
    out = {"sample": len(items),
           "score_p50_ms": M._pct(score_times, 0.5),
           "neural_infer_ms": round(neural_ms, 2),
           "end_to_end_p50_ms": M._pct(total_times, 0.5),
           "end_to_end_p95_ms": M._pct(total_times, 0.95),
           "end_to_end_mean_ms": round(sum(total_times) / len(total_times), 2) if total_times else 0.0}
    log(f"  성능: end-to-end p50={out['end_to_end_p50_ms']}ms p95={out['end_to_end_p95_ms']}ms "
        f"(점수단계 p50={out['score_p50_ms']}ms, 뉴럴추론≈{out['neural_infer_ms']}ms)")
    return out
