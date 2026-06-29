"""metrics.py — 분류 KPI. 보안 맥락의 비대칭(과소분류=유출)을 반영한 안전 우선 목적함수 포함.

지표
----
- 혼동행렬(3×3), 등급별 정밀도/재현율/F1, macro/weighted-F1, 정확도
- under_rate(과소분류율, 실제>예측 — 치명적), over_rate(과대분류율, 실제<예측 — 안전하나 비용)
- c_recall/c_precision (기밀 누락이 핵심 리스크)
- 지연 p50/p95 (GPU 없이 LLM급 속도 증명)
- 포맷별·난이도별 분해 (추출 레이어 결함 분리)

목적함수(objective): macro_f1 를 기준으로, C-recall 이 목표 미달이면 강하게 감점하고
과소분류를 추가 감점, 동률 시 지연이 낮을수록 가점.
"""
from __future__ import annotations

from statistics import median
from typing import Dict, List

RANK = {"O": 0, "S": 1, "C": 2}
GRADES = ["O", "S", "C"]


def _pct(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, int(q * (len(s) - 1) + 0.5))
    return round(s[i], 3)


def compute(rows: List[dict], c_target: float = 0.98) -> dict:
    """rows: [{true_grade, pred_grade, latency_ms?, fmt?, category?}]. 종합 지표 dict."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "objective": 0.0, "macro_f1": 0.0, "accuracy": 0.0,
                "c_recall": 0.0, "c_precision": 0.0, "under_rate": 0.0, "over_rate": 0.0,
                "p50_ms": 0.0, "p95_ms": 0.0, "confusion": {}, "per_grade": {},
                "by_format": {}, "by_category": {}}

    confusion = {t: {p: 0 for p in GRADES} for t in GRADES}
    under = over = correct = 0
    for r in rows:
        t, p = r["true_grade"], r["pred_grade"]
        confusion[t][p] += 1
        if RANK[p] < RANK[t]:
            under += 1
        elif RANK[p] > RANK[t]:
            over += 1
        else:
            correct += 1

    per_grade = {}
    f1s = []
    weighted_f1_num = 0.0
    for g in GRADES:
        tp = confusion[g][g]
        fp = sum(confusion[t][g] for t in GRADES if t != g)
        fn = sum(confusion[g][p] for p in GRADES if p != g)
        support = sum(confusion[g].values())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_grade[g] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": support}
        f1s.append(f1)
        weighted_f1_num += f1 * support

    macro_f1 = sum(f1s) / len(f1s)
    weighted_f1 = weighted_f1_num / n
    accuracy = correct / n
    c_recall = per_grade["C"]["recall"]
    c_precision = per_grade["C"]["precision"]
    under_rate = under / n
    over_rate = over / n

    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    p50 = _pct(lat, 0.50)
    p95 = _pct(lat, 0.95)

    # 포맷별 / 난이도별 정확도 (추출·정밀도 결함 분리)
    def _bucket(key):
        b = {}
        for r in rows:
            k = r.get(key)
            if k is None:
                continue
            d = b.setdefault(k, {"n": 0, "correct": 0, "under": 0, "over": 0})
            d["n"] += 1
            if RANK[r["pred_grade"]] < RANK[r["true_grade"]]:
                d["under"] += 1
            elif RANK[r["pred_grade"]] > RANK[r["true_grade"]]:
                d["over"] += 1
            else:
                d["correct"] += 1
        return {k: {"n": v["n"], "accuracy": round(v["correct"] / v["n"], 4),
                    "under_rate": round(v["under"] / v["n"], 4),
                    "over_rate": round(v["over"] / v["n"], 4)} for k, v in b.items()}

    by_format = _bucket("fmt")
    by_category = _bucket("category")

    # ── 안전 우선 목적함수 ──
    objective = macro_f1
    if c_recall < c_target:
        objective -= 2.0 * (c_target - c_recall)      # 기밀 누락 강한 감점
    objective -= 0.5 * under_rate                      # 과소분류 추가 감점
    # 지연 타이브레이크: p50 1ms 당 1e-5 (동률에서만 영향)
    objective -= 1e-5 * p50

    return {
        "n": n, "objective": round(objective, 5),
        "macro_f1": round(macro_f1, 4), "weighted_f1": round(weighted_f1, 4),
        "accuracy": round(accuracy, 4),
        "c_recall": round(c_recall, 4), "c_precision": round(c_precision, 4),
        "under_rate": round(under_rate, 4), "over_rate": round(over_rate, 4),
        "p50_ms": p50, "p95_ms": p95,
        "confusion": confusion, "per_grade": per_grade,
        "by_format": by_format, "by_category": by_category,
        "c_target": c_target,
    }
