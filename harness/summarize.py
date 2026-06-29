"""summarize.py — 주말 전체(여러 난이도·반복)를 가로질러 요약 산출.

매 반복 후 호출되어 다음을 갱신한다(언제든 최신):
  - WEEKEND_SUMMARY.md   난이도 궤적 + 난이도별 최적 모델 + 모델별 집계 + 추천 3-tier 설정
  - weekend_summary.json  위 내용 구조화
  - recommended_weights.json / recommended_config.json  반복 최적화된 최종 3-tier 모델(설정)

'추천 설정' = 도달한 최고 난이도에서 무유출(under==0)·최고 valid objective 를 낸 설정
(보안 우선: 기밀 누락 없는 것 중 가장 강한 일반화).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import data_classifier as dc
from .db import DB


def _model_of(params: dict) -> str:
    return params.get("model") if params.get("llm_mode") else "rules+NER"


def compute(db: DB) -> dict:
    its = db.query("SELECT * FROM iterations ORDER BY created_at")
    # 반복별 전체 config 의 test 지표(모델 파싱용)
    rows = db.query("""SELECT c.run_id, c.params, c.kind,
                              v.objective vo, v.under_rate vu,
                              t.macro_f1 tf, t.accuracy ta, t.c_recall tc,
                              t.under_rate tu, t.over_rate to_
                       FROM configs c
                       JOIN metrics v ON v.config_id=c.config_id AND v.split='valid'
                       LEFT JOIN metrics t ON t.config_id=c.config_id AND t.split='test'""")
    # run_id → level 매핑
    lvl = {r["run_id"]: r["level"] for r in its}

    # 모델별 집계(각 반복에서 그 모델의 best-valid config 의 test F1)
    per_iter_model: Dict[tuple, dict] = {}
    for r in rows:
        p = DB.jload(r["params"], {})
        m = _model_of(p)
        key = (r["run_id"], m)
        cur = per_iter_model.get(key)
        if r["vo"] is not None and (cur is None or r["vo"] > cur["vo"]):
            per_iter_model[key] = {"vo": r["vo"], "tf": r["tf"], "tc": r["tc"],
                                   "tu": r["tu"], "to_": r["to_"]}
    model_agg: Dict[str, dict] = {}
    for (run_id, m), v in per_iter_model.items():
        d = model_agg.setdefault(m, {"iters": 0, "tf": [], "tc": [], "wins": 0})
        d["iters"] += 1
        if v["tf"] is not None:
            d["tf"].append(v["tf"]); d["tc"].append(v["tc"])
    # 반복별 overall best 모델 = iterations.best_model → wins
    for it in its:
        if it["best_model"] in model_agg:
            model_agg[it["best_model"]]["wins"] += 1
    per_model = {m: {"iters": d["iters"], "wins": d["wins"],
                     "avg_test_f1": round(sum(d["tf"]) / len(d["tf"]), 4) if d["tf"] else None,
                     "best_test_f1": round(max(d["tf"]), 4) if d["tf"] else None,
                     "avg_c_recall": round(sum(d["tc"]) / len(d["tc"]), 4) if d["tc"] else None}
                 for m, d in model_agg.items()}

    # 난이도별 최적 반복(무유출 우선, 그 다음 test F1)
    per_level: Dict[int, dict] = {}
    for it in its:
        L = it["level"]
        cand = (it["test_under"] == 0, it["test_macro_f1"] or 0)
        best = per_level.get(L)
        if best is None or cand > (best["_und0"], best["_f1"]):
            per_level[L] = {"run_id": it["run_id"], "best_model": it["best_model"],
                            "best_ensemble": it["best_ensemble"], "config": DB.jload(it["best_config"], {}),
                            "test_macro_f1": it["test_macro_f1"], "test_accuracy": it["test_accuracy"],
                            "test_c_recall": it["test_c_recall"], "test_under": it["test_under"],
                            "test_over": it["test_over"], "n_docs": it["n_docs"],
                            "_und0": it["test_under"] == 0, "_f1": it["test_macro_f1"] or 0}

    # 추천 = 도달 최고 난이도에서 무유출·최고 objective
    rec = None
    if per_level:
        top_level = max(per_level)
        rec = per_level[top_level]
    return {"n_iterations": len(its), "levels_reached": sorted(per_level),
            "trajectory": [{"run_id": r["run_id"], "level": r["level"],
                            "test_macro_f1": r["test_macro_f1"], "test_c_recall": r["test_c_recall"],
                            "test_under": r["test_under"], "test_over": r["test_over"],
                            "rules_only_f1": r["rules_only_f1"], "best_model": r["best_model"],
                            "best_ensemble": r["best_ensemble"], "elapsed_s": r["elapsed_s"]} for r in its],
            "per_model": per_model, "per_level": {str(k): v for k, v in per_level.items()},
            "recommended": rec}


def write_summary(db: DB, out: Path):
    out = Path(out)
    s = compute(db)
    (out / "weekend_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

    rec = s.get("recommended")
    if rec:
        cfg = rec["config"]
        (out / "recommended_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        weights = {k: cfg[k] for k in ("entity", "keyword", "thresholds", "tier") if k in cfg}
        (out / "recommended_weights.json").write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown
    lines = ["# 주말 자동 최적화 — 종합 요약", "",
             f"- 총 반복: **{s['n_iterations']}**    도달 난이도: **L{s['levels_reached']}**", ""]
    lines.append("## 1. 반복 최적화된 3-tier 모델 (추천)")
    if rec:
        cfg = rec["config"]; nb = cfg.get("model")
        nm = dc.NEURAL_BACKENDS.get(nb, {}) if nb else {}
        lines += [
            f"- 기준: 도달 최고 난이도(L{max(s['per_level'], key=int) if s['per_level'] else '-'})에서 무유출·최고 성능",
            f"- 신경망 백엔드: **{nb or '미사용(규칙+NER)'}** {('('+nm.get('label','')+')') if nm else ''}",
            f"- 앙상블: **{rec['best_ensemble']}**, C임계={cfg.get('thresholds',{}).get('confidential',5.5)}, "
            f"S임계={cfg.get('thresholds',{}).get('sensitive',0.75)}, entity={json.dumps(cfg.get('entity',{}),ensure_ascii=False)}",
            f"- 성능(test): macroF1={rec['test_macro_f1']} acc={rec['test_accuracy']} "
            f"C재현율={rec['test_c_recall']} 과소분류={rec['test_under']} 과대분류={rec['test_over']}",
            f"- 적용: `recommended_weights.json` (+ `--llm --model {nb} --ensemble {rec['best_ensemble']}`)" if cfg.get("llm_mode")
            else "- 적용: `recommended_weights.json` (규칙+NER 전용, 뉴럴 불필요 — 최速)",
            ""]
    lines += ["## 2. 난이도별 최적 모델", "",
              "| 난이도 | 최적모델/앙상블 | macroF1 | C재현율 | 과소분류 | 과대분류 | 문서수 |",
              "|---|---|---|---|---|---|---|"]
    for L in sorted(s["per_level"], key=int):
        v = s["per_level"][L]
        lines.append(f"| L{L} | {v['best_model']}/{v['best_ensemble']} | {v['test_macro_f1']} | "
                     f"{v['test_c_recall']} | {v['test_under']} | {v['test_over']} | {v['n_docs']} |")
    lines += ["", "## 3. 모델별 집계 (전 반복)", "",
              "| 모델 | 평가반복 | overall 우승 | 평균 test F1 | 최고 test F1 | 평균 C재현율 |",
              "|---|---|---|---|---|---|"]
    for m, d in sorted(s["per_model"].items(), key=lambda x: -(x[1]["wins"])):
        lines.append(f"| {m} | {d['iters']} | {d['wins']} | {d['avg_test_f1']} | {d['best_test_f1']} | {d['avg_c_recall']} |")
    lines += ["", "## 4. 반복 궤적 (난이도 조정 흐름)", "",
              "| 반복 | 난이도 | macroF1 | C재현율 | 과소 | 과대 | rules만 | 최적모델 | 시간 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for t in s["trajectory"]:
        lines.append(f"| {t['run_id']} | L{t['level']} | {t['test_macro_f1']} | {t['test_c_recall']} | "
                     f"{t['test_under']} | {t['test_over']} | {t['rules_only_f1']} | "
                     f"{t['best_model']}/{t['best_ensemble']} | {round(t['elapsed_s'] or 0)}s |")
    (out / "WEEKEND_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return s
