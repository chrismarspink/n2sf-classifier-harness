"""final_matrix.py — 최종 종합 매트릭스: 전 모델 × (한국어/다국어/속도/크기) + 교사 비교 + 증류 전후.

각 모델 dir을 CPU로 평가:
  - 뉴럴-단독: 한국어 held-out · L5 다국어 (macroF1, C재현율, 언어별 누락)
  - 전체 파이프라인(튜닝 앙상블 soft·뉴럴가중): macroF1, C재현율
  - 크기(MB), 파라미터(M), 지연 p50(ms)
+ LLM 교사 기준선(gpt-4o/o4-mini/gpt-5.4/gemma) 병합
출력: <out>/final_matrix.json, report.html, report.md
"""
from __future__ import annotations

import argparse, datetime as _dt, json, os, time
from pathlib import Path

import data_classifier as dc
from . import metrics as M
from .eval_l5 import _load, eval_set

TUNED = {"tier": {"rules": 0.3, "ner": 0.3, "neural": 4.0}}


def _size_mb(p):
    t = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
    return round(t / 1e6, 1)


def eval_pipeline(model_key, ho):
    rows = []
    for h in ho:
        r = dc.classify_text(h["text"], locale="ko", llm_mode=True, model=model_key,
                             ensemble_method="soft", weights=TUNED)
        rows.append({"true_grade": h["grade"], "pred_grade": r["gradeCode"]})
    return M.compute(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="최종 종합 매트릭스")
    ap.add_argument("--models", required=True, help="쉼표구분 dir 목록")
    ap.add_argument("--ko", default="distill_soft/heldout.json")
    ap.add_argument("--l5", default="distill_soft/heldout_l5.json")
    ap.add_argument("--out", default="final")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ko = json.loads(Path(args.ko).read_text(encoding="utf-8"))
    l5 = json.loads(Path(args.l5).read_text(encoding="utf-8"))
    paths = [p.strip() for p in args.models.split(",") if p.strip() and os.path.exists(p.strip() + "/config.json")]

    results = {}
    for path in paths:
        name = Path(path).name
        print(f"=== {name} ===", flush=True)
        try:
            tok, mdl = _load(path)
            params = round(sum(p.numel() for p in mdl.parameters()) / 1e6)
            # 뉴럴-단독(한국어/L5) + 지연
            t0 = time.perf_counter()
            kom, _ = eval_set(tok, mdl, ko)
            per_doc = (time.perf_counter() - t0) / max(len(ko), 1) * 1000
            l5m, leak = eval_set(tok, mdl, l5)
            del mdl, tok
            # 전체 파이프라인(튜닝)
            dc.NEURAL_BACKENDS[name] = {"label": name, "kind": "finetuned", "model": path, "langs": "multi"}
            pm = eval_pipeline(name, ko)
            results[name] = {
                "params_m": params, "size_mb": _size_mb(path), "latency_ms": round(per_doc, 1),
                "ko_neural_f1": kom["macro_f1"], "ko_neural_crecall": kom["c_recall"],
                "l5_neural_f1": l5m["macro_f1"], "l5_leak": leak,
                "pipe_f1": pm["macro_f1"], "pipe_crecall": pm["c_recall"], "pipe_over": pm["over_rate"],
            }
            print(f"  koN={kom['macro_f1']} l5N={l5m['macro_f1']} pipe={pm['macro_f1']} "
                  f"Crec(pipe)={pm['c_recall']} size={results[name]['size_mb']}MB", flush=True)
        except Exception as e:
            print(f"  {name} 오류: {str(e)[:160]}", flush=True)
        (out / "final_matrix.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 교사 기준선 병합
    teachers = {}
    for f in ["distill_soft/llm_baselines.json", "distill_soft/gpt5_baseline.json"]:
        if os.path.exists(f):
            teachers.update(json.loads(Path(f).read_text(encoding="utf-8")))

    write_md(out, results, teachers)
    (out / "final_matrix.json").write_text(json.dumps({"students": results, "teachers": teachers},
                                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== 완료 → {out}/final_matrix.json, report.md ==")


def write_md(out: Path, results, teachers):
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = ["# 최종 종합 매트릭스 (학생 모델 × 언어/속도/크기 + 교사 비교)", "",
         f"> 갱신 {ts} · 뉴럴-단독 + 전체 파이프라인(튜닝 soft·뉴럴4.0) · held-out 정직 분포", "",
         "## 학생 모델 (증류)", "",
         "| 모델 | 한국어(뉴럴) | 다국어L5(뉴럴) | 전체F1 | 전체C재현율 | 지연ms | 용량MB | params |",
         "|---|---|---|---|---|---|---|---|"]
    for n, m in results.items():
        L.append(f"| {n} | {m['ko_neural_f1']} | {m['l5_neural_f1']} | {m['pipe_f1']} | "
                 f"{m['pipe_crecall']} | {m['latency_ms']} | {m['size_mb']} | {m['params_m']}M |")
    L += ["", "## LLM 교사 기준선 (클라우드/로컬)", "",
          "| 교사 | macroF1 | C재현율 |", "|---|---|---|"]
    for k, v in teachers.items():
        L.append(f"| {k} | {v.get('macro_f1')} | {v.get('c_recall')} |")
    L += ["", "> 학생=온디바이스(로컬 CPU). 교사=개발단계 라벨링용(추론엔 미사용).",
          "> 전 모델 C재현율(전체 파이프라인) 1.0 목표 — 규칙 floor가 누락 차단."]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
