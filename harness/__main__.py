"""harness 파이프라인 진입점.

    python -m harness [--seed 0] [--per-cell 6] [--out harness_out]
                      [--models minilm,ko-sroberta,mdeberta] [--no-llm]
                      [--llm-max 200] [--c-target 0.98]

generate → detect(캐시) → optimize(2계층 탐색) → LLM 비교 → report(Excel + 최적설정 산출)
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import data_classifier as dc
from .corpus import CorpusGen, render_all, ALL_FORMATS
from .db import DB
from . import detect as DET
from . import optimize as OPT
from . import llm_baseline as LLM
from . import report as RPT


def main(argv=None):
    ap = argparse.ArgumentParser(description="data_classifier 자동 평가·최적화 하네스")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-cell", type=int, default=6, help="등급×난이도 셀당 문서 수")
    ap.add_argument("--difficulty", type=int, default=1, help="코퍼스 난이도 1~4(높을수록 적대 케이스 포함)")
    ap.add_argument("--normalize", action="store_true", help="① 전처리 정규화(전각→반각 등) 적용")
    ap.add_argument("--out", default="harness_out", help="작업/산출 디렉터리")
    ap.add_argument("--formats", default=",".join(ALL_FORMATS))
    ap.add_argument("--models", default="minilm,ko-sroberta,mdeberta",
                    help="평가할 뉴럴 백엔드(쉼표). 빈 값이면 규칙+NER 전용")
    ap.add_argument("--locale", default="ko")
    ap.add_argument("--c-target", type=float, default=0.98, help="C 재현율 목표(안전 제약)")
    ap.add_argument("--llm-provider", choices=["none", "claude", "ollama"], default="ollama",
                    help="LLM 비교군: ollama(로컬, 기본)·claude(API키 필요)·none")
    ap.add_argument("--ollama-model", default="gemma2:9b")
    ap.add_argument("--no-llm", action="store_true", help="(=--llm-provider none)")
    ap.add_argument("--llm-max", type=int, default=200, help="LLM 분류 최대 파일 수(test split)")
    ap.add_argument("--llm-conc", type=int, default=8)
    args = ap.parse_args(argv)
    if args.no_llm:
        args.llm_provider = "none"

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out = Path(args.out)
    corpus_dir = out / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    db = DB(out / "results.db")
    run_id = "run_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"== 하네스 시작  run_id={run_id}  out={out} ==")

    # ── 1. 코퍼스 생성 ──
    print("[1/6] 코퍼스 생성")
    gen = CorpusGen(seed=args.seed, locale=args.locale)
    docs = gen.build(per_cell=args.per_cell, difficulty=args.difficulty)
    manifest = render_all(docs, corpus_dir, formats=formats)
    splits = OPT.assign_splits([m["doc_id"] for m in manifest], seed=args.seed + 7)
    n_err = sum(1 for m in manifest if m["render_error"])
    print(f"  문서 {len(docs)}개 → 파일 {len(manifest)}개 (포맷 {formats}), 렌더오류 {n_err}")
    db.upsert("runs", {"run_id": run_id, "created_at": _dt.datetime.now().isoformat(),
                       "notes": "auto", "corpus_seed": args.seed, "per_cell": args.per_cell,
                       "n_docs": len(docs), "n_files": len(manifest),
                       "meta": {"formats": formats, "models": models, "c_target": args.c_target}})
    db.upsert_many("corpus", [{
        "doc_id": m["doc_id"], "fmt": m["fmt"], "path": m["path"], "grade": m["grade"],
        "category": m["category"], "locale": m["locale"], "split": splits.get(m["doc_id"], "test"),
        "text_len": 0, "render_error": m["render_error"], "expected": m["expected"]}
        for m in manifest])

    corpus_rows = db.query("SELECT * FROM corpus")

    # ── 2. 탐지(추출+NER+뉴럴) 캐시 ──
    print("[2/6] 탐지(추출+Presidio NER+뉴럴) — 1회 캐시")
    DET.run_detection(db, corpus_rows, locale=args.locale, models=models, normalize=args.normalize)

    detections = DET.load_detection(db)
    meta = {(r["doc_id"], r["fmt"]): {"grade": r["grade"], "category": r["category"],
                                      "split": r["split"]} for r in corpus_rows}
    print(f"  탐지 캐시 로드: {len(detections)}건")

    # ── 3. 최적화(2계층 탐색) ──
    print("[3/6] 최적화 — 2계층 탐색")
    opt = OPT.run_optimize(db, run_id, detections, meta, locale=args.locale,
                           c_target=args.c_target, models=models)
    best_cfg = opt["best_config"]
    best_cid = opt["best_config_id"]

    # baseline(default) config_id 찾기
    base_rows = db.query("SELECT config_id FROM configs WHERE run_id=? AND kind='baseline' "
                         "AND label LIKE 'default(rules%'", (run_id,))
    default_cid = base_rows[0]["config_id"] if base_rows else None

    # ── 4. 성능(속도) 벤치 ──
    print("[4/6] 성능(속도) 벤치 — end-to-end 지연")
    perf = OPT.bench_latency(detections, best_cfg, args.locale)

    # ── 5. LLM 비교 (ollama 로컬 / Claude API) ──
    print(f"[5/6] LLM 비교군 — provider={args.llm_provider}")
    if args.llm_provider == "none":
        llm_summary = {"available": False, "reason": "none"}
        print("  생략")
    elif args.llm_provider == "ollama":
        from . import ollama_baseline as OLL
        llm_summary = OLL.run_ollama_baseline(db, meta, detections, split="test",
                                              model=args.ollama_model, max_files=args.llm_max,
                                              concurrency=2)
    else:
        llm_summary = LLM.run_llm_baseline(db, meta, detections, split="test",
                                           max_files=args.llm_max, concurrency=args.llm_conc)

    comparison = RPT.build_comparison(db, best_cid, split="test")

    # ── 6. 리포트 + 최적설정 산출 ──
    print("[6/6] 리포트 + 최적 설정 산출")
    def_test = {}
    if default_cid:
        dm = db.query("SELECT detail FROM metrics WHERE config_id=? AND split='test'", (default_cid,))
        def_test = DB.jload(dm[0]["detail"]) if dm else {}
    xlsx_path = out / "report.xlsx"
    RPT.export_excel(db, run_id, best_cid, str(xlsx_path), default_cid, comparison, perf,
                     llm_summary if llm_summary.get("available") else {}, best_cfg)
    files = RPT.export_config_files(out, best_cfg, opt["best_test"], def_test, comparison,
                                    perf, llm_summary if llm_summary.get("available") else {},
                                    dc.NEURAL_BACKENDS)
    db.close()

    print("\n== 완료 ==")
    print(f"  DB:      {out/'results.db'}")
    print(f"  Excel:   {xlsx_path}")
    print(f"  설정:    {files['weights']}, {files['optimized_config']}")
    print(f"  모델정보: {files['model_md']}")
    bt = opt["best_test"]
    print(f"  최적 설정 test: macroF1={bt.get('macro_f1')} acc={bt.get('accuracy')} "
          f"Crec={bt.get('c_recall')} under={bt.get('under_rate')} over={bt.get('over_rate')}")
    if comparison.get("llm_metrics"):
        lm = comparison["llm_metrics"]
        print(f"  LLM       test: macroF1={lm.get('macro_f1')} acc={lm.get('accuracy')} "
              f"Crec={lm.get('c_recall')}  (일치율 {comparison.get('agreement')})")


if __name__ == "__main__":
    main()
