"""autoloop.py — 난이도 에스컬레이션 무인 루프 (주말 실행용).

매 반복마다 새 seed 로 데이터를 다시 생성(과적합 방지)하고, 비싼 탐지를 캐시한 뒤
점수×앙상블×BERT백엔드 12종을 스윕해 최적을 찾는다. 모델이 풀면 난이도를 올리고(시험을 어렵게),
못 풀면 그 난이도에 데이터·탐색 자원을 더 투입한다. 모든 결과를 results.db 에 (level,iter) 로 누적.

  python -m harness.autoloop --out weekend --hours 48 \
      --models minilm,mpnet,labse,e5,ko-sroberta,mdeberta,xlmr-xnli,mbert,xlm-roberta,koelectra,klue-roberta,kcbert

중단/재개 가능: 같은 --out 으로 다시 실행하면 iterations 테이블을 읽어 이어서 진행한다.
진행 상황은 out/autoloop_status.json 과 out/report.xlsx(최신 반복)로 언제든 확인.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import time
from pathlib import Path

import data_classifier as dc
from .corpus import CorpusGen, render_all, ALL_FORMATS
from .db import DB
from . import detect as DET
from . import optimize as OPT
from . import report as RPT
from . import summarize as SUM

# CPU 에서 동작 가능한 BERT/임베딩 백엔드(영문전용 deberta-large/bart 제외). 무거운 순서 고려.
FULL_CPU_MODELS = ["minilm", "ko-sroberta", "klue-roberta", "koelectra", "kcbert",
                   "mbert", "xlm-roberta", "mpnet", "e5", "mdeberta", "xlmr-xnli", "labse"]


def _per_model(db: DB, run_id: str) -> dict:
    rows = db.query("""SELECT c.params, v.objective vo FROM configs c
                       JOIN metrics v ON v.config_id=c.config_id AND v.split='valid'
                       WHERE c.run_id=?""", (run_id,))
    out = {}
    for r in rows:
        p = DB.jload(r["params"], {})
        key = p.get("model") if p.get("llm_mode") else "rules+NER"
        if r["vo"] is not None and (key not in out or r["vo"] > out[key]):
            out[key] = round(r["vo"], 4)
    return out


def _rules_only_f1(db: DB, run_id: str) -> float:
    r = db.query("""SELECT t.macro_f1 tf FROM configs c
                    JOIN metrics v ON v.config_id=c.config_id AND v.split='valid'
                    LEFT JOIN metrics t ON t.config_id=c.config_id AND t.split='test'
                    WHERE c.run_id=? AND c.params NOT LIKE '%llm_mode%'
                    ORDER BY v.objective DESC LIMIT 1""", (run_id,))
    return r[0]["tf"] if r and r[0]["tf"] is not None else None


def run_iteration(db: DB, out: Path, level: int, git: int, base_seed: int,
                  per_cell: int, models, locale: str, c_target: float, log=print) -> dict:
    tag = f"L{level}i{git}"
    seed = base_seed + git
    t0 = time.perf_counter()
    log(f"\n=== {tag}  난이도={level} per_cell={per_cell} seed={seed} ===")

    # 1) 생성 (새 seed → 신선 데이터). 생성 파일은 난이도별로 **보존**(weekend/corpus/L{level}/{tag}).
    gen = CorpusGen(seed=seed, locale=locale, tag=tag)
    docs = gen.build(per_cell=per_cell, difficulty=level)
    cdir = out / "corpus" / f"L{level}" / tag
    manifest = render_all(docs, cdir, formats=ALL_FORMATS)
    splits = OPT.assign_splits([m["doc_id"] for m in manifest], seed=seed + 7)
    db.upsert_many("corpus", [{
        "doc_id": m["doc_id"], "fmt": m["fmt"], "path": m["path"], "grade": m["grade"],
        "category": m["category"], "locale": m["locale"], "split": splits.get(m["doc_id"], "test"),
        "text_len": 0, "render_error": m["render_error"], "expected": m["expected"]} for m in manifest])
    corpus_rows = db.query("SELECT * FROM corpus WHERE doc_id LIKE ?", (tag + "-%",))

    # 2) 탐지 (캐시). 생성 파일은 corpus/ 에 보존(요구사항: 생성 데이터 파일 산출).
    DET.run_detection(db, corpus_rows, locale=locale, models=models, log=lambda *_: None)
    detections = DET.load_detection(db, doc_prefix=tag + "-")
    meta = {(r["doc_id"], r["fmt"]): {"grade": r["grade"], "category": r["category"],
                                      "split": r["split"]} for r in corpus_rows}

    # 3) 최적화 (점수×앙상블×모델 스윕)
    opt = OPT.run_optimize(db, tag, detections, meta, locale=locale, c_target=c_target,
                           models=models, log=lambda *_: None)
    bt = opt["best_test"]
    bcfg = opt["best_config"]
    perf = OPT.bench_latency(detections, bcfg, locale, log=lambda *_: None)

    # 4) 기록
    per_model = _per_model(db, tag)
    ro_f1 = _rules_only_f1(db, tag)
    elapsed = time.perf_counter() - t0
    db.upsert("iterations", {
        "run_id": tag, "level": level, "iteration": git, "seed": seed,
        "n_docs": len(docs), "n_files": len(manifest),
        "best_config": bcfg, "best_model": bcfg.get("model") if bcfg.get("llm_mode") else "rules+NER",
        "best_ensemble": bcfg.get("ensemble", "escalate") if bcfg.get("llm_mode") else "-",
        "valid_obj": opt["best_valid"]["objective"], "test_macro_f1": bt.get("macro_f1"),
        "test_accuracy": bt.get("accuracy"), "test_c_recall": bt.get("c_recall"),
        "test_under": bt.get("under_rate"), "test_over": bt.get("over_rate"),
        "rules_only_f1": ro_f1, "per_model": per_model, "elapsed_s": round(elapsed, 1),
        "created_at": _dt.datetime.now().isoformat()})
    db.conn.commit()

    log(f"  최적: model={bcfg.get('model') if bcfg.get('llm_mode') else 'rules+NER'} "
        f"ens={bcfg.get('ensemble','-')}  test F1={bt.get('macro_f1')} acc={bt.get('accuracy')} "
        f"Crec={bt.get('c_recall')} over={bt.get('over_rate')} | rules-only F1={ro_f1} "
        f"| {perf.get('end_to_end_p50_ms')}ms | {elapsed:.0f}s")
    log(f"  모델별 best valid obj: " + ", ".join(f"{k}={v}" for k, v in sorted(per_model.items(), key=lambda x: -x[1])))

    return {"tag": tag, "best_test": bt, "best_cfg": bcfg, "perf": perf,
            "per_model": per_model, "rules_only_f1": ro_f1, "opt": opt}


def main(argv=None):
    ap = argparse.ArgumentParser(description="난이도 에스컬레이션 무인 루프(주말 실행)")
    ap.add_argument("--out", default="weekend")
    ap.add_argument("--hours", type=float, default=48.0, help="벽시계 예산(시간)")
    ap.add_argument("--models", default=",".join(FULL_CPU_MODELS))
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--locale", default="ko")
    ap.add_argument("--c-target", type=float, default=0.98)
    ap.add_argument("--target-f1", type=float, default=0.97, help="이상+무유출이면 즉시 난이도 상승")
    ap.add_argument("--floor-f1", type=float, default=0.90, help="미만이면 데이터·탐색 자원 추가(난이도 유지)")
    ap.add_argument("--patience", type=int, default=3,
                    help="같은 난이도 N회 반복 후엔(무유출 & F1>=floor) 난이도 상승 — L1 정체 방지")
    ap.add_argument("--start-level", type=int, default=1)
    ap.add_argument("--max-level", type=int, default=4)
    ap.add_argument("--max-per-cell", type=int, default=24)
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    db = DB(out / "results.db")
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # 재개: 기존 iterations 로부터 level / 전역 반복 인덱스 복원
    prev = db.query("SELECT * FROM iterations ORDER BY created_at DESC LIMIT 1")
    git = db.query("SELECT COUNT(*) n FROM iterations")[0]["n"]
    level = prev[0]["level"] if prev else args.start_level
    per_cell = args.per_cell
    deadline = time.time() + args.hours * 3600
    print(f"== autoloop 시작  out={out}  예산={args.hours}h  모델 {len(models)}종  "
          f"시작레벨={level} (재개 git={git}) ==")

    while time.time() < deadline:
        try:
            res = run_iteration(db, out, level, git, args.seed, per_cell, models,
                                args.locale, args.c_target)
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  반복 {git} 실패: {exc} — 다음으로")
            git += 1
            continue

        # 최신 반복 리포트 갱신(언제든 확인 가능)
        try:
            tag = res["tag"]
            best_cid = res["opt"]["best_config_id"]
            comparison = RPT.build_comparison(db, best_cid, split="test")
            base = db.query("SELECT config_id FROM configs WHERE run_id=? AND kind='baseline' "
                            "AND label LIKE 'default(rules%'", (tag,))
            def_cid = base[0]["config_id"] if base else None
            def_test = {}
            if def_cid:
                dm = db.query("SELECT detail FROM metrics WHERE config_id=? AND split='test'", (def_cid,))
                def_test = DB.jload(dm[0]["detail"]) if dm else {}
            RPT.export_excel(db, tag, best_cid, str(out / "report.xlsx"), def_cid,
                             comparison, res["perf"], {}, res["best_cfg"])
            RPT.export_config_files(out, res["best_cfg"], res["best_test"], def_test,
                                    comparison, res["perf"], {}, dc.NEURAL_BACKENDS)
        except Exception as exc:
            print(f"  리포트 갱신 경고: {exc}")

        # 주말 종합 요약 갱신(난이도별 최적·모델별 집계·추천 3-tier 설정)
        try:
            SUM.write_summary(db, out)
        except Exception as exc:
            print(f"  요약 갱신 경고: {exc}")

        # 발표용 HTML 대시보드 갱신(매 반복 최신화 → 월요일 추가 명령 불필요)
        try:
            from . import visualize as VIZ
            content = VIZ.build_content(db, None)
            (out / "report.html").write_text(
                '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>N²SF 등급 분류 — 주말 자동 최적화 결과</title>'
                '<style>body{margin:0;background:#F6F7F9}</style></head><body>'
                + content + '</body></html>', encoding="utf-8")
        except Exception as exc:
            print(f"  HTML 갱신 경고: {exc}")

        # 상태 파일
        traj = db.query("SELECT run_id,level,iteration,best_model,best_ensemble,"
                        "test_macro_f1,test_over,rules_only_f1,elapsed_s FROM iterations "
                        "ORDER BY created_at")
        (out / "autoloop_status.json").write_text(json.dumps({
            "updated": _dt.datetime.now().isoformat(), "level": level, "git": git,
            "per_cell": per_cell, "hours_left": round((deadline - time.time()) / 3600, 2),
            "trajectory": traj[-30:]}, ensure_ascii=False, indent=2), encoding="utf-8")

        # 에스컬레이션 결정 (patience 로 정체 방지: 같은 난이도 N회면 무유출·F1>=floor 시 진급)
        f1 = res["best_test"].get("macro_f1") or 0.0
        under = res["best_test"].get("under_rate") or 0.0
        lvl_iters = db.query("SELECT COUNT(*) n FROM iterations WHERE level=?", (level,))[0]["n"]
        strong = (f1 >= args.target_f1 and under == 0.0)
        patient = (lvl_iters >= args.patience and under == 0.0 and f1 >= args.floor_f1)
        if strong or patient:
            if level < args.max_level:
                why = "충분히 풂" if strong else f"{lvl_iters}회 반복(patience)"
                level += 1; per_cell = args.per_cell
                print(f"  → {why}(F1={f1}). 난이도 상승 → L{level}")
            else:
                per_cell = min(args.max_per_cell, per_cell + 4)
                print(f"  → 최고 난이도 도달(F1={f1}). 데이터 확대 per_cell={per_cell}")
        elif f1 < args.floor_f1 or under > 0.0:
            per_cell = min(args.max_per_cell, per_cell + 4)
            print(f"  → 어려움(F1={f1}, under={under}). 자원 집중: per_cell={per_cell} (L{level} 유지)")
        else:
            print(f"  → 미세조정(F1={f1}, L{level} {lvl_iters}/{args.patience}회). L{level} 유지")
        git += 1

    db.close()
    print(f"== autoloop 종료 (예산 소진). 총 반복 {git}. 결과: {out/'results.db'}, {out/'report.xlsx'} ==")


if __name__ == "__main__":
    main()
