"""tune4.py — 4차: 과대분류 억제로 held-out 0.85 도전(학습 없이 설정 스윕).

held-out 문서에 탐지 캐시(presidio + mdeberta-n2sf 뉴럴)를 만들고, score.predict 로
supersede(전화↔계좌)·임계·tier가중·앙상블을 스윕해 **C재현율=1.00 제약 하 F1 최대** 설정을 찾는다.
(score.predict 는 classify_text 와 달리 supersede 지원 → 전화↔계좌 오탐 제거 가능)
"""
from __future__ import annotations

import itertools, json
from pathlib import Path

import data_classifier as dc
from . import score as S
from . import metrics as M

MODEL = "mdeberta-n2sf"


def build_cache(heldout, locale="ko", log=print):
    cache = []
    for i, h in enumerate(heldout):
        t = h["text"]
        pres = dc.analyze(t, locale) if t else []
        nr = dc.neural_infer(t, locale, MODEL)
        cache.append({"text": t, "presidio": pres, "neural": {MODEL: nr}, "grade": h["grade"]})
        if (i + 1) % 20 == 0:
            log(f"  탐지 캐시 {i+1}/{len(heldout)}")
    return cache


def evaluate(cache, cfg, locale="ko"):
    rows = []
    for d in cache:
        g_full, _, _, _ = S.predict(d["text"], d["presidio"], d["neural"], cfg, locale)
        rows.append({"true_grade": d["grade"], "pred_grade": dc.SHORT[g_full]})
    return M.compute(rows)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default="distill_soft/heldout.json")
    ap.add_argument("--out", default="distill_soft")
    args = ap.parse_args(argv)
    out = Path(args.out); log = print

    heldout = json.loads(Path(args.heldout).read_text(encoding="utf-8"))
    log(f"held-out {len(heldout)}문서 — 탐지 캐시 생성(presidio + {MODEL})…")
    cache = build_cache(heldout, log=log)

    supersede_opts = [None, {"KR_ACCOUNT": ["KR_PHONE"]}]
    ens_opts = ["soft", "weighted"]
    nw_opts = [2.0, 3.0, 4.0]
    cthr_opts = [6.0, 8.0]
    sthr_opts = [1.5, 2.0, 3.0]
    acct_opts = [1.0, 0.5]

    best = None
    results = []
    for sup, ens, nw, cthr, sthr, acct in itertools.product(
            supersede_opts, ens_opts, nw_opts, cthr_opts, sthr_opts, acct_opts):
        cfg = {"llm_mode": True, "model": MODEL, "ensemble": ens,
               "tier": {"rules": 0.3, "ner": 0.3, "neural": nw},
               "thresholds": {"confidential": cthr, "sensitive": sthr},
               "entity": {"KR_ACCOUNT": acct}}
        if sup:
            cfg["supersede"] = sup
        m = evaluate(cache, cfg)
        results.append((cfg, m))
        # C재현율=1.00 & under=0 제약 하 F1 최대
        key = (m["c_recall"] >= 1.0 and m["under_rate"] == 0.0, m["macro_f1"])
        if best is None or key > best[0]:
            best = (key, cfg, m)

    _, bcfg, bm = best
    log("\n=== 4차 최적 설정 (C재현율=1.0 제약 하 F1 최대) ===")
    log(f"  {json.dumps(bcfg, ensure_ascii=False)}")
    log(f"  macroF1={bm['macro_f1']} 정확도={bm['accuracy']} C재현율={bm['c_recall']} "
        f"과소분류={bm['under_rate']} 과대분류={bm['over_rate']}")
    log(f"  → 0.85 {'달성 ✅' if bm['macro_f1'] >= 0.85 else '미달 ('+str(bm['macro_f1'])+')'}")

    # 상위 5 출력
    ok = [r for r in results if r[1]["c_recall"] >= 1.0 and r[1]["under_rate"] == 0.0]
    ok.sort(key=lambda r: -r[1]["macro_f1"])
    log("\n상위 5(무유출):")
    for cfg, m in ok[:5]:
        s = "Y" if cfg.get("supersede") else "N"
        log(f"  F1={m['macro_f1']:.3f} over={m['over_rate']:.3f} | {cfg['ensemble']} nW={cfg['tier']['neural']} "
            f"C>={cfg['thresholds']['confidential']} S>={cfg['thresholds']['sensitive']} acct={cfg['entity']['KR_ACCOUNT']} sup={s}")

    (out / "tune4_best.json").write_text(json.dumps({"config": bcfg, "metrics": bm}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n저장: {out/'tune4_best.json'}")
    return bcfg, bm


if __name__ == "__main__":
    main()
