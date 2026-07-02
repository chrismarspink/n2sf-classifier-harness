"""lineup.py — 다중 백본 증류·평가: 스몰/베이스/라지(KLUE, XLM-R) + 두 라지 앙상블.

같은 교사 라벨(soft-label)로 백본 4종을 증류하고, 동일 held-out에서 **3축**으로 평가한다:
  ① 정확도(neural-alone macro-F1, 기밀 재현율)  ② 용량(디스크MB, 파라미터M)  ③ 속도(CPU ms/문서)
+ 라지 2종 앙상블(확률 평균)도 평가. 결과 → 선택 매트릭스(JSON/HTML/MD).

neural-alone 기준(모델 자체 분별력) — 실서비스는 여기에 3-tier 규칙 floor(누락 방지)가 더해진다.
"""
from __future__ import annotations

import argparse, datetime as _dt, json, os, re, subprocess, sys, time
from pathlib import Path

from . import metrics as M

SHORT = {"OPEN": "O", "SENSITIVE": "S", "CONFIDENTIAL": "C"}

BACKBONES = [
    {"name": "n2sf-small",      "base": "monologg/koelectra-small-v3-discriminator",
     "batch": 32, "accum": 1, "maxlen": 128, "top": 0, "epochs": 3, "hours": 0.4},
    {"name": "n2sf-base",       "base": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
     "batch": 16, "accum": 2, "maxlen": 128, "top": 0, "epochs": 3, "hours": 0.7},
    {"name": "n2sf-klue-large", "base": "klue/roberta-large",
     "batch": 8,  "accum": 2, "maxlen": 128, "top": 6, "epochs": 3, "hours": 1.3},
    {"name": "n2sf-xlmr-large", "base": "xlm-roberta-large",
     "batch": 6,  "accum": 3, "maxlen": 128, "top": 6, "epochs": 2, "hours": 1.6},
]


def _dir_size_mb(path):
    t = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            t += os.path.getsize(os.path.join(r, f))
    return round(t / 1e6, 1)


def eval_model(path, heldout, log=print):
    """CPU에서 neural-alone 평가 → (metrics, 지연p50ms, 디스크MB, 파라미터M, per-doc probs[3])."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path); mdl.eval().to("cpu")
    id2 = mdl.config.id2label
    params = round(sum(p.numel() for p in mdl.parameters()) / 1e6)
    # 인덱스→등급(O/S/C). id2label 값이 OPEN/SENSITIVE/CONFIDENTIAL 라고 가정.
    def gidx(i):
        lab = id2.get(i, id2.get(str(i), ["OPEN", "SENSITIVE", "CONFIDENTIAL"][i]))
        return SHORT.get(lab, lab)
    rows, lat, probs_all = [], [], []
    # 워밍업
    with torch.no_grad():
        mdl(**tok(["워밍업"], return_tensors="pt", truncation=True, max_length=64))
    for h in heldout:
        t0 = time.perf_counter()
        enc = tok([h["text"]], truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits[0]
        lat.append((time.perf_counter() - t0) * 1000)
        p = torch.softmax(logits, -1).tolist()
        # index→(O,S,C) 정렬
        by_grade = {gidx(i): p[i] for i in range(len(p))}
        vec = [by_grade.get("O", 0), by_grade.get("S", 0), by_grade.get("C", 0)]
        probs_all.append(vec)
        pred = ["O", "S", "C"][int(max(range(3), key=lambda k: vec[k]))]
        rows.append({"true_grade": h["grade"], "pred_grade": pred})
    m = M.compute(rows)
    lat.sort()
    return m, round(lat[len(lat)//2], 1), _dir_size_mb(path), params, probs_all


def eval_ensemble(probs_a, probs_b, heldout):
    rows = []
    for i, h in enumerate(heldout):
        v = [(probs_a[i][k] + probs_b[i][k]) / 2 for k in range(3)]
        pred = ["O", "S", "C"][int(max(range(3), key=lambda k: v[k]))]
        rows.append({"true_grade": h["grade"], "pred_grade": pred})
    return M.compute(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="다중 백본 증류·평가 라인업")
    ap.add_argument("--labels", default="distill_o4/teacher_labels.jsonl")
    ap.add_argument("--heldout", default="distill_soft/heldout.json")
    ap.add_argument("--out", default="lineup")
    ap.add_argument("--only", default="", help="특정 백본만(쉼표) — 재실행용")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = print
    heldout = json.loads(Path(args.heldout).read_text(encoding="utf-8"))
    log(f"== lineup 시작 교사라벨={args.labels} held-out={len(heldout)} ==")

    only = set(x.strip() for x in args.only.split(",") if x.strip())
    results = {}
    probs = {}
    for bb in BACKBONES:
        if only and bb["name"] not in only:
            continue
        mpath = f"models/{bb['name']}"
        # 증류(없으면)
        if not os.path.exists(f"{mpath}/config.json"):
            log(f"\n=== 증류: {bb['name']} ({bb['base']}) ===")
            cmd = [sys.executable, "-m", "harness.train_soft", "--soft-jsonl", args.labels,
                   "--base", bb["base"], "--out", mpath, "--train-top", str(bb["top"]),
                   "--epochs", str(bb["epochs"]), "--batch", str(bb["batch"]),
                   "--accum", str(bb["accum"]), "--maxlen", str(bb["maxlen"]),
                   "--max-hours", str(bb["hours"])]
            try:
                subprocess.run(cmd, check=False, timeout=bb["hours"] * 3600 + 900)
            except Exception as e:
                log(f"  {bb['name']} 학습 경고: {str(e)[:120]}")
        if not os.path.exists(f"{mpath}/config.json"):
            log(f"  {bb['name']} 모델 없음 — 스킵"); continue
        # 평가
        log(f"  평가: {bb['name']} …")
        try:
            m, p50, size, params, pr = eval_model(mpath, heldout, log=log)
            results[bb["name"]] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                                   "c_recall": m["c_recall"], "under_rate": m["under_rate"],
                                   "over_rate": m["over_rate"], "latency_p50_ms": p50,
                                   "size_mb": size, "params_m": params}
            probs[bb["name"]] = pr
            log(f"    F1={m['macro_f1']} Crec={m['c_recall']} p50={p50}ms size={size}MB params={params}M")
        except Exception as e:
            log(f"  {bb['name']} 평가 오류: {str(e)[:160]}")
        # 매 백본 후 중간 저장
        (out / "matrix.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 라지 2종 앙상블
    if "n2sf-klue-large" in probs and "n2sf-xlmr-large" in probs:
        em = eval_ensemble(probs["n2sf-klue-large"], probs["n2sf-xlmr-large"], heldout)
        sz = results["n2sf-klue-large"]["size_mb"] + results["n2sf-xlmr-large"]["size_mb"]
        lt = results["n2sf-klue-large"]["latency_p50_ms"] + results["n2sf-xlmr-large"]["latency_p50_ms"]
        results["n2sf-large-ensemble(KLUE+XLMR)"] = {
            "macro_f1": em["macro_f1"], "accuracy": em["accuracy"], "c_recall": em["c_recall"],
            "under_rate": em["under_rate"], "over_rate": em["over_rate"],
            "latency_p50_ms": round(lt, 1), "size_mb": round(sz, 1),
            "params_m": results["n2sf-klue-large"]["params_m"] + results["n2sf-xlmr-large"]["params_m"]}
        log(f"  앙상블(KLUE+XLMR): F1={em['macro_f1']} Crec={em['c_recall']}")

    (out / "matrix.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out, results)
    log(f"\n== lineup 완료 → {out}/matrix.json, {out}/report.html ==")


def write_report(out: Path, results):
    bl = {}
    p = Path("distill_soft/llm_baselines.json")
    if p.exists():
        bl = json.loads(p.read_text(encoding="utf-8"))
    rows = ""
    for name, m in results.items():
        rows += (f"<tr><td>{name}</td><td class='n'>{m['macro_f1']}</td><td class='n'>{m['c_recall']}</td>"
                 f"<td class='n'>{m['over_rate']}</td><td class='n'>{m.get('latency_p50_ms','-')}</td>"
                 f"<td class='n'>{m.get('size_mb','-')}</td><td class='n'>{m.get('params_m','-')}</td></tr>")
    blrows = "".join(f"<tr style='color:#944'><td>{k} (LLM)</td><td class='n'>{v['macro_f1']}</td>"
                     f"<td class='n'>{v['c_recall']}</td><td class='n'>{v.get('over_rate','-')}</td>"
                     f"<td class='n'>-</td><td class='n'>클라우드</td><td class='n'>-</td></tr>" for k, v in bl.items())
    html = f"""<style>body{{margin:0;background:#F6F7F9}}.w{{max-width:900px;margin:0 auto;padding:28px 20px;
font-family:-apple-system,"Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;color:#161B26}}
h1{{font-size:1.5rem}}table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #E4E7ED;border-radius:10px;overflow:hidden;font-size:.9rem}}
th,td{{padding:8px 10px;border-bottom:1px solid #EEE;text-align:left}}td.n{{text-align:right;font-variant-numeric:tabular-nums}}
th{{background:#F0F2F5;font-size:.78rem;color:#5C6573}}.note{{font-size:.82rem;color:#5C6573;margin-top:10px}}</style>
<div class="w"><h1>등급분류 모델 라인업 — 3축 선택 매트릭스</h1>
<p class="note">정확도(macro-F1, neural-alone)·용량(MB/params)·속도(CPU p50) · held-out 정직 분포 · 갱신 {_dt.datetime.now().strftime('%m-%d %H:%M')}</p>
<table><thead><tr><th>모델</th><th>macroF1</th><th>C재현율</th><th>과대분류</th><th>지연 p50(ms)</th><th>용량(MB)</th><th>params(M)</th></tr></thead>
<tbody>{rows}{blrows}</tbody></table>
<p class="note">neural-alone 기준(모델 자체 분별력). 실서비스는 여기에 3-tier 규칙 floor(강식별자·다국어 키워드)가 더해져
기밀 누락(C재현율)을 추가 보증한다. LLM은 클라우드(외부 반출) — 온프레미스 제약상 비교 기준선.</p></div>"""
    (out / "report.html").write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<title>모델 라인업 매트릭스</title></head><body>' + html + '</body></html>', encoding="utf-8")
    (out / "report_artifact.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
