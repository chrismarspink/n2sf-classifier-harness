"""improve.py — 지속 개선 루프: LLM 다양화 데이터로 재학습 → held-out(다른 분포) 정직 평가 → 모델 재선정.

과적합 해소가 목표. 학습은 템플릿 생성기 A + Azure LLM 생성 데이터, 평가는 **학습에 안 쓴
held-out(Azure+Gemma 생성, 동결)** 로 일반화 측정. 매 사이클 다양한 데이터를 더해 재학습하고
mdeberta-n2sf 포함 후보 모델들을 held-out에서 비교 → 이노티움용 최적 온디바이스 모델을 추적.

리소스: 외부 LLM(Azure)·Gemma는 **데이터 생성/비교 baseline(오프라인)** 에만. 실제 추론은 로컬 BERT.
사이클의 학습데이터 생성은 Azure(네트워크)만 사용해 MPS(학습)와 Metal(gemma) 경합을 피한다.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path

import os

import data_classifier as dc
from . import metrics as M
from .detect import normalize_text
from .corpus import _flat_text
from .gen_llm import generate, LLMFiller, _azure_client, _load_env
from .ollama_baseline import classify_ollama
from .llm_baseline import POLICY

GRADES = ["O", "S", "C"]
FULL = {"O": "OPEN", "S": "SENSITIVE", "C": "CONFIDENTIAL"}
# 평가 후보(온디바이스 BERT). ensemble=soft.
CANDIDATES = [("rules", None), ("minilm", "soft"), ("ko-sroberta", "soft"),
              ("klue-roberta", "soft"), ("mdeberta", "soft"), ("mdeberta-n2sf", "soft")]


def _classify(text, model, ensemble):
    if model == "rules":
        r = dc.classify_text(text, locale="ko")
    else:
        r = dc.classify_text(text, locale="ko", llm_mode=True, model=model, ensemble_method=ensemble)
    return r["grade"]   # O/S/C


def build_heldout(out: Path, per_cell=2, log=print):
    """동결 held-out(LLM 분포) 생성: Azure+Gemma × 등급 × 난이도. 1회만."""
    fp = out / "heldout.json"
    if fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))
    log("  held-out 생성(Azure+Gemma, 동결)…")
    f = LLMFiller(seed=777)
    docs = []
    gens = ["azure:gpt-4o", "azure:o4-mini", "gemma"]
    i = 0
    for gen in gens:
        for g in GRADES:
            for d in (1, 2, 3, 4):
                for _ in range(per_cell):
                    i += 1
                    doc = generate(gen, g, d, f"HO-{i}", f)
                    if not doc:
                        continue
                    txt = normalize_text(_flat_text(doc))
                    docs.append({"text": txt, "grade": g, "gen": gen, "diff": d})
    fp.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    log(f"  held-out {len(docs)}건 동결 저장")
    return docs


def _parse_grade(s: str):
    s = (s or "").upper()
    for g in ("C", "S", "O"):                  # 우선순위: 명시 등급
        if f'"{g}"' in s or f"GRADE: {g}" in s or f"등급: {g}" in s:
            return g
    for ch in s:                                # 첫 등급 문자
        if ch in ("O", "S", "C"):
            return ch
    return None


def azure_classify(text, deployment, apiver):
    """Azure LLM(gpt-4o/o4-mini)으로 등급 분류(O/S/C). 정책 프롬프트 사용."""
    try:
        c = _azure_client(apiver)
        user = POLICY + f"\n\n<문서>\n{text[:6000]}\n</문서>\n\n위 문서의 등급을 O, S, C 중 하나로만 답하라."
        kw = {"model": deployment, "messages": [{"role": "user", "content": user}]}
        if deployment.startswith(("o1", "o3", "o4")):
            kw["max_completion_tokens"] = 2000
        else:
            kw["max_tokens"] = 8; kw["temperature"] = 0
        r = c.chat.completions.create(**kw)
        return _parse_grade(r.choices[0].message.content)
    except Exception as exc:
        return None


def _claude_classify(text):
    try:
        from .llm_baseline import get_client, classify_llm
        cl = get_client()
        if cl is None:
            return None
        r = classify_llm(cl, text)
        return r.get("grade")
    except Exception:
        return None


def llm_baselines(out: Path, heldout, log=print):
    """held-out에서 외부/로컬 LLM을 **분류기**로 평가(1회, 동결). 내 BERT와 비교용 기준선."""
    fp = out / "llm_baselines.json"
    if fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))
    _load_env()
    out_m = {}
    jobs = [("gpt-4o", lambda t: azure_classify(t, os.environ.get("AZURE_GPT4O_DEPLOYMENT", "gpt-4o"),
                                                os.environ.get("AZURE_GPT4O_APIVER", "2025-01-01-preview"))),
            ("o4-mini", lambda t: azure_classify(t, os.environ.get("AZURE_O4MINI_DEPLOYMENT", "o4-mini"),
                                                 os.environ.get("AZURE_O4MINI_APIVER", "2025-04-01-preview"))),
            ("gemma2:9b", lambda t: (classify_ollama(t, "gemma2:9b") or {}).get("grade")),
            ("claude", _claude_classify)]
    for name, fn in jobs:
        rows, errs = [], 0
        for h in heldout:
            g = fn(h["text"])
            if g in ("O", "S", "C"):
                rows.append({"true_grade": h["grade"], "pred_grade": g, "category": f"L{h['diff']}"})
            else:
                errs += 1
        if not rows:
            log(f"  [LLM baseline] {name}: 사용 불가/스킵 (오류 {errs})"); continue
        m = M.compute(rows); out_m[name] = m
        log(f"  [LLM baseline] {name}: F1={m['macro_f1']} Crec={m['c_recall']} under={m['under_rate']} (n={m['n']}, 스킵 {errs})")
    fp.write_text(json.dumps(out_m, ensure_ascii=False), encoding="utf-8")
    # gemma 언로드(학습 메모리 확보)
    try:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/chat",
            data=json.dumps({"model": "gemma2:9b", "keep_alive": 0,
                             "messages": [{"role": "user", "content": "x"}]}).encode(),
            headers={"Content-Type": "application/json"}), timeout=60).read()
    except Exception:
        pass
    return out_m


def gen_train_batch(out: Path, n_per: int, cycle: int, log=print):
    """사이클별 LLM 학습데이터 생성(Azure만 — Metal 경합 회피). train.jsonl 누적."""
    f = LLMFiller(seed=1000 + cycle)
    jsonl = out / "train_llm.jsonl"
    added = 0
    gens = ["azure:gpt-4o", "azure:o4-mini"]
    with open(jsonl, "a", encoding="utf-8") as w:
        for g in GRADES:
            for d in (1, 2, 3, 4):
                for k in range(n_per):
                    gen = gens[(added) % len(gens)]
                    doc = generate(gen, g, d, f"TR-{cycle}-{added}", f)
                    if not doc:
                        continue
                    w.write(json.dumps({"text": _flat_text(doc), "grade": g}, ensure_ascii=False) + "\n")
                    added += 1
    log(f"  사이클{cycle} LLM 학습데이터 +{added}건 (누적 train_llm.jsonl)")
    return jsonl


def eval_candidates(heldout, log=print):
    """held-out에서 후보 모델별 metrics. (model→metrics)"""
    out = {}
    for model, ens in CANDIDATES:
        rows = []
        for h in heldout:
            try:
                pred = _classify(h["text"], model, ens)
            except Exception:
                pred = "O"
            rows.append({"true_grade": h["grade"], "pred_grade": pred, "category": f"L{h['diff']}"})
        out[model] = M.compute(rows)
    return out


def write_report(out: Path, records, baselines, log=print):
    """간단 자체완결 HTML — 사이클별 held-out 일반화 추이 + 내 BERT vs LLM(GPT-4o/o4-mini/Gemma/Claude) 비교."""
    latest = records[-1]["models"] if records else {}
    models = [m for m, _ in CANDIDATES]
    colors = {"rules": "#9AA4B2", "minilm": "#5BA88C", "ko-sroberta": "#5577CC", "klue-roberta": "#7CB342",
              "mdeberta": "#E0A22B", "mdeberta-n2sf": "#3949AB",
              "gemma2:9b": "#B5544E", "gpt-4o": "#8E44AD", "o4-mini": "#C0392B", "claude": "#16A085"}
    bl_names = list(baselines.keys())
    W, H, pl, pb = 720, 240, 40, 30
    iw, ih = W - pl - 20, H - pb - 14
    n = max(1, len(records))
    svg = [f'<svg viewBox="0 0 {W} {H}" class="ch">']
    for gy in (0, .5, 1.0):
        y = 14 + ih * (1 - gy)
        svg.append(f'<line x1="{pl}" y1="{y}" x2="{W-20}" y2="{y}" stroke="#E4E7ED"/><text x="{pl-6}" y="{y+3}" text-anchor="end" font-size="10" fill="#888">{gy:.1f}</text>')
    # BERT 후보: 사이클별 추이 / LLM 기준선: 수평선
    for m in models:
        pts = []
        for i, rec in enumerate(records):
            v = (rec["models"].get(m, {}) or {}).get("macro_f1")
            if v is None:
                continue
            x = pl + iw * (i / max(1, n - 1) if n > 1 else 0.5)
            pts.append(f"{x:.0f},{14 + ih*(1-v):.0f}")
        if pts:
            svg.append(f'<polyline fill="none" stroke="{colors.get(m,"#888")}" stroke-width="2.4" points="{" ".join(pts)}"/>')
    for m in bl_names:
        v = (baselines.get(m) or {}).get("macro_f1")
        if v is None:
            continue
        y = 14 + ih * (1 - v)
        svg.append(f'<line x1="{pl}" y1="{y:.0f}" x2="{W-20}" y2="{y:.0f}" stroke="{colors.get(m,"#888")}" stroke-width="1.4" stroke-dasharray="5 4"/>')
    svg.append(f'<text x="{W-20}" y="{H-6}" text-anchor="end" font-size="10" fill="#888">사이클 {n}회 →</text></svg>')

    def row(name, m, hl=False):
        c = ' style="background:#F5F6FD;font-weight:700"' if hl else ""
        return (f'<tr{c}><td>{name}</td><td class="n">{m.get("macro_f1","-")}</td>'
                f'<td class="n">{m.get("accuracy","-")}</td><td class="n">{m.get("c_recall","-")}</td>'
                f'<td class="n">{m.get("under_rate","-")}</td><td class="n">{m.get("over_rate","-")}</td></tr>')
    rows_html = '<tr><td colspan="6" style="background:#EEF;font-size:.75rem;color:#447">온디바이스 BERT (로컬·CPU)</td></tr>'
    rows_html += "".join(row(m, latest.get(m, {}), hl=(m == "mdeberta-n2sf")) for m, _ in CANDIDATES)
    label_map = {"gpt-4o": "GPT-4o (Azure)", "o4-mini": "o4-mini (Azure)",
                 "gemma2:9b": "Gemma2:9b (로컬)", "claude": "Claude (API)"}
    rows_html += '<tr><td colspan="6" style="background:#FEE;font-size:.75rem;color:#944">LLM 분류기 (비교 기준선)</td></tr>'
    rows_html += "".join(row(label_map.get(m, m), baselines[m]) for m in bl_names)
    best = max(latest.items(), key=lambda kv: (kv[1].get("c_recall", 0) == 1.0, kv[1].get("macro_f1", 0)), default=("-", {})) if latest else ("-", {})

    html = f"""<style>
body{{margin:0;background:#F6F7F9}}.w{{max-width:900px;margin:0 auto;padding:28px 20px;
font-family:-apple-system,"Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;color:#161B26}}
h1{{font-size:1.5rem;margin:.2em 0}}h2{{font-size:1.05rem;margin:1.2em 0 .5em}}
.sub{{color:#5C6573}}.ch{{width:100%;height:auto;background:#fff;border:1px solid #E4E7ED;border-radius:12px;padding:8px}}
table{{width:100%;border-collapse:collapse;font-size:.9rem;background:#fff;border:1px solid #E4E7ED;border-radius:10px;overflow:hidden}}
th,td{{padding:8px 10px;border-bottom:1px solid #EEE;text-align:left}}td.n{{text-align:right;font-variant-numeric:tabular-nums}}
th{{background:#F0F2F5;font-size:.78rem;color:#5C6573}}.note{{font-size:.82rem;color:#5C6573;margin-top:8px}}
.legend{{font-size:.78rem;color:#5C6573;margin-top:6px}}.legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px;vertical-align:-1px}}
</style><div class="w">
<h1>이노티움 등급분류 — 지속 개선(일반화) 대시보드</h1>
<p class="sub">학습=템플릿+Azure LLM · 평가=held-out(Azure+Gemma 생성, 학습 미사용) · 사이클 {n}회 · 갱신 {_dt.datetime.now().strftime('%m-%d %H:%M')}</p>
<h2>held-out 일반화 F1 추이 (과적합 아닌 진짜 성능)</h2>
{''.join(svg)}
<div class="legend">BERT(실선): {''.join(f'<i style="background:{colors.get(m,"#888")}"></i>{m}' for m in models)} · LLM(점선): {''.join(f'<i style="background:{colors.get(m,"#888")}"></i>{m}' for m in bl_names)}</div>
<h2>최신 사이클 — held-out 모델 비교</h2>
<table><thead><tr><th>모델</th><th>macroF1</th><th>정확도</th><th>C재현율</th><th>과소분류</th><th>과대분류</th></tr></thead>
<tbody>{rows_html}</tbody></table>
<p class="note">현재 최적(무유출 우선): <b>{best[0]}</b>. held-out은 학습에 쓰지 않은 LLM 생성 분포라 이 수치가 일반화 성능에 가깝다.
온디바이스(CPU·로컬) 후보 중 C재현율 1.0 유지하며 F1 최고인 모델을 선택한다.</p>
</div>"""
    (out / "report.html").write_text(
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>이노티움 등급분류 — 지속 개선</title></head><body>' + html + '</body></html>',
        encoding="utf-8")
    (out / "report_artifact.html").write_text(html, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="지속 개선 루프(LLM 다양화 → 재학습 → held-out 평가)")
    ap.add_argument("--out", default="improve")
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--gen-per-cycle", type=int, default=2, help="사이클당 (등급×난이도)별 LLM 학습문서 수")
    ap.add_argument("--heldout-per", type=int, default=2, help="held-out (생성기×등급×난이도)별 문서 수")
    ap.add_argument("--train-cap", type=int, default=4000, help="사이클 재학습 시 클래스당 템플릿 표본")
    ap.add_argument("--train-epochs", type=int, default=2)
    ap.add_argument("--train-hours", type=float, default=0.4, help="사이클당 학습 시간 상한")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="시작 시 (등급×난이도)별 다양화 문서 대량 생성(증류 부트스트랩)")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = print
    deadline = time.time() + args.hours * 3600
    log(f"== improve 시작 out={out} 예산={args.hours}h ==")

    # PHASE 0 — held-out 생성 + LLM 분류기 기준선(GPT-4o/o4-mini/Gemma/Claude) 1회
    heldout = build_heldout(out, per_cell=args.heldout_per, log=log)
    baselines = llm_baselines(out, heldout, log=log)

    records = []
    rec_fp = out / "cycles.json"
    if rec_fp.exists():
        records = json.loads(rec_fp.read_text(encoding="utf-8"))
    cycle = len(records)

    # 증류 부트스트랩 — 다양화 학습데이터 대량 확보(1회). 지난 실패(다양화가 템플릿에 묻힘) 교정.
    if args.bootstrap and not (out / "bootstrap.done").exists():
        log(f"  부트스트랩: 다양화 문서 대량 생성((등급×난이도)별 {args.bootstrap})…")
        gen_train_batch(out, args.bootstrap, cycle=-1, log=log)
        (out / "bootstrap.done").write_text("done")

    while time.time() < deadline:
        c0 = time.time()
        log(f"\n=== 사이클 {cycle} === {_dt.datetime.now().strftime('%H:%M')}")
        # 1) 다양화 학습데이터 생성(Azure)
        jsonl = gen_train_batch(out, args.gen_per_cycle, cycle, log=log)
        # 2) 재학습(템플릿 A + 누적 LLM)
        log("  재학습(mdeberta-n2sf)…")
        cmd = [sys.executable, "-m", "harness.train", "--db", "weekend/results.db",
               "--jsonl", str(jsonl), "--out", "models/mdeberta-n2sf",
               "--epochs", str(args.train_epochs), "--per-class-cap", str(args.train_cap),
               "--batch", "16", "--accum", "2", "--maxlen", "128",
               "--max-hours", str(args.train_hours)]
        try:
            subprocess.run(cmd, check=False, timeout=args.train_hours * 3600 + 600)
        except Exception as exc:
            log(f"  학습 경고: {str(exc)[:120]}")
        # 캐시된 finetuned 모델 리로드를 위해 모듈 캐시 비움
        dc._NEURAL_MODELS.pop("mdeberta-n2sf", None)
        # 3) held-out 평가(후보 전종)
        log("  held-out 평가…")
        models_m = eval_candidates(heldout, log=log)
        for m, mm in models_m.items():
            log(f"    {m:14} F1={mm['macro_f1']} Crec={mm['c_recall']} under={mm['under_rate']} over={mm['over_rate']}")
        # 4) 기록 + 리포트
        records.append({"cycle": cycle, "ts": _dt.datetime.now().isoformat(),
                        "models": models_m, "elapsed_s": round(time.time() - c0, 1)})
        rec_fp.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        try:
            write_report(out, records, baselines, log=log)
        except Exception as exc:
            log(f"  리포트 경고: {str(exc)[:120]}")
        log(f"  사이클 {cycle} 완료 ({records[-1]['elapsed_s']:.0f}s)")
        cycle += 1

    log(f"== improve 종료. 총 사이클 {cycle}. 결과: {out}/report.html, {out}/cycles.json ==")


if __name__ == "__main__":
    main()
