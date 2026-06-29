"""visualize.py — results.db → 발표용 자체완결 HTML 통계 대시보드.

외부 의존(CDN/폰트) 0: 차트는 인라인 SVG 로 생성, 한국어는 시스템 폰트 스택 사용.
단일 실행 DB(harness_out) 든 주말 누적 DB(weekend) 든 동작하며, 있는 섹션만 렌더한다.

    python -m harness.visualize --out harness_out                 # 단일 실행(+LLM 비교)
    python -m harness.visualize --out weekend --llm-db harness_out # 주말 궤적 + LLM 비교 병합

산출: <out>/report.html (브라우저로 열기). 발표 자료에 그대로 참조 가능.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

import data_classifier as dc
from .db import DB
from . import summarize as SUM

# ── 팔레트 ──────────────────────────────────────────────────────────────
C = {"bg": "#F6F7F9", "surface": "#FFFFFF", "ink": "#161B26", "muted": "#5C6573",
     "line": "#E4E7ED", "accent": "#3949AB",
     "O": "#2E9E7B", "S": "#E0A22B", "C": "#D64545",
     "default": "#9AA4B2", "optimized": "#3949AB", "llm": "#B5544E"}
GRADES = ["O", "S", "C"]
GRADE_KO = {"O": "공개(O)", "S": "민감(S)", "C": "기밀(C)"}


def _esc(s) -> str:
    return html.escape(str(s))


# ── SVG 차트 ────────────────────────────────────────────────────────────
def grouped_bars(series: List[tuple], cats: List[str], w=560, h=240, ymax=1.0) -> str:
    """series=[(name,color,[v..])] 각 v 는 0..ymax. 범주별 그룹 막대."""
    pad_l, pad_b, pad_t, pad_r = 38, 42, 16, 12
    iw, ih = w - pad_l - pad_r, h - pad_b - pad_t
    n_cat, n_ser = len(cats), len(series)
    gw = iw / n_cat
    bw = min(34, gw / (n_ser + 1))
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">']
    for gy in [0, .25, .5, .75, 1.0]:
        y = pad_t + ih * (1 - gy)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" y2="{y:.1f}" stroke="{C["line"]}"/>')
        out.append(f'<text x="{pad_l-6}" y="{y+3:.1f}" text-anchor="end" class="ax">{gy:.2f}</text>')
    for ci, cat in enumerate(cats):
        gx = pad_l + gw * ci
        for si, (name, color, vals) in enumerate(series):
            v = max(0.0, min(ymax, vals[ci] if ci < len(vals) and vals[ci] is not None else 0))
            bh = ih * (v / ymax)
            x = gx + (gw - bw * n_ser) / 2 + si * bw
            y = pad_t + ih - bh
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-3:.1f}" height="{bh:.1f}" '
                       f'fill="{color}" rx="2"><title>{_esc(name)} {cat}: {v:.3f}</title></rect>')
        out.append(f'<text x="{gx+gw/2:.1f}" y="{h-pad_b+16}" text-anchor="middle" class="ax">{_esc(cat)}</text>')
    out.append("</svg>")
    return "".join(out)


def hbars(items: List[tuple], w=560, rowh=26, ymax=1.0, fmt="{:.3f}") -> str:
    """items=[(label, val, color)] 가로 막대(랭킹)."""
    pad_l, pad_r = 120, 56
    h = rowh * len(items) + 8
    iw = w - pad_l - pad_r
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">']
    for i, (label, val, color) in enumerate(items):
        y = i * rowh + 4
        v = 0 if val is None else max(0.0, min(ymax, val))
        bw = iw * (v / ymax)
        out.append(f'<text x="{pad_l-8}" y="{y+rowh/2+4:.0f}" text-anchor="end" class="lbl">{_esc(label)}</text>')
        out.append(f'<rect x="{pad_l}" y="{y+3:.0f}" width="{iw}" height="{rowh-10}" fill="{C["line"]}" rx="3"/>')
        out.append(f'<rect x="{pad_l}" y="{y+3:.0f}" width="{bw:.1f}" height="{rowh-10}" fill="{color}" rx="3"/>')
        txt = "-" if val is None else fmt.format(val)
        out.append(f'<text x="{pad_l+iw+8}" y="{y+rowh/2+4:.0f}" class="val">{txt}</text>')
    out.append("</svg>")
    return "".join(out)


def confusion_heat(conf: Dict[str, Dict[str, int]]) -> str:
    cell, gap, pad = 78, 6, 54
    w = pad + 3 * (cell + gap) + 20
    h = pad + 3 * (cell + gap) + 24
    total = sum(sum(r.values()) for r in conf.values()) or 1
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">']
    out.append(f'<text x="{pad+3*(cell+gap)/2}" y="20" text-anchor="middle" class="ax">예측 →</text>')
    for j, p in enumerate(GRADES):
        out.append(f'<text x="{pad+j*(cell+gap)+cell/2}" y="{pad-8}" text-anchor="middle" class="lbl">{GRADE_KO[p]}</text>')
    for i, t in enumerate(GRADES):
        out.append(f'<text x="{pad-10}" y="{pad+i*(cell+gap)+cell/2+4}" text-anchor="end" class="lbl">{GRADE_KO[t]}</text>')
        for j, p in enumerate(GRADES):
            v = conf.get(t, {}).get(p, 0)
            x = pad + j * (cell + gap)
            y = pad + i * (cell + gap)
            if t == p:
                base = C["accent"]
            elif GRADES.index(p) < GRADES.index(t):
                base = C["C"]          # 과소분류(유출 위험) 빨강
            else:
                base = C["muted"]      # 과대분류 회색
            op = 0.12 + 0.8 * (v / total) if v else 0.05
            fg = "#fff" if op > 0.45 else C["ink"]
            out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="6" fill="{base}" '
                       f'fill-opacity="{op:.2f}" stroke="{C["line"]}"/>')
            out.append(f'<text x="{x+cell/2}" y="{y+cell/2+6}" text-anchor="middle" '
                       f'style="fill:{fg};font-weight:700;font-size:20px">{v}</text>')
    out.append("</svg>")
    return "".join(out)


def step_line(traj: List[dict], w=720, h=200) -> str:
    """난이도 궤적: x=반복, 좌축=난이도(계단), 막대=test macroF1."""
    if not traj:
        return ""
    pad_l, pad_b, pad_t, pad_r = 30, 30, 14, 30
    iw, ih = w - pad_l - pad_r, h - pad_b - pad_t
    n = len(traj)
    dx = iw / max(1, n)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">']
    # F1 막대(연한 인디고)
    for i, t in enumerate(traj):
        f1 = t.get("test_macro_f1") or 0
        bh = ih * f1
        x = pad_l + dx * i
        out.append(f'<rect x="{x+2:.1f}" y="{pad_t+ih-bh:.1f}" width="{max(3,dx-4):.1f}" height="{bh:.1f}" '
                   f'fill="{C["accent"]}" fill-opacity="0.18"><title>{t["run_id"]} F1={f1}</title></rect>')
    # 난이도 계단선
    pts = []
    for i, t in enumerate(traj):
        lv = t.get("level") or 1
        y = pad_t + ih * (1 - (lv - 1) / 3)   # L1..L4 → 0..1
        x = pad_l + dx * i + dx / 2
        pts.append((x, y, lv))
    path = " ".join(f'{"M" if i==0 else "L"}{x:.1f} {y:.1f}' for i, (x, y, _) in enumerate(pts))
    out.append(f'<path d="{path}" fill="none" stroke="{C["C"]}" stroke-width="2.5"/>')
    for x, y, lv in pts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{C["C"]}"/>')
    for lv in [1, 2, 3, 4]:
        y = pad_t + ih * (1 - (lv - 1) / 3)
        out.append(f'<text x="{pad_l-6}" y="{y+3:.0f}" text-anchor="end" class="ax">L{lv}</text>')
    out.append(f'<text x="{w-pad_r}" y="{h-8}" text-anchor="end" class="ax">반복 {n}회 →</text>')
    out.append("</svg>")
    return "".join(out)


# ── 데이터 수집 ─────────────────────────────────────────────────────────
def _detail(db: DB, cid: str, split="test") -> dict:
    r = db.query("SELECT detail FROM metrics WHERE config_id=? AND split=?", (cid, split))
    return DB.jload(r[0]["detail"], {}) if r else {}


def gather(db: DB, llm_db: Optional[DB]) -> dict:
    s = SUM.compute(db)
    has_iters = s["n_iterations"] > 0
    if has_iters and s.get("recommended"):
        focus_run = s["recommended"]["run_id"]
    else:
        rr = db.query("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1")
        focus_run = rr[0]["run_id"] if rr else None
    best = db.query("SELECT config_id FROM configs WHERE run_id=? AND kind='best'", (focus_run,))
    best_cid = best[0]["config_id"] if best else None
    base = db.query("SELECT config_id FROM configs WHERE run_id=? AND kind='baseline' "
                    "AND label LIKE 'default(rules%'", (focus_run,))
    def_cid = base[0]["config_id"] if base else None
    best_cfg = db.query("SELECT params FROM configs WHERE config_id=?", (best_cid,)) if best_cid else []
    best_cfg = DB.jload(best_cfg[0]["params"], {}) if best_cfg else {}

    best_m = _detail(db, best_cid) if best_cid else {}
    def_m = _detail(db, def_cid) if def_cid else {}

    # BERT 지연(추출+탐지+점수) p50
    bl = db.query("SELECT elapsed_ms FROM predictions WHERE config_id=? AND split='test'", (best_cid,)) if best_cid else []
    bert_lat = round(median([r["elapsed_ms"] for r in bl]), 1) if bl else None

    # LLM 비교
    ldb = llm_db or db
    llm_rows = ldb.query("SELECT * FROM llm_predictions")
    llm_model = llm_rows[0]["model"] if llm_rows else None
    llm_lat = round(median([r["latency_ms"] for r in llm_rows]), 1) if llm_rows else None
    # LLM KPI: best 예측과 같은 (doc,fmt) 기준 정답 대비
    llm_metrics = {}
    agreement = None
    if llm_rows and best_cid:
        bert_pred = {(r["doc_id"], r["fmt"]): r for r in db.query(
            "SELECT * FROM predictions WHERE config_id=? AND split='test'", (best_cid,))}
        from . import metrics as M
        lrows, agree_n, pair_n = [], 0, 0
        lm = {(r["doc_id"], r["fmt"]): r for r in llm_rows}
        for k, b in bert_pred.items():
            l = lm.get(k)
            if not l or not l["llm_grade"]:
                continue
            lrows.append({"true_grade": b["true_grade"], "pred_grade": l["llm_grade"], "fmt": k[1]})
            pair_n += 1
            if l["llm_grade"] == b["pred_grade"]:
                agree_n += 1
        if lrows:
            llm_metrics = M.compute(lrows)
            agreement = round(agree_n / pair_n, 3) if pair_n else None

    # 탐색 리더보드 (focus 반복에서 시도된 설정들 — 최적화 과정)
    lb_rows = db.query("""SELECT c.kind, c.label, c.params, v.objective vo,
                                 v.under_rate vu, v.over_rate ov, t.macro_f1 tf
                          FROM configs c
                          JOIN metrics v ON v.config_id=c.config_id AND v.split='valid'
                          LEFT JOIN metrics t ON t.config_id=c.config_id AND t.split='test'
                          WHERE c.run_id=? ORDER BY v.objective DESC""", (focus_run,)) if focus_run else []
    leaderboard = []
    for r in lb_rows:
        p = DB.jload(r["params"], {})
        leaderboard.append({"kind": r["kind"], "label": r["label"],
                            "model": (p.get("model") if p.get("llm_mode") else "rules+NER"),
                            "ensemble": (p.get("ensemble", "escalate") if p.get("llm_mode") else "-"),
                            "obj": r["vo"], "test_f1": r["tf"], "under": r["vu"], "over": r["ov"]})

    return {"summary": s, "has_iters": has_iters, "focus_run": focus_run,
            "best_cfg": best_cfg, "best_m": best_m, "def_m": def_m,
            "bert_lat": bert_lat, "llm_model": llm_model, "llm_lat": llm_lat,
            "llm_metrics": llm_metrics, "agreement": agreement, "leaderboard": leaderboard,
            "n_configs": len(leaderboard),
            "runs": db.query("SELECT COUNT(*) n FROM corpus")[0]["n"]}


# ── HTML ────────────────────────────────────────────────────────────────
def _stat(label, val, sub="", tone=""):
    cls = f"stat {tone}".strip()
    return (f'<div class="{cls}"><div class="stat-v">{_esc(val)}</div>'
            f'<div class="stat-l">{_esc(label)}</div>'
            + (f'<div class="stat-s">{_esc(sub)}</div>' if sub else "") + '</div>')


def build_content(db: DB, llm_db: Optional[DB]) -> str:
    d = gather(db, llm_db)
    bm, dm, lm = d["best_m"], d["def_m"], d["llm_metrics"]
    cfg = d["best_cfg"]
    nb = cfg.get("model")
    nbm = dc.NEURAL_BACKENDS.get(nb, {}) if nb else {}
    s = d["summary"]

    def pct(m, k):
        return f"{m.get(k,0)*100:.1f}%" if m else "-"

    P = []
    P.append(f'<style>{CSS}</style>')
    P.append('<div class="wrap">')

    # 헤더
    P.append(f'''<header class="hero">
      <div class="eyebrow">N²SF 문서 등급 분류 · 자동 최적화 평가</div>
      <h1>GPU·외부 LLM 없이 LLM 수준 등급 분류</h1>
      <p class="lead">정규식 · NER · 경량 BERT 3-tier 모델을 합성 테스트셋으로 자동 평가하고
      설정을 반복 탐색해 최적화한 결과입니다. 보안 관점에서 <b>기밀 누락(과소분류)</b>을 최우선으로 억제합니다.</p>
    </header>''')

    # 핵심 KPI 카드 (최적 vs 기본 vs LLM)
    over_tone = "good" if (bm.get("over_rate", 1) == 0) else "warn"
    under_tone = "good" if (bm.get("under_rate", 1) == 0) else "bad"
    cards = [
        _stat("최적 macro-F1", pct(bm, "macro_f1"), f"기본 {pct(dm,'macro_f1')}", "accent"),
        _stat("정확도", pct(bm, "accuracy"), f"기본 {pct(dm,'accuracy')}"),
        _stat("기밀(C) 재현율", pct(bm, "c_recall"), "누락 없음" if bm.get("c_recall")==1 else "주의", under_tone),
        _stat("과소분류(유출위험)", pct(bm, "under_rate"), "", under_tone),
        _stat("과대분류", pct(bm, "over_rate"), f"기본 {pct(dm,'over_rate')}", over_tone),
    ]
    if lm:
        cards.append(_stat(f"{d['llm_model'] or 'LLM'} macro-F1", pct(lm, "macro_f1"),
                           f"일치율 {d['agreement']}" if d['agreement'] is not None else "", "llm"))
    P.append('<section class="cards">' + "".join(cards) + '</section>')

    # 모델 구성 배지
    badges = [("신경망", (nb + " · " + nbm.get("label", "")) if nb else "미사용(규칙+NER)"),
              ("앙상블", cfg.get("ensemble", "escalate") if cfg.get("llm_mode") else "—"),
              ("C 임계", cfg.get("thresholds", {}).get("confidential", 5.5)),
              ("S 임계", cfg.get("thresholds", {}).get("sensitive", 0.75)),
              ("BERT 지연(p50)", f"{d['bert_lat']} ms" if d["bert_lat"] is not None else "-")]
    if cfg.get("llm_mode"):
        badges.append(("뉴럴추론(CPU)", "≈ 0.8 s"))
    P.append('<section class="badges">' +
             "".join(f'<div class="badge"><span>{_esc(k)}</span><b>{_esc(v)}</b></div>' for k, v in badges) +
             '</section>')

    # KPI 비교 차트 + 혼동행렬
    cats = ["macro-F1", "정확도", "C재현율", "1-과대분류"]
    def quad(m):
        return [m.get("macro_f1", 0), m.get("accuracy", 0), m.get("c_recall", 0), 1 - m.get("over_rate", 0)] if m else [0, 0, 0, 0]
    series = [("기본", C["default"], quad(dm)), ("최적(BERT 3-tier)", C["optimized"], quad(bm))]
    if lm:
        series.append((d["llm_model"] or "LLM", C["llm"], quad(lm)))
    P.append('<section class="grid2">')
    P.append(f'''<div class="panel"><h2>설정별 핵심 지표 비교</h2>
      {grouped_bars(series, cats)}
      <div class="legend">{"".join(f'<span><i style="background:{c}"></i>{_esc(n)}</span>' for n,c,_ in series)}</div></div>''')
    P.append(f'''<div class="panel"><h2>혼동행렬 — 최적 설정 (test)</h2>
      {confusion_heat(bm.get("confusion", {}))}
      <p class="note">대각선=정답. <b style="color:{C['C']}">빨강 칸</b>=과소분류(실제보다 낮게=유출 위험), 회색=과대분류.</p></div>''')
    P.append('</section>')

    # 포맷별
    bf = bm.get("by_format", {})
    if bf:
        items = [(fmt + (" ★" if fmt in ("xlsx", "hwpx") else ""), bf[fmt]["accuracy"],
                  C["accent"] if fmt in ("xlsx", "hwpx") else C["optimized"]) for fmt in sorted(bf)]
        P.append(f'''<section class="panel"><h2>파일 포맷별 정확도 (최적 설정)</h2>
          <p class="note">동일 콘텐츠를 9개 포맷으로 렌더해 추출 레이어를 분리 검증. ★ = 핵심 포맷(XLSX·HWPX).</p>
          {hbars(items)}</section>''')

    # LLM 비교 — 속도
    if lm and d["llm_lat"]:
        bert_ms = d["bert_lat"] or 0
        mx = max(bert_ms, d["llm_lat"])
        sp = [(f"BERT 3-tier{' (+뉴럴 0.8s)' if cfg.get('llm_mode') else ''}", bert_ms, C["optimized"]),
              (f"{d['llm_model']}", d["llm_lat"], C["llm"])]
        P.append(f'''<section class="grid2">
          <div class="panel"><h2>정확도: 내 모델 vs {_esc(d['llm_model'])}</h2>
            {grouped_bars([("최적 BERT", C["optimized"], quad(bm)), (d['llm_model'], C['llm'], quad(lm))], cats)}
            <p class="note">동일 추출 텍스트 입력. 일치율 {d['agreement']}.</p></div>
          <div class="panel"><h2>분류 속도 (p50, 낮을수록 좋음)</h2>
            {hbars([(n, v, c) for n,v,c in sp], ymax=mx, fmt="{:.0f} ms")}
            <p class="note">로컬 LLM(9B)은 CPU/GPU 자원을 크게 쓰고 느림. 3-tier 는 CPU 만으로 동작.</p></div>
        </section>''')

    # 주말 — 난이도 궤적 + 반복 히스토리 + 난이도별 최적 + 모델별
    if d["has_iters"]:
        P.append(f'''<section class="panel"><h2>난이도 자동 조정 궤적</h2>
          <p class="note">모델이 풀면 난이도↑(시험을 어렵게), 못 풀면 데이터·탐색 집중. 빨강선=난이도, 연한막대=test macro-F1.</p>
          {step_line(s["trajectory"])}</section>''')

        # 반복별 전체 히스토리(시간순) — 최종 결과만이 아닌 전 과정
        hrows = "".join(
            f'<tr><td>{i+1}</td><td>{_esc(t["run_id"])}</td><td>L{t["level"]}</td>'
            f'<td>{_esc(t["best_model"])}/{_esc(t["best_ensemble"])}</td>'
            f'<td class="num">{t["test_macro_f1"]}</td><td class="num">{t["test_c_recall"]}</td>'
            f'<td class="num">{t["test_under"]}</td><td class="num">{t["test_over"]}</td>'
            f'<td class="num">{t["rules_only_f1"]}</td><td class="num">{round(t["elapsed_s"] or 0)}s</td></tr>'
            for i, t in enumerate(s["trajectory"]))
        P.append(f'''<section class="panel"><h2>최적화 히스토리 — 전 반복 (시간순)</h2>
          <p class="note">매 반복: 새 샘플데이터 생성 → 모델 최적화 → 기록. rules만=뉴럴 미사용 빠른 경로 참고치.</p>
          <div class="tw"><table><thead><tr><th>#</th><th>반복</th><th>난이도</th><th>최적 모델/앙상블</th>
          <th>macro-F1</th><th>C재현율</th><th>과소분류</th><th>과대분류</th><th>rules만 F1</th><th>소요</th></tr></thead>
          <tbody>{hrows}</tbody></table></div></section>''')

        pl = s["per_level"]
        if pl:
            rows = "".join(
                f'<tr><td>L{L}</td><td>{_esc(pl[L]["best_model"])}/{_esc(pl[L]["best_ensemble"])}</td>'
                f'<td class="num">{pct(pl[L],"test_macro_f1") if False else (pl[L]["test_macro_f1"] or 0)}</td>'
                f'<td class="num">{pl[L]["test_c_recall"]}</td><td class="num">{pl[L]["test_under"]}</td>'
                f'<td class="num">{pl[L]["test_over"]}</td><td class="num">{pl[L]["n_docs"]}</td></tr>'
                for L in sorted(pl, key=int))
            P.append(f'''<section class="panel"><h2>난이도별 최적 모델</h2>
              <div class="tw"><table><thead><tr><th>난이도</th><th>최적 모델/앙상블</th><th>macro-F1</th>
              <th>C재현율</th><th>과소분류</th><th>과대분류</th><th>문서수</th></tr></thead><tbody>{rows}</tbody></table></div></section>''')

        pm = s["per_model"]
        if pm:
            order = sorted(pm.items(), key=lambda x: -(x[1]["wins"]))
            items = [(m, pm[m]["avg_test_f1"], C["accent"] if pm[m]["wins"] else C["default"]) for m, _ in order]
            P.append(f'''<section class="panel"><h2>BERT 백엔드별 성능 (전 반복 평균 test F1 · 우승 횟수)</h2>
              {hbars(items)}
              <div class="tw"><table><thead><tr><th>모델</th><th>평가반복</th><th>overall 우승</th>
              <th>평균 F1</th><th>최고 F1</th><th>평균 C재현율</th></tr></thead><tbody>''' +
              "".join(f'<tr><td>{_esc(m)}</td><td class="num">{v["iters"]}</td><td class="num">{v["wins"]}</td>'
                      f'<td class="num">{v["avg_test_f1"]}</td><td class="num">{v["best_test_f1"]}</td>'
                      f'<td class="num">{v["avg_c_recall"]}</td></tr>' for m, v in order) +
              '</tbody></table></div></section>')

    # 탐색 리더보드 — 한 반복 내에서 시도된 설정들(최적화 과정)
    lb = d.get("leaderboard") or []
    if lb:
        top = lb[:18]
        bar_items = [((r["model"] + ("/" + r["ensemble"] if r["ensemble"] != "-" else "")), r["obj"],
                      C["accent"] if r["kind"] == "best" else (C["default"] if r["kind"] == "baseline" else C["optimized"]))
                     for r in top[:12]]
        rows = "".join(
            f'<tr><td>{i+1}</td><td><span class="tag {_esc(r["kind"])}">{_esc(r["kind"])}</span></td>'
            f'<td>{_esc(r["model"])}</td><td>{_esc(r["ensemble"])}</td>'
            f'<td class="num">{r["obj"]:.3f}</td><td class="num">{r["test_f1"]}</td>'
            f'<td class="num">{r["under"]}</td><td class="num">{r["over"]}</td></tr>'
            for i, r in enumerate(top))
        P.append(f'''<section class="panel"><h2>최적화 탐색 리더보드 — 시도된 설정 {d["n_configs"]}개 중 상위
          <span class="muted2">(반복 {_esc(d["focus_run"])})</span></h2>
          <p class="note">한 반복 안에서 점수·앙상블·뉴럴 백엔드를 스윕하며 안전 우선 목적함수(objective)로 비교한 과정.
          최종 채택(best)에 이르기까지의 탐색을 보여줍니다.</p>
          {hbars(bar_items)}
          <div class="tw"><table><thead><tr><th>순위</th><th>종류</th><th>모델</th><th>앙상블</th>
          <th>objective</th><th>test F1</th><th>과소</th><th>과대</th></tr></thead><tbody>{rows}</tbody></table></div></section>''')

    # 푸터/주의
    P.append(f'''<footer class="foot">
      <h2>읽는 법 · 주의</h2>
      <ul>
        <li><b>과소분류(유출 위험)</b>가 0 인지를 최우선으로 본다 — 기밀을 낮게 분류하면 유출이다.</li>
        <li>합성 테스트셋은 정책(N²SF) 기준으로 라벨링했다. 만점은 "이 난이도에서 완전 분리 가능"을 뜻하며 실문서 성능 보장이 아니다.</li>
        <li>LLM(로컬 Gemma)은 깨끗한 추출 텍스트를 입력받지만, 3-tier 는 추출+분류를 모두 수행한다(비대칭).</li>
      </ul>
      <p class="src">생성: harness/visualize.py · 데이터: results.db · 문서 {d["runs"]}건 평가</p>
    </footer>''')
    P.append('</div>')
    return "".join(P)


CSS = """
*{box-sizing:border-box}
.wrap{--bg:#F6F7F9;--ink:#161B26;--muted:#5C6573;--line:#E4E7ED;--acc:#3949AB;
  max-width:1080px;margin:0 auto;padding:32px 22px 64px;
  font-family:"Pretendard",-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;
  color:var(--ink);background:var(--bg);line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap h1{font-size:2.05rem;line-height:1.15;margin:.1em 0 .25em;letter-spacing:-.02em;text-wrap:balance}
.wrap h2{font-size:1.06rem;margin:0 0 .7em;letter-spacing:-.01em}
.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:.72rem;font-weight:700;color:var(--acc)}
.lead{color:var(--muted);max-width:62ch;margin:.4em 0 0}
.hero{padding-bottom:18px;border-bottom:1px solid var(--line);margin-bottom:22px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:14px}
.stat{background:var(--surface,#fff);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.stat-v{font-size:1.7rem;font-weight:750;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.stat-l{font-size:.8rem;color:var(--muted);margin-top:2px}
.stat-s{font-size:.72rem;color:var(--muted);margin-top:3px;opacity:.85}
.stat.accent{border-color:#C5CAE9;background:#F5F6FD}.stat.accent .stat-v{color:var(--acc)}
.stat.good .stat-v{color:#2E9E7B}.stat.bad .stat-v{color:#D64545}.stat.warn .stat-v{color:#E0A22B}
.stat.llm{border-color:#E6CFC9}.stat.llm .stat-v{color:#B5544E}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 26px}
.badge{display:flex;flex-direction:column;background:#fff;border:1px solid var(--line);border-radius:9px;padding:7px 12px}
.badge span{font-size:.7rem;color:var(--muted);letter-spacing:.02em}
.badge b{font-size:.92rem;font-variant-numeric:tabular-nums}
.panel{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:18px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:760px){.grid2{grid-template-columns:1fr}.wrap h1{font-size:1.6rem}}
.chart{width:100%;height:auto;display:block}
.chart .ax{font-size:10px;fill:var(--muted)}
.chart .lbl{font-size:11.5px;fill:var(--ink)}
.chart .val{font-size:11.5px;fill:var(--ink);font-weight:650;font-variant-numeric:tabular-nums}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;font-size:.78rem;color:var(--muted)}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.note{font-size:.78rem;color:var(--muted);margin:.7em 0 0}
.tw{overflow-x:auto}
table{width:100%;border-collapse:collapse;margin-top:8px;font-size:.86rem}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-size:.74rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:700}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.foot{margin-top:8px;padding-top:18px;border-top:1px solid var(--line)}
.foot ul{margin:.3em 0;padding-left:1.1em;color:var(--muted);font-size:.86rem}
.foot li{margin:.3em 0}
.foot .src{font-size:.74rem;color:var(--muted);opacity:.8;margin-top:10px}
.muted2{font-weight:400;color:var(--muted);font-size:.82rem}
.tag{font-size:.68rem;padding:1px 7px;border-radius:20px;text-transform:uppercase;letter-spacing:.03em;font-weight:700}
.tag.best{background:#E8EAF6;color:#3949AB}.tag.baseline{background:#EEF0F3;color:#5C6573}.tag.sweep{background:#F3F4F7;color:#8A92A0}
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="results.db → 발표용 HTML 대시보드")
    ap.add_argument("--out", default="harness_out")
    ap.add_argument("--llm-db", default=None, help="LLM 비교 데이터가 든 별도 results.db(선택)")
    ap.add_argument("--file", default="report.html")
    args = ap.parse_args(argv)
    out = Path(args.out)
    db = DB(out / "results.db")
    llm_db = DB(Path(args.llm_db) / "results.db") if args.llm_db else None
    content = build_content(db, llm_db)
    standalone = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
                  '<meta name="viewport" content="width=device-width,initial-scale=1">'
                  '<title>N²SF 등급 분류 — 자동 최적화 결과</title>'
                  '<style>body{margin:0;background:#F6F7F9}</style></head><body>'
                  + content + '</body></html>')
    (out / args.file).write_text(standalone, encoding="utf-8")
    # Artifact 용 본문(헤드/바디 없이)도 별도 저장
    (out / "report_artifact.html").write_text(content, encoding="utf-8")
    db.close()
    if llm_db:
        llm_db.close()
    print(f"생성: {out/args.file}  (+ {out/'report_artifact.html'})")


if __name__ == "__main__":
    main()
