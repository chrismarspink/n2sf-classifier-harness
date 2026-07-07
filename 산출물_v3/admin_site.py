#!/usr/bin/env python3
"""admin_site.py — N²SF 분류 API 서버 + 온사이트 학습 '관리 콘솔' (심플/단일파일).

가정: 분류 모델은 API 서버(/api/classify). 이 콘솔이 그 서버를 '관리'한다.
메뉴(전부): 대시보드 · 분류 테스트 · 데이터 관리(라벨 업로드) · 학습(고객데이터) · 모델 버전(적용/롤백) · 모니터링 · 설정.
학습·적용은 train_onsite + classifier_v3(무중단 핫리로드)로 **실제 동작**. 통계/설정 일부는 더미.
실행: python admin_site.py  →  http://localhost:8080
"""
from __future__ import annotations
import json, os, threading, time, traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import classifier_v3 as C
import train_onsite as T

_ROOT = Path(__file__).resolve().parent.parent
UPLOAD = _ROOT / "onsite_uploads"; UPLOAD.mkdir(exist_ok=True)
CLF = C.N2SFClassifier(model=os.environ.get("N2SF_MODEL", "accurate"), warmup=False)
JOB = {"state": "idle", "events": []}
LOGS = []                      # 최근 분류 로그(모니터링)
DATASETS = {}                  # 업로드된 라벨셋 통계
_lock = threading.Lock()


# ── 학습 데이터 검증(가이드 반영) ───────────────────────────────
def check_dataset(counts: dict) -> dict:
    """등급별 개수 → 학습 가능성/균형 판정 + 권고."""
    o, s, c = counts.get("O", 0), counts.get("S", 0), counts.get("C", 0)
    total = o + s + c
    present = [g for g in ("O", "S", "C") if counts.get(g, 0) > 0]
    mx, mn = max(o, s, c), min(o, s, c)
    if len(present) < 2:
        verdict, level = "학습 불가 — 최소 2개 등급 필요(한 등급만으론 모델 붕괴).", "error"
    elif mn == 0:
        verdict, level = f"경고 — {set('OSC')-set(present)} 등급 0건. 해당 등급은 규칙/기존모델(mix)에 의존.", "warn"
    elif mn < 10:
        verdict, level = "부족 — 등급당 <10건. 전이학습+기존셋 혼합으로 소폭 보정만. (권장 ≥50)", "warn"
    elif mx > mn * 5:
        verdict, level = "불균형 — 최다:최소 >5배. 균형샘플러가 보완하나 소수등급 보강 권장.", "warn"
    else:
        verdict, level = "양호 — 학습 진행 가능.", "ok"
    return {"counts": counts, "total": total, "verdict": verdict, "level": level,
            "guide": "등급당 1건은 '망각 방지 혼합' 덕에 손상은 없으나 학습효과 미미. "
                     "권장: 등급당 50~300건, 대략 균형. C(기밀)는 희소해도 규칙 floor+혼합이 커버."}


def _train_job(labels_file, base, out, mix, mix_ratio):
    def prog(d):
        JOB["events"].append(d); JOB["state"] = d.get("stage", "running")
    try:
        JOB.update({"state": "running", "events": [], "out": out})
        res = T.train_onsite(labels_file, base=base, out=out, mix=mix, mix_ratio=mix_ratio, progress=prog)
        JOB["result"] = res
        if res.get("ok"):
            info = CLF.reload_model(os.path.basename(out), path=out)   # 무중단 적용
            JOB["state"] = "applied"; JOB["reload"] = info
        else:
            JOB["state"] = "failed"
    except Exception as e:
        JOB["state"] = "error"; JOB["error"] = f"{e}\n{traceback.format_exc()[:400]}"


def list_models():
    ms = []
    for d in sorted((_ROOT / "models").glob("*")):
        if (d / "config.json").exists():
            ms.append({"name": d.name, "size_mb": round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file())/1e6, 1),
                       "active": d.name == CLF._active["backend"]})
    return ms


HTML = """<!doctype html><html lang=ko><meta charset=utf-8><title>N²SF 관리 콘솔</title>
<style>
*{box-sizing:border-box}body{margin:0;font-family:system-ui,'Malgun Gothic';color:#1a2233;background:#f5f6f8}
.side{position:fixed;top:0;left:0;width:200px;height:100%;background:#22304a;color:#cfd9ea;padding:16px 0}
.side h2{color:#fff;font-size:1rem;padding:0 18px;margin:6px 0 16px}
.side a{display:block;color:#cfd9ea;text-decoration:none;padding:11px 18px;font-size:.9rem;cursor:pointer}
.side a:hover,.side a.on{background:#2f4468;color:#fff;border-left:3px solid #6ea8fe}
.main{margin-left:200px;padding:22px 28px;max-width:900px}
h1{font-size:1.25rem}.card{background:#fff;border:1px solid #e3e7ee;border-radius:8px;padding:16px;margin:12px 0}
.kpi{display:inline-block;min-width:120px;margin-right:12px}.kpi b{font-size:1.5rem;display:block}
textarea,input,select{width:100%;font:inherit;padding:6px;border:1px solid #ccd;border-radius:5px}
button{background:#2b4c7e;color:#fff;border:0;border-radius:6px;padding:8px 14px;font-weight:700;cursor:pointer}
button.g{background:#6b7280}pre{background:#f5f6f8;padding:10px;border-radius:6px;white-space:pre-wrap;font-size:.82rem;max-height:280px;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:.86rem}td,th{padding:7px;border-bottom:1px solid #eee;text-align:left}
.pill{padding:2px 8px;border-radius:10px;font-size:.72rem;font-weight:700}
.ok{background:#e9f3ec;color:#2e7d57}.warn{background:#fbf3e2;color:#9a7b1f}.error{background:#fbe9e7;color:#c0392b}
.hide{display:none}
</style>
<div class=side><h2>🛡 N²SF Admin</h2>
<a onclick="nav('dash')" id=m_dash class=on>대시보드</a>
<a onclick="nav('cls')" id=m_cls>분류 테스트</a>
<a onclick="nav('data')" id=m_data>데이터 관리</a>
<a onclick="nav('train')" id=m_train>학습</a>
<a onclick="nav('models')" id=m_models>모델 버전</a>
<a onclick="nav('mon')" id=m_mon>모니터링</a>
<a onclick="nav('set')" id=m_set>설정</a></div>
<div class=main>
<section id=s_dash><h1>대시보드</h1><div class=card>
<div class=kpi>현재 모델<b id=d_model>-</b></div><div class=kpi>학습 상태<b id=d_job>-</b></div>
<div class=kpi>누적 분류<b id=d_cnt>0</b></div></div>
<div class=card><b>파이프라인</b><br>T1 정규식 → T2 NER → T3 뉴럴(증류) + 튜닝 앙상블 · 오프라인 · §9(개인정보=S)</div></section>

<section id=s_cls class=hide><h1>분류 테스트</h1><div class=card>
<textarea id=c_txt rows=5 placeholder="문서 텍스트"></textarea>
<p><button onclick=cls()>분류</button></p><pre id=c_out></pre></div></section>

<section id=s_data class=hide><h1>데이터 관리 (고객 라벨)</h1><div class=card>
<p>CSV 붙여넣기 (헤더 <code>text,human_label</code> · 라벨=C/S/O). 또는 서버경로.</p>
<textarea id=d_csv rows=6 placeholder="text,human_label&#10;내부 협의 안내...,S&#10;[대외비] 전략...,C&#10;채용 공고...,O"></textarea>
<p><input id=d_name placeholder="데이터셋 이름 (예: siteA_batch1)"> </p>
<p><button onclick=upl()>업로드·검증</button></p><pre id=d_out></pre></div></section>

<section id=s_train class=hide><h1>학습 (고객 데이터로 모델 향상)</h1><div class=card>
<p>데이터셋 <select id=t_ds></select> · 베이스 <select id=t_base><option>models/n2sf-xlmr-large</option><option>models/n2sf-xlmr-official</option></select></p>
<p>혼합비(기존셋) <input id=t_mix type=number value=0.5 step=0.1 style=width:80px> (망각방지)</p>
<p><button onclick=trn()>학습 시작</button> <button class=g onclick=tst()>상태 새로고침</button></p>
<pre id=t_out></pre></div></section>

<section id=s_models class=hide><h1>모델 버전 (적용/롤백)</h1><div class=card>
<button class=g onclick=lm()>목록 새로고침</button><table id=m_tbl></table>
<p>수동 적용: <input id=m_in placeholder="티어 또는 models/n2sf-custom-vN" style=width:60%> <button onclick=rl()>무중단 적용</button></p>
<pre id=m_out></pre></div></section>

<section id=s_mon class=hide><h1>모니터링</h1><div class=card>
<button class=g onclick=lg()>새로고침</button><pre id=mon_out></pre></div></section>

<section id=s_set class=hide><h1>설정 (더미)</h1><div class=card>
앙상블: soft · 티어가중 rules0.3/ner0.3/neural4.0 · early-exit ON · locale ko<br>
오프라인: HF_HUB_OFFLINE=1 · 모델=로컬 · NER=spaCy 로컬<br>
<small>런타임 변경은 GradeProfile/환경변수로(별도).</small></div></section>
</div>
<script>
function nav(x){for(let s of document.querySelectorAll('section'))s.classList.add('hide');
 document.getElementById('s_'+x).classList.remove('hide');
 for(let a of document.querySelectorAll('.side a'))a.classList.remove('on');document.getElementById('m_'+x).classList.add('on');
 if(x=='dash')stat();if(x=='models')lm();if(x=='mon')lg();if(x=='train')fillds();}
async function J(u,b){let r=await fetch(u,{method:'POST',body:JSON.stringify(b)});return r.json()}
async function G(u){return (await fetch(u)).json()}
async function stat(){let d=await G('/api/status');d_model.textContent=d.model;d_job.textContent=d.job;d_cnt.textContent=d.classified}
async function cls(){c_out.textContent=JSON.stringify(await J('/api/classify',{text:c_txt.value}),0,2)}
async function upl(){let d=await J('/api/upload_labels',{name:d_name.value||'ds',csv:d_csv.value});
 d_out.innerHTML=JSON.stringify(d,0,2)+(d.level?`<br><span class=pill ${d.level}>${d.verdict}</span><br><small>${d.guide}</small>`:'')}
async function fillds(){let d=await G('/api/datasets');t_ds.innerHTML=Object.keys(d).map(k=>`<option>${k}</option>`).join('')||'<option>(업로드 먼저)</option>'}
async function trn(){t_out.textContent=JSON.stringify(await J('/api/train',{dataset:t_ds.value,base:t_base.value,mix_ratio:+t_mix.value}),0,2)}
async function tst(){t_out.textContent=JSON.stringify(await G('/api/train/status'),0,2)}
async function lm(){let d=await G('/api/models');m_tbl.innerHTML='<tr><th>모델</th><th>MB</th><th></th></tr>'+
 d.map(m=>`<tr><td>${m.active?'✅ ':''}${m.name}</td><td>${m.size_mb}</td><td><button onclick="rln('${m.name}')">적용</button></td></tr>`).join('')}
async function rl(){m_out.textContent=JSON.stringify(await J('/api/reload',m_in.value.includes('/')?{model:m_in.value.split('/').pop(),path:m_in.value}:{model:m_in.value}),0,2);lm()}
async function rln(n){m_out.textContent=JSON.stringify(await J('/api/reload',{model:n}),0,2);lm()}
async function lg(){mon_out.textContent=JSON.stringify(await G('/api/logs'),0,2)}
stat();
</script></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _s(self, code, obj, ctype="application/json"):
        b = obj.encode() if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", ctype+"; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        p = self.path
        if p == "/" : return self._s(200, HTML, "text/html")
        if p == "/api/status": return self._s(200, {"model": CLF.model_version, "job": JOB["state"], "classified": len(LOGS)})
        if p == "/api/models": return self._s(200, list_models())
        if p == "/api/datasets": return self._s(200, DATASETS)
        if p == "/api/train/status": return self._s(200, JOB)
        if p == "/api/logs": return self._s(200, LOGS[-30:][::-1])
        return self._s(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._s(400, {"error": f"bad json: {e}"})
        p = self.path
        if p == "/api/classify":
            try:
                r = CLF.classify(req.get("text", ""), is_file=False)
                LOGS.append({"t": time.strftime("%H:%M:%S"), "grade": r["grade"], "conf": r["confidence"],
                             "tiers": r["tiersRun"], "model": r["modelVersion"]})
                return self._s(200, r)
            except Exception as e: return self._s(500, {"error": str(e)})
        if p == "/api/upload_labels":
            name = req.get("name", "ds"); csv = req.get("csv", "")
            f = UPLOAD / f"{name}.csv"; f.write_text(csv, encoding="utf-8")
            try:
                rows = T.read_labels(str(f))
            except Exception as e:
                return self._s(400, {"error": f"라벨 파싱 실패: {e}"})
            from collections import Counter
            counts = dict(Counter(g for _, g in rows))
            chk = check_dataset(counts); chk["file"] = str(f); chk["rows"] = len(rows)
            DATASETS[name] = {"file": str(f), "counts": counts, "rows": len(rows)}
            return self._s(200, chk)
        if p == "/api/train":
            with _lock:
                if JOB["state"] == "running": return self._s(409, {"error": "학습 진행중"})
                ds = DATASETS.get(req.get("dataset", ""))
                if not ds: return self._s(400, {"error": "데이터셋 없음 — 먼저 업로드"})
                out = f"models/n2sf-custom-v{int(time.time())}"
                threading.Thread(target=_train_job, args=(ds["file"], req.get("base", "models/n2sf-xlmr-large"),
                                 out, "distill_o4/teacher_labels_n2sf.jsonl", req.get("mix_ratio", 0.5)),
                                 daemon=True).start()
                return self._s(202, {"started": True, "out": out, "poll": "/api/train/status"})
        if p == "/api/reload":
            try: return self._s(200, CLF.reload_model(req["model"], path=req.get("path")))
            except Exception as e: return self._s(500, {"error": str(e)})
        return self._s(404, {"error": "not found"})


def main():
    port = int(os.environ.get("PORT", "8080"))
    print(f"[admin] N²SF 관리 콘솔 http://localhost:{port} · 모델 {CLF.model_version}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    main()
