#!/usr/bin/env python3
"""serve_v3.py — N²SF 분류 서비스 + 온사이트 학습/무중단 핫리로드 (v3).

표준 라이브러리만. 온프레미스 CPU 서버 진입점.
엔드포인트:
  GET  /                 최소 웹 UI(분류·라벨업로드학습·핫리로드)
  GET  /health           상태·현재 모델버전
  POST /classify         {"text": "..."} 또는 {"file":"/path"} → 등급 결과
  POST /train            {"labels_path":"..."} 백그라운드 학습 시작 → {job_id}
  GET  /train/status      마지막 학습 상태(진행/완료)
  POST /reload           {"model":"티어/백엔드","path":"models/..."} → 무중단 교체

특징: 학습은 백그라운드 스레드 → 서비스 무중단. 학습 완료 후 /reload 로 새 모델 원자적 스왑(다운타임 0).
"""
from __future__ import annotations
import json, os, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import classifier_v3 as C
import train_onsite as T

CLF = C.N2SFClassifier(model=os.environ.get("N2SF_MODEL", "accurate"), warmup=False)
JOB = {"state": "idle", "events": []}
_lock = threading.Lock()


def _train_job(labels_path, base, out, mix, mix_ratio):
    def prog(d):
        JOB["events"].append(d); JOB["state"] = d.get("stage", "running")
    try:
        JOB.update({"state": "running", "events": []})
        res = T.train_onsite(labels_path, base=base, out=out, mix=mix, mix_ratio=mix_ratio, progress=prog)
        JOB["result"] = res
        if res.get("ok"):
            # 학습 성공 → 무중단 핫리로드
            info = CLF.reload_model(os.path.basename(out), path=out)
            JOB["state"] = "reloaded"; JOB["reload"] = info
        else:
            JOB["state"] = "failed"
    except Exception as e:
        JOB["state"] = "error"; JOB["error"] = f"{e}\n{traceback.format_exc()[:500]}"


UI = """<!doctype html><meta charset=utf-8><title>N²SF v3</title>
<style>body{font-family:system-ui,'Malgun Gothic';max-width:760px;margin:24px auto;padding:0 16px;color:#1a2233}
h1{font-size:1.3rem}fieldset{border:1px solid #dde;border-radius:8px;margin:14px 0;padding:14px}
legend{font-weight:700;color:#2b4c7e}textarea,input{width:100%;box-sizing:border-box;font:inherit}
button{background:#2b4c7e;color:#fff;border:0;border-radius:6px;padding:8px 14px;font-weight:700;cursor:pointer}
pre{background:#f5f6f8;padding:10px;border-radius:6px;white-space:pre-wrap;font-size:.85rem}</style>
<h1>N²SF 등급분류 v3 · <span id=ver></span></h1>
<fieldset><legend>1. 문서 분류</legend>
<textarea id=t rows=5 placeholder="문서 텍스트 붙여넣기"></textarea>
<p><button onclick=cls()>분류</button></p><pre id=r></pre></fieldset>
<fieldset><legend>2. 온사이트 학습 (고객 라벨 파일 경로)</legend>
<input id=lp placeholder="서버 내 라벨 파일 경로 (CSV: text,human_label)">
<p><button onclick=trn()>학습 시작</button> <button onclick=st()>상태</button></p><pre id=tr></pre></fieldset>
<fieldset><legend>3. 무중단 모델 교체</legend>
<input id=mp placeholder="티어(accurate) 또는 models/n2sf-custom-v2">
<p><button onclick=rl()>핫리로드</button></p><pre id=rr></pre></fieldset>
<script>
async function j(u,b){let r=await fetch(u,{method:'POST',body:JSON.stringify(b)});return r.json()}
async function cls(){document.getElementById('r').textContent=JSON.stringify(await j('/classify',{text:t.value}),0,2)}
async function trn(){document.getElementById('tr').textContent=JSON.stringify(await j('/train',{labels_path:lp.value}),0,2)}
async function st(){let r=await fetch('/train/status');document.getElementById('tr').textContent=JSON.stringify(await r.json(),0,2)}
async function rl(){let v=mp.value;document.getElementById('rr').textContent=JSON.stringify(await j('/reload',v.includes('/')?{model:v.split('/').pop(),path:v}:{model:v}),0,2)}
fetch('/health').then(r=>r.json()).then(d=>ver.textContent=d.model)
</script>"""


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj, ctype="application/json"):
        b = obj.encode() if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/": self._send(200, UI, "text/html")
        elif self.path == "/health": self._send(200, {"status": "ok", "model": CLF.model_version})
        elif self.path == "/train/status": self._send(200, JOB)
        else: self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, {"error": f"bad json: {e}"}); return
        if self.path == "/classify":
            try:
                if req.get("text") is not None: r = CLF.classify(req["text"], is_file=False)
                elif req.get("file"): r = CLF.classify(req["file"], is_file=True)
                else: return self._send(400, {"error": "text 또는 file 필요"})
                self._send(200, r)
            except Exception as e: self._send(500, {"error": str(e)})
        elif self.path == "/train":
            with _lock:
                if JOB["state"] in ("running",): return self._send(409, {"error": "학습 진행중"})
                lp = req.get("labels_path")
                if not lp or not os.path.exists(lp): return self._send(400, {"error": "labels_path 파일 없음"})
                out = req.get("out", "models/n2sf-custom-v" + str(int(__import__("time").time())))
                threading.Thread(target=_train_job, args=(lp, req.get("base", "models/n2sf-xlmr-large"),
                                 out, req.get("mix", "distill_o4/teacher_labels_n2sf.jsonl"),
                                 req.get("mix_ratio", 0.5)), daemon=True).start()
                self._send(202, {"started": True, "out": out, "status": "/train/status"})
        elif self.path == "/reload":
            try: self._send(200, CLF.reload_model(req["model"], path=req.get("path")))
            except Exception as e: self._send(500, {"error": str(e)})
        else: self._send(404, {"error": "not found"})


def main():
    port = int(os.environ.get("PORT", "8080"))
    print(f"[serve_v3] N²SF 서비스 :{port} · 모델 {CLF.model_version}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    main()
