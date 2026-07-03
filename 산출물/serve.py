#!/usr/bin/env python3
"""serve.py — N²SF 등급분류 HTTP API (표준 라이브러리만, 외부 의존 없음).

로컬/온프레미스 분류 서비스. 모델·엔진을 담은 Docker 이미지의 진입점으로 사용.
  POST /classify   body: {"text": "...", "model": "n2sf-base"}   또는 {"file": "/path"}
  GET  /health
응답: classify_n2sf 결과(grade, confidence, shap, tiers, _tiersRun 등).
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import classify_n2sf as C

DEFAULT_MODEL = os.environ.get("N2SF_MODEL", "n2sf-base")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": DEFAULT_MODEL})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/classify":
            self._send(404, {"error": "not found"}); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, {"error": f"bad json: {e}"}); return
        model = req.get("model", DEFAULT_MODEL)
        try:
            if req.get("text") is not None:
                text = req["text"]
            elif req.get("file"):
                import data_classifier as dc
                text, _, _ = dc.extract_text(req["file"])
            else:
                self._send(400, {"error": "text 또는 file 필요"}); return
            r = C.classify_with_early_exit(text, model=model)
            self._send(200, {"grade": r["gradeCode"], "gradeLabel": r["gradeLabel"],
                             "confidence": r["confidence"], "score": r["score"],
                             "tiersRun": r.get("_tiersRun"), "neuralSkipped": r.get("_neuralSkipped"),
                             "shap": r.get("shap"), "compliance": r.get("compliance")})
        except Exception as e:
            self._send(500, {"error": str(e)})


def main():
    port = int(os.environ.get("PORT", "8080"))
    print(f"[serve] N²SF 분류 API 시작 :{port} (모델 {DEFAULT_MODEL})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
