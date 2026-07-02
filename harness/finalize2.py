"""finalize2.py — 2차 테스트 마무리: Gemma(L3) 비교 측정 + 최종 웹 리포트 생성.

전제: 학습(mdeberta-n2sf) + L3 평가루프가 충분히 돌아 test2/results.db 에 반복이 쌓였고,
      autoloop·학습이 **종료**되어 MPS/메모리가 비어 Gemma 를 안전하게 돌릴 수 있는 상태.

동작:
  1) summarize 가 고른 추천 반복(run_id)을 기준으로 그 반복의 test 문서에 Gemma2:9b 분류 실행
     → llm_predictions 저장 (visualize 의 LLM_vs_BERT 가 동일 반복으로 매칭되도록 정렬)
  2) visualize 로 test2/report.html 재생성 (Gemma 비교 포함)
  3) 완료 후 Gemma 언로드.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .db import DB
from . import detect as DET
from . import summarize as SUM
from . import visualize as VIZ
from . import ollama_baseline as OLL


def main(out="test2", model="gemma2:9b", max_files=240):
    out = Path(out)
    db = DB(out / "results.db")
    s = SUM.compute(db)
    if not s.get("recommended"):
        print("[finalize2] 추천 반복 없음(반복 데이터 부족) — 중단"); return
    run = s["recommended"]["run_id"]
    print(f"[finalize2] 기준 반복(run_id)={run}  (도달난이도 {s['levels_reached']})")

    corpus = db.query("SELECT * FROM corpus WHERE doc_id LIKE ?", (run + "-%",))
    meta = {(r["doc_id"], r["fmt"]): {"grade": r["grade"], "category": r["category"],
                                      "split": r["split"]} for r in corpus}
    det = DET.load_detection(db, doc_prefix=run + "-")
    print(f"[finalize2] 대상 문서 {len(meta)} / 탐지캐시 {len(det)} — Gemma 분류 시작")

    summary = OLL.run_ollama_baseline(db, meta, det, split="test", model=model,
                                      max_files=max_files, concurrency=1)
    print(f"[finalize2] Gemma 결과: {summary}")

    # 최종 웹 리포트 재생성(자체완결 HTML)
    content = VIZ.build_content(db, None)
    (out / "report.html").write_text(
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>2차 테스트 — mdeberta-n2sf · L3 · Gemma 비교</title>'
        '<style>body{margin:0;background:#F6F7F9}</style></head><body>'
        + content + '</body></html>', encoding="utf-8")
    (out / "report_artifact.html").write_text(content, encoding="utf-8")
    print(f"[finalize2] 최종 웹 리포트 생성 → {out/'report.html'}")
    db.close()

    # Gemma 언로드(메모리 반환)
    try:
        import urllib.request, json
        req = urllib.request.Request("http://localhost:11434/api/chat",
            data=json.dumps({"model": model, "keep_alive": 0,
                             "messages": [{"role": "user", "content": "x"}]}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=60).read()
    except Exception:
        pass
    print("[finalize2] 완료.")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(out=a[0] if a else "test2")
