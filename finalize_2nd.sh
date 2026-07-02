#!/bin/bash
# 2차 테스트 자동 마무리(수정판) — 학습 완료(모델 파일) + L3 반복 N회 누적되면
# 평가루프를 멈추고 Gemma(L3) 비교 → 최종 웹 리포트 생성. (pgrep 오탐 회피: 반복수로 트리거)
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"
PY="$DIR/.venv/bin/python"
MIN_ITERS="${1:-3}"

itercount() {
  "$PY" -c "import sqlite3,os;p='test2/results.db';print(sqlite3.connect(p).execute('select count(*) from iterations').fetchone()[0] if os.path.exists(p) else 0)" 2>/dev/null
}

caffeinate -ims nohup bash -c "
  cd '$DIR'
  echo '[finalize2] 대기 시작 '\$(date)
  # 1) 학습 완료(모델 파일 존재) 대기
  while [ ! -f models/mdeberta-n2sf/config.json ]; do sleep 30; done
  echo '[finalize2] 모델 확인 '\$(date)
  # 2) L3 반복 N회 누적 대기
  while true; do
    n=\$('$PY' -c \"import sqlite3,os;p='test2/results.db';print(sqlite3.connect(p).execute('select count(*) from iterations').fetchone()[0] if os.path.exists(p) else 0)\" 2>/dev/null)
    [ \"\$n\" -ge $MIN_ITERS ] && break
    sleep 90
  done
  echo '[finalize2] L3 반복 '\$n' 도달 — 평가루프 정지 후 Gemma 비교'
  # 3) 평가루프 정지(MPS 확보)
  pkill -f 'harness.autoloop'; sleep 6
  # 4) Gemma(L3) 비교 + 최종 웹 리포트
  '$PY' -m harness.finalize2 test2
  touch test2/FINAL_DONE
  echo '[finalize2] 완료 '\$(date)
" >> test2/finalize.log 2>&1 &
echo "PID=$! finalize2 시작 (L3 ${MIN_ITERS}회 후 자동 Gemma 비교+리포트). 로그: test2/finalize.log"
