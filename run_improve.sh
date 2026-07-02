#!/bin/bash
# 지속 개선 무인 루프 — LLM(Azure) 다양화 데이터로 재학습 → held-out(다른 분포) 정직 평가 →
# 모델 재선정 → 사이클 반복. 과적합 해소·일반화 향상이 목표. (detached·caffeinate·중복가드)
#  확인: tail -f improve/improve.log  |  결과: improve/report.html, improve/cycles.json
#  중지: pkill -f harness.improve ; pkill -f harness.train
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"
mkdir -p improve

if pgrep -f "Python -m harness.improve" >/dev/null 2>&1; then
  echo "이미 improve 루프 실행 중. 중복 방지."; exit 0
fi
[ -f .env.azure ] || { echo ".env.azure 없음 — Azure 자격 필요"; exit 1; }

caffeinate -ims nohup "$DIR/.venv/bin/python" -m harness.improve \
    --out improve --hours "${1:-10}" \
    --gen-per-cycle 2 --heldout-per 2 \
    --train-cap 4000 --train-epochs 2 --train-hours 0.4 \
    >> improve/improve.log 2>&1 &
echo "PID=$! improve 루프 시작. 로그: improve/improve.log | 결과: improve/report.html"
