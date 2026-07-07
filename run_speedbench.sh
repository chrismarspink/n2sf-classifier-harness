#!/bin/bash
# 속도 벤치마크: §9 등 진행 학습 종료 대기 → 티어별 처리량 실측 + Gemma/o4-mini 비교 + 100MB 전수검사 검증.
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"; mkdir -p kd final
PY="$DIR/.venv/bin/python"
caffeinate -ims nohup bash -c "
  cd '$DIR'
  echo \"=== speedbench 시작 \$(date '+%m-%d %H:%M') — 진행 학습 종료 대기 ===\"
  while pgrep -x Python >/dev/null 2>&1; do sleep 120; done
  echo \"=== 진행 완료, 벤치 시작 \$(date +%H:%M) ===\"
  # 1) 처리량 실측(소용량 1·2·5MB, 전 등급) + Gemma(동일HW)·o4-mini(클라우드) 청크지연
  '$PY' -m harness.speedbench --sizes 1,2,5 --grades O,S,C --model models/n2sf-small \
     --gemma --llm --out final/speed_throughput.json
  # 2) 100MB 전수검사(누락0) 검증 — S(전체 파이프라인)·C(early-exit) 실제 실행
  '$PY' -m harness.speedbench --sizes 100 --grades S,C --model models/n2sf-small \
     --out final/speed_100mb.json
  echo \"=== speedbench 완료 \$(date '+%m-%d %H:%M') ===\"
" >> kd/speedbench.log 2>&1 &
echo "PID=$! speedbench 예약(학습 후 자동). 로그: kd/speedbench.log | 결과: final/speed_throughput.json, speed_100mb.json"
