#!/bin/bash
# 2차 테스트 무인 실행 — mdeberta-n2sf 파인튜닝 → L3(적대 난독화) 돌파 자동 평가.
#  1) mdeberta-n2sf 학습(체크포인트 저장, 에폭별)         → models/mdeberta-n2sf/
#  2) L3 고정 자동 평가 루프(① 정규화 + mdeberta-n2sf 포함) → test2/
# 1차 자료(weekend/·harness_out/)는 건드리지 않는다. 터미널 닫아도/sleep 돼도 계속.
#  확인: tail -f test2/2차.log   결과: test2/report.html, test2/WEEKEND_SUMMARY.md
#  중지: pkill -f 'harness.train'; pkill -f 'harness.autoloop'
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
mkdir -p test2 models

# 중복 방지
if pgrep -f "Python -m harness.train" >/dev/null 2>&1 || pgrep -f "Python -m harness.autoloop" >/dev/null 2>&1; then
  echo "이미 학습/평가가 실행 중입니다. 중복 실행 방지."
  exit 0
fi

PY="$DIR/.venv/bin/python"

caffeinate -ims nohup bash -c "
  cd '$DIR'
  echo '=== [1/2] mdeberta-n2sf 파인튜닝 시작 ' \$(date) ' ==='
  '$PY' -m harness.train --db weekend/results.db --out models/mdeberta-n2sf \
       --epochs 3 --per-class-cap 6000 --batch 16 --accum 2 --maxlen 128 --max-hours 5
  echo '=== [2/2] L3 자동 평가 루프 시작 ' \$(date) ' ==='
  '$PY' -m harness.autoloop --out test2 --start-level 3 --max-level 3 \
       --per-cell 8 --normalize --hours 10 \
       --models 'mdeberta-n2sf,mdeberta,klue-roberta,ko-sroberta,minilm'
  echo '=== 2차 테스트 완료 ' \$(date) ' ==='
" >> test2/2차.log 2>&1 &

echo "PID=$! 로 2차 테스트 시작됨. 로그: test2/2차.log"
echo "진행: tail -f test2/2차.log   |   결과(완료 후): test2/report.html, test2/WEEKEND_SUMMARY.md"
echo "중지: pkill -f harness.train; pkill -f harness.autoloop"
