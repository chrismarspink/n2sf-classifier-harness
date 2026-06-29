#!/bin/bash
# 주말 무인 실행 — 터미널을 닫아도/로그아웃해도 살아남고, 맥이 잠들지 않게 한다.
# 사용:  bash run_weekend.sh
# 확인:  tail -f weekend/autoloop.log
# 중지:  pkill -f harness.autoloop
# 재개:  같은 명령 다시 실행(중단 지점부터 이어감)
cd "$(dirname "$0")"
mkdir -p weekend

# 중복 실행 방지: 이미 python autoloop 가 돌고 있으면 새로 띄우지 않는다(같은 DB 충돌 방지).
# (caffeinate 래퍼는 소문자 .venv/bin/python, 실제 워커는 대문자 Python 프레임워크 경로로 구분)
if pgrep -f "Python -m harness.autoloop" >/dev/null 2>&1; then
  echo "이미 실행 중입니다 (PID $(pgrep -f 'Python -m harness.autoloop' | tr '\n' ' '))."
  echo "중복 실행을 막았습니다. 중지하려면: pkill -f harness.autoloop"
  exit 0
fi

# caffeinate: 디스플레이/시스템/디스크 잠들기 방지(-i idle, -m disk, -s on-AC, -u user-active)
# nohup + &: 터미널 종료(SIGHUP)에도 계속 실행
caffeinate -ims nohup .venv/bin/python -m harness.autoloop \
    --out weekend \
    --hours 60 \
    --per-cell 8 \
    --start-level 1 --max-level 4 \
    --models "minilm,ko-sroberta,klue-roberta,koelectra,kcbert,mbert,xlm-roberta,mpnet,e5,mdeberta,xlmr-xnli,labse" \
    > weekend/autoloop.log 2>&1 &

echo "PID=$! 로 백그라운드 시작됨. 로그: weekend/autoloop.log"
echo "진행 확인:  tail -f weekend/autoloop.log   또는  cat weekend/autoloop_status.json"
echo "월요일 결과: weekend/report.xlsx, weekend/OPTIMIZED_MODEL.md, weekend/results.db"
