#!/bin/bash
# 다중 백본 라인업 무인 실행 — 스몰/베이스/KLUE-large/XLM-large 증류 + 라지 앙상블 → 3축 평가.
#  교사라벨: distill_o4/teacher_labels.jsonl(3968) · held-out: distill_soft/heldout.json
#  결과: lineup/matrix.json, lineup/report.html, models/n2sf-{small,base,klue-large,xlmr-large}/
#  확인: tail -f lineup/lineup.log | 중지: pkill -f harness.lineup; pkill -f harness.train_soft
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"; mkdir -p lineup
if pgrep -f "Python -m harness.lineup" >/dev/null 2>&1; then echo "lineup 이미 실행중"; exit 0; fi
caffeinate -ims nohup "$DIR/.venv/bin/python" -m harness.lineup \
    --labels distill_o4/teacher_labels.jsonl --heldout distill_soft/heldout.json --out lineup \
    >> lineup/lineup.log 2>&1 &
echo "PID=$! 라인업 시작. 로그: lineup/lineup.log | 결과: lineup/matrix.json, report.html"
