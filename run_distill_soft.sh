#!/bin/bash
# soft-label 지식증류 무인 파이프라인 — 교사(GPT-4o)가 다양 문서에 O/S/C 확률을 매기고
# 학생(mdeberta-n2sf)이 KD로 모방 → held-out(정직 분포) 일반화 향상. (detached·caffeinate·가드)
#  결과: models/mdeberta-n2sf/(재학습), distill_soft/report.html, distill_soft/cycles.json
#  확인: tail -f distill_soft/distill.log | 중지: pkill -f harness.distill; pkill -f harness.train_soft
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"; mkdir -p distill_soft
if pgrep -f "Python -m harness.distill" >/dev/null 2>&1; then echo "distill 이미 실행중. 중복 방지."; exit 0; fi
[ -f .env.azure ] || { echo ".env.azure 없음"; exit 1; }
# 동결 평가셋·LLM 기준선 재사용
[ -f distill/heldout.json ]       && cp -n distill/heldout.json distill_soft/ 2>/dev/null
[ -f distill/llm_baselines.json ] && cp -n distill/llm_baselines.json distill_soft/ 2>/dev/null

caffeinate -ims nohup "$DIR/.venv/bin/python" -m harness.distill \
    --out distill_soft --hours "${1:-12}" \
    --label-per-cycle 400 --train-epochs 3 --train-hours 0.6 \
    >> distill_soft/distill.log 2>&1 &
echo "PID=$! soft-label 증류 시작. 로그: distill_soft/distill.log | 결과: distill_soft/report.html + models/mdeberta-n2sf/"
