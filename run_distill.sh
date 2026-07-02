#!/bin/bash
# 증류(distillation) 무인 파이프라인 — GPT-4o/o4-mini가 만든 다양한 문서를 대량 학습데이터로 써서
# 소형 온디바이스 모델 mdeberta-n2sf를 재학습 → held-out(정직 분포) 일반화 향상.
# 지난 실패(템플릿4000:다양화24로 다양화가 묻힘) 교정: 다양화 대량 + 템플릿 축소(cap 600).
#  결과: models/mdeberta-n2sf/(재학습 모델), distill/report.html, distill/cycles.json
#  확인: tail -f distill/improve.log  | 중지: pkill -f harness.improve; pkill -f harness.train
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"; mkdir -p distill
if pgrep -f "Python -m harness.improve" >/dev/null 2>&1; then echo "improve 이미 실행중. 중복 방지."; exit 0; fi
[ -f .env.azure ] || { echo ".env.azure 없음"; exit 1; }

# 동결 평가셋·LLM 기준선 재사용(재측정 비용 절감) + 기존 다양화 데이터 시드
[ -f improve/heldout.json ]        && cp -n improve/heldout.json distill/ 2>/dev/null
[ -f improve/llm_baselines.json ]  && cp -n improve/llm_baselines.json distill/ 2>/dev/null
[ -f improve/train_llm.jsonl ]     && cp -n improve/train_llm.jsonl distill/ 2>/dev/null

caffeinate -ims nohup "$DIR/.venv/bin/python" -m harness.improve \
    --out distill --hours "${1:-12}" \
    --bootstrap 40 --gen-per-cycle 6 \
    --train-cap 600 --train-epochs 3 --train-hours 0.6 \
    >> distill/improve.log 2>&1 &
echo "PID=$! 증류 파이프라인 시작. 로그: distill/improve.log | 결과: distill/report.html + models/mdeberta-n2sf/"
