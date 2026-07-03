#!/bin/bash
# item1(GPT-5.4 교사): S+C경계+O 균형 증강 → n2sf-base 재증류(누락0 유지하며 0.8 도전).
#  1) aug_s: gpt-4o 생성 + GPT-5.4 라벨(최강 교사) → distill_o4/teacher_labels_gpt5.jsonl
#  2) train_kd: 균형샘플+온도+maxlen128 → models/n2sf-base-kd3 (C재현율 우선 best 저장)
#  확인: tail -f kd/aug5.log | 중지: pkill -f harness.aug_s; pkill -f harness.train_kd
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"; mkdir -p kd
if pgrep -f "Python -m harness.aug_s" >/dev/null 2>&1 || pgrep -f "Python -m harness.train_kd" >/dev/null 2>&1; then
  echo "이미 실행중"; exit 0; fi
PY="$DIR/.venv/bin/python"

caffeinate -ims nohup bash -c "
  cd '$DIR'
  echo \"=== [1/2] GPT-5.4 균형 증강 시작 \$(date +%H:%M) ===\"
  '$PY' -m harness.aug_s \
      --base-jsonl distill_o4/teacher_labels.jsonl \
      --out distill_o4/teacher_labels_gpt5.jsonl \
      --n-s 250 --n-c 200 --n-o 50 --gen azure:gpt-4o --teacher gpt-5 --max-hours 2.5
  echo \"=== [2/2] 재증류 시작 \$(date +%H:%M) ===\"
  '$PY' -m harness.train_kd \
      --soft-jsonl distill_o4/teacher_labels_gpt5.jsonl \
      --heldout distill_soft/heldout.json \
      --out models/n2sf-base-kd3 \
      --epochs 6 --batch 16 --accum 2 --maxlen 128 \
      --temperature 1.5 --alpha 0.1 --balance \
      --lr 2e-5 --warmup 0.1 --max-hours 2.0
  echo \"=== 완료 \$(date +%H:%M) → models/n2sf-base-kd3 ===\"
" >> kd/aug5.log 2>&1 &
echo "PID=$! GPT-5.4 증강+재증류 시작. 로그: kd/aug5.log | 결과: models/n2sf-base-kd3"
