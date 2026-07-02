#!/bin/bash
# 옵션1: S·경계 문서 증강 → n2sf-base 재증류(누락0 유지하며 0.8 도전).
#  1) aug_s: gpt-4o 생성 + o4-mini 라벨 → distill_o4/teacher_labels_aug.jsonl
#  2) train_kd: 증강셋으로 재증류 → models/n2sf-base-kd2 (best-on-heldout, C재현율 우선)
#  확인: tail -f kd/aug.log | 중지: pkill -f harness.aug_s; pkill -f harness.train_kd
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"; mkdir -p kd
if pgrep -f "Python -m harness.aug_s" >/dev/null 2>&1 || pgrep -f "Python -m harness.train_kd" >/dev/null 2>&1; then
  echo "이미 실행중"; exit 0; fi
PY="$DIR/.venv/bin/python"

caffeinate -ims nohup bash -c "
  cd '$DIR'
  echo \"=== [1/2] S·경계 증강 시작 \$(date +%H:%M) ===\"
  '$PY' -m harness.aug_s \
      --base-jsonl distill_o4/teacher_labels.jsonl \
      --out distill_o4/teacher_labels_aug.jsonl \
      --n-s 300 --n-bound 100 --gen azure:gpt-4o --teacher o4-mini --max-hours 1.5
  echo \"=== [2/2] 재증류 시작 \$(date +%H:%M) ===\"
  '$PY' -m harness.train_kd \
      --soft-jsonl distill_o4/teacher_labels_aug.jsonl \
      --heldout distill_soft/heldout.json \
      --out models/n2sf-base-kd2 \
      --epochs 6 --batch 16 --accum 2 --maxlen 128 \
      --temperature 1.5 --alpha 0.1 --balance \
      --lr 2e-5 --warmup 0.1 --max-hours 2.0
  echo \"=== 완료 \$(date +%H:%M) → models/n2sf-base-kd2 ===\"
" >> kd/aug.log 2>&1 &
echo "PID=$! 증강+재증류 시작. 로그: kd/aug.log | 결과: models/n2sf-base-kd2"
