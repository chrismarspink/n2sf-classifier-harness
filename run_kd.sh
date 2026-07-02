#!/bin/bash
# base 0.8 돌파 재증류 — 개선 KD(균형샘플링+온도T+하드혼합+maxlen256+cosine+best저장).
#  교사라벨: distill_o4/teacher_labels.jsonl(3968) · held-out: distill_soft/heldout.json
#  산출: models/n2sf-base-kd/  (기존 n2sf-base는 보존 → 좋으면 수동 승격)
#  확인: tail -f kd/kd.log | 중지: pkill -f harness.train_kd
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"; mkdir -p kd
if pgrep -f "Python -m harness.train_kd" >/dev/null 2>&1; then echo "train_kd 이미 실행중"; exit 0; fi
caffeinate -ims nohup "$DIR/.venv/bin/python" -m harness.train_kd \
    --soft-jsonl distill_o4/teacher_labels.jsonl \
    --heldout distill_soft/heldout.json \
    --out models/n2sf-base-kd \
    --epochs 5 --batch 16 --accum 2 --maxlen 128 \
    --temperature 1.5 --alpha 0.1 --balance \
    --lr 2e-5 --warmup 0.1 --max-hours 2.0 \
    >> kd/kd.log 2>&1 &
echo "PID=$! KD 재증류 시작. 로그: kd/kd.log | best 모델: models/n2sf-base-kd/"
