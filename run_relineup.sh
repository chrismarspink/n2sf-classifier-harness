#!/bin/bash
# 전 백본 GPT-5.4 재증류 — 균형증강 교사셋 + train_kd 개선. **재개 가능(idempotent)**.
#  이미 완성된 models/n2sf-{name}-v2/config.json 은 건너뜀 → 크래시/재부팅 후 재실행하면 이어서 진행.
#  교사셋: distill_o4/teacher_labels_gpt5.jsonl · 확인: tail -f kd/relineup.log
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"; mkdir -p kd
if pgrep -f "Python -m harness.train_kd" >/dev/null 2>&1; then echo "train_kd 실행중"; exit 0; fi
PY="$DIR/.venv/bin/python"; SET="distill_o4/teacher_labels_gpt5.jsonl"; HO="distill_soft/heldout.json"

caffeinate -ims nohup bash -c "
  cd '$DIR'
  echo \"=== 재증류(재개형) 시작 \$(date +%H:%M) ===\"

  train() {  # \$1=name \$2=base \$3=batch \$4=accum \$5=top \$6=epochs \$7=hours
    if [ -f \"models/n2sf-\$1-v2/config.json\" ]; then echo \"--- \$1: 이미 완료(건너뜀) ---\"; return; fi
    echo \"--- \$1 (\$2) \$(date +%H:%M) ---\"
    '$PY' -m harness.train_kd --soft-jsonl $SET --heldout $HO \
       --base \"\$2\" --out \"models/n2sf-\$1-v2\" \
       --epochs \"\$6\" --batch \"\$3\" --accum \"\$4\" --maxlen 128 \
       --temperature 1.5 --alpha 0.1 --balance --train-top \"\$5\" \
       --lr 2e-5 --warmup 0.1 --max-hours \"\$7\"
  }

  train small       monologg/koelectra-small-v3-discriminator 32 1 0 6 0.8
  train klue-large  klue/roberta-large                         8 2 6 4 1.8
  train xlmr-large  xlm-roberta-large                          6 3 6 4 2.2

  echo \"=== 재증류 완료 \$(date +%H:%M) ===\"
" >> kd/relineup.log 2>&1 &
echo "PID=$! 재증류(재개형) 시작. 로그: kd/relineup.log"
