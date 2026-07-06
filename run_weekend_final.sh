#!/bin/bash
# 주말 자율 실행(재개형) — 다국어 스몰 포함 전 모델 증류 + 최종 매트릭스(교사 비교·전후).
#  0) 현재 train_kd(있으면) 종료 대기
#  1) 다국어 교사라벨 생성 → teacher_labels_multi.jsonl (재개형)
#  2) 증류(각 config.json 있으면 건너뜀):
#       small-multi-e5 / small-multi-minilm / base-ml / xlmr-ml  (다국어 셋)
#  3) 최종 매트릭스: 전 모델 × 한국어/L5/속도/크기 + 교사 비교  → final/
#  재부팅 후: 그냥 다시 ./run_weekend_final.sh (완료분 자동 skip)
#  확인: tail -f kd/weekend.log
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"; mkdir -p kd final
if pgrep -f "Python -m harness.(gen_multi|final_matrix)" >/dev/null 2>&1; then echo "weekend 실행중"; exit 0; fi
PY="$DIR/.venv/bin/python"; MULTI="distill_o4/teacher_labels_multi.jsonl"; HO="distill_soft/heldout.json"

caffeinate -ims nohup bash -c "
  cd '$DIR'
  echo \"=== 주말 실행 시작 \$(date '+%m-%d %H:%M') ===\"

  # 1) 다국어 교사 라벨(영/중/일) — o4-mini(최강 교사)
  echo \"--- [1] 다국어 라벨 생성 \$(date +%H:%M) ---\"
  '$PY' -m harness.gen_multi --base-jsonl distill_o4/teacher_labels_gpt5.jsonl --out $MULTI \
     --langs en,zh,ja --per-cell 60 --gen azure:gpt-4o --teacher o4-mini --max-hours 4.0

  # 2) 증류 (재개형: 완료분 skip)
  train() {  # name base batch accum top epochs hours
    if [ -f \"models/n2sf-\$1/config.json\" ]; then echo \"--- \$1: 완료(skip) ---\"; return; fi
    echo \"--- 증류 \$1 (\$2) \$(date +%H:%M) ---\"
    '$PY' -m harness.train_kd --soft-jsonl $MULTI --heldout $HO --base \"\$2\" --out \"models/n2sf-\$1\" \
       --epochs \"\$6\" --batch \"\$3\" --accum \"\$4\" --maxlen 128 --temperature 1.5 --alpha 0.1 \
       --balance --train-top \"\$5\" --lr 2e-5 --warmup 0.1 --max-hours \"\$7\"
  }
  train small-multi-e5     intfloat/multilingual-e5-small                16 2 0 6 1.0
  train small-multi-minilm microsoft/Multilingual-MiniLM-L12-H384        16 2 0 6 1.0
  train base-ml            MoritzLaurer/mDeBERTa-v3-base-mnli-xnli       16 2 0 6 1.2
  train xlmr-ml            xlm-roberta-large                              6 3 6 4 2.2

  # 3) 최종 매트릭스(전후·교사비교 포함)
  echo \"--- [3] 최종 매트릭스 \$(date +%H:%M) ---\"
  '$PY' -m harness.final_matrix --out final --models \"\
models/n2sf-small,models/n2sf-base-orig,\
models/n2sf-small-v2,models/n2sf-base,models/n2sf-klue-large-v2,models/n2sf-xlmr-large-v2,\
models/n2sf-small-multi-e5,models/n2sf-small-multi-minilm,models/n2sf-base-ml,models/n2sf-xlmr-ml\"

  echo \"=== 주말 실행 완료 \$(date '+%m-%d %H:%M') → final/report.md ===\"
" >> kd/weekend.log 2>&1 &
echo "PID=$! 주말 자율 실행 시작. 로그: kd/weekend.log | 결과: final/report.md, final/final_matrix.json"
