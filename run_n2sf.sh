#!/bin/bash
# N²SF 공식(§9) 전환 파이프라인: §9 데이터 생성·라벨 → xlmr 재증류 → §9 재평가. 재개형.
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"; mkdir -p kd
if pgrep -f "Python -m harness.(gen_n2sf|train_kd)" >/dev/null 2>&1; then echo "실행중"; exit 0; fi
PY="$DIR/.venv/bin/python"; SET="distill_o4/teacher_labels_n2sf.jsonl"; HO="distill_soft/heldout_n2sf.json"
caffeinate -ims nohup bash -c "
  cd '$DIR'
  echo \"=== N²SF §9 전환 시작 \$(date '+%m-%d %H:%M') ===\"
  # 1) §9 데이터 생성·라벨(gpt-4o 생성 + o4-mini §9 라벨)
  if [ ! -f $SET ]; then
    '$PY' -m harness.gen_n2sf --per-cell 35 --langs ko,ko,en,zh,ja --teacher o4-mini \
      --out-train $SET --out-heldout $HO --heldout-frac 0.2 --max-hours 2.0
  fi
  # 2) xlmr §9 재증류
  if [ ! -f models/n2sf-xlmr-official/config.json ]; then
    '$PY' -m harness.train_kd --soft-jsonl $SET --heldout $HO --base xlm-roberta-large \
      --out models/n2sf-xlmr-official --epochs 4 --batch 6 --accum 3 --maxlen 128 \
      --temperature 1.5 --alpha 0.1 --balance --train-top 6 --lr 2e-5 --warmup 0.1 --max-hours 2.5
  fi
  # 3) §9 재평가(전체 파이프라인, §9 규칙)
  '$PY' - <<'PYEOF'
import json
import data_classifier as dc
from harness import metrics as M
ho=json.load(open('distill_soft/heldout_n2sf.json'))
W={'tier':{'rules':0.3,'ner':0.3,'neural':4.0}}
dc.NEURAL_BACKENDS['off']={'label':'off','kind':'finetuned','model':'models/n2sf-xlmr-official','langs':'multi'}
rows=[{'true_grade':h['grade'],'pred_grade':dc.classify_text(h['text'],locale='ko',llm_mode=True,model='off',ensemble_method='soft',weights=W)['gradeCode']} for h in ho]
m=M.compute(rows)
print('=== N²SF §9 재평가(전체 파이프라인) ===')
print('macroF1=%s C재현율=%s 과소(유출)=%s 과대=%s n=%d'%(m['macro_f1'],m['c_recall'],m['under_rate'],m['over_rate'],len(ho)))
print('per_grade:',{g:{'P':v['precision'],'R':v['recall'],'F1':v['f1'],'n':v['support']} for g,v in m['per_grade'].items()})
json.dump(m,open('final/n2sf_official_eval.json','w'),ensure_ascii=False,indent=2)
print('저장: final/n2sf_official_eval.json')
PYEOF
  echo \"=== N²SF §9 전환 완료 \$(date '+%m-%d %H:%M') ===\"
" >> kd/n2sf.log 2>&1 &
echo "PID=$! N²SF §9 파이프라인 시작. 로그: kd/n2sf.log | 결과: final/n2sf_official_eval.json"
