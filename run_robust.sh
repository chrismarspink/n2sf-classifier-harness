#!/bin/bash
# 보완#1(근본책): 다중 생성기 학습셋 → xlmr 재증류 → OOD 재평가. 재개형.
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"; mkdir -p kd
if pgrep -f "Python -m harness.(gen_robust|train_kd)" >/dev/null 2>&1; then echo "실행중"; exit 0; fi
PY="$DIR/.venv/bin/python"; SET="distill_o4/teacher_labels_robust.jsonl"; HO="distill_soft/heldout.json"
caffeinate -ims nohup bash -c "
  cd '$DIR'
  echo \"=== 강건 재증류 시작 \$(date '+%m-%d %H:%M') ===\"
  echo '--- [1] 다중 생성기 학습셋(템플릿+Gemma) ---'
  '$PY' -m harness.gen_robust --base-jsonl distill_o4/teacher_labels_multi.jsonl --out $SET \
     --n-template 250 --n-gemma 150 --teacher o4-mini --max-hours 3.0
  echo '--- [2] xlmr 강건 재증류 ---'
  if [ ! -f models/n2sf-xlmr-robust/config.json ]; then
    '$PY' -m harness.train_kd --soft-jsonl $SET --heldout $HO --base xlm-roberta-large \
       --out models/n2sf-xlmr-robust --epochs 4 --batch 6 --accum 3 --maxlen 128 \
       --temperature 1.5 --alpha 0.1 --balance --train-top 6 --lr 2e-5 --warmup 0.1 --max-hours 2.5
  else echo 'xlmr-robust 완료(skip)'; fi
  echo '--- [3] OOD/동일분포 재평가 ---'
  '$PY' - <<'PYEOF'
import json
import data_classifier as dc
from harness import metrics as M
ood=json.load(open('distill_soft/heldout_ood.json')); ind=json.load(open('distill_soft/heldout.json'))
W={'tier':{'rules':0.3,'ner':0.3,'neural':4.0}}
dc.NEURAL_BACKENDS['r']={'label':'r','kind':'finetuned','model':'models/n2sf-xlmr-robust','langs':'multi'}
def ev(d):
    rows=[{'true_grade':h['grade'],'pred_grade':dc.classify_text(h['text'],locale='ko',llm_mode=True,model='r',ensemble_method='soft',weights=W)['gradeCode']} for h in d]
    return M.compute(rows)
o=ev(ood); i=ev(ind)
print('xlmr-robust OOD f1=%s C=%s 과대=%s | 동일 f1=%s C=%s'%(o['macro_f1'],o['c_recall'],o['over_rate'],i['macro_f1'],i['c_recall']))
json.dump({'ood':o,'ind':i},open('final/robust_eval.json','w'),ensure_ascii=False,indent=2)
PYEOF
  echo \"=== 강건 재증류 완료 \$(date '+%m-%d %H:%M') ===\"
" >> kd/robust.log 2>&1 &
echo "PID=$! 강건 재증류 시작. 로그: kd/robust.log | 결과: models/n2sf-xlmr-robust, final/robust_eval.json"
