#!/bin/bash
# 보완 A(평가 안정화): #1 종료 대기 → 다시드(3) 학습으로 분산 측정 + 부트스트랩 신뢰구간.
#  "오락가락"을 수치로: CI(측정 노이즈) + 시드간 std(학습 노이즈).
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"; mkdir -p kd
PY="$DIR/.venv/bin/python"; SET="distill_o4/teacher_labels_multi.jsonl"; HO="distill_soft/heldout.json"
caffeinate -ims nohup bash -c "
  cd '$DIR'
  echo \"=== A 시작 \$(date '+%m-%d %H:%M') — #1(Python 학습) 종료 대기 ===\"
  while pgrep -x Python >/dev/null 2>&1; do sleep 120; done
  echo \"=== #1 종료 확인, A 진행 \$(date +%H:%M) ===\"

  # 1) 다시드 학습(small-multi-minilm, 빠름) — 시드간 분산 측정
  for S in 1 2 3; do
    OUT=models/ssm-seed\$S
    if [ ! -f \$OUT/config.json ]; then
      echo \"--- seed \$S 학습 ---\"
      '$PY' -m harness.train_kd --soft-jsonl $SET --heldout $HO \
        --base microsoft/Multilingual-MiniLM-L12-H384 --out \$OUT \
        --epochs 4 --batch 16 --accum 2 --maxlen 128 --temperature 1.5 --alpha 0.1 \
        --balance --seed \$S --lr 2e-5 --warmup 0.1 --max-hours 0.8
    fi
  done

  # 2) 부트스트랩 CI + 다시드 분산
  echo '--- 분석: 부트스트랩 CI + 시드 분산 ---'
  '$PY' - <<'PYEOF'
import json, random
import data_classifier as dc
from harness import metrics as M
random.seed(42)
ood=json.load(open('distill_soft/heldout_ood.json')); ind=json.load(open('distill_soft/heldout.json'))
W={'tier':{'rules':0.3,'ner':0.3,'neural':4.0}}
def preds(model, data):
    return [(h['grade'], dc.classify_text(h['text'],locale='ko',llm_mode=True,model=model,ensemble_method='soft',weights=W)['gradeCode']) for h in data]
def f1_of(pairs):
    return M.compute([{'true_grade':t,'pred_grade':p} for t,p in pairs])['macro_f1']
def boot(pairs, n=2000):
    N=len(pairs); vals=[]
    for _ in range(n):
        s=[pairs[random.randrange(N)] for _ in range(N)]
        vals.append(f1_of(s))
    vals.sort(); return round(vals[int(n*.025)],3), round(vals[int(n*.975)],3)
out={}
# 대표 모델 CI (xlmr-large-v2)
dc.NEURAL_BACKENDS['x']={'label':'x','kind':'finetuned','model':'models/n2sf-xlmr-large-v2','langs':'multi'}
for nm,data in [('OOD',ood),('동일분포',ind)]:
    pr=preds('x',data); pt=f1_of(pr); lo,hi=boot(pr)
    out[f'xlmr_{nm}']={'f1':round(pt,3),'ci95':[lo,hi],'n':len(data)}
    print(f'[xlmr-large {nm}] F1={pt:.3f}  95%CI=[{lo},{hi}]  n={len(data)}',flush=True)
# 다시드 분산(small-multi-minilm seed1/2/3) — OOD
seed_f1=[]
for S in (1,2,3):
    key=f's{S}'; dc.NEURAL_BACKENDS[key]={'label':key,'kind':'finetuned','model':f'models/ssm-seed{S}','langs':'multi'}
    import os
    if not os.path.exists(f'models/ssm-seed{S}/config.json'): continue
    f=f1_of(preds(key,ood)); seed_f1.append(f); print(f'[seed {S}] OOD F1={f:.3f}',flush=True)
if seed_f1:
    import statistics as st
    mean=st.mean(seed_f1); sd=st.pstdev(seed_f1)
    out['seed_variance']={'seeds':[round(x,3) for x in seed_f1],'mean':round(mean,3),'std':round(sd,3),'range':round(max(seed_f1)-min(seed_f1),3)}
    print(f'[다시드] OOD F1 mean={mean:.3f} std={sd:.3f} range={max(seed_f1)-min(seed_f1):.3f}',flush=True)
json.dump(out,open('final/stability.json','w'),ensure_ascii=False,indent=2)
print('저장: final/stability.json')
PYEOF
  echo \"=== A 완료 \$(date '+%m-%d %H:%M') ===\"
" >> kd/stability.log 2>&1 &
echo "PID=$! A(평가 안정화) 예약. #1 끝나면 자동 진행. 로그: kd/stability.log"
