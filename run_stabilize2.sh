#!/bin/bash
# 밤샘 안정화 v2: (진행 Python 종료 대기) → 청정 OOD 생성 → xlmr 다시드3 → CI·분산·앙상블 분석. 재개형.
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"; mkdir -p kd
PY="$DIR/.venv/bin/python"; ROB="distill_o4/teacher_labels_robust.jsonl"; HO="distill_soft/heldout.json"
caffeinate -ims nohup bash -c "
  cd '$DIR'
  echo \"=== 안정화 v2 시작 \$(date '+%m-%d %H:%M') — 진행 학습 종료 대기 ===\"
  while pgrep -x Python >/dev/null 2>&1; do sleep 120; done
  echo \"=== 진행 완료, v2 시작 \$(date +%H:%M) ===\"

  # 1) 청정 미지분포 OOD(o4-mini 생성 — 학습 미사용 스타일)
  if [ ! -f distill_soft/heldout_clean.json ]; then
    '$PY' -m harness.gen_cleaneval --out distill_soft/heldout_clean.json --gen azure:o4-mini --per-grade 40 --langs ko,en,zh,ja
  fi

  # 2) xlmr 배포모델 다시드(강건셋으로) — 분산 측정 + 앙상블 재료
  for S in 1 2 3; do
    OUT=models/xlmr-rob-s\$S
    if [ ! -f \$OUT/config.json ]; then
      echo \"--- xlmr seed \$S ---\"
      '$PY' -m harness.train_kd --soft-jsonl $ROB --heldout $HO --base xlm-roberta-large \
        --out \$OUT --epochs 4 --batch 6 --accum 3 --maxlen 128 --temperature 1.5 --alpha 0.1 \
        --balance --train-top 6 --seed \$S --lr 2e-5 --warmup 0.1 --max-hours 2.5
    fi
  done

  # 3) 분석: 3분포(동일/템플릿OOD/청정OOD) × (개별시드 CI + 시드앙상블)
  '$PY' - <<'PYEOF'
import json, random, os
import data_classifier as dc
from harness import metrics as M
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
random.seed(1)
SETS={'동일분포':'distill_soft/heldout.json','템플릿OOD':'distill_soft/heldout_ood.json','청정OOD':'distill_soft/heldout_clean.json'}
data={k:json.load(open(v)) for k,v in SETS.items() if os.path.exists(v)}
SHORT={'OPEN':'O','SENSITIVE':'S','CONFIDENTIAL':'C'}
seeds=[s for s in (1,2,3) if os.path.exists(f'models/xlmr-rob-s{s}/config.json')]
# 각 시드 모델 로드(뉴럴-단독, maxlen256) — 앙상블은 확률 평균
def load(p):
    t=AutoTokenizer.from_pretrained(p); m=AutoModelForSequenceClassification.from_pretrained(p).eval().to('cpu'); return t,m
mods=[load(f'models/xlmr-rob-s{s}') for s in seeds]
def probs(t,m,text):
    enc=t([text],truncation=True,max_length=256,return_tensors='pt')
    with torch.no_grad(): pr=torch.softmax(m(**enc).logits[0],-1).tolist()
    id2=m.config.id2label; by={SHORT.get(id2.get(i,id2.get(str(i),'')),''):pr[i] for i in range(len(pr))}
    return [by.get('O',0),by.get('S',0),by.get('C',0)]
def f1(pairs): return M.compute([{'true_grade':a,'pred_grade':b} for a,b in pairs])['macro_f1']
def crec(pairs): return M.compute([{'true_grade':a,'pred_grade':b} for a,b in pairs])['c_recall']
def boot(pairs,n=1500):
    N=len(pairs); v=[]
    for _ in range(n): v.append(f1([pairs[random.randrange(N)] for _ in range(N)]))
    v.sort(); return round(v[int(n*.025)],3),round(v[int(n*.975)],3)
res={}
for dn,ds in data.items():
    # 개별 시드 F1
    per=[]
    for (t,m) in mods:
        pr=[(h['grade'],['O','S','C'][max(range(3),key=lambda k:probs(t,m,h['text'])[k])]) for h in ds]
        per.append(f1(pr))
    # 시드 앙상블(확률 평균)
    epairs=[]
    for h in ds:
        avg=[sum(probs(t,m,h['text'])[k] for (t,m) in mods)/len(mods) for k in range(3)]
        epairs.append((h['grade'],['O','S','C'][max(range(3),key=lambda k:avg[k])]))
    lo,hi=boot(epairs)
    import statistics as st
    res[dn]={'seed_f1':[round(x,3) for x in per],'seed_std':round(st.pstdev(per),3) if len(per)>1 else 0,
             'ensemble_f1':round(f1(epairs),3),'ensemble_crec':round(crec(epairs),3),'ensemble_ci95':[lo,hi],'n':len(ds)}
    print(f\"[{dn}] 시드F1={res[dn]['seed_f1']} std={res[dn]['seed_std']} | 앙상블F1={res[dn]['ensemble_f1']} CI={res[dn]['ensemble_ci95']} Crec={res[dn]['ensemble_crec']} n={len(ds)}\",flush=True)
json.dump(res,open('final/stability_v2.json','w'),ensure_ascii=False,indent=2)
print('저장: final/stability_v2.json')
PYEOF
  echo \"=== 안정화 v2 완료 \$(date '+%m-%d %H:%M') ===\"
" >> kd/stabilize2.log 2>&1 &
echo "PID=$! 안정화 v2 예약. 진행중 학습 끝나면 자동 시작. 로그: kd/stabilize2.log"
