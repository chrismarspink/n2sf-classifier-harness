#!/bin/bash
# bundle_offline.sh — 완전 오프라인 배포 번들 생성 (v4, 단일파일 엔진).
#  (1) 선택 모델을 산출물_v4/models/ 로 복사(로컬 로드) (2) 의존 wheel·spaCy 모델 오프라인 아카이브.
#  온라인 PC에서 1회 실행 → 산출물_v4/ 통째로 폐쇄망 반입 → 오프라인 실행.
#  ※ 추론 엔진은 classifier_v4.py '한 파일' — 별도 data_classifier.py 복사 불필요.
#    (온사이트 재학습을 쓰려면 harness/ 도 함께 반입: train_onsite.py 가 harness.train_kd 호출)
#  사용: ./bundle_offline.sh [모델명...]   (기본: 권장 티어 = official/large/small)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"; cd "$ROOT"
MODELS=("${@:-}"); [ -z "${MODELS[*]}" ] && MODELS=(n2sf-xlmr-official n2sf-xlmr-large n2sf-small)

echo "== 1) 모델 로컬 번들 → 산출물_v4/models/"
mkdir -p "$HERE/models"
for m in "${MODELS[@]}"; do
  if [ -d "$HERE/models/$m" ]; then echo "  이미 있음: $m (스킵)"
  elif [ -d "models/$m" ]; then cp -R "models/$m" "$HERE/models/$m"; echo "  복사: $m ($(du -sh models/$m|cut -f1))"
  else echo "  ⚠️ 없음: models/$m (스킵)"; fi
done

echo "== 2) 파이썬 의존 wheel 오프라인 아카이브 → 산출물_v4/offline_wheels/"
mkdir -p "$HERE/offline_wheels"
"$ROOT/.venv/bin/pip" download -d "$HERE/offline_wheels" \
  transformers torch presidio-analyzer presidio-anonymizer spacy openpyxl pypdf protobuf sentencepiece 2>&1 | tail -3 || \
  echo "  (pip download 일부 실패 가능 — 온라인 PC에서 재실행 권장)"
# (옵션) Fable T4 티어를 쓸 경우에만 anthropic 포함 — 기본 배포는 불필요(온프레미스)
# "$ROOT/.venv/bin/pip" download -d "$HERE/offline_wheels" anthropic 2>&1 | tail -1 || true

echo "== 3) spaCy 한국어 모델(NER) 오프라인 wheel"
"$ROOT/.venv/bin/python" -m spacy download ko_core_news_sm 2>/dev/null || true
"$ROOT/.venv/bin/pip" download -d "$HERE/offline_wheels" ko_core_news_sm 2>&1 | tail -2 || \
  echo "  spaCy 모델 wheel은 온라인 PC에서: python -m spacy download ko_core_news_sm && pip download ko_core_news_sm"

echo "== 완료. 폐쇄망 설치: pip install --no-index --find-links offline_wheels -r requirements_offline.txt"
echo "   실행 시 HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1 자동 설정(classifier_v4), 모델은 산출물_v4/models/ 에서 로드."
