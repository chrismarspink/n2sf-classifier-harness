#!/bin/bash
# bundle_offline.sh — 완전 오프라인 배포 번들 생성.
#  (1) 선택 모델을 산출물_v3/models/ 로 복사(로컬 로드) (2) 의존 wheel·spaCy 모델 오프라인 아카이브.
#  온라인 PC에서 1회 실행 → 산출물_v3/ 통째로 폐쇄망 반입 → 오프라인 실행.
#  사용: ./bundle_offline.sh [모델명...]   (기본: 권장 3티어)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"; cd "$ROOT"
MODELS=("${@:-}"); [ -z "${MODELS[*]}" ] && MODELS=(n2sf-xlmr-large n2sf-small-multi-minilm n2sf-small)

echo "== 1) 모델 로컬 번들 → 산출물_v3/models/"
mkdir -p "$HERE/models"
for m in "${MODELS[@]}"; do
  if [ -d "models/$m" ]; then cp -R "models/$m" "$HERE/models/$m"; echo "  복사: $m ($(du -sh models/$m|cut -f1))"
  else echo "  ⚠️ 없음: models/$m (스킵)"; fi
done

echo "== 2) 파이썬 의존 wheel 오프라인 아카이브 → 산출물_v3/offline_wheels/"
mkdir -p "$HERE/offline_wheels"
"$ROOT/.venv/bin/pip" download -d "$HERE/offline_wheels" \
  transformers torch presidio-analyzer presidio-anonymizer spacy openpyxl pypdf protobuf sentencepiece 2>&1 | tail -3 || \
  echo "  (pip download 일부 실패 가능 — 온라인 PC에서 재실행 권장)"

echo "== 3) spaCy 한국어 모델(NER) 오프라인 wheel"
"$ROOT/.venv/bin/python" -m spacy download ko_core_news_sm 2>/dev/null || true
"$ROOT/.venv/bin/pip" download -d "$HERE/offline_wheels" ko_core_news_sm 2>&1 | tail -2 || \
  echo "  spaCy 모델 wheel은 온라인 PC에서: python -m spacy download ko_core_news_sm && pip download ko_core_news_sm"

echo "== 완료. 폐쇄망 설치: pip install --no-index --find-links offline_wheels -r requirements_offline.txt"
echo "   실행 시 HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1 자동 설정(classifier_v3), 모델은 산출물_v3/models/ 에서 로드."
