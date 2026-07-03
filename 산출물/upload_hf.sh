#!/bin/bash
# Hugging Face Hub 업로드 — "docker push"의 모델판. 실행하면 어디서나 from_pretrained 로 받아 씀.
# 준비: pip install huggingface_hub ; export HF_TOKEN=hf_xxx (write 권한 토큰)
#       ORG=innotium (본인 계정/조직명으로)
# 사용: ./upload_hf.sh            (공개)
#       PRIVATE=1 ./upload_hf.sh  (사설 repo)
set -e
ORG="${ORG:-innotium}"
PRIV=""; [ "${PRIVATE:-0}" = "1" ] && PRIV="--private"
: "${HF_TOKEN:?HF_TOKEN 환경변수 필요 (huggingface.co 토큰, write)}"
export HF_TOKEN
DIR="$(cd "$(dirname "$0")/.." && pwd)"   # 리포 루트(models/ 위치)

# 업로드할 모델(용도별). 존재하는 것만 올림.
for M in n2sf-small n2sf-base n2sf-xlmr-large n2sf-small-multi-e5; do
  SRC="$DIR/models/$M"
  [ -d "$SRC" ] || { echo "skip(없음): $M"; continue; }
  echo "== 업로드: $ORG/$M =="
  # 모델 카드가 없으면 템플릿 복사
  [ -f "$SRC/README.md" ] || cp "$DIR/산출물/MODEL_CARD_템플릿.md" "$SRC/README.md" 2>/dev/null || true
  huggingface-cli upload "$ORG/$M" "$SRC" . $PRIV
done
echo "완료. 사용: AutoModelForSequenceClassification.from_pretrained('$ORG/n2sf-base')"
