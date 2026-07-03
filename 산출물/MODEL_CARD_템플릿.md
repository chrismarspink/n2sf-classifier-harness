---
language: [ko, en, zh, ja]
license: apache-2.0
tags: [text-classification, document-security, n2sf, on-premise, knowledge-distillation]
pipeline_tag: text-classification
---

# N²SF 문서 보안등급 분류 모델

문서를 **C(기밀) / S(민감) / O(공개)** 3등급으로 분류하는 온디바이스 모델.
클라우드 LLM(GPT-4o/o4-mini)의 판단을 **soft-label 지식증류**로 이전 → **외부 LLM·GPU 없이 로컬 CPU**로 LLM급 분류.

## 라벨
`0=OPEN(공개)`, `1=SENSITIVE(민감)`, `2=CONFIDENTIAL(기밀)` (config.id2label 참조)

## 사용
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
tok = AutoTokenizer.from_pretrained("innotium/n2sf-base")
mdl = AutoModelForSequenceClassification.from_pretrained("innotium/n2sf-base").eval()
enc = tok("주민등록번호 900101-1234567 포함 대외비 문서", return_tensors="pt", truncation=True, max_length=512)
p = torch.softmax(mdl(**enc).logits[0], -1)  # [O, S, C]
```

> 실서비스는 3-tier(정규식+NER+본 모델) 파이프라인 권장 — `classify_n2sf.py` 참조.
> 앙상블 권장 설정: soft, 티어가중 {rules:0.3, ner:0.3, neural:4.0}.

## 성능 (held-out 정직 분포, macro-F1)
| 관점 | 값 |
|---|---|
| 전체 파이프라인 F1 | (모델별 기입) |
| 기밀 재현율(C-recall) | 1.00 목표(누락0) |

## 안전·한계
- **기밀 재현율 최우선**(과소분류=유출 최소화). 실서비스는 규칙 floor로 이중 방어.
- 증류엔 **합성 문서만** 사용(실기밀 미사용). 추론 시 외부 호출 0.
- 소형 한국어 모델은 다국어 문서에서 과대분류 경향 → 다국어는 base/large 또는 multilingual 변형 사용.

## 인용/출처
이노티움 N²SF 등급분류 하네스(내부). 상세: 리포지토리 문서 참조.
