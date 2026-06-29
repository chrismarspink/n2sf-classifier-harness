# 최적화된 등급 분류 모델 — 참조 설정 · 모델 정보

자동 탐색으로 선정된 data_classifier.py 최적 설정과 성능 근거를 정리한다.

## 1. 모델 구성 (Model Info)

| 항목 | 값 |
|---|---|
| 신경망(BERT) 백엔드 | `mdeberta` — mDeBERTa-v3 NLI (제로샷) |
| 백엔드 종류 | zeroshot |
| HF 모델 | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` |
| 앙상블 방식 | `vote` |
| C(기밀) 임계값 | 5.5 |
| S(민감) 임계값 | 1.5 |
| 엔티티 가중치 override | `{"KR_ACCOUNT": 2.5}` |
| tier 가중치 | `{}` |
| supersede 규칙 | `없음` |

## 2. 성능 KPI (test split, 정답=정책 라벨)

| 지표 | 기본 설정 | **최적 설정(BERT 3-tier)** | LLM(Claude gemma2:9b) |
|---|---|---|---|
| macro F1 | 0.4678 | **1.0000** | 0.8917 |
| accuracy | 0.5556 | **1.0000** | 0.9012 |
| C 재현율(기밀 누락 방지) | 1.0000 | **1.0000** | 1.0000 |
| C 정밀도 | 0.5833 | **1.0000** | 1.0000 |
| 과소분류율(유출 위험) | 0.0000 | **0.0000** | 0.0000 |
| 과대분류율 | 0.4444 | **0.0000** | 0.0988 |

LLM 대비 일치율(agreement): **0.9012**  (비교 파일 162건)

## 3. 성능(속도) — GPU 없이 LLM 대비

| | BERT 3-tier(CPU) | LLM(Claude) |
|---|---|---|
| end-to-end p50 | 827.287 ms | 19514.4 ms |
| end-to-end p95 | 833.918 ms | 24823.7 ms |

> LLM 은 깨끗한 추출 텍스트를 입력받고, 3-tier 모델은 추출+분류를 모두 수행한다(비대칭). 그럼에도
> 외부 LLM/GPU 없이 위 KPI/속도를 달성한다.

## 4. 적용법 (참조 코드)

### CLI
```bash
python data_classifier.py <file> --locale ko --weights weights.json --llm --model mdeberta --ensemble vote
```

### Python
```python
import json
from data_classifier import classify
weights = json.load(open("weights.json", encoding="utf-8"))
r = classify("문서.xlsx", n2sf_mode=True, locale="ko",
             weights=weights, llm_mode=True, model="mdeberta", ensemble_method="vote",
             )
print(r["grade"], r["score"])
```
