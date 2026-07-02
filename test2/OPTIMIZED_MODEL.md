# 최적화된 등급 분류 모델 — 참조 설정 · 모델 정보

자동 탐색으로 선정된 data_classifier.py 최적 설정과 성능 근거를 정리한다.

## 1. 모델 구성 (Model Info)

| 항목 | 값 |
|---|---|
| 신경망(BERT) 백엔드 | `mdeberta-n2sf` — mDeBERTa-n2sf (파인튜닝 3-class) |
| 백엔드 종류 | finetuned |
| HF 모델 | `models/mdeberta-n2sf` |
| 앙상블 방식 | `soft` |
| C(기밀) 임계값 | 5.5 |
| S(민감) 임계값 | 1.5 |
| 엔티티 가중치 override | `{"KR_ACCOUNT": 6.0}` |
| tier 가중치 | `{"rules": 1.0, "ner": 1.0, "neural": 1.5}` |
| supersede 규칙 | `{"KR_ACCOUNT": ["KR_PHONE"]}` |

## 2. 성능 KPI (test split, 정답=정책 라벨)

| 지표 | 기본 설정 | **최적 설정(BERT 3-tier)** | LLM(Claude -) |
|---|---|---|---|
| macro F1 | 0.3274 | **1.0000** | - |
| accuracy | 0.4125 | **1.0000** | - |
| C 재현율(기밀 누락 방지) | 0.7097 | **1.0000** | - |
| C 정밀도 | 0.4151 | **1.0000** | - |
| 과소분류율(유출 위험) | 0.1125 | **0.0000** | - |
| 과대분류율 | 0.4750 | **0.0000** | - |

LLM 대비 일치율(agreement): **0.0**  (비교 파일 0건)

## 3. 성능(속도) — GPU 없이 LLM 대비

| | BERT 3-tier(CPU) | LLM(Claude) |
|---|---|---|
| end-to-end p50 | 94.3 ms | - ms |
| end-to-end p95 | 101.417 ms | - ms |

> LLM 은 깨끗한 추출 텍스트를 입력받고, 3-tier 모델은 추출+분류를 모두 수행한다(비대칭). 그럼에도
> 외부 LLM/GPU 없이 위 KPI/속도를 달성한다.

## 4. 적용법 (참조 코드)

### CLI
```bash
python data_classifier.py <file> --locale ko --weights weights.json --llm --model mdeberta-n2sf --ensemble soft
```

### Python
```python
import json
from data_classifier import classify
weights = json.load(open("weights.json", encoding="utf-8"))
r = classify("문서.xlsx", n2sf_mode=True, locale="ko",
             weights=weights, llm_mode=True, model="mdeberta-n2sf", ensemble_method="soft",
             )
print(r["grade"], r["score"])
```

### 5. 탐지 레이어 패치 (supersede — 권장)

최적 설정의 `supersede={"KR_ACCOUNT": ["KR_PHONE"]}` 는 공개 `weights` API 로 표현되지
않는다(탐지 단계 규칙). data_classifier.py 의 `SUPERSEDED_BY` 에 아래 한 줄을 추가하면
동일 효과(겹치는 일반 인식기를 더 구체적인 인식기가 억제)를 내부적으로 얻는다:

```python
SUPERSEDED_BY = {
    ...,
    "KR_ACCOUNT": ["KR_PHONE"],   # 전화번호와 겹친 계좌 오탐 제거
}
```
이 패치는 전화번호가 느슨한 계좌 정규식에 오탐되어 S→C 로 과대분류되던 문제를 교정한다.
