# N²SF 등급분류 — 산출물 패키지

> 온디바이스(외부 LLM·GPU 없이) 문서 보안등급 분류. N²SF 3등급: **C(기밀) / S(민감) / O(공개)**.
> 이 폴더는 배포용 **참조 구현 + 사용 가이드/백서**입니다.

## 구성

| 파일 | 설명 |
|---|---|
| `classify_n2sf.py` | 참조 분류 스크립트(엔진 `data_classifier.py`를 감싸는 얇은 CLI/함수). 티어 early-exit 포함 |
| `N2SF_백서_사용가이드.md` | 백서·사용 가이드(설계 포인트 + **SHAP 설명가능성 별도 섹션** + 참조 코드 설명) |
| `requirements.txt` | 추론 의존성(CPU만으로 동작) |

> 엔진 본체(`data_classifier.py`)와 증류 모델(`models/n2sf-*`)은 리포지토리 루트에 있습니다.
> 참조 스크립트는 상위 폴더의 엔진·모델을 참조합니다.

## 빠른 시작

```bash
pip install -r requirements.txt
python -m spacy download ko_core_news_lg     # 한국어 NER

# 파일 분류(기본 n2sf-base, 티어 early-exit on)
python classify_n2sf.py 문서.pdf

# 정확도 최우선(라지, 다국어 강건)
python classify_n2sf.py 문서.docx --model n2sf-xlmr-large

# 텍스트 직접 + 원시 JSON
python classify_n2sf.py --text "홍길동 주민번호 900101-1234567" --json
```

## 용도별 모델 선택

| 모델 | 티어 | 크기 | 지연(p50) | 정확도(held-out) | 권장 환경 |
|---|---|---|---|---|---|
| `n2sf-small` | Fast | 57MB / 14M | ~13ms | 0.71 | 엣지·실시간·저사양 |
| `n2sf-base` (기본) | Balanced | 1.1GB / 279M | ~120ms | 0.72~0.80 | 표준 업무 PC |
| `n2sf-klue-large` | Korean | 1.3GB / 337M | ~205ms | 0.87 | 한국어 특화·용량 절감 |
| `n2sf-xlmr-large` | Accurate | 2.3GB / 560M | ~203ms | **0.92** | 정확도 최우선·서버·다국어 |

- 전 모델 **기밀 재현율 1.00(누락 0)**. 앙상블은 이득이 없어 비권장.
- 정확도 = 뉴럴-단독 macro-F1(정직 held-out). 실서비스는 3-tier 규칙 floor가 누락을 추가 차단.

## 핵심 포인트 (자세한 내용은 백서 참조)

1. **외부 LLM·GPU 불필요** — 추론은 100% 로컬 CPU. LLM은 학습(증류) 단계에만, 추론엔 호출 0.
2. **티어 early-exit** — 정규식(T1)에서 기밀 확정 시 뉴럴(T3) 미수행 → 명백한 기밀은 빠르게 종료.
3. **확장성** — `--model`로 라지 승격, 필요 시 LLM 확장 가능. 규칙·정책(정규식/키워드/GradeProfile)로 고객사 맞춤.
4. **설명가능성(SHAP)** — 결과 `shap` 블록 = 어떤 검출이 등급을 얼마나 올렸는지 가법적 기여도.
