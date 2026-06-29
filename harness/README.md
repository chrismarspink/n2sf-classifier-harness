# classifier_harness — data_classifier 자동 평가·최적화 하네스

`data_classifier.py`(정규식 + Presidio NER + BERT 신경망 3-tier)가 **외부 LLM·GPU 없이**
LLM 수준의 등급(C/S/O) 분류 성능에 도달하도록, 테스트셋 생성 → 분류 → KPI 측정 →
최적 설정 자동 탐색 → 결과 정리를 한 번에 수행한다. 비교군으로 Claude(LLM) 분류를 돌려
BERT-모델과 정량 비교 데이터를 만든다.

## 왜 단일 스크립트가 아니라 하네스인가

분류 파이프라인은 단계별 비용이 극단적으로 다르다:

| 단계 | 비용 | 설정 의존성 |
|---|---|---|
| 추출 + Presidio NER + 뉴럴 추론 | **높음** | 정규식/NER/뉴럴 모델 |
| 점수화 + 앙상블 + 임계값 | 거의 0 | entity 가중치·threshold·tier·앙상블 |

비싼 탐지를 **1회 캐시**(SQLite `detection`)하고, 싼 점수·앙상블을 **수천 조합 스윕**하면
조합 폭발이 CPU 에서 감당 가능해진다(단일 스크립트의 매번 재실행 대비 100~1000배). 그래서
모듈형 하네스 + 결과 DB 가 정답이다. (멀티-에이전트 오케스트레이션은 결정론적 수치 탐색에는 과잉.)

## 실행

```bash
# 가상환경(최초 1회): presidio/spacy/torch/transformers/anthropic/openpyxl/reportlab/pypdf
python -m spacy download ko_core_news_sm

# 전체 파이프라인 (규칙+NER+뉴럴 최적화)
python -m harness --seed 0 --per-cell 6 --models minilm,ko-sroberta,mdeberta --out harness_out

# 규칙+NER 만 (가장 빠름, 뉴럴 다운로드 없음)
python -m harness --models "" --out harness_out

# LLM(Claude) 비교까지 — 키 필요
ANTHROPIC_API_KEY=sk-ant-... python -m harness --out harness_out --llm-max 200
```

탐지 캐시는 `(doc_id, fmt)` 단위라, 키를 설정하고 같은 seed 로 재실행하면 추출·뉴럴은
캐시에서 즉시 재사용하고 LLM 비교만 새로 채운다.

## 단계 (generate → detect → optimize → llm → report)

1. **generate** (`corpus.py`) — 등급(O/S/C) × 난이도(normal/hard_neg/format_stress)로 합성 문서를
   만들고 **동일 콘텐츠를 9개 포맷**(txt/csv/json/md/docx/**xlsx**/pptx/**hwpx**/pdf)으로 렌더.
   라벨은 분류기 규칙이 아니라 **N²SF 정책**으로 부여(순환오류 방지). PII 는 체크섬·포맷이
   유효하게 생성(주민번호 검증자리, 카드 Luhn 등).
2. **detect** (`detect.py`) — 추출+NER+뉴럴을 1회 실행해 `detection` 테이블에 캐시. 뉴럴은
   정규화 텍스트 해시로 포맷 변이를 통합.
3. **optimize** (`optimize.py`) — 2계층 탐색. valid 에서 안전 우선 목적함수로 튜닝, test 로 최종 보고.
   - Stage 0 기준선 → Stage 1 규칙+NER 점수설정(supersede·임계값·가중치) → Stage 2 뉴럴×앙상블 →
     Stage 3 tier 가중치.
4. **llm** (`llm_baseline.py`) — Claude `claude-opus-4-8` 로 동일 텍스트 분류(구조화 출력, 정책
   프롬프트 캐싱). 지연·토큰 측정.
5. **report** (`report.py`) — Excel + 최적 설정 산출.

## 산출물 (`harness_out/`)

| 파일 | 내용 |
|---|---|
| `results.db` | 코퍼스·탐지캐시·설정·예측·지표·LLM 예측 (SQLite, 재현·쿼리용) |
| `report.xlsx` | Summary / Leaderboard / ConfusionMatrix / **PerFormat(xlsx·hwpx 강조)** / PerCategory / Misclassified / **LLM_vs_BERT** |
| `weights.json` | `classify(..., weights=...)` 에 바로 쓰는 최적 가중치/임계값 |
| `optimized_config.json` | 하네스 네이티브 전체 최적 설정 |
| `OPTIMIZED_MODEL.md` | 모델 정보 + KPI(기본 vs 최적 vs LLM) + 성능 + 적용법(참조 코드) |

## KPI / 목적함수 (보안 우선)

기밀(C)을 낮게 보는 **과소분류 = 유출**이 과대분류보다 치명적이다. 목적함수는 macro-F1 을
기준으로 C-재현율이 목표(기본 0.98) 미달이면 강하게 감점하고, 과소분류를 추가 감점, 동률 시
지연이 낮을수록 가점한다. 포맷별 분해로 "분류 실패"와 "추출 실패"를 분리한다.
