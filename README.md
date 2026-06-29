# 문서 등급 분류(3-tier) 자동 평가·최적화 하네스

`data_classifier.py` — 문서를 **N²SF 3등급(C 기밀 / S 민감 / O 공개)** 으로 분류하는 모델.
**외부 LLM·GPU 없이** 정규식 + Presidio NER + 경량 BERT 3-tier 만으로 LLM 수준 성능을 목표로 한다.

`harness/` — 그 모델을 **자동 평가·최적화**하는 하네스: 등급·난이도·포맷별 합성 테스트셋 생성 →
분류 → KPI 측정 → 정규식·NER·키워드·뉴럴·앙상블·임계값 자동 탐색 → 결과를 DB/Excel/HTML 로 정리.
난이도를 점진적으로 올리며(L1→L4) 무인 반복 최적화하는 주말 러너 포함.

## 빠른 시작

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install presidio-analyzer spacy numpy openpyxl pyyaml reportlab pypdf \
            torch transformers sentence-transformers anthropic sentencepiece
python -m spacy download ko_core_news_sm

# 전체 파이프라인 (규칙+NER+뉴럴 최적화 + 로컬 LLM 비교)
python -m harness --out harness_out --models minilm,ko-sroberta,mdeberta --llm-provider ollama

# 발표용 HTML 대시보드 생성
python -m harness.visualize --out harness_out      # → harness_out/report.html

# 주말 무인 최적화(난이도 에스컬레이션, 60h)
bash run_weekend.sh
```

## 구성

| 경로 | 내용 |
|---|---|
| `data_classifier.py` / `.md` | 분류 모델(단일 소스) + 매뉴얼 |
| `harness/` | 평가·최적화 하네스 (corpus/detect/score/metrics/optimize/llm·ollama_baseline/summarize/report/visualize/autoloop) |
| `harness/README.md` | 하네스 상세 설명 |
| `PROJECT_STATUS.md` | 작업 현황·결과 공유 문서 |
| `run_weekend.sh` | 주말 무인 최적화 러너(중복 방지·sleep 방지) |

> 대용량 산출물(`results.db`, `*/corpus/`)과 `.venv` 는 `.gitignore` 로 제외(재생성 가능).
> 소형 산출물(report.html, WEEKEND_SUMMARY.md, recommended_weights.json 등)은 추적한다.

## 핵심 결과 (요약)

- 기본 설정 macro-F1 0.47 → **자동 최적화 후 1.00**(정상·경계 난이도), 기밀 누락 0.
- 로컬 LLM 비교: 최적 3-tier **F1 1.00 / Gemma2:9b 0.89**, early-exit 시 **평균 ~45배 빠름**(GPU 없이).
- 적대 난독화(L3)는 정규식 회피로 한계(F1 ~0.54) → 전처리·체크섬·뉴럴 파인튜닝이 다음 단계.

자세한 내용은 `PROJECT_STATUS.md` 참조.
