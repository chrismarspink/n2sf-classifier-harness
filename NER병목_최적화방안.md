{{toc}}

# NER 병목 — 원인 분석 & 최적화 방안 (측정 기반)

> 배경: 속도 벤치에서 NER(Presidio+spaCy)이 ~30s/MB로 전체의 ~96% 차지(뉴럴 1s/MB, 정규식 무료).
> 방법: 구성요소를 분리 실측해 원인 규명 → 우선순위 최적화안.

---

## 1. 원인 (실측 분리)

30KB 텍스트 기준:

| 구성 | 시간 | 관찰 |
|---|---|---|
| spaCy **전체** 파이프라인 | 1.50s | tok2vec·tagger·morphologizer·parser·lemmatizer·attribute_ruler·ner 전부 실행 |
| spaCy **NER-only**(불필요 제거) | 0.94s | **1.6× 빠름** — tagger/parser/lemmatizer는 NER에 불필요 |
| **Presidio analyze(전체)** | **4.6~7.8s** | **spaCy의 5배!** — 병목의 핵심 |
| Presidio(엔티티 제한) | 2.5s | **1.8× 빠름** — 불필요 인식기 스킵 |

**진짜 원인 = Presidio 오케스트레이션**:
1. **내장 인식기 30개** 전부 실행(US_SSN·IBAN·암호화폐·각국 여권 등 우리가 안 쓰는 것 다수) → 텍스트마다 수십 개 정규식·문맥분석.
2. **문맥 강화(context enhancement)**·중복제거·스팬정렬 등 per-call 오버헤드.
3. spaCy가 **불필요 컴포넌트**(tagger/parser/lemmatizer) 실행(1.6× 낭비).
4. **청크별 단건 호출**(배치·병렬 없음) → 오버헤드 반복.

---

## 2. 최적화 방안 (우선순위 · 측정/기대 이득)

| # | 최적화 | 이득 | 위험 |
|---|---|---|---|
| **1** | **인식기 slim** — KR·사용 엔티티만 등록/`entities=` 지정, 불필요 30→~8개 | **1.8× (측정)** | 낮음 |
| **2** | **spaCy NER-only** — parser/tagger/lemmatizer/morphologizer 비활성 | **1.6× (측정)** | 낮음(NER 품질 유지) |
| **3** | **배치+멀티프로세싱** — `nlp.pipe(batch_size, n_process)` 또는 청크 프로세스풀 | **~코어수×**(8코어≈4~6×) | 낮음 |
| 4 | **NER 게이팅** — 규칙(강식별자·기밀표지)이 이미 확정(C early-exit)하거나 후보 없으면 NER 스킵 | 실데이터에서 **큼**(대부분 스킵) | 중(정책) |
| 5 | **Apple 가속** — `thinc-apple-ops` 설치(M-시리즈 spaCy 가속) | ~1.5~2× | 낮음 |
| 6 | **청크 확대** — 4k→50k자(호출수↓, spaCy max_length 내) | ~1.2~1.5× | 낮음 |
| 7 | **(전략) NER 티어 축소/대체** — §9에선 강식별자=규칙, 의미=뉴럴이 담당 → NER 한계효용 대비 96% 비용. 정확도 영향 측정 후 **선택적/샘플 NER** 또는 제거 | 최대(제거 시 병목 소멸) | 중(정확도 검증 필요) |

**결합 기대**: 1×2 ≈ **~2.5~3×**, +3(병렬) ≈ **~10~15×** → NER 30s/MB → **~2~3s/MB**, 100MB 51분 → **~4~5분**. +4(게이팅) 시 실데이터 대부분 NER 회피.

---

## 3. 구현 체크리스트 (엔진 변경점)

`data_classifier.get_engine()` / `analyze()`:
1. **spaCy NlpEngine에 NER-only 구성** — NlpEngineProvider에 `disable=[parser,tagger,lemmatizer,morphologizer,attribute_ruler]` 전달(또는 model config로).
2. **인식기 레지스트리 slim** — 기본 predefined 제거하고 **우리 PATTERN_RECOGNIZERS + spaCy + Email/Phone**만 등록.
3. `analyze(..., entities=USED_ENTITIES)` — 사용 엔티티만.
4. **배치 분석기** — `analyze_batch(chunks)`: `nlp.pipe(n_process=코어-2, batch_size=64)` + 프로세스풀.
5. **게이팅 훅** — classify_text에서 규칙이 C 확정 시 NER 생략(이미 early-exit 존재), + 후보토큰 없으면 스킵.

## 4. 검증 계획
- 각 최적화 적용 후 **speedbench 재실행**으로 s/MB·100MB 시간 재측정(회귀 없는지).
- **정확도 회귀 확인**(특히 #7 NER 축소): §9 held-out·OOD로 macro-F1·C재현율 재평가 → 속도↑가 정확도/누락0을 해치지 않는지.
- 목표: NER ~2~3s/MB, 전체 파이프라인 100MB S 기준 5분 내, C early-exit 3초 유지.

---

## 5. 결론
- **병목은 spaCy가 아니라 Presidio(불필요 인식기 30개·문맥분석·단건호출).** 측정: 인식기 slim 1.8×, NER-only 1.6×.
- **저위험 조합(slim+NER-only+배치·병렬)로 ~10×** 기대 → 100MB 51분→~5분.
- **게이팅/티어 축소**가 실데이터 최대 이득이나 **정확도 검증 필수**(안전 누락0 유지).
- 모든 개선은 **speedbench 재측정 + 정확도 회귀 평가**로 확정.

> 근거: 프로파일 실측(spaCy 전체 1.5s/NER-only 0.94s/Presidio 4.6~7.8s, 인식기 30개, 엔티티제한 1.8×).
