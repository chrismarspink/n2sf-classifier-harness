{{toc}}

# N²SF 온디바이스 등급분류 — 백서 & 사용 가이드

> 일자: 2026-07-02 · 대상: 도입 검토자·개발자·감사자
> 요지: **외부 LLM·GPU 없이**, 고객 디바이스 자원만으로 문서 보안등급(기밀 C/민감 S/공개 O)을
> LLM에 근접한 정확도로 분류한다. 클라우드 LLM의 지식은 **개발 단계 증류로 로컬 모델에 이전**하고,
> **추론에는 LLM을 전혀 호출하지 않는다.**

---

## 1. 개요

N²SF 등급분류기는 문서를 3등급으로 자동 분류한다.

| 코드 | 등급 | 의미 | 예 |
|---|---|---|---|
| **C** | 기밀(Confidential) | 유출 시 중대 피해. 강식별자·기밀표지 | 주민번호·카드·API키, "대외비", 개인정보 명부 |
| **S** | 민감(Sensitive) | 제한적 개인정보·내부용 | 단일 연락처, 인사·계약 맥락 |
| **O** | 공개(Open) | 유출돼도 무해 | 공지·안내·홍보 |

**설계 원칙(안전 우선)**: 과소분류(기밀을 낮게 = 유출)가 최악. 과대분류(공개를 높게)는 감수.
→ 전 구성에서 **기밀 재현율(C-recall) = 1.00(누락 0)** 을 최우선 불변식으로 유지.

---

## 2. 핵심 포인트

### 2.1 외부 LLM·GPU 없이 분류 (데이터 주권)
- **추론은 100% 로컬 CPU.** 운영 중 외부 API 호출·인터넷 반출 **0건**.
- 클라우드 LLM(GPT-4o 등)은 **개발 단계의 "교사"로 증류에만** 사용하고, **합성 문서만** 전송(실기밀 금지).
- 결과: 데이터가 고객 경계를 벗어나지 않음(에어갭 가능). 추론당 API 과금 0.

### 2.2 티어 판단속도 유리 — early-exit
3-tier 파이프라인은 **값싼 티어부터** 수행하고, **확정되면 이후 티어를 생략**한다.

```
문서 → [T1 정규식·deny-list] → [T2 Presidio NER] → [T3 뉴럴(증류 모델)] → 앙상블
         강식별자(주민번호·카드·키·기밀표지)
         여기서 기밀 확정 시 ─────────────▶ ★ early-exit: T3(뉴럴) 미수행
```

- 주민번호·카드번호·AWS키 같은 **강식별자는 언어무관 정규식**이 즉시 검출 → 명백한 기밀은 T1에서 종료.
- 뉴럴(수십~수백 ms, 라지는 수백 MB 로드)은 **경계 문서(O/S 애매)에만** 돌아 평균 지연↓.
- 참조 구현 `classify_n2sf.py`의 `classify_with_early_exit()`가 이 동작을 코드로 보여준다.

### 2.3 확장성 (필요 시 LLM·라지·정책)
- **모델 승격**: `--model n2sf-xlmr-large`로 정확도 0.92 라지로 교체(코드 변경 없음).
- **LLM 확장(옵션)**: 온프레미스 정책이 허용하면 T3 뒤에 로컬/사내 LLM 티어를 덧붙일 수 있는 구조.
  기본값은 LLM 미사용.
- **규칙·정책 확장**: 정규식/키워드/deny-list 추가, 회사 규정을 GradeProfile(YAML)·floor_rules로 주입
  (무재배포). → 고객사별 맞춤 = 제품 차별화.

### 2.4 고객 디바이스 자원 사용
- 소형(57MB)부터 라지(2.3GB)까지 **디바이스에 맞춰 선택**. 엣지·노트북·서버 어디서든 CPU로 동작.
- GPU 없어도 실용 지연(소형 ~13ms, 라지 ~200ms/문서). 고객 인프라 그대로 활용.

---

## 3. 아키텍처 (3-tier)

| 티어 | 기술 | 역할 | 강점 |
|---|---|---|---|
| **T1 규칙** | 정규식 + deny-list + 키워드 (+체크섬) | 강식별자·기밀표지 검출, 대량 PII 집계 | 즉효·언어무관·설명가능·재현율 하한 보장 |
| **T2 NER** | Presidio + spaCy | 문맥적 개체(이름·주소·기관) 인식 | 규칙이 놓친 문맥 PII 포착 |
| **T3 뉴럴** | 증류된 N²SF 모델(mDeBERTa/KoELECTRA/KLUE/XLM-R) | 문서 의미로 O/S/C 판정 | 위장·완곡·경계 문서 대응 |
| **앙상블** | 티어 결합(soft) + 컴플라이언스 floor | 최종 등급. floor는 **하향 불가**(안전) | 규칙 안전성 + 뉴럴 분별력 |

- **컴플라이언스 floor**: 대량 PII·마이넘버 등은 앙상블이 **하향 못 함** → 규정 최저등급 항상 보장.
- 등급 결정 임계: `C_THRESHOLD`, `S_THRESHOLD`, 엔티티 가중치 `ENTITY_WEIGHTS`(런타임 `weights`로 조정 가능).

---

## 4. 성능 (정직 held-out · 뉴럴-단독 macro-F1)

held-out = 학습에 쓰지 않은 LLM 생성 분포(과적합 착시 배제). 72문서, CPU 측정.

| 모델 | macroF1 | 기밀재현율 | 지연 p50 | 용량 | 파라미터 |
|---|---|---|---|---|---|
| n2sf-small (KoELECTRA-small) | 0.709 | 1.00 | 13ms | 57MB | 14M |
| n2sf-base (mDeBERTa) | 0.72~0.80 | 1.00 | 122ms | 1.1GB | 279M |
| n2sf-klue-large (KLUE-RoBERTa-large) | 0.872 | 1.00 | 205ms | 1.3GB | 337M |
| **n2sf-xlmr-large (XLM-R-large)** | **0.917** | 1.00 | 203ms | 2.3GB | 560M |

### 4.1 일반 LLM 대비
| 분류기 | macroF1 | 기밀재현율 | 반출 | 과금 |
|---|---|---|---|---|
| **n2sf-xlmr-large (본 모델)** | **0.917** | 1.00 | **없음(로컬)** | **0** |
| GPT-4o (교사, 클라우드) | 0.915 | 1.00 | 있음 | 호출당 |
| o4-mini (클라우드) | 0.972 | 1.00 | 있음 | 호출당 |
| Gemma2:9b (로컬 LLM) | 0.873 | 0.917 | 없음 | 0 |

> **n2sf-xlmr-large(0.917)가 교사 GPT-4o(0.915)와 동급.** "외부 LLM 없이 로컬에서 LLM급"을 실측 입증.
> 로컬 Gemma2:9b(0.873)보다 정확하고, 기밀 누락도 없음(Gemma는 C재현율 0.917로 누락 발생).

- **뉴럴-단독** 기준이며, 실서비스는 3-tier 규칙 floor가 더해져 기밀 누락을 추가 차단.
- base 티어의 추가 개선(0.8 확실화)은 균형샘플링·온도증류로 진행(참조: `base_0.8돌파_전략.md`).

---

## 5. 모델 실체 & 증류 (요약)

- **모델 실체**: `models/n2sf-*/` = HuggingFace `AutoModelForSequenceClassification` 가중치 폴더
  (config.json + safetensors + tokenizer). N²SF 3-class(O/S/C) 헤드.
- **증류(KD)**: 교사 LLM이 문서에 O/S/C **확률분포**를 부여(soft-label) → 학생이 그 분포를 모방 학습
  (`L = −Σ p_교사·log softmax(학생)`). 정답만이 아니라 **교사의 결정경계**를 이전 → 일반화↑.
- 추론엔 교사 불필요. 릴리즈는 모델 폴더 배포(사내 레지스트리/파일). 상세: `모델_실체_및_릴리즈_가이드.md`,
  `LLM증류_방식_및_도구.md`.

---

## 6. 설명가능성 — SHAP (별도 섹션)

> "왜 이 등급인가"를 사람이 검증·감사할 수 있어야 한다. N²SF는 **규칙 티어에서 정확한 가법적 기여도**를
> 제공하고, **뉴럴 티어는 SHAP 확장**으로 근거를 보강한다. (LLM 블랙박스 대비 핵심 강점)

### 6.1 지금 제공: 규칙 티어 가법적 기여도 (결과 `shap` 블록)
등급 점수는 검출들의 가중 합이다. `_shap_block`이 **각 검출(feature)의 기여도와 비율**을 산출한다.

```json
"shap": {
  "baseline": 0.0, "total": 17.0,
  "contributions": [
    {"feature": "KR_RRN",     "label": "주민등록번호", "contribution": 6.0, "percent": 0.35},
    {"feature": "KR_ACCOUNT",  "label": "계좌번호",     "contribution": 6.0, "percent": 0.35},
    {"feature": "KEYWORD",     "label": "대외비",       "contribution": 3.0, "percent": 0.17}
  ]
}
```

- **의미**: "주민번호(35%) + 계좌(35%) + '대외비'(17%)가 기밀 판정을 만들었다" — 감사·소명에 직접 사용.
- **가법성**: 기여도 합 = 점수(total). SHAP의 핵심 성질(additive attribution)을 규칙 티어에서 **정확히** 만족.
- 참조 스크립트가 이를 사람이 읽는 형태로 출력(근거 패널).

### 6.2 뉴럴 티어 SHAP (확장)
뉴럴(증류 모델)은 의미 기반이라 기여도가 자명하지 않다. 필요 시 다음으로 근거를 산출:
- **토큰 기여도**: KernelSHAP/Gradient×Input 으로 문장·토큰별 등급 기여 시각화(`shap` 패키지, 옵션 의존성).
- **exemplar 근접도**: 판정에 가까웠던 등급 예시문과의 유사도 제시.
- 운영 권장: **결정은 규칙 floor가 보증**(설명가능), 뉴럴은 경계 보조 → 감사 시 규칙 근거를 1차 제시.

### 6.3 왜 중요한가
- 규제 대응(개인정보보호법·감사)에서 **판단 근거 제시 의무**를 충족.
- LLM 단독은 근거가 불투명 → N²SF는 규칙 기여도로 **재현·소명 가능**.

---

## 7. 참조 코드 설명 (`classify_n2sf.py`)

배포용 참조 구현은 엔진(`data_classifier.py`)을 얇게 감싼다.

### 7.1 진입점
- `classify_with_early_exit(text, model, ensemble_method, early_exit)` — 3-tier + early-exit 핵심.
- CLI: `python classify_n2sf.py <파일|--text> [--model ...] [--json] [--no-early-exit]`.

### 7.2 early-exit 로직 (속도 포인트를 코드로)
```python
# 1) 티어 1·2만 먼저 (뉴럴 미로드)
rules = dc.classify_text(text, llm_mode=False, ensemble_method=method)
# 2) 규칙만으로 기밀 확정 → 뉴럴 생략(early-exit)
if early_exit and rules["gradeFull"] == "CONFIDENTIAL":
    return rules                       # T3 미수행 → 빠름
# 3) 경계 문서만 뉴럴(T3) 수행 후 앙상블
final = dc.classify_text(text, llm_mode=True, model=model, ensemble_method=method)
```
- `llm_mode`는 레거시 명칭이며 **"뉴럴 티어 사용" 스위치**(외부 LLM 아님). 뉴럴은 로컬 CPU 추론.

### 7.3 엔진 핵심 함수(참고)
| 함수 | 역할 |
|---|---|
| `extract_text(file)` | 포맷별 텍스트 추출(txt/csv/md/json/docx/xlsx/pptx/hwpx/pdf) |
| `analyze` / `_aggregate_findings` | Presidio 검출 → 개체 집계 |
| `_scan_keywords` | 등급 키워드(+런타임 추가) 스캔 |
| `_score` | 검출 가중합 → 점수·등급·신뢰도 |
| `neural_infer` | 증류 모델 O/S/C 확률(로컬) |
| `_combine` | 티어 앙상블 + 컴플라이언스 floor |
| `_shap_block` | 가법적 기여도(§6) |

### 7.4 반환 스키마(요약)
`grade`(C/S/O) · `confidence` · `score` · `shap` · `tiers`(rules/ner/neural) ·
`compliance`(regulations/violations) · `_tiersRun`·`_neuralSkipped`(early-exit 여부).

---

## 8. 운영·확장 가이드

- **모델 선택**: 디바이스·정확도 요구에 맞춰 `--model`. 기본 base, 정확도 최우선 xlmr-large.
- **규칙 추가**: 새 정규식→`PATTERN_RECOGNIZERS`, 사내어휘→`DENY_LIST_RECOGNIZERS`, 키워드→런타임 `weights`.
  다국어 기밀표지(机密/極秘/CONFIDENTIAL) 추가로 비한국어 누락 방지 강화(참조: `L5_다국어_누락률_평가.md`).
- **정책 외부화**: 회사별 등급체계·규정을 GradeProfile(YAML)·floor_rules로(참조: `규칙_정책_확장_방안.md`).
- **재평가 필수**: 규칙·정책 변경 후 하네스 held-out 재평가로 재현율↑ vs 과대분류 trade-off 확인·회귀 방지.
- **에어갭**: 모델·프로필 모두 로컬 파일 로드. 런타임 외부 접근 금지.

---

## 9. 한계 & 주의

- 증류 상한 = 교사 성능·학생 용량. 더 높이려면 더 강한 교사(GPT-5 등) + 라지 학생.
- 소형/base는 외국어 O/S 과대분류 경향(안전엔 무해). 정확도·다국어는 xlmr-large 권장.
- 뉴럴-단독 수치이며 실서비스는 규칙 floor가 안전을 추가 보증. 도입 시 고객 데이터로 재보정 권장.

---

## 부록 · 관련 문서
- `모델_실체_및_릴리즈_가이드.md` — 모델 물리적 실체·릴리즈
- `LLM증류_방식_및_도구.md` — 증류 개념·도구·절차
- `모델_라인업_선택구조_제안.md` §9 — 실측 라인업 매트릭스
- `규칙_정책_확장_방안.md` — 정규식/키워드/GradeProfile 확장
- `L5_다국어_누락률_평가.md` — 다국어 적대공격 누락률
- `base_0.8돌파_전략.md` — base 정확도 추가 개선
