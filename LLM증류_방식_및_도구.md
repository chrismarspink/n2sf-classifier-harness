{{toc}}

# LLM 지식증류 — 개념 & 우리 프로젝트의 방식·도구

> 일자: 2026-07-02 · 목적: (1) LLM 증류가 무엇인지, (2) **우리가 어떤 도구·절차로** 증류하는지 정리.
> 교사 후보: GPT-4o·o4-mini(현재, Azure) / **GPT-5(키 제공 예정)** / Fable 5(키 없음) / 로컬 Gemma.

---

## 1. LLM 지식증류란 (개념)

**크고 똑똑한 교사(LLM)의 판단을 작은 학생(로컬 모델)에 이전**하는 기법.
학생은 정답만 외우는 게 아니라 **교사가 "어떻게·얼마나 확신하며" 판단했는지**를 모방한다.

- 목표: **"클라우드 LLM 성능을 외부 반출·GPU 없이 로컬에서"** — 추론엔 LLM 불필요.
- **soft-label 증류(우리 방식)**: 교사가 최종 등급 1개(hard)가 아니라 **확률분포**(예: O=0.1, S=0.3, **C=0.6**)를
  주고, 학생이 그 분포를 맞추도록 학습 → 뉘앙스·불확실성까지 전이(데이터 효율↑).

```
[교사 LLM]  문서 → O/S/C 확률분포        (판단 지식)
                     │  라벨로 사용
                     ▼
[학생 BERT] 문서 → 출력분포 ≈ 교사분포   (KD로 모방 학습)
                     ▼
[실서비스]  로컬 학생만으로 분류 (LLM·GPU·외부반출 없음)
```

---

## 2. 우리가 쓰는 도구 (스택)

| 역할 | 도구 | 우리 코드/모델 |
|---|---|---|
| **교사(라벨 생성)** | Azure OpenAI (`openai` SDK, AzureOpenAI) | gpt-4o / o4-mini / **GPT-5(예정)**. 로컬 Gemma(ollama)도 가능 |
| **문서 생성** | 교사 LLM + 우리 합성기 | `harness/gen_llm.py` (LLM 문체 + `synth.py` 유효 PII 주입=라벨 신뢰) |
| **학생(로컬 모델)** | HuggingFace `transformers` + `torch`(Apple MPS) | `AutoModelForSequenceClassification` — mDeBERTa/KoELECTRA/KLUE/XLM-R |
| **KD 학습** | torch(소프트 CE 손실) | `harness/train_soft.py` (레이어 동결로 대형 OOM 방지) |
| **교사 라벨링 루프** | Azure 호출 오케스트레이션 | `harness/distill.py` (`teacher_soft`) |
| **다중 백본·평가** | 하네스 | `harness/lineup.py` (3축 매트릭스) |
| **정직 평가** | 하네스 | held-out(학습 미사용) + `harness/metrics.py` |

- **KD 손실**: `L = − Σ_g p_교사(g)·log softmax(학생)_g` (g∈{O,S,C}).
- **추론(실서비스)**: `data_classifier.py` 3-tier(정규식+NER+학생) — **LLM 호출 없음**.

---

## 3. 우리 증류 절차 (5단계)

```
1) 문서 생성   gen_llm.py — 교사 LLM(Azure)이 다양한 문체·난이도 문서 작성
               + synth.py가 등급 결정요소(주민번호·카드·키워드)를 유효 포맷으로 주입 → 라벨 신뢰
2) 교사 라벨링 distill.py teacher_soft — 교사 LLM이 각 문서에 O/S/C 확률분포 부여
               → teacher_labels.jsonl  ({"text":..., "probs":[pO,pS,pC]})
3) 학생 학습   train_soft.py — transformers+torch(MPS)로 학생이 KD 손실 최소화
               → models/n2sf-{small,base,large}/  (558MB급 가중치 폴더)
4) 정직 평가   held-out(학습에 안 쓴 LLM 분포)로 일반화 측정 (metrics.py)
               → 과적합 아닌 실제 성능. LLM 기준선과 비교
5) 반복/라인업 distill 루프(교사라벨 누적 재학습) / lineup(백본별 증류·3축 비교)
```

**핵심 산출 파일**
- `teacher_labels.jsonl` — 교사 soft-라벨(증류 학습셋)
- `heldout.json` — 동결 평가셋(정직 분포)
- `models/n2sf-*/` — 증류된 학생 모델(온디바이스)
- `cycles.json`·`matrix.json`·`report.html` — 성능 궤적·선택 매트릭스

---

## 4. 교사 교체 (GPT-5 / Fable 5 / 기타)

파이프라인은 **교사-불가지론적**이라 교사만 바꾸면 된다.

### 4.1 GPT-5 (키 제공 예정) — **바로 가능**
- Azure 배포명·API 버전만 추가하면 됨:
  ```
  # .env.azure 에 추가
  AZURE_GPT5_DEPLOYMENT=gpt-5
  AZURE_GPT5_APIVER=<제공 버전>
  ```
- `distill.py --teacher gpt-5` 로 실행(교사 함수에 배포명 분기 1줄 추가).
- GPT-5는 더 강한 교사 → **증류 상한↑**. 단 상한을 다 쓰려면 **라지 학생**이 필요(소형은 용량 병목).

### 4.2 Fable 5 (Anthropic) — 키 없음
- Anthropic API 키 필요(Azure 키로는 불가). 키 확보 시 `anthropic` SDK로 교사 함수 추가하면 동일하게 동작.

### 4.3 로컬 Gemma
- 오프라인 교사로 사용 가능하나 품질(0.873)이 낮아 **비교 기준선** 위주로 사용.

> **주의(공통)**: 교사에는 **합성 문서만** 전송(실기밀 금지). 데이터 주권 원칙 유지.

---

## 5. 왜 이 방식인가 (요지)

- **온프레미스·CPU**: 추론에 LLM·GPU 불필요 — 교사는 개발단계 오프라인 라벨링에만.
- **데이터 주권**: 실문서 외부 반출 0. 교사엔 합성만.
- **비용**: 추론당 API 과금 0. 교사 비용은 학습 1회성.
- **성능**: "외부 LLM 없이 로컬에서 LLM 근접"을 실측(학생 0.53→0.78~0.80, 교사 gpt-4o 0.915).
- **한계**: 증류 상한 = 교사 성능 · 학생 용량. 더 높이려면 **더 강한 교사(GPT-5) + 더 큰 학생(라지)** 조합.

---

## 6. 현황 & GPT-5 도입 시 계획

- 현재: gpt-4o/o4-mini 교사로 소형/base/라지 4백본 증류·평가 진행 중(lineup).
- **GPT-5 키 제공 시**:
  1. `.env.azure`에 GPT-5 배포 추가 → `--teacher gpt-5`.
  2. **라지 학생 + GPT-5 교사** 재증류 → 0.85 이상 도전(교사·학생 상한 동시 상향).
  3. held-out으로 gpt-4o/o4-mini/GPT-5 교사별 학생 성능 비교 → 교사 선택 근거.

> 준비되면 키를 `.env.azure`(gitignore됨)에 넣어 주시면 `--teacher gpt-5`로 바로 증류하겠습니다.
