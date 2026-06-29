# data_classifier.py — 문서 등급 분류 단일 소스 모델 · 개발 가이드

문서를 입력받아 **C/S/O 등급**으로 분류하는 모델을 한 파일(`data_classifier.py`)로
통합한 것이다. 기존 `classifier-svc` 의 분류 로직(패턴·Presidio NER·신경망 tier·
점수화·앙상블·SHAP)과 파일 추출(오피스 계열 + HWPX)을 전부 포함한다.

> 이 문서는 **소스를 기반으로 신규 기능을 추가**하기 위한 개발 가이드를 겸한다.
> 빠른 사용은 §1–3, 모델 선택은 §7, **기능 확장은 §11** 을 보라.

기존 서비스 파일(`classify.py`/`presidio_setup.py`/`neural.py`/`custom_patterns.yaml`)은
그대로 두고, 이 모듈은 **독립 실행 가능한 단일 소스**로 별도 제공된다.

---

## 목차
1. 빠른 시작
2. API
3. 반환 JSON 규격
4. 등급 체계 (N²SF)
5. 가중치 / 임계값 override
6. 지원 파일 포맷 (오피스 + HWPX 우선)
7. 신경망(LLM) 모델 카탈로그 — 15종
8. 사용 사례 (시나리오별)
9. 파이프라인 내부 동작
10. 의존성 · 성능
11. **신규 기능 추가 가이드 (확장)**
12. 트러블슈팅

---

## 1. 빠른 시작

```bash
cd classifier-svc
pip install -r requirements.txt              # presidio + spaCy
python -m spacy download ko_core_news_sm     # 로케일별 1회

# 신경망(LLM) tier 를 쓰려면(옵션):
pip install "sentence-transformers" "transformers" "torch"

# CLI
python data_classifier.py 직원명부.xlsx --verbose
python data_classifier.py 발표자료.pptx --no-n2sf          # 풀네임 등급
python data_classifier.py 계약서.hwpx --llm --model ko-sroberta
```

```python
from data_classifier import classify

result = classify("직원명부.xlsx", n2sf_mode=True, verbose=True)
print(result["grade"], result["score"])        # 예: 'C' 12.0
```

---

## 2. API

### `classify(file, n2sf_mode=True, verbose=False, llm_mode=False, *, model="minilm", weights=None, locale="ko", ensemble_method="escalate") -> dict`

문서 **파일 경로**를 받아 텍스트 추출 → 분류까지 수행하고 결과 dict 를 돌려준다.

| 파라미터 | 타입 | 기본 | 설명 |
|---|---|---|---|
| `file` | str \| Path | — | 분류할 파일. 오피스(docx/xlsx/pptx) + HWPX 우선 |
| `n2sf_mode` | bool | `True` | `True` → `grade` 를 N²SF 코드 `C/S/O`. `False` → 풀네임 |
| `verbose` | bool | `False` | `True` → `pii`/`keywords` 에 **실제 검출 원문**·offset 포함 |
| `llm_mode` | bool | `False` | `True` → 신경망(LLM) tier 활성 (`model` 로 백엔드 선택) |
| `model` | str | `"minilm"` | 신경망 백엔드 키 (§7 의 15종 중 택1) |
| `weights` | dict | `None` | 가중치/임계값 override (§5) |
| `locale` | str | `"ko"` | `ko`·`ja`·`en`·`zh-CN`·`auto`(본문 기반 추정) |
| `ensemble_method` | str | `"escalate"` | `escalate`·`vote`·`weighted`·`max-rank`·`soft` |

### `classify_text(text, *, locale="ko", n2sf_mode=True, verbose=False, llm_mode=False, model="minilm", weights=None, ensemble_method="escalate") -> dict`

이미 추출된 텍스트를 분류 (파일 추출 단계 생략). OCR/STT 결과 등 외부에서 뽑은
텍스트를 그대로 넣을 때 사용. 반환 규격은 `classify()` 와 동일(단 `file` 키 없음).

### `extract_text(file) -> (text, file_type, warnings)`

파일 → 텍스트만 추출. 분류 없이 추출만 검증할 때.

---

## 3. 반환 JSON 규격

화면에 검출 결과를 출력하는 데 필요한 정보로 구성된다.

```jsonc
{
  "schemaVersion": "2.0.0",
  "n2sfMode": true,
  "grade": "C",                  // n2sf_mode=true → C/S/O, false → CONFIDENTIAL/...
  "gradeCode": "C",              // 항상 코드도 제공
  "gradeFull": "CONFIDENTIAL",   // 항상 풀네임도 제공
  "gradeLabel": "기밀",           // 한국어 표시명
  "score": 12.0,                 // 가중 합산 점수
  "confidence": 0.95,            // 0~1
  "locale": "ko",

  "pii": {                       // 검출된 개인정보 (타입별)
    "totalCount": 4,
    "byType": [
      {
        "type": "KR_RRN", "label": "주민등록번호",
        "count": 1, "confidence": 0.9, "weight": 6.0,
        "source": "presidio", "rule": "KR_RRN",
        "spans": [[24, 38]],
        "items": [                         // verbose=true 일 때만
          {"text": "800101-1234567", "start": 24, "end": 38, "score": 0.9}
        ]
      }
    ]
  },

  "keywords": [                  // 등급 키워드 (대외비/극비/Confidential …)
    {"type": "KEYWORD_SECRET", "label": "대외비", "count": 1, "weight": 3.0,
     "items": [{"text": "대외비", "start": 0, "end": 3}]}   // verbose 일 때만
  ],

  "shap": {                      // 등급 분류 기여도 (화면 "왜 이 등급?" 패널용)
    "baseline": 0.0, "total": 12.0,
    "contributions": [
      {"feature": "KR_RRN", "label": "주민등록번호", "count": 1,
       "contribution": 6.0, "percent": 0.5, "direction": "up"},
      {"feature": "KEYWORD_SECRET", "label": "대외비", "count": 1,
       "contribution": 3.0, "percent": 0.25, "direction": "up"}
    ]
  },

  "ensemble": { "grade": "...", "method": "escalate", "votes": {...}, "weights": {...} },
  "tiers": { "rules": {...}, "ner": {...}, "neural": {...} },
  "compliance": { "regulations": ["KR-PIPA"],
                  "violations": [{"code":"KR-PIPA-Art23","msg":"...","severity":"warn"}] },
  "jpCompliance": null,          // locale=='ja' 일 때만 채워짐
  "model": { "rules":"2.0","presidio":"2.2.355","spacy":"ko_core_news_sm",
             "neural": null, "ensembleMethod":"escalate" },
  "stats": { "textLength":52, "elapsedMs":40, "findingsCount":4,
             "bulkPii":false, "bulkPiiCount":2,
             "thresholds":{"confidential":5.5,"sensitive":0.75} },

  "file": { "name":"직원명부.xlsx", "type":"xlsx", "textLength":52, "warnings":[] }
}
```

**화면 출력 매핑 가이드**
- 등급 배지 → `grade` / `gradeLabel`, 신뢰도 → `confidence`
- 검출 개인정보 목록 → `pii.byType` (타입·건수, verbose 면 `items[].text`)
- 등급 근거(기여도 막대) → `shap.contributions` (`label`, `percent`)
- 규제 경고 → `compliance.violations`
- tier별 판정(감사) → `tiers.{rules,ner,neural}`

---

## 4. 등급 체계 (N²SF)

| 코드 | 풀네임 | 라벨 | 트리거 예 |
|---|---|---|---|
| `O` | OPEN | 공개 | 민감정보 없음 |
| `S` | SENSITIVE | 민감 | 이메일·전화·이름·주소 등 |
| `C` | CONFIDENTIAL | 기밀 | 주민번호·카드·여권·마이넘버·API 키, 대량 PII, 기밀 키워드 |

점수 임계값: `score ≥ 5.5` → C, `≥ 0.75` → S, 그 외 O.
강제 상향 규칙(앙상블이 하향 불가):
- **대량 PII**: 개인식별자 합계 ≥ 10건 → C (명부/내보내기 탐지)
- **JP 마이넘버**: 검출 시 → C + suppress 강제

---

## 5. 가중치 / 임계값 override (`weights`)

모든 키 선택적. 미지정 항목은 기본(보정된) 값 사용.

```python
weights = {
  "tier":     {"rules": 1.0, "ner": 1.0, "neural": 1.5},   # 앙상블 tier 가중치
  "entity":   {"KR_RRN": 7.0, "EMAIL_ADDRESS": 0.5},       # 타입별 점수 가중치
  "keyword":  [{"keyword": "내부자료", "weight": 1.5, "label": "내부자료"}],  # 추가 키워드
  "thresholds": {"confidential": 6.0, "sensitive": 1.0},   # 등급 임계값
}
classify("file.docx", weights=weights)
```

CLI 는 `--weights weights.json` 으로 동일 구조의 JSON 파일을 받는다.

> 점수 가중치/임계값을 바꾸면 보정(calibration)이 깨질 수 있다. 운영 적용 전
> `sample-data/calibrate.py` 로 재검증 권장.

---

## 6. 지원 파일 포맷 (오피스 + HWPX 우선)

| 분류 | 확장자 | 추출 범위 | 비고 |
|---|---|---|---|
| **워드** | `.docx` `.docm` | 본문 + 머리/꼬리말 + 각주/미주 + 주석 | zip-of-xml `<w:t>` |
| **엑셀** | `.xlsx` `.xlsm` | 공유문자열 + **모든 셀(문자열·숫자·수식·inline)**, 시트별 행 보존 | 전용 파서 |
| **파워포인트** | `.pptx` `.pptm` | 슬라이드 + **표 셀** + **발표자 노트** + 다이어그램/차트 | zip-of-xml `<a:t>` |
| **HWPX** | `.hwpx` | `Contents/section*.xml` + header | `<hp:t>` |
| 평문 | `.txt` `.csv` `.md` `.log` `.json` `.tsv` | 전체 | UTF-8 |
| PDF(옵션) | `.pdf` | 최대 30p 텍스트 | `pypdf` → `pdfminer.six` |
| 구형 한글(옵션) | `.hwp` | best-effort | `hwp5txt`(pyhwp) → olefile PrvText, HWPX 권장 |
| 이미지/음성 | png/jpg/mp3/… | **미지원** | OCR/STT 별도 서비스 → `classify_text()` 사용 |

추출 강화 포인트
- **xlsx**: 숫자로 저장된 계좌번호·전화번호·사번 등도 복원(공유문자열만 읽던 한계 제거).
  셀은 공백, 행은 줄바꿈으로 이어 PII 인식기가 경계를 정확히 잡게 한다.
- **pptx**: 표 셀과 발표자 노트까지 포함 — 노트에 숨은 PII 누락 방지.
- **docx**: 각주/미주/주석 포함 — 본문 외 영역의 민감정보 포착.
- 오피스·HWPX 추출은 **표준 라이브러리(zipfile/re)만으로** 동작(외부 의존 0).

> 이미지(OCR)·음성(STT)은 본 모듈 범위 밖. 해당 서비스로 텍스트를 뽑은 뒤
> `classify_text(text, locale=...)` 에 넣으면 동일 규격으로 분류된다(§8.5).

---

## 7. 신경망(LLM) 모델 카탈로그 — 15종

`llm_mode=True` 일 때만 동작. `model=` 또는 CLI `--model` 로 선택. 미설치/오류 시
자동 무력화(규칙·NER 만으로 분류, `tiers.neural.error` 에 사유 표기).

**방식(kind)**
- `exemplar` — 문장 임베딩 + 등급 예시문 코사인. 가볍고 무난. **기본 권장.**
- `zeroshot` — NLI 제로샷. 등급별 **확률**을 직접 산출(soft 앙상블에 유리).
- `embed` — 원시 인코더 평균풀링 → exemplar 와 동일 코사인. 파인튜닝 없이 신호 제공.

| key | 모델 | kind | 언어 | 메모 |
|---|---|---|---|---|
| `minilm` ★기본 | paraphrase-multilingual-MiniLM-L12-v2 | exemplar | 다국어 | 빠름·경량(~120MB), 실시간 권장 |
| `mpnet` | paraphrase-multilingual-mpnet-base-v2 | exemplar | 다국어 | minilm 대비 정확↑, ~1GB |
| `labse` | LaBSE | exemplar | 109개 | 다국어 혼재 문서 최강, ~1.8GB |
| `e5` | intfloat/multilingual-e5-base | exemplar | 다국어 | 의미유사도 SOTA급, ~1.1GB |
| `bge-m3` | BAAI/bge-m3 | exemplar | 다국어 | 장문·다국어 강함, ~2.2GB |
| `ko-sroberta` | jhgan/ko-sroberta-multitask | exemplar | 한국어 | **국내 문서 특화** 임베딩 |
| `mdeberta` | mDeBERTa-v3-base-mnli-xnli | zeroshot | 다국어 | 다국어 제로샷, 확률 산출 |
| `xlmr-xnli` | joeddav/xlm-roberta-large-xnli | zeroshot | 다국어 | 고정확 제로샷, ~2.2GB |
| `deberta-large-mnli` | DeBERTa-v3-large-mnli-… | zeroshot | 영문 | 영문 제로샷 최상급 |
| `bart-mnli` | facebook/bart-large-mnli | zeroshot | 영문 | 영문 제로샷 표준 |
| `mbert` | bert-base-multilingual-cased | embed | 다국어 | 표준 다국어 BERT |
| `xlm-roberta` | xlm-roberta-base | embed | 다국어 | 다국어 표현력↑ |
| `koelectra` | monologg/koelectra-base-v3 | embed | 한국어 | 한국어 ELECTRA |
| `klue-roberta` | klue/roberta-base | embed | 한국어 | KLUE 벤치마크 RoBERTa |
| `kcbert` | beomi/kcbert-base | embed | 한국어 | 댓글·구어체 한국어 |

**선택 가이드**
- 한국어 위주, 정확도 중요 → `ko-sroberta`(exemplar) 또는 `koelectra`/`klue-roberta`(embed)
- 다국어 혼재 → `labse` / `e5` / `mpnet`
- 등급 확률(soft 앙상블, 설명력) 필요 → `mdeberta`(다국어) / `xlmr-xnli`(정확) / `bart-mnli`(영문)
- 빠른 실시간·데모 → `minilm`(기본)

모델은 최초 호출 시 HF 허브에서 로드(캐시). 오프라인 환경은 사전 다운로드 필요.
`exemplar`/`embed` 는 `sentence-transformers`·`torch`, `zeroshot` 은 `transformers`·`torch` 필요.

**앙상블 방식(`ensemble_method`)**
- `escalate`(기본): 규칙+NER 등급을 신경망이 **상향만** 가능(보수적, 오탐 억제)
- `vote`: tier 다수결(동률 시 상위 등급)
- `weighted`: tier 가중치×신뢰도 합산 argmax
- `max-rank`: 어느 tier든 최고 등급 채택(최고 민감, recall↑)
- `soft`: tier별 등급 확률분포 가중합 argmax(가장 부드러운 결합, 확률 모델)

---

## 8. 사용 사례 (시나리오별)

### 8.1 직원 명부 xlsx → 대량 PII 자동 기밀화
```python
r = classify("직원명부.xlsx", n2sf_mode=True, verbose=True)
# 전화·이메일·주소가 행마다 누적 → bulkPiiCount ≥ 10 → grade='C'
assert r["grade"] == "C"
print(r["stats"]["bulkPiiCount"], "건 →", r["compliance"]["violations"][0]["msg"])
```

### 8.2 발표자료 pptx → 노트에 숨은 연락처 탐지
```python
r = classify("Q3전략.pptx", verbose=True)
for p in r["pii"]["byType"]:
    print(p["label"], p["count"], [i["text"] for i in p.get("items", [])])
# 슬라이드 표 + 발표자 노트의 전화번호까지 검출
```

### 8.3 계약서 hwpx → 한국어 특화 모델로 정밀 분류
```python
r = classify("임대차계약서.hwpx", llm_mode=True, model="ko-sroberta",
             ensemble_method="soft")
print(r["grade"], r["tiers"]["neural"])   # 신경망 등급 확률 포함
```

### 8.4 등급 근거를 화면에 표시 (SHAP)
```python
r = classify("문서.docx")
print(f"{r['gradeLabel']} (신뢰도 {r['confidence']:.0%})")
for c in r["shap"]["contributions"]:
    print(f"  {c['label']:14s} {'█'*int(c['percent']*20)} {c['percent']:.0%}")
# 주민등록번호  ██████████ 50%
# 대외비        █████      25% ...
```

### 8.5 OCR/STT 결과를 동일 규격으로 분류
```python
from data_classifier import classify_text
text = run_my_ocr("scan.png")            # 외부 OCR/STT
r = classify_text(text, locale="ko", verbose=True)
```

### 8.6 조직 커스텀 키워드 + 임계값 강화
```python
weights = {
  "keyword": [{"keyword": "사외유출금지", "weight": 3.0, "label": "사외유출금지"},
              {"keyword": "Project Nova", "weight": 2.0, "label": "Project Nova"}],
  "thresholds": {"sensitive": 1.0},
}
classify("기획안.docx", weights=weights)
```

### 8.7 배치 분류 (디렉터리 일괄)
```python
from pathlib import Path
from data_classifier import classify
SUPPORTED = {".docx",".docm",".xlsx",".xlsm",".pptx",".pptm",".hwpx",".txt",".pdf"}
for f in Path("docs").rglob("*"):
    if f.suffix.lower() in SUPPORTED:
        r = classify(f, locale="auto")
        print(f"{r['grade']}  {r['score']:5.1f}  {f.name}")
```

### 8.8 일본어 마이넘버 → 강제 기밀 + suppress
```python
r = classify("従業員リスト.xlsx", locale="ja")
print(r["grade"], r["jpCompliance"])   # 'C', myNumberSuppressForced=True
```

---

## 9. 파이프라인 내부 동작

```
파일 → extract_text() → 텍스트
     → Presidio(패턴+spaCy NER) → findings
     → 키워드 스캔(등급 라벨)    → keyword findings
     → _score() 가중 합산        → score, base grade
     → 대량PII/JP 강제 상향
     → 신경망(opt) → _combine() 앙상블 → 최종 grade
     → _shap_block() 기여도 분해
     → JSON
```

- **점수식**: `Σ entity_weight·min(count, cap) + Σ keyword_weight·min(count, 3)`
  (CONFIDENTIAL driver 는 cap 없이 1건만으로 임계 도달)
- **로케일 우선순위**: 활성 로케일의 자국 PII 를 타국 PII 보다 우선(겹치면 타국 제거).
- **중복 제거**: 더 구체적인 인식기가 일반 인식기를 억제(예: KR_PHONE > PHONE_NUMBER).
- **SHAP**: 신경망 그래디언트가 아닌 **점수 기여 분해**(각 finding 가중 기여를 전체로 정규화).

---

## 10. 의존성 · 성능

| 용도 | 패키지 | 비고 |
|---|---|---|
| 필수 | `presidio-analyzer`, `spacy`(+로케일 모델), `numpy` | |
| 신경망 exemplar/embed | `sentence-transformers`, `torch` | llm_mode |
| 신경망 zeroshot | `transformers`, `torch` | llm_mode |
| PDF | `pypdf` 또는 `pdfminer.six` | |
| 구형 hwp | `pyhwp`, `olefile` | |

- spaCy 모델은 로케일별 최초 1회 로드(~1–3s) 후 메모이즈.
- 오피스/HWPX 추출은 표준 라이브러리만 사용 → 별도 설치 불필요.
- 신경망 모델은 첫 호출 시 다운로드/로드(수백 MB~2GB) 후 캐시.

---

## 11. 신규 기능 추가 가이드 (확장)

본 모듈은 **단일 소스**라 확장 지점이 한 파일 안에 모여 있다. 아래 패턴대로 추가한다.

### 11.1 새 PII 패턴(정규식) 추가
`PATTERN_RECOGNIZERS` 리스트에 dict 한 개를 추가하면 즉시 Presidio 인식기로 등록된다.
```python
PATTERN_RECOGNIZERS.append({
    "name": "KR_HEALTH_INSURANCE", "supported_entity": "KR_HEALTH_INSURANCE",
    "patterns": [{"name": "hi_no", "regex": r"\b\d-\d{10}\b", "score": 0.6}],
    "context": ["건강보험", "보험증", "요양기관"],   # 주변에 있으면 score 부스트
})
```
deny-list(고정 어휘) 방식은 `DENY_LIST_RECOGNIZERS` 에 추가한다.

### 11.2 새 엔티티 타입의 점수·표시명 등록
1. `ENTITY_WEIGHTS["KR_HEALTH_INSURANCE"] = 3.0` — 등급 점수 가중치(없으면 기본 0.05).
2. `ENTITY_LABELS["KR_HEALTH_INSURANCE"] = "건강보험번호"` — 화면 표시명.
3. (민감→기밀 상향이 필요하면) 6.0 이상으로 두면 1건만으로 CONFIDENTIAL.
4. (대량 PII 집계 대상이면) `BULK_PII_TYPES` 에 추가.

### 11.3 새 등급 키워드 추가
`GRADE_KEYWORDS` 에 `(키워드, 가중치, 라벨)` 추가. `≥2.5` 면 KEYWORD_SECRET(기밀),
미만이면 KEYWORD_INTERNAL(내부)로 분류된다. **런타임 추가**는 `weights["keyword"]`(§5).

### 11.4 새 신경망(LLM) 모델 추가
`NEURAL_BACKENDS` 에 한 줄 추가하면 `--model <key>` 로 바로 선택된다.
```python
NEURAL_BACKENDS["my-model"] = {
    "label": "사내 파인튜닝 모델", "kind": "embed",   # exemplar|zeroshot|embed 중 택1
    "model": "myorg/my-finetuned-encoder",           # HF 허브 id 또는 로컬 경로
    "langs": "ko", "note": "사내 분류 데이터로 파인튜닝",
}
```
- `kind` 만 맞으면 로딩/추론 코드(`_neural_load`/`_neural_encode`/`neural_infer`)가 재사용된다.
- 등급 예시문(exemplar/embed)을 바꾸려면 `EXEMPLARS`, 제로샷 라벨 문구는 `ZS_LABELS` 수정.

### 11.5 새 파일 포맷 추가
- **zip-of-xml 계열**(odt/odp 등): `_ZIP_XML_TARGETS[".odt"] = [r"content\.xml"]` 추가 → 끝.
- **특수 구조**: `extract_text()` 에 분기 추가 후 전용 추출 함수 작성(`_extract_xlsx` 참고).
- **평문 확장자**: `_TEXT_EXTS` 에 추가.

### 11.6 새 로케일(언어) 추가
1. `SPACY_MODELS["xx"] = "xx_core_web_sm"` (해당 spaCy 모델 설치).
2. `LOCALE_REGS["xx"] = ["XX-LAW"]`, `_LOCALE_PREFIX["xx"] = "XX_"`.
3. 신경망 쓰면 `ZS_LABELS["xx"]`, `EXEMPLARS["xx"]` 추가.
4. 자국 PII 패턴은 §11.1 로 추가(접두사 `XX_`).

### 11.7 임계값·앙상블 튜닝
- 코드 기본값: `C_THRESHOLD`/`S_THRESHOLD`/`ENTITY_COUNT_CAP`/`BULK_PII_THRESHOLD`.
- 런타임 override: `weights["thresholds"]`, `weights["tier"]`, `ensemble_method`.
- **튜닝 후 반드시** `sample-data/calibrate.py` 로 회귀 검증(샘플 등급 분포 유지 확인).

### 11.8 컴플라이언스 규칙 추가
`classify_text()` 의 violations 블록에 조건 추가(예: 특정 타입 검출 시 규제 코드/메시지).
강제 상향이 필요하면 "컴플라이언스 floor"(앙상블 직후)에 grade 고정 로직을 둔다.

---

## 12. 트러블슈팅

| 증상 | 원인 / 조치 |
|---|---|
| `OSError: [E050] ... ko_core_news_sm` | `python -m spacy download ko_core_news_sm` |
| `tiers.neural.error` 채워짐 | llm_mode 모델 미설치/다운로드 실패 — 규칙·NER 로 정상 분류됨 |
| xlsx 에서 숫자 PII 누락 | 전용 파서가 숫자 셀을 복원함. 그래도 없으면 셀이 차트/이미지일 가능성 |
| pptx 텍스트 비어있음 | 슬라이드가 이미지뿐 — OCR 필요(별도 서비스) |
| 등급이 과하게 높음/낮음 | `weights["thresholds"]`/`["entity"]` 로 조정 후 calibrate 재검증 |
| 한국어가 깨짐(구형 .hwp) | HWPX 로 변환해 사용 권장(`_extract_hwp` 는 best-effort) |

---

## 부록. 설계 메모

- `classify.py`/`presidio_setup.py`/`neural.py`/`custom_patterns.yaml` 의 로직과
  **보정된 가중치·임계값을 그대로 이식**했다. 동일 입력에 동일 등급을 낸다.
- 패턴을 YAML 대신 파일 내부(`PATTERN_RECOGNIZERS`)에 인라인해 "단일 소스"를 유지한다.
- SHAP 은 규칙 기반 분류의 설명 가능성을 화면에 직관적으로 보여주는 점수 기여 분해다.
