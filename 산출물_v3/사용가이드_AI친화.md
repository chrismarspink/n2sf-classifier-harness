{{toc}}

# N²SF classifier_v3 — 사용 가이드 (AI 친화 · 기계 파싱용)

> 목적: 사람과 **AI 에이전트가 모두** 이 함수를 정확히 호출하도록, 계약(contract)·스키마·예시를 명시.
> 버전: v3 (N²SF 공식 §9 정합) · 추론: 100% 로컬 CPU, 외부 LLM/GPU 없음.

---

## 1. 무엇을 하는가 (한 줄)

문서 텍스트/파일을 입력받아 **N²SF 보안등급 C(기밀)/S(민감)/O(공개)** 로 분류한다. 근거(SHAP)·티어·모델버전 포함.

---

## 2. 진입점 (함수 계약)

### 2.1 클래스 `N2SFClassifier` (서비스 권장 — 재사용)
```python
from classifier_v3 import N2SFClassifier
clf = N2SFClassifier(model="accurate")     # 로드 1회
res = clf.classify("문서.pdf")             # 또는 원문 텍스트 문자열
```

**생성자**: `N2SFClassifier(model="accurate", ensemble="soft", tier_weights=None, early_exit=True, locale="ko", warmup=True)`
- `model` (str): 아래 §3 티어 프리셋명 | 백엔드명 | 모델 디렉토리 경로.
- `early_exit` (bool): 규칙(T1)에서 기밀 확정 시 뉴럴(T3) 생략(속도↑). 기본 True.
- `locale` (str): "ko"|"ja"|"en"|"zh-CN".

**메서드**:
- `classify(source: str, *, is_file: bool=None) -> dict` — §4 스키마 반환. `is_file` 생략 시 자동판별.
- `reload_model(model: str, *, path: str=None, evict_old=True) -> dict` — **무중단 교체**(§6).
- `register_model(name: str, path: str)` — 새 모델 백엔드 등록.
- `model_version` (property) → 예: `"accurate#v0"`.

### 2.2 편의 함수(1회성)
```python
from classifier_v3 import classify
res = classify("주민등록번호 900101-1234567", model="accurate")
```

### 2.3 CLI
```bash
python classifier_v3.py 문서.pdf --model accurate --json
python classifier_v3.py --text "..." --model compact-multi
python classifier_v3.py --demo
```

---

## 3. 모델 티어 (enum: model)

| 프리셋 | 백엔드 | 용도 | 크기/지연 |
|---|---|---|---|
| `fast-ko` | n2sf-small | 엣지·한국어 전용 | 57MB/~14ms |
| `compact-multi` | n2sf-small-multi-minilm | 소형 다국어 | 488MB/~40ms |
| `balanced` | n2sf-base | 표준 | 1.1GB/~120ms |
| **`accurate`** | n2sf-xlmr-large | **정확도·강건성 최우선(권장 기본)** | 2.3GB/~200ms |
| `n2sf-official` | n2sf-xlmr-official | §9 공식 재증류판(있으면) | 2.3GB |

> OOD 검증상 **라지(accurate)가 강건·누출0**. 소형/base는 미지분포에서 안전 약화 → 안전 최우선이면 accurate.

---

## 4. 반환 스키마 (classify 결과)

```json
{
  "grade": "C|S|O",
  "gradeLabel": "기밀|민감|공개",
  "confidence": 0.0-1.0,
  "score": <float 규칙점수>,
  "format": "pdf|docx|xlsx|hwpx|text|...",
  "tiersRun": ["rules","ner"] | ["rules","ner","neural"],
  "neuralSkipped": true|false,
  "shap": {"total": <float>, "contributions": [{"feature","label","contribution","percent"}]},
  "compliance": {"regulations": [...], "violations": [{"code","msg","severity"}]},
  "modelVersion": "accurate#v0",
  "elapsedMs": <int>
}
```
- **핵심 필드**: `grade`(최종 등급), `confidence`, `shap.contributions`(근거), `modelVersion`(감사).

---

## 5. 분류 기준 (N²SF §9 — 판정 규칙, 결정론적 부분)

| 등급 | 트리거 | 근거 |
|---|---|---|
| **C 기밀** | '기밀/대외비/극비/사외비/機密/極秘/绝密' 표지(하드 floor) · 또는 뉴럴이 국가안보·외교·수사 '영향'으로 판정 | 정보공개법 §9 1~4호 |
| **S 민감** | 개인정보(주민번호·성명·연락처·주소)·계좌·카드·키 · 내부문서 · 로그/백업 · 대량 개인정보(명부) | §9 5~8호 |
| **O 공개** | 개인정보·기밀표지 없음 · 공개 안내/보도자료(발표후)/통계 | 잔여 |

- **중요(§9)**: 개인정보(주민번호 포함)는 **기본 S**. 국가안보·수사·외교 맥락일 때만 C.
- 규칙(정규식/키워드)은 **S floor + 기밀표지 C floor**만 담당. **C의 의미 판단은 뉴럴**.
- 전각·띄어쓰기 난독(９０…, 대 외 비)도 정규화로 검출.

---

## 6. 무중단 핫리로드 (요약 — 상세 §온사이트 가이드)

```python
clf.reload_model("n2sf-official")                       # 티어/백엔드
clf.reload_model("cust-v2", path="models/n2sf-custom-v2")  # 온사이트 학습 결과
```
- 새 모델을 **먼저 로드·검증한 뒤** 활성 참조를 **원자적으로 교체** → 진행 중 요청은 구 모델로 완료, 신규 요청부터 신 모델. **다운타임 0.**

---

## 7. AI 에이전트용 호출 레시피 (복붙)

```python
# 1) 단발 분류
from classifier_v3 import classify
r = classify(open_text_or_path, model="accurate")
grade = r["grade"]              # "C"/"S"/"O"
why   = r["shap"]["contributions"]

# 2) 서비스(재사용 + 핫리로드)
from classifier_v3 import N2SFClassifier
clf = N2SFClassifier("accurate")
r = clf.classify(text)
# ... 고객 재학습 후 ...
clf.reload_model("cust-v2", path="models/n2sf-custom-v2")   # 무중단
```

**주의(계약)**
- 입력이 파일 경로면 `is_file=True` 권장(자동판별은 개행 없는 짧은 경로만). 원문은 `is_file=False`.
- 반환은 항상 dict. 실패 시 예외 → 호출측 try/except.
- 안전 원칙: 과소분류(유출)=최악. 애매하면 상위. 대량 오분류는 사람 검토 큐로.

---

## 8. 의존성·실행
- `pip install transformers torch presidio-analyzer presidio-anonymizer spacy openpyxl pypdf`
- `python -m spacy download ko_core_news_lg`
- 엔진 `data_classifier.py`와 `models/` 는 리포 루트. 본 파일은 상위 폴더를 자동 import.
