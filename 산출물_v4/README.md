# N²SF 등급분류 — 산출물 v4 (상세 설명형 · GUI/XAI)

> v3(통합·오프라인·핫리로드) 위에 **판단 근거 전체를 JSON으로 노출**. 화면에서 "왜 C/S/O인지" 그대로 렌더.

## 구성
| 파일 | 역할 |
|---|---|
| `classifier_v4.py` | 상세 설명형 분류(`N2SFExplainClassifier`) — 정규식·키워드·NER·SHAP·§9원칙·파일메타·설명문 |
| `사용가이드_v4_상세.md` | 반환 JSON 전 필드 + **정규식/키워드/§9 확장 변수** + **GUI 렌더링** 가이드 |
| `models/` | 번들 모델(오프라인 로드): n2sf-small·xlmr-large·**xlmr-official(§9)** |

## v3 대비 추가 (요청 반영)
- **정규식/키워드/NER(Presidio·spaCy) 검출 전체 노출** — `findings.regexEntities / nerEntities / keywords`
- **N²SF §9 등급 원칙** — `decision.n2sfPrinciple`(개인정보→S / 기밀표지→C / …)
- **SHAP 기여도** — `shap.contributions`(각 근거가 등급에 몇 %)
- **파일 정보** — `file`(포맷·크기·해시·추출길이)
- **사람이 읽는 설명문** — `explanation` (GUI 상단)
- 리턴값 = **GUI로 그릴 수 있는 상세 JSON**

## 빠른 시작
```bash
python classifier_v4.py --demo
python classifier_v4.py 문서.pdf --model accurate       # 상세 JSON 출력
```
```python
from classifier_v4 import N2SFExplainClassifier
clf = N2SFExplainClassifier("accurate")
r = clf.classify("문서.pdf")     # r: grade·file·decision·tiers·findings·shap·explanation …
```

## 특성
- **완전 오프라인**(HF 미접속·NER 로컬), **무중단 핫리로드**, **§9 정합**(개인정보=S). 추론 100% 로컬 CPU.
- 근거·SHAP·§9원칙 노출 → **설명가능(XAI)**, LLM 블랙박스 대비 강점.

> 엔진·확장은 리포 `data_classifier.py`, `규칙_정책_확장_방안.md`, §9는 `N2SF공식기준_적용방안.md` 참조.
