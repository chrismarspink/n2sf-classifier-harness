# N²SF 등급분류 — 산출물 v4 (3-tier 통합 단일 파일 · 연구소 표준 · GUI/XAI)

> `classifier_v4.py` **한 파일**에 3-tier 전체(규칙 정의 포함)가 들어있다. 연구소가 이 파일을 표준으로 편집·배포.
> 추론 100% 로컬(외부 LLM/GPU 0). **기본 모델=§9 학습판(n2sf-official)**. 반환은 GUI/XAI용 상세 JSON.

## 구성
| 파일 | 역할 |
|---|---|
| `classifier_v4.py` | **3-tier 통합 단일 파일** — 규칙(PATTERN/DENY_LIST/GRADE_KEYWORDS/SECRET_FLOOR_KW)+엔진+뉴럴+SHAP+§9+설명층+(옵션)Fable+핫리로드 |
| `serve_v4.py` | 온프레미스 분류 서비스(:8091) + 학습/무중단 핫리로드 API (표준 라이브러리만) |
| `admin_site.py` | 관리 콘솔(:8091) — 대시보드·분류(XAI)·데이터·학습·모델버전(적용/롤백)·모니터링·설정 |
| `train_onsite.py` | 고객 라벨(CSV/JSONL)→전이+혼합 재학습 → 새 버전 모델(무중단 적용용) |
| `bundle_offline.sh`·`requirements_offline.txt` | 폐쇄망 오프라인 번들(모델 복사 + wheel 아카이브) |
| `사용가이드_v4_상세.md` | 편집 지점·반환 JSON 전 필드·정규식/키워드 확장·GUI 렌더링·학습/핫스왑(v3)·한계 |
| `오프라인_배포_가이드.md`·`온사이트학습_핫리로드_가이드.md`·`학습데이터_가이드.md` | 배포·재학습·데이터 준비 상세 가이드 |
| `models/` | 번들 모델(오프라인): n2sf-small · xlmr-large · **xlmr-official(§9, 기본)** |

## v4 핵심 (요청 반영)
- **단일 파일**: 정규식·deny-list·키워드·§9 원칙·NER·뉴럴·SHAP 전부 이 파일에. 연구소가 상단 규칙 변수만 고치면 반영.
- **상세 JSON(GUI/XAI)**: `findings.regexEntities/nerEntities/keywords` · `shap` · `decision.n2sfPrinciple` · `file` · `explanation`.
- **§9 정합 기본**: 기본 모델 n2sf-official(개인정보=S). 다른 티어는 구 rubric → §9엔 official.
- **(옵션) Fable5 T4 티어**: `use_fable=True`+키 일 때만(기본 off, 온프레미스 유지).
- **학습/핫스왑 v3 준용**: `train_onsite.py` + `reload_model`(무중단).

## 빠른 시작
```bash
python classifier_v4.py --demo
python classifier_v4.py 문서.pdf --model n2sf-official       # 상세 JSON
python serve_v4.py            # 분류 서비스 + 학습/핫리로드 (:8091)
python admin_site.py          # 관리 콘솔 (:8091)
cd 산출물_v4 && ./bundle_offline.sh   # 폐쇄망 오프라인 번들 생성(온라인 PC 1회)
```
```python
from classifier_v4 import N2SFExplainClassifier
clf = N2SFExplainClassifier()          # 기본 §9(n2sf-official)
r = clf.classify("문서.pdf")            # grade·file·decision·findings·shap·explanation …
clf.reload_model("cust-v2", path="models/n2sf-custom-v2")   # 무중단 교체
```

## 한계(정직)
- n2sf-official은 합성 §9 PoC(400건)라 **O→C 과대분류 경향**(macroF1 0.788·C재현율 1.0·과대 0.20). 안전측이나 **실문서 재학습 권장**.
- 상세: `사용가이드_v4_상세.md` §8, 리포 `N2SF공식기준_적용방안.md`·`N2SF§9_재평가_결과.md`.
