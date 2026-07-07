# N²SF 등급분류 — 산출물 v3 (통합 함수 · 온사이트 학습 · 무중단 핫리로드 · 완전 오프라인)

> N²SF 공식 §9 정합. 문서를 **C(기밀)/S(민감)/O(공개)** 로 분류. 추론·NER·학습 **전부 로컬(외부 접속 0)**.

## 구성
| 파일 | 역할 |
|---|---|
| `classifier_v3.py` | **통합 분류 함수/클래스**(N2SFClassifier) + 티어 early-exit + 무중단 핫리로드 + 샘플 |
| `train_onsite.py` | **고객 라벨 데이터로 현장 재학습**(전이+안정화 혼합) — 웹 UI 연동 |
| `serve_v3.py` | HTTP 서비스 + **웹 UI**(분류/학습/핫리로드) |
| `bundle_offline.sh` | **오프라인 번들**(모델·wheel·NER 모델 로컬 아카이브) |
| `requirements_offline.txt` | 폐쇄망 의존성 |
| `models/` | 번들된 로컬 모델(오프라인 로드) — bundle_offline.sh 생성 |
| 가이드 MD | 아래 |

## 가이드 문서
- `사용가이드_AI친화.md` — 함수 계약·스키마·예시(사람·AI 에이전트 공용)
- `온사이트학습_핫리로드_가이드.md` — 고객 라벨→재학습→무중단 교체 흐름
- `오프라인_배포_가이드.md` — HF 미접속·NER 로컬·폐쇄망/Docker

## 빠른 시작
```bash
# (오프라인 번들: 온라인 PC 1회)  cd 산출물_v3 && ./bundle_offline.sh
python classifier_v3.py --demo
python classifier_v3.py 문서.pdf --model accurate --json
python serve_v3.py          # 웹 UI :8080 (분류·학습·핫리로드)
```

## 핵심 특성
- **완전 오프라인**: `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` 자동, 모델은 `models/`에서, NER은 설치 spaCy에서 로드. 외부 LLM/GPU/네트워크 0.
- **§9 정합**: 개인정보=S, 국가안보·기밀표지=C, 공개=O. 규칙 floor(기밀표지→C, 개인정보→S) 상시.
- **무중단 핫리로드**: 새 모델 사전 로드 후 원자적 스왑 → 다운타임 0, 실패 시 기존 유지.
- **온사이트 학습**: 고객 소량 라벨로 전이 재학습 + 망각 방지 혼합 → 새 버전 → 즉시 리로드.
- **모델 티어**: fast-ko(57MB) / compact-multi(488MB) / balanced(1.1GB) / **accurate(2.3GB·권장)**.

> 정확도의 정직한 기준·검증(OOD·사람 골드셋)은 리포 루트 문서 참조(OOD_검증_결과·근본_성능향상_방안·N2SF공식기준_적용방안).
