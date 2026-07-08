"""classifier_v4.py — N²SF 3-tier 통합 단일 파일 (연구소 표준화용, 상세 설명형/GUI·XAI).

이 파일 '하나'에 3-tier 전체가 들어있다:
  · T1 규칙: 정규식(PATTERN_RECOGNIZERS)·deny-list(DENY_LIST_RECOGNIZERS)·키워드(GRADE_KEYWORDS)·기밀표지(SECRET_FLOOR_KW)
  · T2 NER: Presidio+spaCy(오프라인)   · T3 뉴럴: 증류 모델(오프라인, 배치)   · 앙상블 + §9 등급원칙
  · (옵션) T4 Fable5 LLM 티어 — 기본 off, N2SF_FABLE=1 + ANTHROPIC_API_KEY 시에만(확장성).
연구소는 상단의 규칙 변수(PATTERN_RECOGNIZERS/DENY_LIST_RECOGNIZERS/GRADE_KEYWORDS/ENTITY_WEIGHTS/임계)를
이 파일에서 직접 편집·표준화한다. 추론 100% 로컬(외부 LLM/GPU 없음, Fable 티어만 opt-in).
반환은 GUI/XAI용 상세 JSON(근거·SHAP·§9원칙·파일메타·설명문). 학습/핫스왑은 v3 준용(train_onsite + reload_model).
"""
from __future__ import annotations

import os as _os
# ── 완전 오프라인 강제(HF/네트워크 미접속). transformers 지연 import 전에 설정 ──
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# ════════════════════════════════════════════════════════════════════════
# 1. 등급 정의 + 표시용 라벨
# ════════════════════════════════════════════════════════════════════════
GRADE_RANK = {"OPEN": 0, "SENSITIVE": 1, "CONFIDENTIAL": 2}
RANK_GRADE = {0: "OPEN", 1: "SENSITIVE", 2: "CONFIDENTIAL"}
SHORT = {"OPEN": "O", "SENSITIVE": "S", "CONFIDENTIAL": "C"}   # N²SF 코드
GRADE_LABEL_KO = {"OPEN": "공개", "SENSITIVE": "민감", "CONFIDENTIAL": "기밀"}

# 검출 타입 → 한국어 표시명 (화면 출력용). 미정의 타입은 그대로 노출.
ENTITY_LABELS: Dict[str, str] = {
    "KR_RRN": "주민등록번호", "KR_PASSPORT": "여권번호", "KR_BIZ_NO": "사업자등록번호",
    "KR_ACCOUNT": "계좌번호", "KR_ADDRESS": "주소", "KR_PHONE": "전화번호",
    "KR_NAME": "성명", "KR_MONEY": "금액",
    "JP_MY_NUMBER": "마이넘버", "JP_PASSPORT": "여권번호(JP)", "JP_PHONE": "전화번호(JP)",
    "JP_POSTAL_CODE": "우편번호(JP)", "JP_CORPORATE_NUMBER": "법인번호(JP)",
    "JP_BANK_ACCOUNT": "은행계좌(JP)", "JP_ADDRESS": "주소(JP)",
    "JP_DRIVERS_LICENSE": "운전면허(JP)",
    "US_SSN": "미국 SSN", "CREDIT_CARD": "신용카드번호", "IBAN_CODE": "IBAN 계좌",
    "AWS_ACCESS_KEY": "AWS 액세스 키", "GENERIC_API_KEY": "API 키/비밀토큰",
    "EMAIL_ADDRESS": "이메일", "PHONE_NUMBER": "전화번호", "CN_PHONE": "전화번호(CN)",
    "PERSON": "인명(NER)", "VIP_PERSON": "주요인물", "INTERNAL_PROJECT": "내부 프로젝트명",
    "LOCATION": "지명", "ORGANIZATION": "조직명", "IP_ADDRESS": "IP 주소",
    "DATE_TIME": "일시", "URL": "URL",
    "KEYWORD_SECRET": "기밀 등급 키워드", "KEYWORD_INTERNAL": "내부 등급 키워드",
}


# ════════════════════════════════════════════════════════════════════════
# 2. 가중치 · 임계값 (sample-data/calibrate.py 로 보정된 값 — 변경 시 재보정 권장)
# ════════════════════════════════════════════════════════════════════════
ENTITY_WEIGHTS: Dict[str, float] = {
    # ── N²SF §9-6호: 개인정보(주민번호·여권·카드·계좌·키)는 기본 S(민감). C 아님. ──
    #   C(기밀)는 국가안보·외교·수사 '영향'이며 뉴럴(의미)+기밀표지 floor가 담당.
    #   규칙은 개인정보를 S 대역으로만 올림(단독으론 C_THRESHOLD 미달).
    "KR_RRN": 2.0, "KR_PASSPORT": 2.0, "JP_MY_NUMBER": 2.0, "JP_PASSPORT": 2.0,
    "US_SSN": 2.0, "CREDIT_CARD": 2.0, "IBAN_CODE": 2.0, "AWS_ACCESS_KEY": 2.5,
    "KR_ACCOUNT": 2.0,
    # SENSITIVE drivers
    "KR_NAME": 0.6, "KR_MONEY": 0.5, "KR_BIZ_NO": 2.5, "JP_CORPORATE_NUMBER": 2.0,
    "JP_BANK_ACCOUNT": 2.5, "GENERIC_API_KEY": 2.5, "VIP_PERSON": 2.0,
    "INTERNAL_PROJECT": 2.0, "KR_ADDRESS": 1.5, "JP_ADDRESS": 1.0,
    "JP_POSTAL_CODE": 0.3, "KR_PHONE": 1.0, "JP_PHONE": 1.0, "CN_PHONE": 1.0,
    "PHONE_NUMBER": 1.0, "EMAIL_ADDRESS": 1.0, "PERSON": 0.5, "IP_ADDRESS": 0.4,
    "LOCATION": 0.0, "ORGANIZATION": 0.15, "DATE_TIME": 0.0, "URL": 0.0,
}
DEFAULT_ENTITY_WEIGHT = 0.05
ENTITY_COUNT_CAP = 2          # 타입당 2건 초과는 점수 기여 체감 (CONFIDENTIAL driver 제외)

# (키워드, 가중치, 라벨) — 등급 표시 라벨/문서 분류 키워드
GRADE_KEYWORDS: List[Tuple[str, float, str]] = [
    ("극비", 4.0, "극비"), ("top secret", 4.0, "Top Secret"),
    ("대외비", 3.0, "대외비"), ("기밀", 3.0, "기밀"), ("confidential", 3.0, "Confidential"),
    ("secret", 2.5, "Secret"), ("機密", 3.0, "機密"), ("社外秘", 3.0, "社外秘"),
    ("極秘", 4.0, "極秘"), ("绝密", 4.0, "绝密"), ("사외비", 3.0, "사외비"),
    ("机密", 3.0, "机密"), ("内部", 1.5, "内部"), ("内部用", 1.5, "内部用"),
    ("内部资料", 1.5, "内部资料"), ("内部資料", 1.5, "内部資料"), ("社内限", 1.5, "社内限"),
    ("internal use", 1.5, "Internal Use"), ("restricted", 1.5, "Restricted"),
    ("private", 1.0, "Private"), ("개인정보", 1.0, "개인정보 라벨"),
]
KW_COUNT_CAP = 3
# 명시적 기밀 라벨 — 검출 시 CONFIDENTIAL 하드 floor(앙상블이 하향 못 함). 언어무관.
SECRET_FLOOR_KW = {"극비", "대외비", "기밀", "사외비", "top secret", "confidential",
                   "機密", "社外秘", "極秘", "机密", "绝密"}

C_THRESHOLD = 5.5
S_THRESHOLD = 0.75

# 대량 개인정보(명부/내보내기) escalation — 합산 PII 건수가 임계 이상이면 CONFIDENTIAL.
BULK_PII_THRESHOLD = 10
BULK_PII_TYPES = {
    "KR_PHONE", "JP_PHONE", "CN_PHONE", "PHONE_NUMBER", "PHONE",
    "EMAIL_ADDRESS", "EMAIL", "PERSON", "VIP_PERSON",
    "KR_ADDRESS", "JP_ADDRESS", "KR_RRN", "JP_MY_NUMBER",
    "KR_PASSPORT", "JP_PASSPORT", "US_SSN", "CREDIT_CARD",
    "JP_BANK_ACCOUNT", "KR_BIZ_NO",
}

LOCALE_REGS = {"ko": ["KR-PIPA"], "ja": ["JP-APPI"], "en": ["US-CCPA"], "zh-CN": ["CN-PIPL"]}


# ════════════════════════════════════════════════════════════════════════
# 3. 패턴 (custom_patterns.yaml 를 인라인 임베드 — 단일 소스 유지)
#    새 룰 추가 시 여기에 dict 를 추가하면 즉시 Presidio 인식기로 등록된다.
# ════════════════════════════════════════════════════════════════════════
PATTERN_RECOGNIZERS: List[dict] = [
    {"name": "KR_RRN", "supported_entity": "KR_RRN", "patterns": [
        {"name": "rrn_with_hyphen", "regex": r"(?<!\d)\d{6}-[1-4]\d{6}(?!\d)", "score": 0.9},
        {"name": "rrn_no_hyphen", "regex": r"(?<!\d)\d{6}[1-4]\d{6}(?!\d)", "score": 0.6}],
     "context": ["주민등록번호", "주민번호", "RRN"]},
    {"name": "KR_PHONE", "supported_entity": "KR_PHONE", "patterns": [
        {"name": "mobile_with_hyphen", "regex": r"(?<!\d)01[016-9]-\d{3,4}-\d{4}(?!\d)", "score": 0.9},
        {"name": "mobile_no_hyphen", "regex": r"(?<!\d)01[016-9]\d{7,8}(?!\d)", "score": 0.5}],
     "context": ["휴대폰", "전화", "연락처", "phone", "tel"]},
    {"name": "KR_BIZ_NO", "supported_entity": "KR_BIZ_NO", "patterns": [
        {"name": "biz_no", "regex": r"\b\d{3}-\d{2}-\d{5}\b", "score": 0.85}],
     "context": ["사업자", "사업자등록번호"]},
    {"name": "KR_PASSPORT", "supported_entity": "KR_PASSPORT", "patterns": [
        {"name": "passport_kr", "regex": r"\b[MSRODmsrod]\d{8}\b", "score": 0.6}],
     "context": ["여권", "passport"]},
    {"name": "KR_ADDRESS", "supported_entity": "KR_ADDRESS", "patterns": [
        {"name": "kr_address",
         "regex": r"(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)"
                  r"(?:특별자치시|특별시|광역시|특별자치도|도)?\s*[가-힣]+(?:시|군|구)\s*[가-힣]+(?:동|로|길|읍|면)(?:\s*\d+(?:-\d+)?)?",
         "score": 0.75}],
     "context": ["주소", "거주지", "소재지", "본사"]},
    {"name": "KR_ACCOUNT", "supported_entity": "KR_ACCOUNT", "patterns": [
        {"name": "account_hyphen", "regex": r"(?<!\d)\d{2,6}-\d{2,6}-\d{2,7}(?:-\d{1,6})?(?!\d)", "score": 0.4},
        {"name": "account_packed", "regex": r"(?<!\d)\d{10,16}(?!\d)", "score": 0.2}],
     "context": ["계좌", "계좌번호", "예금주", "입금", "은행", "송금", "account"]},
    {"name": "KR_MONEY", "supported_entity": "KR_MONEY", "patterns": [
        {"name": "money_hangul", "regex": r"(?:금\s*)?[일이삼사오육칠팔구십백천만억]{1,}\s*원(?:정)?", "score": 0.5},
        {"name": "money_numeric", "regex": r"(?:금\s*)?\d{1,3}(?:,\d{3})+\s*원(?:정)?", "score": 0.55}],
     "context": ["금액", "보증금", "차임", "월세", "대금", "계약금", "잔금", "관리비"]},
    {"name": "KR_NAME", "supported_entity": "KR_NAME", "patterns": [
        {"name": "name_labeled", "regex": r"(?:임대인|임차인|대표자|성명|이름|예금주)\s*[:：(]?\s*[가-힣]{2,4}", "score": 0.5}],
     "context": ["임대인", "임차인", "대표자", "성명", "이름", "예금주"]},
    {"name": "AWS_ACCESS_KEY", "supported_entity": "AWS_ACCESS_KEY", "patterns": [
        {"name": "aws_access_key", "regex": r"\b(AKIA|ASIA)[0-9A-Z]{16}\b", "score": 0.95}],
     "context": ["aws", "access", "key"]},
    {"name": "GENERIC_API_KEY", "supported_entity": "GENERIC_API_KEY", "patterns": [
        {"name": "long_secret_isolated", "regex": r"\b[A-Za-z0-9_\-]{40,}\b", "score": 0.15},
        {"name": "prefixed_api_key",
         "regex": r"(?i)(?:api[_-]?key|secret|token|bearer)\s*[:=]\s*[\"']?([A-Za-z0-9_\-]{20,})", "score": 0.85}],
     "context": ["api_key", "secret", "token", "비밀", "키"]},
    {"name": "JP_MY_NUMBER", "supported_entity": "JP_MY_NUMBER", "patterns": [
        {"name": "my_number_grouped", "regex": r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b", "score": 0.55},
        {"name": "my_number_packed", "regex": r"\b\d{12}\b", "score": 0.15}],
     "context": ["マイナンバー", "個人番号", "my_number", "마이넘버"]},
    {"name": "JP_PHONE", "supported_entity": "JP_PHONE", "patterns": [
        {"name": "mobile_jp", "regex": r"\b0[789]0-\d{4}-\d{4}\b", "score": 0.85},
        {"name": "mobile_jp_packed", "regex": r"\b0[789]0\d{8}\b", "score": 0.5},
        {"name": "landline_jp", "regex": r"\b0\d{1,3}-\d{2,4}-\d{4}\b", "score": 0.7}],
     "context": ["電話", "携帯", "連絡先", "TEL", "tel"]},
    {"name": "JP_POSTAL_CODE", "supported_entity": "JP_POSTAL_CODE", "patterns": [
        {"name": "postal_with_mark", "regex": r"〒\s*\d{3}-?\d{4}", "score": 0.9},
        {"name": "postal_plain", "regex": r"\b\d{3}-\d{4}\b", "score": 0.4}],
     "context": ["〒", "郵便番号", "〶"]},
    {"name": "JP_PASSPORT", "supported_entity": "JP_PASSPORT", "patterns": [
        {"name": "passport_jp", "regex": r"\b[A-Z]{2}\d{7}\b", "score": 0.65}],
     "context": ["パスポート", "旅券", "passport"]},
    {"name": "JP_DRIVERS_LICENSE", "supported_entity": "JP_DRIVERS_LICENSE", "patterns": [
        {"name": "drivers_license_jp", "regex": r"\b\d{12}\b", "score": 0.15}],
     "context": ["運転免許", "免許証", "drivers_license"]},
    {"name": "JP_CORPORATE_NUMBER", "supported_entity": "JP_CORPORATE_NUMBER", "patterns": [
        {"name": "corporate_number_jp", "regex": r"\b\d{13}\b", "score": 0.4}],
     "context": ["法人番号", "会社番号", "corporate_number"]},
    {"name": "JP_BANK_ACCOUNT", "supported_entity": "JP_BANK_ACCOUNT", "patterns": [
        {"name": "bank_account_jp", "regex": r"(?:普通|当座|貯蓄|定期)\s*\d{6,8}", "score": 0.7}],
     "context": ["口座", "銀行", "支店"]},
    {"name": "JP_ADDRESS", "supported_entity": "JP_ADDRESS", "patterns": [
        {"name": "jp_address_full",
         "regex": r"(?:東京都|京都府|大阪府|北海道|(?:[一-龥]{2,3})県)[一-龥ぁ-んァ-ヶー\s]+?(?:市|区|郡)[一-龥ぁ-んァ-ヶー\d\-\s]+?(?:町|村|丁目|番地|号|\d)",
         "score": 0.75},
        {"name": "jp_address_loose",
         "regex": r"(?:東京都|京都府|大阪府|北海道|(?:[一-龥]{2,3})県)[一-龥ぁ-んァ-ヶー\s\d\-]+", "score": 0.45}],
     "context": ["住所", "所在地", "本社"]},
]

DENY_LIST_RECOGNIZERS: List[dict] = [
    {"name": "INTERNAL_PROJECTS", "supported_entity": "INTERNAL_PROJECT",
     "deny_list": ["ProjectAlpha", "ProjectOmega", "프로젝트 사일런스"], "score": 0.9},
    {"name": "VIP_NAMES", "supported_entity": "VIP_PERSON",
     "deny_list": ["홍길동", "김철수", "이영희"], "score": 0.85},
]


# ════════════════════════════════════════════════════════════════════════
# 4. Presidio + spaCy NER 엔진 (로케일별 lazy 빌드 · 메모이즈)
# ════════════════════════════════════════════════════════════════════════
SPACY_MODELS = {
    "en": "en_core_web_sm", "ko": "ko_core_news_sm",
    "ja": "ja_core_news_sm", "zh-CN": "zh_core_web_sm",
}
_ENGINES: dict = {}


def _presidio_lang(locale: str) -> str:
    return locale.split("-")[0] if locale.startswith("zh") else locale


def _build_recognizers():
    """인라인 패턴 정의 → Presidio PatternRecognizer 리스트."""
    from presidio_analyzer import PatternRecognizer, Pattern
    out = []
    for spec in PATTERN_RECOGNIZERS:
        patterns = [Pattern(name=p["name"], regex=p["regex"], score=float(p["score"]))
                    for p in spec.get("patterns", [])]
        out.append(PatternRecognizer(
            supported_entity=spec["supported_entity"],
            supported_language=spec.get("supported_language", "any"),
            patterns=patterns, context=spec.get("context", []), name=spec["name"]))
    for spec in DENY_LIST_RECOGNIZERS:
        out.append(PatternRecognizer(
            supported_entity=spec["supported_entity"],
            supported_language=spec.get("supported_language", "any"),
            deny_list=spec["deny_list"], name=spec["name"]))
    return out


def get_engine(locale: str):
    """로케일별 AnalyzerEngine 반환 (최초 호출 시 spaCy 로드 ~1-3s)."""
    if locale not in SPACY_MODELS:
        locale = "en"
    if locale in _ENGINES:
        return _ENGINES[locale]

    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.recognizer_registry import RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    lang = _presidio_lang(locale)
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": lang, "model_name": SPACY_MODELS[locale]}],
    }).create_engine()

    # [최적화 2] spaCy NER-only — parser/tagger/lemmatizer 등 불필요 컴포넌트 비활성(속도 1.6×)
    try:
        nlp_map = getattr(nlp_engine, "nlp", {})
        for _nlp in (nlp_map.values() if isinstance(nlp_map, dict) else []):
            keep = {"tok2vec", "transformer", "ner"}
            drop = [p for p in _nlp.pipe_names if p not in keep]
            if drop:
                _nlp.disable_pipes(*drop)
    except Exception:
        pass

    registry = RecognizerRegistry(supported_languages=[lang])
    registry.load_predefined_recognizers(languages=[lang], nlp_engine=nlp_engine)
    for r in _build_recognizers():
        if r.supported_language == "any":
            r.supported_language = lang
        try:
            registry.add_recognizer(r)
        except Exception:
            pass

    # [최적화 1] 인식기 slim — 로케일별 사용 엔티티를 지원하는 인식기만 남김(불필요 인식기 제거)
    try:
        used = _used_entities(locale)
        keep_recs = []
        for r in registry.recognizers:
            se = set(getattr(r, "supported_entities", []) or [])
            if not se or (se & used):        # spaCy 등 엔티티 미표기는 유지
                keep_recs.append(r)
        registry.recognizers = keep_recs
    except Exception:
        pass

    engine = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine,
                            supported_languages=[lang])
    _ENGINES[locale] = engine
    return engine


_COMMON_ENTS = {"PERSON", "LOCATION", "ORGANIZATION", "NRP", "EMAIL_ADDRESS",
                "PHONE_NUMBER", "DATE_TIME", "URL", "IP_ADDRESS", "CREDIT_CARD"}
_LOCALE_ENTS = {
    "ko": {"KR_RRN", "KR_PHONE", "KR_ACCOUNT", "KR_PASSPORT", "KR_BIZ_NO", "KR_ADDRESS",
           "KR_NAME", "KR_MONEY", "VIP_PERSON", "INTERNAL_PROJECT", "GENERIC_API_KEY", "AWS_ACCESS_KEY"},
    "ja": {"JP_MY_NUMBER", "JP_PASSPORT", "JP_PHONE", "JP_ADDRESS", "JP_BANK_ACCOUNT",
           "JP_CORPORATE_NUMBER", "JP_POSTAL_CODE", "GENERIC_API_KEY", "AWS_ACCESS_KEY"},
    "en": {"US_SSN", "IBAN_CODE", "GENERIC_API_KEY", "AWS_ACCESS_KEY"},
    "zh-CN": {"CN_PHONE", "GENERIC_API_KEY", "AWS_ACCESS_KEY"},
}


def _used_entities(locale: str = "ko") -> set:
    """로케일별로 실제 기여하는 엔티티만(불필요 인식기 스킵). 좁을수록 빠름."""
    return _COMMON_ENTS | _LOCALE_ENTS.get(locale, _LOCALE_ENTS["ko"])


def analyze(text: str, locale: str) -> List[dict]:
    """Presidio 분석 → finding dict 리스트 (entity_type/start/end/score/recognizer)."""
    engine = get_engine(locale)
    lang = _presidio_lang(locale)
    # [최적화 3] 사용 엔티티만 분석(불필요 인식기 실행 스킵)
    _ents = sorted(_used_entities(locale) & set(engine.get_supported_entities(language=lang) or []))
    raw = engine.analyze(text=text, language=lang, entities=_ents or None)
    return [{
        "entity_type": r.entity_type, "start": r.start, "end": r.end,
        "score": r.score,
        "recognizer": (r.recognition_metadata or {}).get("recognizer_name", "unknown"),
    } for r in raw]


# ════════════════════════════════════════════════════════════════════════
# 5. 신경망 / LLM tier (선택 가능한 모델 · llm_mode 일 때만)
#    백엔드별 lazy 로드. transformers/sentence-transformers 미설치 시 무력화.
# ════════════════════════════════════════════════════════════════════════
# 선택 가능한 신경망(LLM) 백엔드 레지스트리.
#   kind=exemplar  문장 임베딩(sentence-transformers) + 등급 예시문 코사인. 가볍고 무난.
#   kind=zeroshot  NLI 제로샷 분류(transformers pipeline) — 실제 등급 확률 산출.
#   kind=embed     원시 인코더(transformers AutoModel) 평균풀링 → exemplar 와 동일 코사인.
# 새 모델 추가: 아래 dict 에 한 줄 추가하면 즉시 --model 로 선택 가능(매뉴얼 §11 참조).
# langs 는 권장 언어, note 는 선택 가이드(둘 다 동작에는 영향 없는 메타데이터).
NEURAL_BACKENDS: Dict[str, Dict[str, str]] = {
    # ── exemplar: 다국어 문장 임베딩 ──────────────────────────────────────
    "minilm":    {"label": "MiniLM-L12 (다국어, 경량)", "kind": "exemplar",
                  "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                  "langs": "ko,ja,en,zh", "note": "기본값. 빠르고 가벼움(~120MB). 데모/실시간 권장"},
    "mpnet":     {"label": "mpnet-base (다국어, 고품질)", "kind": "exemplar",
                  "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                  "langs": "ko,ja,en,zh", "note": "minilm 보다 정확, 약 2~3배 무거움(~1GB)"},
    "labse":     {"label": "LaBSE (109개 언어)", "kind": "exemplar",
                  "model": "sentence-transformers/LaBSE",
                  "langs": "multi", "note": "언어 커버리지 최강. 다국어 혼재 문서에 강함(~1.8GB)"},
    "e5":        {"label": "multilingual-e5-base", "kind": "exemplar",
                  "model": "intfloat/multilingual-e5-base",
                  "langs": "multi", "note": "검색·의미유사도 SOTA급 임베딩(~1.1GB)"},
    "bge-m3":    {"label": "BGE-M3 (다국어)", "kind": "exemplar",
                  "model": "BAAI/bge-m3",
                  "langs": "multi", "note": "장문·다국어 강함. 무거움(~2.2GB)"},
    "ko-sroberta": {"label": "Ko-SRoBERTa (한국어 특화)", "kind": "exemplar",
                  "model": "jhgan/ko-sroberta-multitask",
                  "langs": "ko", "note": "한국어 문장 임베딩 특화. 국내 문서 정확도↑"},
    # ── zeroshot: NLI 제로샷 (등급 확률 직접 산출) ────────────────────────
    "mdeberta":  {"label": "mDeBERTa-v3 NLI (제로샷)", "kind": "zeroshot",
                  "model": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                  "langs": "ko,ja,en,zh", "note": "다국어 제로샷. 등급 확률 필요 시 권장"},
    "xlmr-xnli": {"label": "XLM-R-large XNLI (제로샷)", "kind": "zeroshot",
                  "model": "joeddav/xlm-roberta-large-xnli",
                  "langs": "multi", "note": "고정확 다국어 제로샷. 무거움(~2.2GB)"},
    "deberta-large-mnli": {"label": "DeBERTa-v3-large MNLI (제로샷, EN)", "kind": "zeroshot",
                  "model": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                  "langs": "en", "note": "영문 제로샷 최상급 정확도"},
    "bart-mnli": {"label": "BART-large MNLI (제로샷, EN)", "kind": "zeroshot",
                  "model": "facebook/bart-large-mnli",
                  "langs": "en", "note": "영문 제로샷 표준 베이스라인"},
    # ── embed: 원시 인코더(평균풀링) — 파인튜닝 헤드 없이도 신호 제공 ──────
    "mbert":     {"label": "mBERT (다국어 인코더)", "kind": "embed",
                  "model": "bert-base-multilingual-cased",
                  "langs": "multi", "note": "표준 다국어 BERT"},
    "xlm-roberta": {"label": "XLM-RoBERTa-base (다국어 인코더)", "kind": "embed",
                  "model": "xlm-roberta-base",
                  "langs": "multi", "note": "mBERT 후속, 다국어 표현력↑"},
    "koelectra": {"label": "KoELECTRA-base-v3 (한국어 인코더)", "kind": "embed",
                  "model": "monologg/koelectra-base-v3-discriminator",
                  "langs": "ko", "note": "한국어 ELECTRA"},
    "klue-roberta": {"label": "KLUE-RoBERTa-base (한국어 인코더)", "kind": "embed",
                  "model": "klue/roberta-base",
                  "langs": "ko", "note": "KLUE 벤치마크 한국어 RoBERTa"},
    "kcbert":    {"label": "KcBERT-base (한국어 구어체)", "kind": "embed",
                  "model": "beomi/kcbert-base",
                  "langs": "ko", "note": "댓글·구어체 한국어 특화"},
    # ── finetuned: 우리 데이터로 학습한 직접 3-class(O/S/C) 분류기 (harness/train.py 산출) ──
    "mdeberta-n2sf": {"label": "mDeBERTa-n2sf (파인튜닝 3-class)", "kind": "finetuned",
                  "model": "models/mdeberta-n2sf",
                  "langs": "ko", "note": "N²SF 라벨로 파인튜닝. 위장(L3) 대응 강화"},
    # ── N²SF 라인업 (soft-label 증류, harness/train_soft·train_kd 산출) — 용도별 3-티어 ──
    "n2sf-small": {"label": "N²SF-small (KoELECTRA-small 증류)", "kind": "finetuned",
                  "model": "models/n2sf-small",
                  "langs": "ko", "note": "Fast 티어. 14M/57MB/~13ms. 엣지·실시간·저사양"},
    "n2sf-base": {"label": "N²SF-base (mDeBERTa 증류, 기본)", "kind": "finetuned",
                  "model": "models/n2sf-base",
                  "langs": "ko,ja,en,zh", "note": "Balanced 티어(기본). 279M/1.1GB. 표준 업무 PC"},
    "n2sf-klue-large": {"label": "N²SF-klue-large (KLUE-RoBERTa-large 증류)", "kind": "finetuned",
                  "model": "models/n2sf-klue-large",
                  "langs": "ko", "note": "한국어 특화 라지. 337M/1.3GB. 정확도 0.87"},
    "n2sf-xlmr-large": {"label": "N²SF-xlmr-large (XLM-R-large 증류)", "kind": "finetuned",
                  "model": "models/n2sf-xlmr-large",
                  "langs": "multi", "note": "Accurate 티어. 560M/2.3GB. 정확도 0.92, 다국어 강건"},
}
_GRADES = ["OPEN", "SENSITIVE", "CONFIDENTIAL"]

ZS_LABELS = {
    "ko": {"OPEN": "공개 가능한 일반 문서",
           "SENSITIVE": "이름·전화·이메일 등 개인정보가 포함된 민감 문서",
           "CONFIDENTIAL": "주민등록번호·금융 정보·기밀 라벨이 포함된 기밀 문서"},
    "ja": {"OPEN": "公開可能な一般文書",
           "SENSITIVE": "氏名・電話・メール等の個人情報を含む機微文書",
           "CONFIDENTIAL": "マイナンバー・金融情報・機密ラベルを含む機密文書"},
    "en": {"OPEN": "a public document",
           "SENSITIVE": "a document containing personal information like emails or phones",
           "CONFIDENTIAL": "a confidential document with SSNs or financial credentials"},
    "zh-CN": {"OPEN": "可公开的一般文档",
              "SENSITIVE": "包含姓名、电话、邮箱等个人信息的敏感文档",
              "CONFIDENTIAL": "包含身份证号、金融信息、机密标签的机密文档"},
}
EXEMPLARS: Dict[str, Dict[str, List[str]]] = {
    "ko": {"OPEN": ["오늘 회의 일정은 다음과 같습니다.", "분기 매출 보고서가 안정적인 흐름을 보입니다.", "오픈소스 라이선스 가이드라인입니다."],
           "SENSITIVE": ["참석자 명단과 이메일 주소를 첨부합니다.", "사내 임직원 연락처 안내입니다.", "사업자등록번호와 담당자 정보가 포함됩니다."],
           "CONFIDENTIAL": ["대외비 — 주민등록번호와 금융 정보가 포함된 명부입니다.", "극비 인사 자료 및 보안 키 정보.", "기밀 거래내역 및 신용카드 번호."]},
    "ja": {"OPEN": ["本日の会議スケジュールを共有します。", "四半期業績レポート(一般公開向け)。", "オープンソースライセンスのご案内。"],
           "SENSITIVE": ["参加者リストとメールアドレスを添付します。", "社内連絡先一覧。", "法人番号と担当者情報。"],
           "CONFIDENTIAL": ["機密 — マイナンバーと金融情報が含まれます。", "社外秘の人事資料および秘密鍵情報。", "機密 — 取引履歴とクレジットカード情報。"]},
    "en": {"OPEN": ["Here is today's meeting schedule.", "Quarterly business report for public release.", "Open-source license guidelines."],
           "SENSITIVE": ["Attendee list with email addresses attached.", "Internal contact directory.", "Vendor info with business registration numbers."],
           "CONFIDENTIAL": ["Confidential — contains SSN and financial account info.", "Top secret HR data and API credentials.", "Restricted — credit card transactions and personal IDs."]},
    "zh-CN": {"OPEN": ["今日会议日程通知。", "季度业绩报告(对外公开)。", "开源许可证使用指南。"],
              "SENSITIVE": ["参与者名单及邮箱地址附件。", "公司内部联系信息。", "供应商和业务联系人资料。"],
              "CONFIDENTIAL": ["机密 — 包含身份证号和金融账户信息。", "绝密人事档案及 API 密钥。", "内部限制 — 信用卡交易和个人身份证。"]},
}

_NEURAL_MODELS: dict = {}
_EXEMPLAR_EMB: dict = {}


def _neural_load(backend: str):
    if backend in _NEURAL_MODELS:
        return _NEURAL_MODELS[backend]
    spec = NEURAL_BACKENDS[backend]
    name, kind = spec["model"], spec["kind"]
    if kind == "zeroshot":
        from transformers import pipeline
        _NEURAL_MODELS[backend] = pipeline("zero-shot-classification", model=name, device=-1)
    elif kind == "embed":
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModel.from_pretrained(name); mdl.eval()
        _NEURAL_MODELS[backend] = (tok, mdl)
    elif kind == "finetuned":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForSequenceClassification.from_pretrained(name); mdl.eval()
        _NEURAL_MODELS[backend] = (tok, mdl)
    else:
        from sentence_transformers import SentenceTransformer
        _NEURAL_MODELS[backend] = SentenceTransformer(name)
    return _NEURAL_MODELS[backend]


def _neural_encode(backend: str, texts):
    import numpy as np
    kind = NEURAL_BACKENDS[backend]["kind"]
    single = isinstance(texts, str)
    batch = [texts] if single else list(texts)
    if kind == "embed":
        import torch
        tok, mdl = _neural_load(backend)
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            out = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vecs = ((out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy()
    else:
        vecs = np.asarray(_neural_load(backend).encode(batch, normalize_embeddings=True))
    vecs = np.atleast_2d(vecs)
    vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
    return vecs[0] if single else vecs


def _neural_exemplars(backend: str, locale: str):
    import numpy as np
    key = f"{backend}:{locale}"
    if key in _EXEMPLAR_EMB:
        return _EXEMPLAR_EMB[key]
    src = EXEMPLARS.get(locale) or EXEMPLARS["en"]
    out = {g: np.mean(_neural_encode(backend, s), axis=0).tolist() for g, s in src.items()}
    _EXEMPLAR_EMB[key] = out
    return out


def neural_infer(text: str, locale: str, backend: str) -> Optional[dict]:
    """{grade, confidence, scores, version, backend} 또는 None(미설치/오류)."""
    if backend not in NEURAL_BACKENDS:
        backend = "minilm"
    name = NEURAL_BACKENDS[backend]["model"]
    if not text or len(text) < 8:
        return {"grade": "OPEN", "confidence": 0.0,
                "scores": {g: 0.0 for g in _GRADES}, "version": name,
                "backend": backend, "note": "text too short"}
    try:
        if NEURAL_BACKENDS[backend]["kind"] == "zeroshot":
            import numpy as np  # noqa: F401
            pipe = _neural_load(backend)
            labels = ZS_LABELS.get(locale) or ZS_LABELS["en"]
            inv = {v: k for k, v in labels.items()}
            res = pipe(text[:4096], candidate_labels=list(labels.values()), multi_label=False)
            scores = {inv[l]: float(s) for l, s in zip(res["labels"], res["scores"])}
            best = inv[res["labels"][0]]
            return {"grade": best, "confidence": round(scores[best], 3),
                    "scores": {k: round(v, 3) for k, v in scores.items()},
                    "version": name, "backend": backend}
        if NEURAL_BACKENDS[backend]["kind"] == "finetuned":
            import numpy as np  # noqa: F401
            import torch
            tok, mdl = _neural_load(backend)
            enc = tok([text[:4096]], padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                logits = mdl(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            id2label = mdl.config.id2label
            raw = {}
            for i, p in enumerate(probs):
                g = id2label.get(i, id2label.get(str(i), _GRADES[i] if i < len(_GRADES) else str(i)))
                raw[g] = float(p)
            scores = {g: float(raw.get(g, 0.0)) for g in _GRADES}
            best = max(_GRADES, key=lambda g: scores[g])
            return {"grade": best, "confidence": round(scores[best], 3),
                    "scores": {k: round(v, 3) for k, v in scores.items()},
                    "version": name, "backend": backend}
        import numpy as np
        exemplars = _neural_exemplars(backend, locale)
        emb = np.asarray(_neural_encode(backend, text[:4096]))
        scores = {g: float(np.dot(emb, np.asarray(v))) for g, v in exemplars.items()}
        vals = np.array(list(scores.values()))
        e = np.exp(vals - vals.max()); probs = e / e.sum()
        bi = int(np.argmax(vals)); best = list(scores.keys())[bi]
        return {"grade": best, "confidence": round(float(probs[bi]), 3),
                "scores": {k: round(float(v), 3) for k, v in scores.items()},
                "version": name, "backend": backend}
    except Exception as exc:                                       # noqa: BLE001
        return {"grade": "OPEN", "confidence": 0.0,
                "scores": {g: 0.0 for g in _GRADES}, "version": name,
                "backend": backend, "error": str(exc)}


# ════════════════════════════════════════════════════════════════════════
# 6. 점수화 · 앙상블 · SHAP 기여도
# ════════════════════════════════════════════════════════════════════════
SUPERSEDED_BY = {
    "PHONE_NUMBER": ["KR_PHONE", "JP_PHONE", "CN_PHONE"],
    "LOCATION": ["JP_ADDRESS", "KR_ADDRESS"],
    "DATE_TIME": ["JP_POSTAL_CODE"],
}
_LOCALE_PREFIX = {"ko": "KR_", "ja": "JP_", "en": "US_", "zh-CN": "CN_"}
_LOCALE_PREFIXES = ("KR_", "JP_", "CN_", "US_")
ENSEMBLE_METHODS = ["escalate", "vote", "weighted", "max-rank", "soft"]
DEFAULT_TIER_WEIGHTS = {"rules": 1.0, "ner": 1.0, "neural": 1.0}


def _dedupe_overlaps(raws: List[dict]) -> List[dict]:
    out = []
    for r in raws:
        specifics = SUPERSEDED_BY.get(r["entity_type"])
        if specifics and any(
            (o["entity_type"] in specifics and r["start"] < o["end"] and r["end"] > o["start"])
            for o in raws):
            continue
        out.append(r)
    return out


def _resolve_locale_priority(raws: List[dict], locale: str) -> List[dict]:
    home = _LOCALE_PREFIX.get(locale)
    if not home:
        return raws
    out = []
    for r in raws:
        et = r["entity_type"]
        is_foreign = et.startswith(_LOCALE_PREFIXES) and not et.startswith(home)
        if is_foreign and any(
            o["entity_type"].startswith(home) and r["start"] < o["end"] and r["end"] > o["start"]
            for o in raws):
            continue
        out.append(r)
    return out


def _aggregate_findings(presidio_findings: List[dict], text: str, locale: str,
                        verbose: bool) -> List[dict]:
    presidio_findings = _resolve_locale_priority(presidio_findings, locale)
    presidio_findings = _dedupe_overlaps(presidio_findings)
    by_type: Dict[str, dict] = {}
    for r in presidio_findings:
        et = r["entity_type"]
        slot = by_type.setdefault(et, {
            "type": et, "count": 0, "spans": [], "confidence": 0.0,
            "source": "ner" if et in ("PERSON", "LOCATION", "ORGANIZATION", "DATE_TIME") else "presidio",
            "rule": r.get("recognizer", "presidio"), "items": []})
        slot["count"] += 1
        slot["spans"].append([r["start"], r["end"]])
        slot["confidence"] = max(slot["confidence"], float(r["score"]))
        if verbose:
            slot["items"].append({"text": text[r["start"]:r["end"]],
                                  "start": r["start"], "end": r["end"],
                                  "score": round(float(r["score"]), 3)})
    return list(by_type.values())


def _scan_keywords(text: str, extra: List[tuple], verbose: bool) -> List[dict]:
    out = []
    lower = text.lower()
    compact = re.sub(r"\s+", "", lower)          # 띄어쓴 키워드(대 외 비) 방어용
    for kw, w, label in list(GRADE_KEYWORDS) + list(extra or []):
        needle = kw.lower()
        spans, idx = [], 0
        while True:
            j = lower.find(needle, idx)
            if j < 0:
                break
            spans.append([j, j + len(needle)])
            idx = j + len(needle)
            if len(spans) > KW_COUNT_CAP * 2:
                break
        # 원문에 없고 고위험 표지면 공백제거본에서 재탐(난독 대응) — 과탐은 안전측
        obfus = False
        if not spans and w >= 2.5 and re.sub(r"\s+", "", needle) in compact:
            spans = [[0, 0]]; obfus = True
        if spans:
            etype = "KEYWORD_SECRET" if w >= 2.5 else "KEYWORD_INTERNAL"
            f = {"type": etype, "count": len(spans), "spans": spans[:KW_COUNT_CAP],
                 "confidence": min(1.0, 0.5 + w * 0.1), "source": "keyword",
                 "rule": f"KW_{label}", "weight": w, "label": label, "items": [],
                 "secretFloor": kw in SECRET_FLOOR_KW, "obfuscated": obfus}
            if verbose:
                f["items"] = [{"text": text[s[0]:s[1]], "start": s[0], "end": s[1]}
                              for s in spans[:KW_COUNT_CAP]]
            out.append(f)
    return out


def _entity_weight(etype: str, overrides: dict) -> float:
    if etype in overrides:
        return float(overrides[etype])
    return ENTITY_WEIGHTS.get(etype, DEFAULT_ENTITY_WEIGHT)


def _finding_contribution(f: dict, entity_overrides: dict) -> float:
    """단일 finding 의 점수 기여(= SHAP 근사). _score 의 합산 항과 동일 규칙."""
    if f["source"] == "keyword":
        return f.get("weight", 0.0) * min(f["count"], KW_COUNT_CAP)
    w = _entity_weight(f["type"], entity_overrides)
    cap = max(f["count"], 1) if w >= 6.0 else min(f["count"], ENTITY_COUNT_CAP)
    return w * cap


def _score(findings: List[dict], entity_overrides: dict,
           c_thr: float, s_thr: float) -> Tuple[float, str, float]:
    s = sum(_finding_contribution(f, entity_overrides) for f in findings)
    # N²SF §9: 규칙(형태 기반)은 C(기밀)를 결정하지 않는다. C는 유출 '영향'(외교·수사·국방)이므로
    # 뉴럴(의미)과 명시적 기밀표지 하드 floor가 담당. 규칙 점수의 등급 상한 = S(민감).
    if s >= s_thr:
        grade, margin = "SENSITIVE", min(s - s_thr, max(c_thr - s, 0.5))
    else:
        grade, margin = "OPEN", s_thr - s
    confidence = max(0.0, min(1.0, 0.55 + 0.4 * math.tanh(margin / 2.0)))
    return s, grade, round(confidence, 3)


def _neural_is_real(nr: dict) -> bool:
    return bool(nr) and not nr.get("note") and not nr.get("error") \
        and float(nr.get("confidence", 0)) > 0


def _dist_from_grade(grade: str, conf: float) -> Dict[str, float]:
    c = max(0.0, min(1.0, conf or 0.0))
    others = [g for g in _GRADES if g != grade]
    p = {g: (1.0 - c) / len(others) for g in others}
    p[grade] = c
    return p


def _dist_from_scores(scores: Dict[str, float]) -> Dict[str, float]:
    vals = {g: max(float(scores.get(g, 0.0)), 0.0) for g in _GRADES}
    total = sum(vals.values())
    if total <= 0:
        return {g: 1.0 / len(_GRADES) for g in _GRADES}
    return {g: v / total for g, v in vals.items()}


def _combine(base_grade, base_conf, tier_grades, tier_confs, neural_result,
             method, weights) -> dict:
    has_neural = _neural_is_real(neural_result)
    tiers = {"rules": tier_grades["rules"], "ner": tier_grades["ner"]}
    if has_neural:
        tiers["neural"] = neural_result["grade"]
    soft_probs = None

    if method == "vote":
        buckets: Dict[str, int] = {}
        for g in tiers.values():
            buckets[g] = buckets.get(g, 0) + 1
        top = max(buckets.values())
        grade = max([g for g, c in buckets.items() if c == top], key=lambda g: GRADE_RANK[g])
    elif method == "weighted":
        scored: Dict[str, float] = {}
        for tier, g in tiers.items():
            w = float(weights.get(tier, 1.0))
            conf = tier_confs.get(tier) if tier != "neural" else float(neural_result.get("confidence", 0))
            scored[g] = scored.get(g, 0.0) + w * max(conf or 0.0, 0.1)
        grade = max(scored, key=lambda g: (scored[g], GRADE_RANK[g]))
    elif method == "max-rank":
        grade = max(tiers.values(), key=lambda g: GRADE_RANK[g])
    elif method == "soft":
        dists = {"rules": _dist_from_grade(tier_grades["rules"], tier_confs.get("rules", 0.0)),
                 "ner": _dist_from_grade(tier_grades["ner"], tier_confs.get("ner", 0.0))}
        if has_neural:
            dists["neural"] = (_dist_from_scores(neural_result["scores"]) if neural_result.get("scores")
                               else _dist_from_grade(neural_result["grade"], neural_result.get("confidence", 0.0)))
        agg = {g: 0.0 for g in _GRADES}; wsum = 0.0
        for tier, dist in dists.items():
            w = float(weights.get(tier, 1.0)); wsum += w
            for g in _GRADES:
                agg[g] += w * dist[g]
        if wsum:
            agg = {g: v / wsum for g, v in agg.items()}
        grade = max(_GRADES, key=lambda g: (agg[g], GRADE_RANK[g]))
        soft_probs = {g: round(agg[g], 3) for g in _GRADES}
    else:  # escalate
        grade = base_grade
        if has_neural and neural_result.get("confidence", 0) >= 0.55:
            ng = neural_result["grade"]
            if GRADE_RANK.get(ng, 0) > GRADE_RANK[grade]:
                grade = ng

    confidence = base_conf
    if method == "soft":
        confidence = soft_probs[grade]
    elif has_neural and neural_result["grade"] == grade:
        confidence = max(confidence, float(neural_result["confidence"]))
    out = {"grade": grade, "confidence": round(min(1.0, max(0.0, confidence)), 3),
           "method": method, "votes": tiers,
           "weights": {k: float(weights.get(k, 1.0)) for k in tiers},
           "neuralCounted": has_neural}
    if method == "soft":
        out["probs"] = soft_probs
    return out


def _shap_block(findings: List[dict], score: float, entity_overrides: dict) -> dict:
    """등급 분류 기여도 — feature(검출 타입) 별 점수 기여 + 비율. 화면 'why' 패널용."""
    contribs = []
    for f in findings:
        c = _finding_contribution(f, entity_overrides)
        if c == 0:
            continue
        contribs.append({
            "feature": f["type"],
            "label": f.get("label") or ENTITY_LABELS.get(f["type"], f["type"]),
            "count": f["count"],
            "contribution": round(c, 3),
            "percent": round(c / score, 3) if score > 0 else 0.0,
            "direction": "up",   # 모든 가중치가 0 이상 → 항상 등급을 올리는 방향
        })
    contribs.sort(key=lambda x: -x["contribution"])
    return {"baseline": 0.0, "total": round(score, 3), "contributions": contribs}


# ════════════════════════════════════════════════════════════════════════
# 7. 파일 → 텍스트 추출 (오피스 계열 + HWPX 우선)
# ════════════════════════════════════════════════════════════════════════
# docx(<w:t>)·pptx(<a:t>)·hwpx(<hp:t>) 는 zip-of-xml 이므로 동일 제너릭 `<*:t>`
# 추출기로 처리한다. xlsx 는 셀 구조(공유문자열 + 숫자 셀)를 살리려 전용 파서 사용.
_ZIP_XML_TARGETS = {
    ".docx": [r"word/document\.xml", r"word/header\d*\.xml", r"word/footer\d*\.xml",
              r"word/footnotes\.xml", r"word/endnotes\.xml", r"word/comments\.xml"],
    ".pptx": [r"ppt/slides/slide\d+\.xml", r"ppt/notesSlides/notesSlide\d+\.xml",
              r"ppt/diagrams/data\d*\.xml", r"ppt/charts/chart\d+\.xml"],
    ".hwpx": [r"Contents/section\d+\.xml", r"Contents/header\.xml"],
    # .docm/.pptx 매크로 변형도 동일 구조 → 같은 타깃 사용
    ".docm": [r"word/document\.xml", r"word/header\d*\.xml", r"word/footer\d*\.xml",
              r"word/footnotes\.xml", r"word/endnotes\.xml", r"word/comments\.xml"],
    ".pptm": [r"ppt/slides/slide\d+\.xml", r"ppt/notesSlides/notesSlide\d+\.xml",
              r"ppt/diagrams/data\d*\.xml", r"ppt/charts/chart\d+\.xml"],
}
_T_TAG_RE = re.compile(r"<(?:[\w]+:)?t(?:\s[^>]*)?>([\s\S]*?)</(?:[\w]+:)?t>")
_TEXT_EXTS = {".txt", ".csv", ".md", ".log", ".json", ".tsv", ".text"}


def _decode_entities(s: str) -> str:
    return (s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
             .replace("&#39;", "'").replace("&apos;", "'").replace("&amp;", "&"))


def _extract_zip_xml(path: Path, patterns: List[str]) -> str:
    import zipfile
    res = [re.compile(p) for p in patterns]
    parts: List[str] = []
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if any(r.match(n) for r in res))
        for n in names:
            xml = z.read(n).decode("utf-8", "ignore")
            buf = [_decode_entities(m.group(1)).strip() for m in _T_TAG_RE.finditer(xml)]
            buf = [t for t in buf if t]
            if buf:
                parts.append(" ".join(buf))
    return "\n\n".join(parts)


def _extract_xlsx(path: Path) -> str:
    """xlsx/xlsm 전용 — 공유문자열 + 워크시트 셀(문자열·숫자·inline)을 모두 추출.

    제너릭 `<t>` 추출은 공유문자열만 잡아 숫자 셀(계좌번호·전화번호 등이 숫자로
    저장된 경우)을 놓친다. 여기서는 셀 타입을 해석해 숫자 값까지 복원한다.
    시트별로 행을 줄바꿈, 셀을 공백으로 이어 PII 인식기가 경계를 잡게 한다.
    """
    import zipfile
    cell_re = re.compile(r"<c\b([^>]*)>([\s\S]*?)</c>")
    v_re = re.compile(r"<v>([\s\S]*?)</v>")
    t_attr_re = re.compile(r'\bt="([^"]+)"')
    row_re = re.compile(r"<row\b[^>]*>([\s\S]*?)</row>")
    parts: List[str] = []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        shared: List[str] = []
        if "xl/sharedStrings.xml" in names:
            sxml = z.read("xl/sharedStrings.xml").decode("utf-8", "ignore")
            for si in re.findall(r"<si>([\s\S]*?)</si>", sxml):
                shared.append("".join(_decode_entities(m.group(1))
                                      for m in _T_TAG_RE.finditer(si)))
        sheets = sorted(n for n in names if re.match(r"xl/worksheets/sheet\d+\.xml", n))
        for n in sheets:
            xml = z.read(n).decode("utf-8", "ignore")
            rows: List[str] = []
            for rm in row_re.finditer(xml):
                cells: List[str] = []
                for cm in cell_re.finditer(rm.group(1)):
                    attrs, inner = cm.group(1), cm.group(2)
                    tm = t_attr_re.search(attrs)
                    ctype = tm.group(1) if tm else "n"
                    if ctype == "s":                       # 공유문자열 인덱스
                        vm = v_re.search(inner)
                        if vm and vm.group(1).isdigit():
                            i = int(vm.group(1))
                            if 0 <= i < len(shared):
                                cells.append(shared[i])
                    elif ctype in ("inlineStr", "str"):     # 인라인/수식문자열
                        cells.append("".join(_decode_entities(m.group(1))
                                             for m in _T_TAG_RE.finditer(inner)) or
                                     (_decode_entities(v_re.search(inner).group(1))
                                      if v_re.search(inner) else ""))
                    else:                                   # 숫자/일반 값
                        vm = v_re.search(inner)
                        if vm:
                            cells.append(_decode_entities(vm.group(1)))
                cells = [c for c in cells if c]
                if cells:
                    rows.append(" ".join(cells))
            if rows:
                parts.append("\n".join(rows))
    return "\n\n".join(parts)


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        return "\n\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages[:30])
    except Exception:
        pass
    try:
        from pdfminer.high_level import extract_text as _pm
        return _pm(str(path)) or ""
    except Exception as exc:
        raise RuntimeError(f"PDF 추출 실패(pypdf/pdfminer 미설치?): {exc}")


def _extract_hwp(path: Path) -> str:
    """구형 .hwp(OLE 바이너리) best-effort: hwp5txt → 실패 시 olefile PrvText."""
    import subprocess
    try:
        out = subprocess.run(["hwp5txt", str(path)], capture_output=True, timeout=30)
        if out.returncode == 0:
            txt = out.stdout.decode("utf-8", "ignore").strip()
            if txt:
                return txt
    except Exception:
        pass
    try:
        import olefile
        if olefile.isOleFile(str(path)):
            ole = olefile.OleFileIO(str(path))
            try:
                if ole.exists("PrvText"):
                    return ole.openstream("PrvText").read().decode("utf-16-le", "ignore").strip()
            finally:
                ole.close()
    except Exception:
        pass
    raise RuntimeError(".hwp 추출 실패 (pyhwp/olefile 필요). HWPX 사용 권장.")


def extract_text(file: Union[str, Path]) -> Tuple[str, str, List[str]]:
    """파일 → (text, file_type, warnings). file_type 은 확장자(점 제외)."""
    path = Path(file)
    ext = path.suffix.lower()
    ftype = ext.lstrip(".") or "unknown"
    warnings: List[str] = []

    if ext in _TEXT_EXTS:
        return path.read_text(encoding="utf-8", errors="ignore"), ftype, warnings
    if ext in (".xlsx", ".xlsm"):
        text = _extract_xlsx(path)
        if not text:
            warnings.append("xlsx: 추출된 텍스트 없음 (빈 시트이거나 차트/이미지만 존재).")
        return text, ftype, warnings
    if ext in _ZIP_XML_TARGETS:
        text = _extract_zip_xml(path, _ZIP_XML_TARGETS[ext])
        if not text:
            warnings.append(f"{ftype}: 추출된 텍스트 없음 (이미지/도형만 있거나 비표준 구조).")
        return text, ftype, warnings
    if ext == ".pdf":
        return _extract_pdf(path), ftype, warnings
    if ext == ".hwp":
        warnings.append("구형 .hwp 는 best-effort 추출. HWPX 권장.")
        return _extract_hwp(path), ftype, warnings
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"):
        raise ValueError(f"이미지({ftype})는 OCR 서비스(별도) 필요 — 이 모듈은 텍스트/문서 전용.")
    if ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"):
        raise ValueError(f"음성({ftype})은 STT 서비스(별도) 필요 — 이 모듈은 텍스트/문서 전용.")
    # 알 수 없는 확장자 → 평문 시도
    warnings.append(f"미지원 확장자 '{ext}' — 평문으로 읽기 시도.")
    return path.read_text(encoding="utf-8", errors="ignore"), ftype, warnings


def _detect_locale(text: str) -> str:
    """간단 휴리스틱 로케일 추정 (locale='auto' 일 때)."""
    sample = text[:4000]
    kana = sum(1 for c in sample if 0x3040 <= ord(c) <= 0x30FF)
    hangul = sum(1 for c in sample if 0xAC00 <= ord(c) <= 0xD7A3)
    han = sum(1 for c in sample if 0x4E00 <= ord(c) <= 0x9FFF)
    if hangul > 5:
        return "ko"
    if kana > 5:
        return "ja"
    if han > 5:
        return "zh-CN"
    return "en"


# ════════════════════════════════════════════════════════════════════════
# 8. 공개 API — classify()
# ════════════════════════════════════════════════════════════════════════
_FULLWIDTH = {ord('０') + i: ord('0') + i for i in range(10)}
_FULLWIDTH.update({ord('－'): ord('-'), ord('　'): ord(' ')})


def _normalize_obfuscation(text: str) -> str:
    """적대 난독 정규화: 전각 숫자·하이픈 → 반각(전각 주민번호/카드 검출).
    NER/스팬에 영향 없이 문자 치환만 수행."""
    return text.translate(_FULLWIDTH)


def classify_text(text: str, *, locale: str = "ko", n2sf_mode: bool = True,
                  verbose: bool = False, llm_mode: bool = False,
                  model: str = "minilm", weights: Optional[dict] = None,
                  ensemble_method: str = "escalate") -> dict:
    text = _normalize_obfuscation(text)          # 전각→반각(난독 방어)
    """이미 추출된 텍스트를 분류 (파일 추출 없이). classify() 내부 코어."""
    started = time.perf_counter()
    weights = weights or {}
    entity_overrides = {k: float(v) for k, v in (weights.get("entity") or {}).items()}
    tier_weights = {**DEFAULT_TIER_WEIGHTS, **(weights.get("tier") or {})}
    thr = weights.get("thresholds") or {}
    c_thr = float(thr.get("confidential", C_THRESHOLD))
    s_thr = float(thr.get("sensitive", S_THRESHOLD))
    extra_kw = [(k.get("keyword", ""), float(k.get("weight", 0) or 0), k.get("label", ""))
                for k in (weights.get("keyword") or []) if k.get("keyword")]

    presidio_findings = analyze(text, locale)
    pii_findings = _aggregate_findings(presidio_findings, text, locale, verbose)
    kw_findings = _scan_keywords(text, extra_kw, verbose)
    findings = pii_findings + kw_findings

    score, grade, confidence = _score(findings, entity_overrides, c_thr, s_thr)

    # 대량 개인정보(명부) — N²SF §9-6: 개인정보이므로 기밀(C) 아님. 최소 S(민감) floor.
    bulk_count = sum(f["count"] for f in findings if f["type"] in BULK_PII_TYPES)
    bulk_pii = bulk_count >= BULK_PII_THRESHOLD
    if bulk_pii and GRADE_RANK[grade] < GRADE_RANK["SENSITIVE"]:
        grade, confidence = "SENSITIVE", max(confidence, 0.8)

    # JP 컴플라이언스 (마이넘버=개인정보 → §9-6 S. 취급주의 표시는 유지하되 등급 강제 안 함)
    jp_compliance = None
    if locale == "ja":
        my_number = next((f for f in findings if f["type"] == "JP_MY_NUMBER"), None)
        jp_compliance = {
            "myNumberSuppressForced": bool(my_number),
            "specialCareDetected": False,
            "addressDecomposed": any(f["type"] == "JP_ADDRESS" for f in findings),
        }
        if my_number and GRADE_RANK[grade] < GRADE_RANK["SENSITIVE"]:
            grade = "SENSITIVE"

    # 규제 · 위반
    regulations = list(LOCALE_REGS.get(locale, []))
    violations = []
    if bulk_pii:
        violations.append({"code": "BULK-PII",
                           "msg": f"대량 개인정보({bulk_count}건) — 명부. §9-6 개인정보→민감(S) 상향",
                           "severity": "warn"})
    if locale == "ko" and any(f["type"] == "KR_RRN" for f in findings):
        violations.append({"code": "KR-PIPA-Art23", "msg": "민감정보 비식별 필요", "severity": "warn"})
    if locale == "ja" and any(f["type"] == "JP_MY_NUMBER" for f in findings):
        regulations.append("JP-MyNumberAct")
        violations.append({"code": "JP-MyNumberAct",
                           "msg": "マイナンバーは収集・保管に厳格制限。suppress 必須", "severity": "error"})

    # tier 결과
    rules_grade = grade if any(f["source"] != "ner" for f in findings) else "OPEN"
    ner_grade = grade if any(f["source"] in ("presidio", "ner") for f in findings) else "OPEN"
    ner_conf = min(1.0, max((f["confidence"] for f in pii_findings), default=0.0))

    # 신경망(LLM) tier
    neural_result = None
    if llm_mode:
        neural_result = neural_infer(text, locale, model)

    method = ensemble_method if ensemble_method in ENSEMBLE_METHODS else "escalate"
    ensemble = _combine(base_grade=grade, base_conf=confidence,
                        tier_grades={"rules": rules_grade, "ner": ner_grade},
                        tier_confs={"rules": confidence, "ner": ner_conf},
                        neural_result=neural_result, method=method, weights=tier_weights)
    grade, confidence = ensemble["grade"], ensemble["confidence"]

    # N²SF §9 floor — 앙상블이 하향 못 함
    # 개인정보(마이넘버·대량PII)는 §9-6 → 최소 S(기밀 아님)
    if jp_compliance and jp_compliance["myNumberSuppressForced"] and GRADE_RANK[grade] < GRADE_RANK["SENSITIVE"]:
        grade = "SENSITIVE"
    if bulk_pii and GRADE_RANK[grade] < GRADE_RANK["SENSITIVE"]:
        grade = "SENSITIVE"
    # 명시적 국가비밀 라벨(극비/대외비/기밀/機密…)만 CONFIDENTIAL 하드 floor(§9 비밀/보안업무규정)
    if any(f.get("secretFloor") for f in kw_findings):
        grade = "CONFIDENTIAL"

    shap = _shap_block(findings, score, entity_overrides)

    # ── 화면 출력용 정리 ──
    def _strip(f: dict) -> dict:
        out = {"type": f["type"],
               "label": f.get("label") or ENTITY_LABELS.get(f["type"], f["type"]),
               "count": f["count"], "confidence": round(f["confidence"], 3),
               "source": f["source"], "rule": f.get("rule"),
               "weight": (f.get("weight") if f["source"] == "keyword"
                          else _entity_weight(f["type"], entity_overrides)),
               "spans": f["spans"]}
        if verbose:
            out["items"] = f.get("items", [])
        return out

    pii_out = [_strip(f) for f in pii_findings]
    kw_out = [_strip(f) for f in kw_findings]

    grade_code = SHORT[grade]
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return {
        "schemaVersion": "2.0.0",
        "n2sfMode": n2sf_mode,
        "grade": grade_code if n2sf_mode else grade,
        "gradeCode": grade_code,
        "gradeFull": grade,
        "gradeLabel": GRADE_LABEL_KO[grade],
        "score": round(score, 3),
        "confidence": confidence,
        "locale": locale,
        "pii": {
            "totalCount": sum(f["count"] for f in pii_findings),
            "byType": pii_out,
        },
        "keywords": kw_out,
        "shap": shap,
        "ensemble": ensemble,
        "tiers": {
            "rules": {"grade": rules_grade, "confidence": confidence, "version": "rules-2.0"},
            "ner": {"grade": ner_grade, "confidence": ner_conf, "version": "presidio-2.2.355"},
            "neural": (neural_result and {
                "grade": neural_result.get("grade", "OPEN"),
                "confidence": neural_result.get("confidence", 0.0),
                "version": neural_result.get("version", "neural"),
                "scores": neural_result.get("scores"),
                "backend": neural_result.get("backend"),
                "note": neural_result.get("note"), "error": neural_result.get("error"),
            }) or {"grade": "OPEN", "confidence": 0.0, "version": "disabled"},
        },
        "compliance": {"regulations": regulations, "violations": violations},
        "jpCompliance": jp_compliance,
        "model": {
            "rules": "2.0", "presidio": "2.2.355", "spacy": SPACY_MODELS.get(locale, "-"),
            "neural": (NEURAL_BACKENDS.get(model, {}).get("model") if llm_mode else None),
            "ensembleMethod": method,
        },
        "stats": {
            "textLength": len(text), "elapsedMs": elapsed_ms,
            "findingsCount": sum(f["count"] for f in findings),
            "bulkPii": bulk_pii, "bulkPiiCount": bulk_count,
            "thresholds": {"confidential": c_thr, "sensitive": s_thr},
        },
    }


def classify(file: Union[str, Path], n2sf_mode: bool = True, verbose: bool = False,
             llm_mode: bool = False, *, model: str = "minilm",
             weights: Optional[dict] = None, locale: str = "ko",
             ensemble_method: str = "escalate") -> dict:
    """문서 파일 → 등급 분류 결과 JSON(dict).

    file       파일 경로 (오피스/HWPX 우선). 파라미터·반환 규격은 모듈 docstring 참조.
    """
    text, ftype, warnings = extract_text(file)
    if locale == "auto":
        locale = _detect_locale(text)

    result = classify_text(
        text, locale=locale, n2sf_mode=n2sf_mode, verbose=verbose,
        llm_mode=llm_mode, model=model, weights=weights,
        ensemble_method=ensemble_method)

    result["file"] = {
        "name": Path(file).name, "type": ftype,
        "textLength": len(text), "warnings": warnings,
    }
    if verbose:
        result["file"]["textSnippet"] = text[:300]
    return result


# ════════════════════════════════════════════════════════════════════════
# 9. CLI
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
#  v4 설명층 — 상세 JSON(GUI/XAI) + 오프라인 번들 + 무중단 핫리로드 + (옵션)Fable 티어
#  위 엔진(동일 파일)의 classify_text/analyze/extract_text/NEURAL_BACKENDS 를 직접 사용.
# ════════════════════════════════════════════════════════════════════════
import sys as _sys, time as _time, hashlib as _hashlib
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
_BUNDLED = _HERE / "models"              # 산출물_v4/models (오프라인 로컬 로드 우선)

MODEL_TIERS = {"fast-ko": "n2sf-small", "compact-multi": "n2sf-small-multi-minilm",
               "balanced": "n2sf-base", "accurate": "n2sf-xlmr-large", "n2sf-official": "n2sf-xlmr-official"}
TUNED_TIER_WEIGHTS = {"rules": 0.3, "ner": 0.3, "neural": 4.0}
GRADE_KO = {"C": "기밀", "S": "민감", "O": "공개"}


# ── (옵션) T4 Fable 5 LLM 티어 — 기본 off. 확장성 지점(온프레미스 기본 유지) ──
def fable_available() -> bool:
    return _os.environ.get("N2SF_FABLE") == "1" and bool(_os.environ.get("ANTHROPIC_API_KEY"))


def fable_classify(text: str) -> Optional[dict]:
    """Fable 5로 O/S/C 판정(§9 프롬프트). N2SF_FABLE=1 + ANTHROPIC_API_KEY 일 때만. 실패 시 None.
    ※ 클라우드 호출 → 데이터 반출 발생(opt-in). 기본 파이프라인은 로컬 유지."""
    if not fable_available():
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        sys_p = ("너는 대한민국 N²SF(정보공개법 §9) 문서 등급분류기다. "
                 "C=국가안보·외교·수사(§9 1~4호)/보안업무규정 비밀, "
                 "S=개인정보(주민번호 등 §9-6)·내부문서·로그, O=공개. "
                 "개인정보만 있으면 C가 아니라 S. 한 글자로만 답: C, S, O.")
        msg = client.messages.create(
            model="claude-fable-5", max_tokens=8,
            betas=["server-side-fallback-2026-06-01"],
            fallbacks=[{"model": "claude-opus-4-8"}],
            system=sys_p, messages=[{"role": "user", "content": text[:6000]}])
        out = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip().upper()
        g = next((c for c in out if c in "CSO"), None)
        return {"grade": g, "raw": out} if g else None
    except Exception as e:
        return {"error": str(e)[:120]}


class N2SFExplainClassifier:
    """3-tier 상세 분류 + 무중단 핫리로드. 반환은 GUI/XAI용 상세 JSON."""

    def __init__(self, model="n2sf-official", *, ensemble="soft", tier_weights=None,
                 early_exit=True, locale="ko", use_fable=False):
        self.ensemble = ensemble
        self.tier_weights = dict(tier_weights or TUNED_TIER_WEIGHTS)
        self.early_exit = early_exit
        self.locale = locale
        self.use_fable = use_fable
        self._ver = 0
        self._active = {"backend": self._resolve(model), "name": model, "ver": 0}

    @staticmethod
    def _resolve(model):
        b = MODEL_TIERS.get(model, model)
        d = _BUNDLED / b
        if d.exists():                                   # 번들(오프라인)에 있으면 등록/재지정 후 사용
            NEURAL_BACKENDS[b] = {"label": b, "kind": "finetuned", "model": str(d), "langs": "multi"}
            return b
        if b in NEURAL_BACKENDS:
            return b
        if _Path(model).exists():
            n = _Path(model).name
            NEURAL_BACKENDS[n] = {"label": n, "kind": "finetuned", "model": model, "langs": "multi"}
            return n
        raise ValueError(f"알 수 없는 모델: {model} (티어/백엔드명/경로)")

    def register_model(self, name, path):
        NEURAL_BACKENDS[name] = {"label": name, "kind": "finetuned", "model": path, "langs": "multi"}

    def reload_model(self, model, *, path=None, evict_old=True):
        """무중단 교체(v3 준용): 새 모델 사전 로드→활성 참조 원자적 스왑. 진행 요청은 구모델로 완료."""
        if path:
            self.register_model(model, path)
        b = self._resolve(model)
        classify_text("워밍업", locale=self.locale, llm_mode=True, model=b,
                      ensemble_method=self.ensemble, weights={"tier": self.tier_weights})
        old = self._active["backend"]; self._ver += 1
        self._active = {"backend": b, "name": model, "ver": self._ver}
        if evict_old and old != b:
            _NEURAL_MODELS.pop(old, None)
        return {"reloaded": model, "backend": b, "version": self._ver}

    @property
    def model_version(self):
        a = self._active; return f"{a['name']}#v{a['ver']}"

    def classify(self, source: str, *, is_file: bool = None) -> dict:
        active = self._active; backend = active["backend"]; t0 = _time.perf_counter()
        finfo = {"source": "text", "format": "text", "sizeBytes": None, "sha1_12": None}
        if is_file is None:
            is_file = ("\n" not in source) and len(source) < 500 and _Path(source).exists()
        if is_file:
            p = _Path(source); text, fmt, _ = extract_text(source)
            finfo = {"source": str(p), "format": fmt,
                     "sizeBytes": p.stat().st_size if p.exists() else None,
                     "sha1_12": _hashlib.sha1(p.read_bytes()).hexdigest()[:12] if p.exists() else None}
        else:
            text = source
        finfo["extractedChars"] = len(text)

        w = {"tier": self.tier_weights}
        rules = classify_text(text, locale=self.locale, llm_mode=False, verbose=True,
                              ensemble_method=self.ensemble, weights=w)
        skipped = self.early_exit and rules["gradeFull"] == "CONFIDENTIAL"
        final = rules if skipped else classify_text(text, locale=self.locale, llm_mode=True,
                                    model=backend, verbose=True, ensemble_method=self.ensemble, weights=w)
        try:
            ner = analyze(text[:20000], self.locale)
            ner_ents = [{"entity": e["entity_type"], "text": text[e["start"]:e["end"]][:60],
                         "start": e["start"], "end": e["end"], "score": round(e["score"], 3),
                         "recognizer": e["recognizer"]} for e in ner[:50]]
        except Exception:
            ner_ents = []

        fable = None
        if self.use_fable and not skipped:
            fable = fable_classify(text)     # 옵션 티어(클라우드) — 참고/확장

        return self._pack(final, skipped, finfo, ner_ents, active, t0, fable)

    def _pack(self, r, skipped, finfo, ner_ents, active, t0, fable):
        grade = r["gradeCode"]
        pii = r.get("pii", {}).get("byType", [])
        kws = r.get("keywords", [])
        spacy_types = {"PERSON", "LOCATION", "ORGANIZATION", "NRP", "GPE", "DATE_TIME"}
        regex_ents = [e for e in ner_ents if e["entity"] not in spacy_types]
        ner_ctx = [e for e in ner_ents if e["entity"] in spacy_types]
        n2sf_rule, reason = self._n2sf_reason(grade, pii, kws, r)
        out = {
            "schema": "n2sf-explain-v4",
            "grade": grade, "gradeLabel": GRADE_KO.get(grade, grade),
            "confidence": r["confidence"], "score": r["score"], "file": finfo,
            "decision": {"finalGrade": grade,
                         "decidedByTier": ("T1(규칙)" if skipped else "T3(뉴럴)+앙상블"),
                         "earlyExit": skipped, "n2sfPrinciple": n2sf_rule, "reason": reason},
            "tiers": {"rules": r["tiers"]["rules"], "ner": r["tiers"]["ner"],
                      "neural": r["tiers"]["neural"], "ensemble": r.get("ensemble", {})},
            "findings": {
                "regexEntities": regex_ents,     # 강식별자(정규식·deny-list)
                "nerEntities": ner_ctx,          # 문맥 개체(spaCy NER)
                "keywords": [self._fin(f) for f in kws],
                "piiByType": [self._fin(f) for f in pii],
                "piiTotal": r.get("pii", {}).get("totalCount", 0)},
            "shap": r.get("shap", {}),
            "compliance": r.get("compliance", {}),
            "explanation": self._explain(grade, r, n2sf_rule),
            "modelVersion": f"{active['name']}#v{active['ver']}",
            "elapsedMs": int((_time.perf_counter() - t0) * 1000),
        }
        if fable is not None:
            out["fableTier"] = fable            # (옵션) 클라우드 LLM 참고 판정
        return out

    @staticmethod
    def _fin(f):
        return {"type": f.get("type"), "label": f.get("label"), "count": f.get("count"),
                "source": f.get("source"), "weight": f.get("weight"),
                "spans": (f.get("spans") or [])[:5],
                "samples": [it.get("text") for it in (f.get("items") or [])][:3]}

    @staticmethod
    def _n2sf_reason(grade, pii, kws, r):
        if any((f.get("type") == "KEYWORD_SECRET") or (f.get("label") in SECRET_FLOOR_KW) for f in kws):
            return "명시적 기밀표지 → C(§9 비밀/보안업무규정)", "기밀/대외비/극비 등 국가 비밀 표지 검출 → 기밀 하드 floor."
        if any(v.get("code") == "BULK-PII" for v in (r.get("compliance") or {}).get("violations", [])):
            return "대량 개인정보 → S(§9-6)", "개인정보 다수(명부) → 민감 floor(개인정보는 기밀 아님)."
        if grade == "C":
            return "국가안보·외교·수사 영향 → C(§9 1~4호)", "뉴럴이 국가업무 기밀성으로 판단."
        if grade == "S":
            has = any(f.get("source") in ("pattern", "regex", "ner", "presidio", "keyword") for f in pii)
            return ("개인정보 → S(§9-6)" if has else "비공개 업무정보 → S(§9 5~8호)",
                    "개인정보/내부문서 → 민감(국가안보 맥락 아님).")
        return "공개 정보 → O", "개인정보·기밀표지 없음(또는 공개 조치)."

    @staticmethod
    def _explain(grade, r, n2sf_rule):
        top = (r.get("shap") or {}).get("contributions", [])[:3]
        why = ", ".join(f"{c['label']}({int(c.get('percent', 0)*100)}%)" for c in top) or "명시적 근거 없음(의미 판단)"
        return f"이 문서는 {GRADE_KO.get(grade, grade)}({grade}) 등급입니다. 근거: {why}. 원칙: {n2sf_rule}."


def classify(source, model="n2sf-official", **kw):
    return N2SFExplainClassifier(model=model).classify(source, **kw)


if __name__ == "__main__":
    import argparse, json as _json
    ap = argparse.ArgumentParser(description="N²SF 3-tier 단일파일 분류 v4(상세)")
    ap.add_argument("source", nargs="?"); ap.add_argument("--text")
    ap.add_argument("--model", default="n2sf-official"); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--fable", action="store_true", help="옵션 Fable 티어 사용(N2SF_FABLE=1+키 필요)")
    a = ap.parse_args()
    clf = N2SFExplainClassifier(model=a.model, use_fable=a.fable)
    if a.demo or (not a.source and not a.text):
        for t in ["[대외비] 사내 전략 자료. 무단유출 금지.",
                  "담당자 홍길동 주민번호 900101-1234567 010-1234-5678.",
                  "2024 공개채용 안내입니다."]:
            r = clf.classify(t, is_file=False)
            print(f"\n{r['grade']}({r['gradeLabel']}) — {r['explanation']}")
            print("  regexEntities:", [e['entity'] for e in r['findings']['regexEntities']],
                  "keywords:", [f['label'] for f in r['findings']['keywords']])
    else:
        src = a.text if a.text is not None else a.source
        print(_json.dumps(clf.classify(src, is_file=(a.text is None)), ensure_ascii=False, indent=2))
