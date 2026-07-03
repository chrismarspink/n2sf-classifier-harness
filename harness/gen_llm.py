"""gen_llm.py — 외부/로컬 LLM으로 **다양한 분포**의 등급 문서 생성 (과적합 해소용 생성기 B).

생성기 A(corpus.py 템플릿)와 다른 문체·구조·난독을 LLM이 만들어 일반화를 검증·향상한다.
**라벨 신뢰성**: LLM은 자연스러운 본문(다양성)만 쓰고, 등급을 결정하는 식별자/키워드는
우리 synth.py의 **유효 포맷 값을 주입**한다(플레이스홀더 치환) → 의도 등급이 정확히 고정된다.

생성기:
  - azure:gpt-4o / azure:o4-mini  (Azure OpenAI, .env.azure)
  - gemma                          (로컬 ollama gemma2:9b)

외부 LLM은 **데이터 생성(오프라인)** 에만 쓰인다. 실제 분류(추론)는 로컬 3-tier 유지.
LLM에는 합성 시나리오만 전송한다(실문서 금지).
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

from .corpus import LogicalDoc
from .synth import Synth

PLACEHOLDERS = ["{{NAME}}", "{{PHONE}}", "{{EMAIL}}", "{{ADDRESS}}", "{{BIZNO}}",
                "{{RRN}}", "{{CARD}}", "{{ACCOUNT}}", "{{PASSPORT}}", "{{APIKEY}}", "{{KEYWORD}}"]

# 등급별 필수 플레이스홀더(라벨 보장) — 본문에 없으면 한 줄 보강 주입
REQUIRED = {
    "C": [["{{RRN}}"], ["{{CARD}}"], ["{{ACCOUNT}}"], ["{{PASSPORT}}"], ["{{APIKEY}}"], ["{{KEYWORD}}"]],
    "S": [["{{NAME}}", "{{PHONE}}"], ["{{NAME}}", "{{EMAIL}}"]],
    "O": [],
}
GRADE_DESC = {
    "C": "기밀(CONFIDENTIAL) — 주민등록번호·신용카드·계좌·여권·API키 같은 강한 개인식별자나 '대외비/극비/기밀' 라벨이 포함",
    "S": "민감(SENSITIVE) — 소수의 이름·연락처·이메일 같은 제한적 개인정보만 포함(강식별자·기밀라벨 없음)",
    "O": "공개(OPEN) — 개인식별 정보도 기밀 라벨도 없는 일반 공개 문서",
}
DIFF_DESC = {
    1: "평이하고 명확한 일반 문서",
    2: "경계가 모호한 문서(정보가 적당히 섞임)",
    3: "탐지를 피하려는 듯 표현을 비틀거나 띄어쓰기/완곡어법을 쓴 까다로운 문서",
    4: "정보가 길고 산만한 본문 속에 드물게 흩어져 있어 추출이 어려운 문서",
}


def _load_env(path=".env.azure"):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k, v)


# ── 백엔드별 본문 생성 ───────────────────────────────────────────────────
def _azure_client(api_version: str, endpoint: str = None, key: str = None):
    from openai import AzureOpenAI
    _load_env()
    return AzureOpenAI(azure_endpoint=endpoint or os.environ["AZURE_OPENAI_ENDPOINT"],
                       api_key=key or os.environ["AZURE_OPENAI_KEY"], api_version=api_version)


def _gpt5_client():
    """GPT-5(별도 cognitiveservices 엔드포인트) 클라이언트."""
    _load_env()
    return _azure_client(os.environ.get("AZURE_GPT5_APIVER", "2025-04-01-preview"),
                         endpoint=os.environ["AZURE_GPT5_ENDPOINT"],
                         key=os.environ["AZURE_GPT5_KEY"])


LANG_DESC = {"ko": "한국어", "en": "영어(English)", "zh": "중국어(简体中文)", "ja": "일본어(日本語)"}


def _prompt(grade: str, difficulty: int, lang: str = "ko") -> str:
    ph = ", ".join(PLACEHOLDERS)
    lname = LANG_DESC.get(lang, "한국어")
    return (
        f"너는 기업의 실제 문서를 흉내 내는 합성 데이터 생성기다.\n"
        f"다음 조건의 **{lname}** 문서 1건을 자연스럽게 작성하라(본문은 {lname}로).\n"
        f"- 등급: {GRADE_DESC[grade]}\n"
        f"- 난이도: {DIFF_DESC[difficulty]}\n"
        f"- 개인정보·식별자·기밀어가 들어갈 자리는 **반드시 다음 플레이스홀더 토큰**으로 표기: {ph}\n"
        f"  (실제 번호/이름을 쓰지 말 것. 토큰만 사용. 예: '담당자 {{{{NAME}}}} ({{{{PHONE}}}})')\n"
        f"- 등급에 맞는 플레이스홀더만 사용하라(공개 문서엔 어떤 토큰도 넣지 말 것).\n"
        f"- 기밀 라벨 키워드는 해당 언어로도 자연스럽게(예: en=CONFIDENTIAL, zh=机密, ja=極秘).\n"
        f"- 제목 한 줄 + 본문 3~8문장. 다양한 업종·상황·문체로. 머리말/설명 없이 문서 본문만 출력."
    )


def _gen_text_azure(grade: str, difficulty: int, deployment: str, api_version: str, lang: str = "ko") -> Optional[str]:
    try:
        is_gpt5 = deployment.startswith("gpt-5")
        c = _gpt5_client() if is_gpt5 else _azure_client(api_version)
        kw = {"model": deployment, "messages": [{"role": "user", "content": _prompt(grade, difficulty, lang)}]}
        if is_gpt5 or deployment.startswith(("o4", "o3", "o1")):
            kw["max_completion_tokens"] = 3000          # reasoning 모델(gpt-5 포함)
        else:
            kw["max_tokens"] = 600; kw["temperature"] = 1.0
        r = c.chat.completions.create(**kw)
        return (r.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"  [gen azure:{deployment}] 오류: {str(exc)[:120]}")
        return None


def _gen_text_gemma(grade: str, difficulty: int, url="http://localhost:11434/api/chat") -> Optional[str]:
    import json, urllib.request
    try:
        payload = {"model": "gemma2:9b", "stream": False, "options": {"temperature": 1.0},
                   "messages": [{"role": "user", "content": _prompt(grade, difficulty)}]}
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        return (r.get("message") or {}).get("content", "").strip()
    except Exception as exc:
        print(f"  [gen gemma] 오류: {str(exc)[:120]}")
        return None


# ── 플레이스홀더 → 유효 PII 치환(라벨 보장) + 난이도별 난독 ────────────────
class LLMFiller:
    def __init__(self, seed=0):
        self.s = Synth(seed); self.r = self.s.r

    def _val(self, token: str, difficulty: int) -> str:
        obf = difficulty >= 3
        if token == "{{NAME}}": return self.s.name()
        if token == "{{PHONE}}": return self.s.phone()
        if token == "{{EMAIL}}": return self.s.email()
        if token == "{{ADDRESS}}": return self.s.address()
        if token == "{{BIZNO}}": return self.s.biz_no()
        if token == "{{RRN}}":
            v = self.s.rrn()
            return self.s.fullwidth_digits(v) if (obf and self.r.random() < 0.5) else v
        if token == "{{CARD}}": return self.s.credit_card()
        if token == "{{ACCOUNT}}": return self.s.account()
        if token == "{{PASSPORT}}": return self.s.passport()
        if token == "{{APIKEY}}": return self.s.api_key()
        if token == "{{KEYWORD}}":
            kw = self.r.choice(["대외비", "극비", "기밀"])
            return self.s.spaced_keyword(kw) if obf else kw
        return ""

    def fill(self, text: str, grade: str, difficulty: int) -> str:
        # 등장한 토큰 치환
        for tok in PLACEHOLDERS:
            while tok in text:
                text = text.replace(tok, self._val(tok, difficulty), 1)
        # 등급 필수 요소 보강(라벨 보장)
        present = lambda toks: all(False for _ in toks)  # placeholder already replaced; check by re-scan below
        # 이미 치환됐으므로, 등급별 필수 요소가 텍스트에 실제로 들어갔는지 보장하기 위해
        # 필수 조합 중 하나라도 충족 안 되면 한 줄 추가.
        return text


def _ensure_label(doc_text: str, grade: str, difficulty: int, filler: LLMFiller) -> str:
    """치환 후에도 등급 결정 요소가 비면 한 줄 보강(라벨 신뢰성)."""
    if grade == "O":
        return doc_text
    combos = REQUIRED[grade]
    # 간이 검사: 강식별자/키워드 흔적이 있는지(치환된 실제 값 기준은 어려우니, 보강 라인을 무조건 1개 추가)
    extra_tokens = filler.r.choice(combos)
    extra = " ".join(filler._val(t, difficulty) for t in extra_tokens)
    label = {"C": "[부속] ", "S": "[문의] "}[grade]
    return doc_text + "\n" + label + extra


def generate(generator: str, grade: str, difficulty: int, doc_id: str,
             filler: LLMFiller, locale="ko", lang: str = "ko") -> Optional[LogicalDoc]:
    """generator: 'azure:gpt-4o' | 'azure:o4-mini' | 'azure:gpt-5' | 'gemma'. lang: ko/en/zh/ja."""
    if generator == "azure:gpt-4o":
        raw = _gen_text_azure(grade, difficulty, os.environ.get("AZURE_GPT4O_DEPLOYMENT", "gpt-4o"),
                              os.environ.get("AZURE_GPT4O_APIVER", "2025-01-01-preview"), lang)
    elif generator == "azure:o4-mini":
        raw = _gen_text_azure(grade, difficulty, os.environ.get("AZURE_O4MINI_DEPLOYMENT", "o4-mini"),
                              os.environ.get("AZURE_O4MINI_APIVER", "2025-04-01-preview"), lang)
    elif generator == "azure:gpt-5":
        raw = _gen_text_azure(grade, difficulty, os.environ.get("AZURE_GPT5_DEPLOYMENT", "gpt-5.4"),
                              os.environ.get("AZURE_GPT5_APIVER", "2025-04-01-preview"), lang)
    elif generator == "gemma":
        raw = _gen_text_gemma(grade, difficulty)
    else:
        return None
    if not raw or len(raw) < 15:
        return None
    # 마크다운 펜스/헤더 정리
    raw = re.sub(r"```[a-zA-Z]*", "", raw).replace("```", "")
    raw = re.sub(r"^#{1,6}\s*", "", raw, flags=re.MULTILINE).strip()
    filled = filler.fill(raw, grade, difficulty)
    filled = _ensure_label(filled, grade, difficulty, filler)
    lines = [l for l in filled.splitlines() if l.strip()]
    title = lines[0][:80] if lines else f"{grade} 문서"
    paras = lines[1:] if len(lines) > 1 else lines
    cat = f"llm-{generator.split(':')[-1]}-L{difficulty}"
    return LogicalDoc(doc_id=doc_id, grade=grade, category=cat, locale=locale,
                      title=title, paragraphs=paras)


_load_env()
