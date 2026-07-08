#!/usr/bin/env python3
"""classifier_v4.py — N²SF 등급분류 '상세 설명형' 함수 (GUI/XAI용).

v3(통합·오프라인·핫리로드) 위에, **판단 근거 전체**를 JSON으로 노출한다:
  · 정규식/키워드 검출(source별)  · NER(Presidio+spaCy) 엔티티  · §9 등급 원칙(적용된 floor)
  · SHAP 기여도(왜 이 등급)       · 파일 메타(포맷/크기/추출)   · 사람이 읽는 설명문
→ 화면에서 "왜 C/S/O인지"를 그대로 그릴 수 있는 수준.

핵심: clf = N2SFExplainClassifier("accurate"); r = clf.classify("문서.pdf"); r 는 GUI용 상세 dict.
추론 100% 로컬(외부 LLM/GPU 없음). 오프라인·핫리로드는 v3와 동일.
"""
from __future__ import annotations
import hashlib, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    sys.path.insert(0, str(_p))
import data_classifier as dc  # noqa: E402
_BUNDLED = _HERE / "models"

MODEL_TIERS = {"fast-ko": "n2sf-small", "compact-multi": "n2sf-small-multi-minilm",
               "balanced": "n2sf-base", "accurate": "n2sf-xlmr-large", "n2sf-official": "n2sf-xlmr-official"}
TUNED_TIER_WEIGHTS = {"rules": 0.3, "ner": 0.3, "neural": 4.0}
GRADE_KO = {"C": "기밀", "S": "민감", "O": "공개"}


class N2SFExplainClassifier:
    def __init__(self, model="accurate", *, ensemble="soft", tier_weights=None,
                 early_exit=True, locale="ko"):
        self.ensemble = ensemble
        self.tier_weights = dict(tier_weights or TUNED_TIER_WEIGHTS)
        self.early_exit = early_exit
        self.locale = locale
        self._ver = 0
        self._active = {"backend": self._resolve(model), "name": model, "ver": 0}

    # ── 모델 관리(v3와 동일: 오프라인 로컬 + 무중단 핫리로드) ──
    @staticmethod
    def _resolve(model):
        b = MODEL_TIERS.get(model, model)
        d = _BUNDLED / b
        if d.exists() and b in dc.NEURAL_BACKENDS:
            dc.NEURAL_BACKENDS[b]["model"] = str(d)
        if b in dc.NEURAL_BACKENDS:
            return b
        if Path(model).exists():
            n = Path(model).name
            dc.NEURAL_BACKENDS[n] = {"label": n, "kind": "finetuned", "model": model, "langs": "multi"}
            return n
        raise ValueError(f"알 수 없는 모델: {model}")

    def register_model(self, name, path):
        dc.NEURAL_BACKENDS[name] = {"label": name, "kind": "finetuned", "model": path, "langs": "multi"}

    def reload_model(self, model, *, path=None):
        if path: self.register_model(model, path)
        b = self._resolve(model)
        dc.classify_text("워밍업", locale=self.locale, llm_mode=True, model=b,
                         ensemble_method=self.ensemble, weights={"tier": self.tier_weights})
        old = self._active["backend"]; self._ver += 1
        self._active = {"backend": b, "name": model, "ver": self._ver}
        if old != b: dc._NEURAL_MODELS.pop(old, None)
        return {"reloaded": model, "backend": b, "version": self._ver}

    # ── 상세 분류 ──
    def classify(self, source: str, *, is_file: bool = None) -> dict:
        active = self._active; backend = active["backend"]; t0 = time.perf_counter()
        # 파일 메타
        finfo = {"source": "text", "format": "text", "sizeBytes": None, "sha1_12": None}
        if is_file is None:
            is_file = ("\n" not in source) and len(source) < 500 and Path(source).exists()
        if is_file:
            p = Path(source)
            text, fmt, _ = dc.extract_text(source)
            finfo = {"source": str(p), "format": fmt,
                     "sizeBytes": p.stat().st_size if p.exists() else None,
                     "sha1_12": hashlib.sha1(p.read_bytes()).hexdigest()[:12] if p.exists() else None}
        else:
            text = source; finfo["format"] = "text"
        finfo["extractedChars"] = len(text)

        w = {"tier": self.tier_weights}
        rules = dc.classify_text(text, locale=self.locale, llm_mode=False, verbose=True,
                                 ensemble_method=self.ensemble, weights=w)
        skipped = self.early_exit and rules["gradeFull"] == "CONFIDENTIAL"
        final = rules if skipped else dc.classify_text(text, locale=self.locale, llm_mode=True,
                                     model=backend, verbose=True, ensemble_method=self.ensemble, weights=w)
        # 원시 NER 엔티티(Presidio+spaCy) — 근거 시각화용(길면 앞부분만)
        try:
            ner = dc.analyze(text[:20000], self.locale)
            ner_ents = [{"entity": e["entity_type"], "text": text[e["start"]:e["end"]][:60],
                         "start": e["start"], "end": e["end"], "score": round(e["score"], 3),
                         "recognizer": e["recognizer"]} for e in ner[:50]]
        except Exception:
            ner_ents = []

        return self._pack(final, skipped, finfo, ner_ents, active, t0)

    # ── GUI용 JSON 조립 ──
    def _pack(self, r, skipped, finfo, ner_ents, active, t0):
        grade = r["gradeCode"]
        pii = r.get("pii", {}).get("byType", [])
        kws = r.get("keywords", [])
        # 엔티티 타입으로 정규식(강식별자) vs NER(문맥 개체) 분리 — GUI 근거표시용
        spacy_types = {"PERSON", "LOCATION", "ORGANIZATION", "NRP", "GPE", "DATE_TIME"}
        regex_ents = [e for e in ner_ents if e["entity"] not in spacy_types]
        ner_ctx = [e for e in ner_ents if e["entity"] in spacy_types]
        n2sf_rule, reason = self._n2sf_reason(grade, pii, kws, r)
        expl = self._explain(grade, r, pii, kws, n2sf_rule)
        return {
            "schema": "n2sf-explain-v4",
            "grade": grade, "gradeLabel": GRADE_KO.get(grade, grade),
            "confidence": r["confidence"], "score": r["score"],
            "file": finfo,
            "decision": {
                "finalGrade": grade, "decidedByTier": ("T1(규칙)" if skipped else "T3(뉴럴)+앙상블"),
                "earlyExit": skipped, "n2sfPrinciple": n2sf_rule, "reason": reason,
            },
            "tiers": {
                "rules": r["tiers"]["rules"], "ner": r["tiers"]["ner"],
                "neural": r["tiers"]["neural"], "ensemble": r.get("ensemble", {}),
            },
            "findings": {
                "regexEntities": regex_ents,       # 강식별자(주민번호·카드·전화·이메일·키…) = 정규식 tier
                "nerEntities": ner_ctx,            # 문맥 개체(인물·기관·지명) = Presidio/spaCy NER tier
                "keywords": [self._fin(f) for f in kws],   # 등급 키워드(기밀표지 등)
                "piiByType": [self._fin(f) for f in pii],  # 집계(등급 점수 기여)
                "piiTotal": r.get("pii", {}).get("totalCount", 0),
            },
            "shap": r.get("shap", {}),          # baseline/total/contributions(왜 이 등급)
            "compliance": r.get("compliance", {}),
            "explanation": expl,                # 사람이 읽는 설명(GUI 상단)
            "modelVersion": f"{active['name']}#v{active['ver']}",
            "elapsedMs": int((time.perf_counter() - t0) * 1000),
        }

    @staticmethod
    def _fin(f):
        return {"type": f.get("type"), "label": f.get("label"), "count": f.get("count"),
                "source": f.get("source"), "weight": f.get("weight"), "spans": f.get("spans", [])[:5],
                "samples": [it.get("text") for it in (f.get("items") or [])][:3]}

    @staticmethod
    def _n2sf_reason(grade, pii, kws, r):
        """§9 원칙 중 무엇이 등급을 만들었나."""
        secret = any((f.get("type") == "KEYWORD_SECRET") or (f.get("label") in dc.SECRET_FLOOR_KW)
                     for f in kws)
        if secret:
            return "명시적 기밀표지 → C(§9 비밀/보안업무규정)", "기밀/대외비/극비 등 국가 비밀 표지 검출 → 기밀 하드 floor."
        viol = (r.get("compliance") or {}).get("violations", [])
        if any(v.get("code") == "BULK-PII" for v in viol):
            return "대량 개인정보 → S(§9-6)", "개인정보 다수(명부) → 민감 floor(개인정보는 기밀 아님)."
        if grade == "C":
            return "국가안보·외교·수사 영향 → C(§9 1~4호)", "뉴럴이 국가업무 기밀성으로 판단."
        if grade == "S":
            has_pii = any(f.get("source") in ("pattern", "regex", "ner", "presidio") for f in pii)
            return ("개인정보 → S(§9-6)" if has_pii else "비공개 업무정보 → S(§9 5~8호)",
                    "개인정보/내부문서 → 민감(국가안보 맥락 아님).")
        return "공개 정보 → O", "개인정보·기밀표지 없음(또는 공개 조치)."

    @staticmethod
    def _explain(grade, r, pii, kws, n2sf_rule):
        top = (r.get("shap") or {}).get("contributions", [])[:3]
        why = ", ".join(f"{c['label']}({int(c.get('percent',0)*100)}%)" for c in top) or "명시적 근거 없음(의미 판단)"
        return (f"이 문서는 {GRADE_KO.get(grade, grade)}({grade}) 등급입니다. "
                f"근거: {why}. 원칙: {n2sf_rule}. "
                f"{'규칙에서 확정되어 뉴럴 생략(early-exit).' if r['tiers']['neural']['grade']=='OPEN' and grade=='C' else '규칙+NER+뉴럴 종합 판단.'}")


def classify(source, model="accurate", **kw):
    return N2SFExplainClassifier(model=model).classify(source, **kw)


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="N²SF 상세 설명형 분류 v4")
    ap.add_argument("source", nargs="?"); ap.add_argument("--text")
    ap.add_argument("--model", default="accurate"); ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    clf = N2SFExplainClassifier(model=a.model)
    if a.demo or (not a.source and not a.text):
        for t in ["[대외비] 사내 전략 자료. 무단유출 금지.",
                  "담당자 홍길동 주민번호 900101-1234567 010-1234-5678.",
                  "2024 공개채용 안내입니다."]:
            r = clf.classify(t, is_file=False)
            print(f"\n{r['grade']}({r['gradeLabel']}) — {r['explanation']}")
            print("  findings.regex:", [f['label'] for f in r['findings']['regex']],
                  "keywords:", [f['label'] for f in r['findings']['keywords']])
    else:
        src = a.text if a.text is not None else a.source
        print(json.dumps(clf.classify(src, is_file=(a.text is None)), ensure_ascii=False, indent=2))
