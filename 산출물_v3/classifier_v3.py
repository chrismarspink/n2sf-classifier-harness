#!/usr/bin/env python3
"""classifier_v3.py — N²SF 문서 등급분류 통합 함수 (v3, N²SF §9 정합).

지금까지의 결과(3-tier 엔진 + 증류 뉴럴 + 튜닝 앙상블 + §9 규칙 + 무중단 핫리로드)를
**하나의 클래스/함수**로 정리한 참조 구현.

핵심 API
  clf = N2SFClassifier(model="accurate")        # 티어 프리셋 또는 백엔드명
  result = clf.classify("문서.pdf")             # 파일 경로 또는 원문 텍스트
  clf.reload_model("n2sf-custom-v2")            # 무중단(zero-downtime) 모델 교체

반환(dict): grade(C/S/O)·gradeLabel·confidence·score·tiersRun·neuralSkipped·shap·compliance·modelVersion
설계 포인트: 추론 100% 로컬 CPU(외부 LLM/GPU 없음) · 티어 early-exit · §9 영향기반(개인정보=S).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

# ── 완전 오프라인 강제: Hugging Face/네트워크 접속 없이 로컬 파일만 사용 ──
#   (transformers import 전에 설정해야 효과. 모델은 로컬 디렉토리에서 로드.)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 엔진(data_classifier.py): 리포 루트(상위) 또는 같은 폴더 모두 지원
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    sys.path.insert(0, str(_p))
import data_classifier as dc  # noqa: E402

# 번들 모델 위치(산출물_v3/models/<name>)가 있으면 그걸 우선(자족적 오프라인 배포)
_BUNDLED = _HERE / "models"

# ── 용도별 티어 프리셋 → 실제 백엔드명 ───────────────────────────────
MODEL_TIERS = {
    "fast-ko":       "n2sf-small",             # 초경량 한국어 전용(엣지)
    "compact-multi": "n2sf-small-multi-minilm",  # 소형 다국어(≈488MB)
    "balanced":      "n2sf-base",              # 표준
    "accurate":      "n2sf-xlmr-large",        # 정확도·강건성 최우선(권장 기본)
    "n2sf-official": "n2sf-xlmr-official",      # §9 공식기준 재증류판(있으면)
}
# 튜닝된 실서비스 앙상블(규칙/NER는 floor, 뉴럴 주도). held-out 실측 최적.
TUNED_TIER_WEIGHTS = {"rules": 0.3, "ner": 0.3, "neural": 4.0}


class N2SFClassifier:
    """스레드 안전 · 무중단 핫리로드 지원 분류기."""

    def __init__(self, model: str = "accurate", *, ensemble: str = "soft",
                 tier_weights: dict = None, early_exit: bool = True,
                 locale: str = "ko", warmup: bool = True):
        self.ensemble = ensemble
        self.tier_weights = dict(tier_weights or TUNED_TIER_WEIGHTS)
        self.early_exit = early_exit
        self.locale = locale
        self._version = 0
        # _active 는 단일 참조 → 원자적 스왑(무중단). 읽는 쪽은 스냅샷을 잡는다.
        self._active = {"backend": self._resolve(model), "name": model, "ver": self._version}
        if warmup:
            try: self._warm(self._active["backend"])
            except Exception: pass

    # ── 모델 관리 ────────────────────────────────────────────────
    @staticmethod
    def _bundle_local(backend: str):
        """번들(산출물_v3/models/<backend>)이 있으면 백엔드 경로를 그쪽으로 재지정(오프라인 자족)."""
        d = _BUNDLED / backend
        if d.exists() and backend in dc.NEURAL_BACKENDS:
            dc.NEURAL_BACKENDS[backend]["model"] = str(d)

    @staticmethod
    def _resolve(model: str) -> str:
        """티어 프리셋명 또는 백엔드명/경로 → data_classifier 백엔드명. 로컬/오프라인 우선."""
        if model in MODEL_TIERS:
            b = MODEL_TIERS[model]; N2SFClassifier._bundle_local(b); return b
        if model in dc.NEURAL_BACKENDS:
            N2SFClassifier._bundle_local(model); return model
        # 경로면 즉석 등록
        if Path(model).exists():
            name = Path(model).name
            dc.NEURAL_BACKENDS[name] = {"label": name, "kind": "finetuned",
                                        "model": model, "langs": "multi"}
            return name
        raise ValueError(f"알 수 없는 모델: {model} (티어/백엔드명/경로 중 하나)")

    def register_model(self, name: str, path: str, kind: str = "finetuned"):
        """새 모델(온사이트 학습 결과)을 백엔드로 등록."""
        dc.NEURAL_BACKENDS[name] = {"label": name, "kind": kind, "model": path, "langs": "multi"}

    def _warm(self, backend: str):
        """새 모델을 미리 로드(검증) — 스왑 전에 무거운 로딩을 끝낸다."""
        dc.classify_text("워밍업 문서", locale=self.locale, llm_mode=True,
                         model=backend, ensemble_method=self.ensemble,
                         weights={"tier": self.tier_weights})

    def reload_model(self, model: str, *, path: str = None, evict_old: bool = True) -> dict:
        """무중단 모델 교체. path 지정 시 등록 후 교체.
        1) 새 모델을 백그라운드에서 로드/검증(_warm) — 기존 모델은 계속 서빙.
        2) self._active 를 원자적으로 교체(단일 dict 참조 대입 = GIL상 원자적).
        진행 중 요청은 기존 모델로 완료, 신규 요청부터 새 모델 → 다운타임 0.
        """
        if path:
            self.register_model(model, path)
        backend = self._resolve(model)
        self._warm(backend)                       # 느린 로딩을 스왑 밖에서 완료
        old = self._active.get("backend")
        self._version += 1
        self._active = {"backend": backend, "name": model, "ver": self._version}  # ← 원자적 스왑
        if evict_old and old and old != backend:
            dc._NEURAL_MODELS.pop(old, None)      # 구 모델 메모리 해제
        return {"reloaded": model, "backend": backend, "version": self._version}

    @property
    def model_version(self) -> str:
        a = self._active
        return f"{a['name']}#v{a['ver']}"

    # ── 분류 ────────────────────────────────────────────────────
    def classify(self, source: str, *, is_file: bool = None) -> dict:
        active = self._active                      # 스냅샷(스왑과 무관하게 일관)
        backend = active["backend"]
        t0 = time.perf_counter()

        if is_file is None:
            is_file = ("\n" not in source) and Path(source).exists()
        if is_file:
            text, fmt, _ = dc.extract_text(source)
        else:
            text, fmt = source, "text"

        w = {"tier": self.tier_weights}
        # 티어 1·2(규칙+NER)만 — 뉴럴 미로드
        rules = dc.classify_text(text, locale=self.locale, llm_mode=False,
                                 ensemble_method=self.ensemble, weights=w)
        if self.early_exit and rules["gradeFull"] == "CONFIDENTIAL":
            return self._pack(rules, ["rules", "ner"], True, fmt, active, t0)

        # 티어 3(뉴럴) + 앙상블
        final = dc.classify_text(text, locale=self.locale, llm_mode=True, model=backend,
                                 ensemble_method=self.ensemble, weights=w)
        return self._pack(final, ["rules", "ner", "neural"], False, fmt, active, t0)

    def _pack(self, r, tiers, skipped, fmt, active, t0) -> dict:
        return {
            "grade": r["gradeCode"], "gradeLabel": r["gradeLabel"],
            "confidence": r["confidence"], "score": r["score"],
            "format": fmt, "tiersRun": tiers, "neuralSkipped": skipped,
            "shap": r.get("shap"), "compliance": r.get("compliance"),
            "modelVersion": f"{active['name']}#v{active['ver']}",
            "elapsedMs": int((time.perf_counter() - t0) * 1000),
        }


# ── 편의 함수(1회성) ─────────────────────────────────────────────
_DEFAULT = None
def classify(source: str, model: str = "accurate", **kw) -> dict:
    """단발 분류. 서비스는 N2SFClassifier 인스턴스 재사용 권장."""
    global _DEFAULT
    if _DEFAULT is None or _DEFAULT._active["name"] != model:
        _DEFAULT = N2SFClassifier(model=model, warmup=False)
    return _DEFAULT.classify(source, **kw)


# ── 샘플 프로그램 ────────────────────────────────────────────────
def _demo():
    clf = N2SFClassifier(model="accurate", warmup=False)
    samples = [
        ("개인정보(→S)", "담당자 홍길동, 주민등록번호 900101-1234567, 연락처 010-1234-5678."),
        ("기밀표지(→C)", "[대외비] 사내 전략 검토 자료. 무단 열람·배포 금지."),
        ("공개(→O)", "2024 하반기 공개채용 안내. 지원서는 홈페이지에서 확인하세요."),
    ]
    print(f"# N²SF classifier v3 · model={clf.model_version}\n")
    for name, txt in samples:
        r = clf.classify(txt, is_file=False)
        print(f"[{name}] → {r['grade']}({r['gradeLabel']}) conf={r['confidence']} "
              f"tiers={'>'.join(r['tiersRun'])} skip={r['neuralSkipped']} {r['elapsedMs']}ms")
    # 무중단 핫리로드 데모(모델이 있으면)
    print("\n# 핫리로드 예시:")
    print("  clf.reload_model('n2sf-official')          # 티어 프리셋")
    print("  clf.reload_model('cust-v2', path='models/n2sf-custom-v2')  # 온사이트 학습 결과")


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="N²SF 등급분류 v3")
    ap.add_argument("source", nargs="?", help="문서 경로 또는 --text")
    ap.add_argument("--text")
    ap.add_argument("--model", default="accurate",
                    help="티어(fast-ko/compact-multi/balanced/accurate/n2sf-official) 또는 백엔드명/경로")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo or (not a.source and not a.text):
        _demo()
    else:
        clf = N2SFClassifier(model=a.model, warmup=False)
        src = a.text if a.text is not None else a.source
        r = clf.classify(src, is_file=(a.text is None))
        print(json.dumps(r, ensure_ascii=False, indent=2) if a.json
              else f"{r['grade']} ({r['gradeLabel']}) · conf {r['confidence']} · {r['modelVersion']}")
