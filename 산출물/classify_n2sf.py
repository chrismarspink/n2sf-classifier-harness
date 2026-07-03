#!/usr/bin/env python3
"""classify_n2sf.py — N²SF 등급분류 참조 소스 (외부 LLM·GPU 없이 온디바이스 분류).

이 스크립트는 배포용 "참조 구현"입니다. 핵심 엔진(`data_classifier.py`)을 얇게 감싸,
아래 세 가지 포인트를 실제 코드로 보여줍니다.

  ① 외부 LLM·GPU 불필요 — 추론은 100% 로컬 CPU. (LLM은 학습 단계 증류에만, 추론엔 호출 0)
  ② 티어 early-exit — 정규식(T1)에서 기밀이 확정되면 값비싼 뉴럴(T3)을 아예 수행하지 않음(속도 유리).
  ③ 확장성 — 정확도가 필요하면 --model 로 라지 모델(n2sf-xlmr-large)로 교체, 필요 시 LLM 확장도 가능.
  (설명가능성 SHAP: 결과의 shap 블록 = 어떤 검출이 등급을 얼마나 올렸는지 가법적 기여도)

용도별 모델(--model):
  n2sf-small       Fast     14M/57MB/~13ms   엣지·실시간·저사양
  n2sf-base        Balanced 279M/1.1GB       표준 업무 PC (기본)
  n2sf-klue-large  Korean   337M/1.3GB       한국어 특화 라지
  n2sf-xlmr-large  Accurate 560M/2.3GB       정확도 최우선(0.92)·다국어 강건

사용:
  python classify_n2sf.py 문서.pdf                       # 기본(n2sf-base), early-exit on
  python classify_n2sf.py 문서.docx --model n2sf-xlmr-large
  python classify_n2sf.py --text "홍길동 주민번호 900101-1234567" --json
  python classify_n2sf.py 문서.hwpx --no-early-exit      # 항상 전 티어 수행(비교용)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 엔진 data_classifier.py 위치: 리포 루트(상위 폴더) 또는 Docker(같은 폴더) 모두 지원.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    sys.path.insert(0, str(_p))
import data_classifier as dc  # noqa: E402


# 튜닝된 앙상블 가중치 — 규칙/NER은 floor(안전)로만 낮게, 뉴럴 판단을 주도로.
# held-out 실측: 기본 가중(1.0) 대비 전체 파이프라인 F1 0.54→0.80(base)·0.86(xlmr), C재현율 1.0 유지.
TUNED_TIER_WEIGHTS = {"rules": 0.3, "ner": 0.3, "neural": 4.0}


def classify_with_early_exit(text: str, *, locale: str = "ko",
                             model: str = "n2sf-base",
                             ensemble_method: str = "soft",
                             tier_weights: dict = None,
                             early_exit: bool = True) -> dict:
    """3-tier 분류 + early-exit.

    1) T1/T2(정규식·NER·키워드)만 먼저 수행(llm_mode=False) — 뉴럴 로드/추론 없음.
    2) 규칙만으로 이미 CONFIDENTIAL 확정 → **뉴럴 생략(early-exit)**. 강식별자(주민번호·카드·키)는
       언어무관으로 규칙이 잡으므로, 대부분의 명백한 기밀은 여기서 끝나 속도가 빠르다.
    3) 그 외(OPEN/SENSITIVE 경계)만 뉴럴(T3)을 수행해 최종 앙상블.
    """
    t0 = time.perf_counter()
    weights = {"tier": tier_weights or TUNED_TIER_WEIGHTS}

    # ── 티어 1·2: 규칙 + NER (값싼 검출) ──────────────────────────────
    rules = dc.classify_text(text, locale=locale, llm_mode=False,
                             ensemble_method=ensemble_method, weights=weights)

    if early_exit and rules["gradeFull"] == "CONFIDENTIAL":
        rules["_tiersRun"] = ["rules", "ner"]
        rules["_neuralSkipped"] = True
        rules["_elapsedMs"] = int((time.perf_counter() - t0) * 1000)
        return rules

    # ── 티어 3: 뉴럴 (경계 문서만) ────────────────────────────────────
    final = dc.classify_text(text, locale=locale, llm_mode=True, model=model,
                             ensemble_method=ensemble_method, weights=weights)
    final["_tiersRun"] = ["rules", "ner", "neural"]
    final["_neuralSkipped"] = False
    final["_elapsedMs"] = int((time.perf_counter() - t0) * 1000)
    return final


def _print_human(r: dict, model: str):
    tiers = " → ".join(r.get("_tiersRun", []))
    skipped = "예(early-exit)" if r.get("_neuralSkipped") else "아니오"
    print(f"\n  등급        : {r['grade']} ({r['gradeLabel']})  · 신뢰도 {r['confidence']}")
    print(f"  점수        : {r['score']}")
    print(f"  수행 티어   : {tiers}   · 뉴럴 생략: {skipped}   · 모델: {model}")
    print(f"  소요        : {r.get('_elapsedMs', '?')}ms")
    contribs = (r.get("shap") or {}).get("contributions", [])
    if contribs:
        print("  근거(SHAP 기여도):")
        for c in contribs[:6]:
            print(f"    - {c['label']:<20} 기여 {c['contribution']:>6}  ({int(c['percent']*100)}%)  ×{c['count']}")
    viol = (r.get("compliance") or {}).get("violations", [])
    for v in viol:
        print(f"  ⚠ 컴플라이언스: [{v['code']}] {v['msg']}")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description="N²SF 등급분류 참조 구현(온디바이스, 티어 early-exit)")
    ap.add_argument("file", nargs="?", help="분류할 문서 경로(txt/csv/md/json/docx/xlsx/pptx/hwpx/pdf)")
    ap.add_argument("--text", help="파일 대신 직접 텍스트 분류")
    ap.add_argument("--model", default="n2sf-base",
                    choices=["n2sf-small", "n2sf-base", "n2sf-klue-large", "n2sf-xlmr-large"],
                    help="뉴럴 티어 모델(용도별). 기본 n2sf-base")
    ap.add_argument("--locale", default="ko")
    ap.add_argument("--ensemble", default="soft", help="앙상블 방식(soft 권장)")
    ap.add_argument("--no-early-exit", action="store_true", help="early-exit 끄고 항상 전 티어 수행")
    ap.add_argument("--json", action="store_true", help="원시 JSON 출력")
    args = ap.parse_args(argv)

    if args.text is not None:
        text, fmt = args.text, "text"
    elif args.file:
        text, fmt, _ = dc.extract_text(args.file)
    else:
        ap.error("file 또는 --text 중 하나가 필요합니다")

    r = classify_with_early_exit(text, locale=args.locale, model=args.model,
                                 ensemble_method=args.ensemble,
                                 early_exit=not args.no_early_exit)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        _print_human(r, args.model)
    return r["gradeCode"]


if __name__ == "__main__":
    main()
