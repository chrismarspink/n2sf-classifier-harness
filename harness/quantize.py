"""quantize.py — int8 동적 양자화로 최소 메모리 탐색(크기·정확도 trade-off 측정).

CPU 배포에서 메모리↓ 목적. Linear 레이어를 int8로 동적 양자화 → held-out 정확도·모델 크기 비교.
  python -m harness.quantize --model models/n2sf-base
프로덕션 최적 배포는 ONNX int8(별도) 권장 — 본 스크립트는 "얼마나 줄고 얼마나 손해인가" 실측용.
"""
from __future__ import annotations

import argparse, io, json
from pathlib import Path

import data_classifier as dc
from . import metrics as M
from .eval_l5 import eval_set


def _statedict_mb(model):
    buf = io.BytesIO(); import torch; torch.save(model.state_dict(), buf)
    return round(buf.getbuffer().nbytes / 1e6, 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="int8 동적 양자화 크기·정확도 측정")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ko", default="distill_soft/heldout.json")
    args = ap.parse_args(argv)
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    ko = json.loads(Path(args.ko).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.model).eval().to("cpu")

    fp32_mb = _statedict_mb(mdl)
    m0, _ = eval_set(tok, mdl, ko)
    q = torch.quantization.quantize_dynamic(mdl, {torch.nn.Linear}, dtype=torch.qint8)
    int8_mb = _statedict_mb(q)
    m1, _ = eval_set(tok, q, ko)

    print(f"[quantize] {args.model}")
    print(f"  fp32: {fp32_mb}MB  한국어F1={m0['macro_f1']} C재현율={m0['c_recall']}")
    print(f"  int8: {int8_mb}MB  한국어F1={m1['macro_f1']} C재현율={m1['c_recall']}  (축소 {round(fp32_mb/max(int8_mb,1),1)}x)")


if __name__ == "__main__":
    main()
