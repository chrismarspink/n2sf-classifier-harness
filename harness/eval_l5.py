"""eval_l5.py — 모델별 L5 다국어 누락률 평가(뉴럴-단독).

각 모델을 CPU에서 held-out(한국어)과 heldout_l5(영/중/일)에 평가하고,
**언어별 기밀 누락률(leak rate = C를 C 미만으로 오분류한 비율)** 을 보고한다.
뉴럴-단독 기준(모델 자체 대응력). 실서비스는 T1 정규식 floor가 강식별자를 언어무관으로 잡아 누락을 추가 차단.
"""
from __future__ import annotations

import argparse, json, os
from collections import defaultdict
from pathlib import Path

from . import metrics as M

SHORT = {"OPEN": "O", "SENSITIVE": "S", "CONFIDENTIAL": "C"}
ORDER = {"O": 0, "S": 1, "C": 2}


def _load(path):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path); mdl.eval().to("cpu")
    return tok, mdl


def _predict(tok, mdl, texts, maxlen=256):
    import torch
    id2 = mdl.config.id2label
    def gidx(i):
        lab = id2.get(i, id2.get(str(i), ["OPEN", "SENSITIVE", "CONFIDENTIAL"][i]))
        return SHORT.get(lab, lab)
    preds = []
    with torch.no_grad():
        for t in texts:
            enc = tok([t], truncation=True, max_length=maxlen, return_tensors="pt")
            p = torch.softmax(mdl(**enc).logits[0], -1).tolist()
            by = {gidx(i): p[i] for i in range(len(p))}
            vec = [by.get("O", 0), by.get("S", 0), by.get("C", 0)]
            preds.append(["O", "S", "C"][int(max(range(3), key=lambda k: vec[k]))])
    return preds


def eval_set(tok, mdl, data, maxlen=256):
    preds = _predict(tok, mdl, [d["text"] for d in data], maxlen)
    rows = [{"true_grade": d["grade"], "pred_grade": p} for d, p in zip(data, preds)]
    m = M.compute(rows)
    # 언어별 C 누락률
    per_lang = defaultdict(lambda: {"c_total": 0, "c_leak": 0})
    for d, p in zip(data, preds):
        lang = d.get("lang", "ko")
        if d["grade"] == "C":
            per_lang[lang]["c_total"] += 1
            if ORDER[p] < ORDER["C"]:            # C를 C미만으로 → 누락(유출)
                per_lang[lang]["c_leak"] += 1
    leak = {lg: round(v["c_leak"] / max(v["c_total"], 1), 3) for lg, v in per_lang.items()}
    return m, leak


def main(argv=None):
    ap = argparse.ArgumentParser(description="L5 다국어 누락률 평가")
    ap.add_argument("--models", default="", help="쉼표구분 모델경로(기본: models/n2sf-*)")
    ap.add_argument("--l5", default="distill_soft/heldout_l5.json")
    ap.add_argument("--heldout", default="distill_soft/heldout.json")
    ap.add_argument("--out", default="lineup/l5_matrix.json")
    args = ap.parse_args(argv)

    if args.models:
        paths = [p.strip() for p in args.models.split(",") if p.strip()]
    else:
        paths = sorted(str(p) for p in Path("models").glob("n2sf-*") if (p / "config.json").exists())
    l5 = json.loads(Path(args.l5).read_text(encoding="utf-8"))
    ho = json.loads(Path(args.heldout).read_text(encoding="utf-8")) if Path(args.heldout).exists() else []

    results = {}
    for path in paths:
        name = Path(path).name
        print(f"\n=== {name} 평가 ===")
        tok, mdl = _load(path)
        l5m, leak = eval_set(tok, mdl, l5)
        row = {"l5_macro_f1": l5m["macro_f1"], "l5_c_recall": l5m["c_recall"],
               "l5_under_rate": l5m["under_rate"], "leak_by_lang": leak}
        if ho:
            hom, _ = eval_set(tok, mdl, ho)
            row["ko_macro_f1"] = hom["macro_f1"]; row["ko_c_recall"] = hom["c_recall"]
        results[name] = row
        print(f"  L5 macroF1={l5m['macro_f1']} C재현율={l5m['c_recall']} 언어별누락={leak}")
        if ho:
            print(f"  (한국어 held-out) macroF1={row['ko_macro_f1']} C재현율={row['ko_c_recall']}")
        del mdl, tok

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== L5 평가 완료 → {args.out} ==")


if __name__ == "__main__":
    main()
