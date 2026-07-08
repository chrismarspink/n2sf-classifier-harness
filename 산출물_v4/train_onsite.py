#!/usr/bin/env python3
"""train_onsite.py — 고객 현장(on-site) 라벨 데이터로 모델 신규 학습 (v4, v3 준용).

실데이터 부족 문제를, 고객이 자기 문서에 라벨한 데이터로 **현장에서 직접 재학습**해 해소한다.
- 입력: 고객 라벨 CSV/JSONL (text + label[C/S/O]). (골드셋 CSV 포맷 호환: human_label 컬럼)
- 처리: 라벨→확률(라벨 스무딩) 변환, 선택적으로 기존 증류셋과 혼합(파국적 망각 방지),
        기존 배포 모델에서 **이어서 학습**(전이) → 새 버전 모델 저장.
- 실행: 100% 로컬(CPU/MPS), 외부 호출 0. 웹 UI에서 호출 가능(진행 콜백·상태파일).
- 학습 후: classifier_v4.N2SFExplainClassifier.reload_model(name, path=out) 로 **무중단 교체**.

※ 학습 자체는 리포 루트의 harness/train_kd.py 를 subprocess 로 호출한다(증류 학습기 재사용).
   폐쇄망 배포 시 harness/ 도 함께 반입해야 재학습 가능(추론은 classifier_v4.py 단일 파일로 충분).

CLI:
  python train_onsite.py --labels customer_labels.csv --base models/n2sf-xlmr-official \
      --out models/n2sf-custom-v2 --mix distill_o4/teacher_labels_n2sf.jsonl --mix-ratio 0.5
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
GRADE_IDX = {"O": 0, "S": 1, "C": 2, "OPEN": 0, "SENSITIVE": 1, "CONFIDENTIAL": 2}


def read_labels(path: str):
    """CSV(text,human_label|label,grade) 또는 JSONL({text,label|grade|human_label}) → [(text, grade)]."""
    p = Path(path); rows = []
    if p.suffix.lower() in (".csv", ".tsv"):
        delim = "\t" if p.suffix.lower() == ".tsv" else ","
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f, delimiter=delim):
                lab = (r.get("human_label") or r.get("label") or r.get("grade") or "").strip().upper()
                txt = (r.get("text") or "").strip()
                if txt and lab and lab[0] in "OSC":
                    rows.append((txt, lab[0]))
    else:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            lab = str(o.get("label") or o.get("grade") or o.get("human_label") or "").strip().upper()
            txt = (o.get("text") or "").strip()
            if txt and lab and lab[0] in "OSC":
                rows.append((txt, lab[0]))
    return rows


def to_soft(grade: str, smoothing: float = 0.1):
    """하드 라벨 C/S/O → 확률벡터(라벨 스무딩). 스무딩은 과신 방지·일반화."""
    p = [smoothing / 3] * 3
    p[GRADE_IDX[grade]] += 1 - smoothing
    s = sum(p)
    return [x / s for x in p]


def prepare(labels_path, out_jsonl, smoothing=0.1, mix=None, mix_ratio=0.5, progress=None):
    """고객 라벨 → soft-jsonl. mix(기존 증류셋) 지정 시 안정화를 위해 섞음."""
    rows = read_labels(labels_path)
    if not rows:
        raise ValueError(f"유효 라벨 0건: {labels_path} (text + C/S/O 라벨 필요)")
    out = []
    for txt, g in rows:
        out.append({"text": txt[:2000], "probs": to_soft(g, smoothing), "src": "customer"})
    n_cust = len(out)
    if mix and Path(mix).exists() and mix_ratio > 0:
        base = [json.loads(l) for l in open(mix, encoding="utf-8") if l.strip()]
        import random
        random.Random(0).shuffle(base)
        take = int(n_cust * mix_ratio / max(1 - mix_ratio, 1e-6))
        out += [{"text": b["text"], "probs": b["probs"], "src": "base"} for b in base[:take] if b.get("probs")]
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as w:
        for r in out:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    if progress:
        progress({"stage": "prepared", "customer": n_cust, "total": len(out)})
    return {"customer": n_cust, "total": len(out), "jsonl": out_jsonl}


def train_onsite(labels_path, base="models/n2sf-xlmr-official", out="models/n2sf-custom-v1",
                 mix="distill_o4/teacher_labels_n2sf.jsonl", mix_ratio=0.5, smoothing=0.1,
                 epochs=3, batch=6, accum=3, maxlen=128, train_top=6, max_hours=2.0,
                 progress=None) -> dict:
    """현장 재학습 실행. 기존 배포 모델(base)에서 이어 학습 → 새 버전(out).
    progress(dict) 콜백으로 웹 UI에 단계 통지. 반환: {model_dir, ...}."""
    started = time.time()
    prep = prepare(labels_path, f"{out}.train.jsonl", smoothing, mix, mix_ratio, progress)
    if progress:
        progress({"stage": "training", **prep})
    cmd = [sys.executable, "-m", "harness.train_kd", "--soft-jsonl", f"{out}.train.jsonl",
           "--base", base, "--out", out, "--epochs", str(epochs), "--batch", str(batch),
           "--accum", str(accum), "--maxlen", str(maxlen), "--temperature", "1.5",
           "--alpha", "0.1", "--balance", "--train-top", str(train_top),
           "--lr", "2e-5", "--warmup", "0.1", "--max-hours", str(max_hours)]
    # 로그를 실시간으로 진행 콜백에 흘림
    proc = subprocess.Popen(cmd, cwd=str(_ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = {}
    for line in proc.stdout:
        line = line.rstrip()
        if "held-out F1=" in line and progress:
            progress({"stage": "epoch", "log": line})
        last["log"] = line
    proc.wait()
    ok = Path(f"{out}/config.json").exists()
    res = {"ok": ok, "model_dir": out, "customer_labels": prep["customer"],
           "train_size": prep["total"], "elapsed_s": int(time.time() - started),
           "next": f"clf.reload_model('{Path(out).name}', path='{out}')"}
    if progress:
        progress({"stage": "done", **res})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="온사이트 고객 라벨 재학습 (v4)")
    ap.add_argument("--labels", required=True, help="고객 라벨 CSV/JSONL (text + C/S/O)")
    ap.add_argument("--base", default="models/n2sf-xlmr-official", help="이어서 학습할 기존 모델(전이)")
    ap.add_argument("--out", default="models/n2sf-custom-v1")
    ap.add_argument("--mix", default="distill_o4/teacher_labels_n2sf.jsonl", help="안정화용 기존 증류셋(빈값=미혼합)")
    ap.add_argument("--mix-ratio", type=float, default=0.5)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-hours", type=float, default=2.0)
    a = ap.parse_args()
    r = train_onsite(a.labels, a.base, a.out, a.mix or None, a.mix_ratio, a.smoothing,
                     a.epochs, max_hours=a.max_hours, progress=lambda d: print("[진행]", d, flush=True))
    print(json.dumps(r, ensure_ascii=False, indent=2))
