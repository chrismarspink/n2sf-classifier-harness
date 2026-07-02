"""distill.py — 진짜 soft-label 지식증류 무인 루프.

교사(GPT-4o)가 다양한 문서에 O/S/C **확률분포**를 매기고 → 학생(mdeberta-n2sf)이 이를 KD로 모방.
우리 주입 라벨이 아닌 **LLM의 결정경계**를 학습 → held-out(다양 분포) 일반화 향상이 목표.
평가·held-out·LLM 기준선은 improve 모듈을 재사용(동결 셋 재사용).

  python -m harness.distill --out distill_soft --hours 12 --label-per-cycle 400
"""
from __future__ import annotations

import argparse, datetime as _dt, json, os, subprocess, sys, time, hashlib, re
from pathlib import Path

import data_classifier as dc
from . import improve as IMP
from .gen_llm import _azure_client, _load_env, generate, LLMFiller
from .corpus import _flat_text
from .llm_baseline import POLICY

GRADES = ["O", "S", "C"]


def _hash(t): return hashlib.md5(re.sub(r"\s+", " ", t).strip().encode()).hexdigest()


TEACHER = "gpt-4o"   # main에서 설정


def teacher_soft(text, deployment=None, apiver=None):
    """교사 LLM으로 O/S/C 확률분포 [pO,pS,pC] 반환. TEACHER=gpt-4o|o4-mini. 실패 시 None."""
    _load_env()
    if TEACHER == "o4-mini":
        deployment = os.environ.get("AZURE_O4MINI_DEPLOYMENT", "o4-mini")
        apiver = os.environ.get("AZURE_O4MINI_APIVER", "2025-04-01-preview")
    else:
        deployment = deployment or os.environ.get("AZURE_GPT4O_DEPLOYMENT", "gpt-4o")
        apiver = apiver or os.environ.get("AZURE_GPT4O_APIVER", "2025-01-01-preview")
    try:
        c = _azure_client(apiver)
        user = (POLICY + f"\n\n<문서>\n{text[:6000]}\n</문서>\n\n"
                "이 문서가 각 등급일 확률을 추정해 JSON으로만 답하라. "
                '형식: {"O": 0.x, "S": 0.x, "C": 0.x} (합 1.0). 다른 말 금지.')
        kw = {"model": deployment, "messages": [{"role": "user", "content": user}]}
        if deployment.startswith(("o1", "o3", "o4")):
            kw["max_completion_tokens"] = 2000
        else:
            kw["max_tokens"] = 40; kw["temperature"] = 0
        r = c.chat.completions.create(**kw)
        txt = r.choices[0].message.content
        m = re.search(r"\{.*\}", txt, re.S)
        d = json.loads(m.group(0))
        p = [float(d.get("O", 0)), float(d.get("S", 0)), float(d.get("C", 0))]
        return p if sum(p) > 0 else None
    except Exception:
        return None


def load_pool(out: Path):
    """교사 라벨링 대상 다양화 텍스트 풀(기존 distill/train_llm.jsonl + out/gen 축적)."""
    texts, seen = [], set()
    for src in [Path("distill/train_llm.jsonl"), out / "gen_texts.jsonl"]:
        if src.exists():
            for line in open(src, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line).get("text", "")
                except Exception:
                    continue
                h = _hash(t)
                if t and len(t) > 15 and h not in seen:
                    seen.add(h); texts.append(t)
    return texts


def gen_more(out: Path, per: int, cycle: int, log):
    """풀 고갈 시 다양화 문서 추가 생성(Azure) → out/gen_texts.jsonl (교사 라벨링용)."""
    f = LLMFiller(seed=5000 + cycle)
    added = 0
    with open(out / "gen_texts.jsonl", "a", encoding="utf-8") as w:
        for g in GRADES:
            for d in (1, 2, 3, 4):
                for _ in range(per):
                    gen = "azure:gpt-4o" if added % 2 == 0 else "azure:o4-mini"
                    doc = generate(gen, g, d, f"G-{cycle}-{added}", f)
                    if doc:
                        w.write(json.dumps({"text": _flat_text(doc)}, ensure_ascii=False) + "\n"); added += 1
    log(f"  다양화 문서 +{added} 생성(gen_texts.jsonl)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="soft-label 지식증류 루프")
    ap.add_argument("--out", default="distill_soft")
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--label-per-cycle", type=int, default=400, help="사이클당 교사 라벨링 문서 수")
    ap.add_argument("--train-epochs", type=int, default=3)
    ap.add_argument("--train-hours", type=float, default=0.6)
    ap.add_argument("--teacher", choices=["gpt-4o", "o4-mini"], default="gpt-4o")
    args = ap.parse_args(argv)
    global TEACHER
    TEACHER = args.teacher

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = print
    deadline = time.time() + args.hours * 3600
    log(f"== distill(soft-label) 시작 out={out} 예산={args.hours}h ==")

    # 동결 held-out + LLM 기준선 재사용(improve 모듈)
    heldout = IMP.build_heldout(out, log=log)
    baselines = IMP.llm_baselines(out, heldout, log=log)

    tl_fp = out / "teacher_labels.jsonl"
    labeled = set()
    if tl_fp.exists():
        for line in open(tl_fp, encoding="utf-8"):
            try:
                labeled.add(_hash(json.loads(line)["text"]))
            except Exception:
                pass
    records = []
    rec_fp = out / "cycles.json"
    if rec_fp.exists():
        records = json.loads(rec_fp.read_text(encoding="utf-8"))
    cycle = len(records)

    while time.time() < deadline:
        c0 = time.time()
        log(f"\n=== 증류 사이클 {cycle} === {_dt.datetime.now().strftime('%H:%M')}")
        # 1) 교사 라벨링 배치
        pool = [t for t in load_pool(out) if _hash(t) not in labeled]
        if len(pool) < args.label_per_cycle:
            gen_more(out, per=8, cycle=cycle, log=log)
            pool = [t for t in load_pool(out) if _hash(t) not in labeled]
        batch = pool[:args.label_per_cycle]
        got = 0
        with open(tl_fp, "a", encoding="utf-8") as w:
            for t in batch:
                p = teacher_soft(t)
                if p:
                    w.write(json.dumps({"text": t, "probs": p}, ensure_ascii=False) + "\n")
                    labeled.add(_hash(t)); got += 1
        total_labeled = len(labeled)
        log(f"  교사 라벨 +{got} (누적 {total_labeled})")
        if total_labeled < 50:
            log("  라벨 부족 — 대기 후 재시도"); time.sleep(30); continue
        # 2) 학생 KD 학습
        log("  학생 KD 학습(mdeberta-n2sf)…")
        cmd = [sys.executable, "-m", "harness.train_soft", "--soft-jsonl", str(tl_fp),
               "--out", "models/mdeberta-n2sf", "--epochs", str(args.train_epochs),
               "--batch", "16", "--accum", "2", "--maxlen", "128", "--max-hours", str(args.train_hours)]
        try:
            subprocess.run(cmd, check=False, timeout=args.train_hours * 3600 + 600)
        except Exception as exc:
            log(f"  학습 경고: {str(exc)[:120]}")
        dc._NEURAL_MODELS.pop("mdeberta-n2sf", None)
        # 3) held-out 평가(후보 전종) + 기록 + 리포트
        log("  held-out 평가…")
        models_m = IMP.eval_candidates(heldout)
        for m, mm in models_m.items():
            log(f"    {m:14} F1={mm['macro_f1']} Crec={mm['c_recall']} under={mm['under_rate']} over={mm['over_rate']}")
        records.append({"cycle": cycle, "ts": _dt.datetime.now().isoformat(),
                        "labeled": total_labeled, "models": models_m,
                        "elapsed_s": round(time.time() - c0, 1)})
        rec_fp.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        try:
            IMP.write_report(out, records, baselines, log=log)
        except Exception as exc:
            log(f"  리포트 경고: {str(exc)[:120]}")
        log(f"  사이클 {cycle} 완료 ({records[-1]['elapsed_s']:.0f}s, 라벨 {total_labeled})")
        cycle += 1

    log(f"== distill 종료. 사이클 {cycle}, 교사라벨 {len(labeled)}. 결과 {out}/report.html ==")


if __name__ == "__main__":
    main()
