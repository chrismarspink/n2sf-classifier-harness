"""gen_l5.py — L5 다국어 적대공격 평가셋 생성(비한국어 누락률 측정용).

목적: 기밀/민감 내용을 **영어·중국어·일본어**로 표현했을 때 분류기가 누락(유출)하는지 측정.
 - 강식별자(주민번호·카드·API/AWS키·여권)는 언어 무관 → T1 정규식이 잡아야 함(안전 하한).
 - 다국어 기밀표지(机密/極秘/CONFIDENTIAL 등) + 외국어 산문 → 뉴럴/키워드의 비한국어 대응력 시험.

라벨은 정책 기반(분류기 규칙 아님):
 - C: 강식별자 or 최고기밀표지+내부프로젝트
 - S: 단일 개인정보(이메일/전화) + 민감맥락
 - O: 개인정보·기밀표지 없음(순수 안내문)

네트워크·GPU 불필요. 결정론적(seed). 출력: distill_soft/heldout_l5.json  (heldout.json과 동일 스키마 + lang)
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

from .synth import Synth

# 언어별 기밀/민감/공개 표지·문구
MARK = {
    "en": {
        "c": ["CONFIDENTIAL — Internal Use Only", "TOP SECRET", "Do Not Distribute Externally"],
        "s": ["For internal review", "Personal data enclosed", "Handle with care"],
        "ctx_c": "This memo contains restricted material related to project {proj}. Unauthorized disclosure is prohibited.",
        "ctx_s": "Please find below the contact details for the applicant handled by HR.",
        "ctx_o": "This newsletter shares our quarterly community activities and upcoming public events open to everyone.",
        "id_lead": "Attached identifier: ", "contact_lead": "Contact: ",
        "filler": "The following section summarizes background information and next steps for the team.",
    },
    "zh": {
        "c": ["机密 — 仅限内部使用", "绝密", "严禁外传"],
        "s": ["内部审阅", "含个人信息", "请谨慎处理"],
        "ctx_c": "本备忘录包含与项目{proj}相关的受限资料，未经授权不得披露。",
        "ctx_s": "以下为人力资源部处理的申请人联系方式。",
        "ctx_o": "本通讯分享我们本季度的社区活动以及面向公众的公开活动信息。",
        "id_lead": "所附标识：", "contact_lead": "联系方式：",
        "filler": "以下部分概述了团队的背景信息与后续步骤。",
    },
    "ja": {
        "c": ["極秘 — 社内限り", "取扱注意", "社外秘"],
        "s": ["社内確認用", "個人情報を含む", "取り扱い注意"],
        "ctx_c": "本メモはプロジェクト{proj}に関する制限資料を含みます。無断開示を禁じます。",
        "ctx_s": "以下は人事部が扱う応募者の連絡先です。",
        "ctx_o": "本ニュースレターは四半期の地域活動と一般公開イベントの情報をお届けします。",
        "id_lead": "添付識別子：", "contact_lead": "連絡先：",
        "filler": "次のセクションでは、チームの背景情報と今後のステップを要約します。",
    },
}


def _conf_doc(s: Synth, m: dict, lang: str) -> str:
    # 강식별자 1~2개(언어무관) + 최고기밀표지 + 내부 프로젝트
    idv = s.r.choice([s.rrn(), s.credit_card(), s.aws_key(), s.api_key(), s.passport()])
    idv2 = s.r.choice([s.rrn(), s.credit_card()])
    proj = s.r.choice(["Alpha", "Omega", "Silence"])
    return (f"{s.r.choice(m['c'])}\n\n{m['ctx_c'].format(proj=proj)}\n"
            f"{m['id_lead']}{idv}\n{m['id_lead']}{idv2}\n{m['filler']}")


def _sens_doc(s: Synth, m: dict, lang: str) -> str:
    contact = s.r.choice([s.email(), s.phone()])
    return (f"{s.r.choice(m['s'])}\n\n{m['ctx_s']}\n{m['contact_lead']}{contact}\n{m['filler']}")


def _open_doc(s: Synth, m: dict, lang: str) -> str:
    return f"{m['ctx_o']}\n{m['filler']}\n{m['filler']}"


MAKERS = {"C": _conf_doc, "S": _sens_doc, "O": _open_doc}


def build(per_cell: int = 8, seed: int = 7):
    s = Synth(seed)
    out = []
    for lang, m in MARK.items():
        for grade in ("O", "S", "C"):
            for i in range(per_cell):
                text = MAKERS[grade](s, m, lang)
                out.append({"text": text, "grade": grade, "gen": f"template:{lang}",
                            "diff": 5, "lang": lang})
    s.r.shuffle(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="L5 다국어 적대공격 평가셋 생성")
    ap.add_argument("--per-cell", type=int, default=8, help="언어×등급 셀당 문서 수")
    ap.add_argument("--out", default="distill_soft/heldout_l5.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    data = build(args.per_cell, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter
    lc = Counter((d["lang"], d["grade"]) for d in data)
    print(f"[gen_l5] {len(data)}건 생성 → {args.out}")
    print(f"[gen_l5] 언어×등급 분포: {dict(lc)}")


if __name__ == "__main__":
    main()
