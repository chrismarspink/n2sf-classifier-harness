"""synth.py — 포맷·체크섬이 유효한 한국어 합성 PII 생성기.

분류기가 실제로 검출하는 정규식/Presidio 인식기에 맞춰, **포맷·체크섬이 유효한** 값을 만든다.
(예: 주민번호 검증자리, 신용카드 Luhn) — 그래야 "현실적 문서 ↔ 정확한 라벨" 관계가 성립한다.
모든 생성은 seed 고정 random.Random 으로 재현 가능하다.
"""
from __future__ import annotations

import random
from typing import List

# ── 한국어 성씨/이름 음절 (deny-list VIP 와 겹치지 않게) ───────────────────
_SURNAMES = ["강", "권", "김", "남", "노", "문", "박", "배", "백", "서", "송", "신",
             "안", "양", "오", "유", "윤", "임", "장", "전", "정", "조", "차", "최", "한", "황"]
_GIVEN = ["민준", "서연", "도윤", "지우", "예준", "하은", "주원", "지호", "서준", "수아",
          "지훈", "현우", "은서", "건우", "민서", "유진", "성민", "다은", "준영", "소율"]
VIP_NAMES = ["홍길동", "김철수", "이영희"]          # 분류기 deny-list 와 일치
INTERNAL_PROJECTS = ["ProjectAlpha", "ProjectOmega", "프로젝트 사일런스"]

_BANKS = ["국민", "신한", "우리", "하나", "농협", "기업", "카카오뱅크", "토스뱅크"]
_REGION = ["서울특별시", "부산광역시", "경기도", "인천광역시", "대전광역시"]
_SIGU = {"서울특별시": ["강남구", "서초구", "송파구", "마포구", "종로구"],
         "부산광역시": ["해운대구", "수영구", "남구", "동래구"],
         "경기도": ["성남시", "수원시", "용인시", "고양시"],
         "인천광역시": ["연수구", "남동구", "부평구"],
         "대전광역시": ["유성구", "서구", "중구"]}
_DONG = ["역삼동", "삼성동", "정자동", "서현동", "신촌동", "송도동", "둔산동"]
_ROADS = ["테헤란로", "강남대로", "올림픽로", "세종대로", "월드컵로"]


class Synth:
    """seed 고정 합성 PII 생성기."""

    def __init__(self, seed: int = 0):
        self.r = random.Random(seed)

    # ── 이름 ─────────────────────────────────────────────────────────────
    def name(self) -> str:
        return self.r.choice(_SURNAMES) + self.r.choice(_GIVEN)

    def vip(self) -> str:
        return self.r.choice(VIP_NAMES)

    # ── 주민등록번호 (검증자리 유효) ─────────────────────────────────────
    def rrn(self) -> str:
        yy = self.r.randint(60, 99)
        mm = self.r.randint(1, 12)
        dd = self.r.randint(1, 28)
        gender = self.r.choice([1, 2])           # 1900년대 출생
        mid = self.r.randint(0, 999999)
        digits = [int(c) for c in f"{yy:02d}{mm:02d}{dd:02d}{gender}{mid:06d}"]
        weights = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
        check = (11 - sum(d * w for d, w in zip(digits, weights)) % 11) % 10
        front = f"{yy:02d}{mm:02d}{dd:02d}"
        back = f"{gender}{mid:06d}{check}"
        return f"{front}-{back}"

    # ── 휴대폰 ──────────────────────────────────────────────────────────
    def phone(self) -> str:
        return f"010-{self.r.randint(1000, 9999)}-{self.r.randint(1000, 9999)}"

    # ── 사업자등록번호 ──────────────────────────────────────────────────
    def biz_no(self) -> str:
        return f"{self.r.randint(100, 999)}-{self.r.randint(10, 99)}-{self.r.randint(10000, 99999)}"

    # ── 계좌번호 ────────────────────────────────────────────────────────
    def account(self) -> str:
        bank = self.r.choice(_BANKS)
        acc = f"{self.r.randint(100, 999)}-{self.r.randint(100, 9999)}-{self.r.randint(100000, 9999999)}"
        return f"{bank} {acc}"

    # ── 신용카드 (Luhn 유효) ─────────────────────────────────────────────
    def credit_card(self) -> str:
        prefix = self.r.choice(["4", "51", "52", "53", "54", "55"])
        body = prefix + "".join(str(self.r.randint(0, 9)) for _ in range(15 - len(prefix)))
        digits = [int(c) for c in body]
        s = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 0:
                d *= 2
                if d > 9:
                    d -= 9
            s += d
        check = (10 - s % 10) % 10
        num = body + str(check)
        return "-".join(num[i:i + 4] for i in range(0, 16, 4))

    # ── 여권번호 ────────────────────────────────────────────────────────
    def passport(self) -> str:
        return self.r.choice("MSRO") + "".join(str(self.r.randint(0, 9)) for _ in range(8))

    # ── 이메일 ──────────────────────────────────────────────────────────
    def email(self) -> str:
        user = "".join(self.r.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(self.r.randint(5, 9)))
        dom = self.r.choice(["gmail.com", "naver.com", "daum.net", "company.co.kr", "kakao.com"])
        return f"{user}{self.r.randint(1, 99)}@{dom}"

    # ── 주소 (분류기 KR_ADDRESS 정규식 매칭) ─────────────────────────────
    def address(self) -> str:
        region = self.r.choice(_REGION)
        gu = self.r.choice(_SIGU[region])
        dong = self.r.choice(_DONG)
        return f"{region} {gu} {dong} {self.r.randint(1, 300)}-{self.r.randint(1, 99)}"

    def address_road(self) -> str:
        region = self.r.choice(_REGION)
        gu = self.r.choice(_SIGU[region])
        road = self.r.choice(_ROADS)
        return f"{region} {gu} {road}"

    # ── 금액 ────────────────────────────────────────────────────────────
    def money(self) -> str:
        won = self.r.choice([1_000_000, 5_000_000, 12_500_000, 30_000_000, 250_000])
        return f"{won:,}원"

    # ── API/시크릿 키 ───────────────────────────────────────────────────
    def api_key(self) -> str:
        body = "".join(self.r.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
                       for _ in range(self.r.randint(28, 40)))
        return f"api_key={body}"

    def aws_key(self) -> str:
        body = "".join(self.r.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567") for _ in range(16))
        return "AKIA" + body

    # ── 비식별 잡음 (오탐 유발용: 13자리 제품코드 등) ─────────────────────
    def product_code(self) -> str:
        return "P" + "".join(str(self.r.randint(0, 9)) for _ in range(self.r.randint(10, 13)))

    def order_no(self) -> str:
        return f"ORD-{self.r.randint(2020, 2026)}-{self.r.randint(100000, 999999)}"

    # ── 적대(adversarial) 케이스 ────────────────────────────────────────
    def near_miss_rrn(self) -> str:
        """주민번호 형식이지만 검증자리가 틀린 값(실제 PII 아님 → 정밀도 트랩)."""
        r = self.rrn()
        wrong = str((int(r[-1]) + 1) % 10)
        return r[:-1] + wrong

    def decoy_card_number(self) -> str:
        """Luhn 유효 16자리지만 제품/상품 코드 문맥에서 쓰이는 미끼(정밀도 트랩)."""
        return self.credit_card()

    def spaced_keyword(self, kw: str) -> str:
        """'대외비' → '대 외 비' 공백 난독화(키워드 substring 매칭 회피 → 재현율 트랩)."""
        return " ".join(list(kw))

    def fullwidth_digits(self, s: str) -> str:
        """ASCII 숫자를 전각으로(정규식 회피)."""
        tbl = {ord(str(d)): chr(0xFF10 + d) for d in range(10)}
        return s.translate(tbl)

    def filler(self, n: int = 1) -> List[str]:
        pool = [
            "본 안내는 사내 전 구성원에게 공유됩니다.",
            "관련 절차는 표준 운영 지침을 따릅니다.",
            "문의 사항은 담당 부서로 전달해 주시기 바랍니다.",
            "세부 일정은 추후 별도 공지될 예정입니다.",
            "본 문서의 효력은 게시일로부터 발생합니다.",
            "이전 버전과의 차이는 변경 이력에 정리되어 있습니다.",
            "관련 양식은 사내 포털에서 내려받을 수 있습니다.",
        ]
        return [self.r.choice(pool) for _ in range(n)]
