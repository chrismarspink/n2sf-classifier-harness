"""db.py — SQLite 결과 저장소. 코퍼스·탐지 캐시·설정·예측·지표·LLM 비교를 담는다.

비싼 탐지(추출+NER+뉴럴)는 detection 테이블에 1회 캐시하고, 점수 설정 스윕은 그 위에서
수천 번 돈다. 모든 실행이 비교·재현 가능하도록 run/config 단위로 기록한다.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    created_at    TEXT,
    notes         TEXT,
    corpus_seed   INTEGER,
    per_cell      INTEGER,
    n_docs        INTEGER,
    n_files       INTEGER,
    meta          TEXT
);
CREATE TABLE IF NOT EXISTS corpus (
    doc_id        TEXT,
    fmt           TEXT,
    path          TEXT,
    grade         TEXT,           -- 정책 라벨 O/S/C
    category      TEXT,           -- normal | hard_neg | format_stress
    locale        TEXT,
    split         TEXT,           -- train | valid | test
    text_len      INTEGER,
    render_error  TEXT,
    expected      TEXT,           -- 주입 PII (감사)
    PRIMARY KEY (doc_id, fmt)
);
CREATE TABLE IF NOT EXISTS detection (
    doc_id        TEXT,
    fmt           TEXT,
    det_config    TEXT,           -- 탐지 설정 키 (기본 'base')
    locale        TEXT,
    text          TEXT,
    extract_ms    REAL,
    detect_ms     REAL,           -- presidio analyze 시간
    presidio_json TEXT,           -- 원시 findings 리스트
    neural_json   TEXT,           -- {model_key: result, ...}
    warnings      TEXT,
    PRIMARY KEY (doc_id, fmt, det_config)
);
CREATE TABLE IF NOT EXISTS configs (
    config_id     TEXT PRIMARY KEY,
    run_id        TEXT,
    kind          TEXT,           -- baseline | sweep | best | llm
    label         TEXT,
    params        TEXT            -- 점수/탐지/모델/앙상블 설정 JSON
);
CREATE TABLE IF NOT EXISTS predictions (
    run_id        TEXT,
    config_id     TEXT,
    doc_id        TEXT,
    fmt           TEXT,
    true_grade    TEXT,
    pred_grade    TEXT,
    score         REAL,
    confidence    REAL,
    elapsed_ms    REAL,
    split         TEXT,
    PRIMARY KEY (config_id, doc_id, fmt)
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id        TEXT,
    config_id     TEXT,
    split         TEXT,
    objective     REAL,           -- 안전성 제약 반영 종합 점수
    macro_f1      REAL,
    accuracy      REAL,
    c_recall      REAL,
    c_precision   REAL,
    under_rate    REAL,           -- 과소분류율 (실제 등급 > 예측)
    over_rate     REAL,           -- 과대분류율 (실제 등급 < 예측)
    p50_ms        REAL,
    p95_ms        REAL,
    detail        TEXT,           -- 전체 지표 JSON (혼동행렬·포맷별 등)
    PRIMARY KEY (config_id, split)
);
CREATE TABLE IF NOT EXISTS iterations (
    run_id        TEXT PRIMARY KEY,   -- 예: 'L2i7'
    level         INTEGER,
    iteration     INTEGER,
    seed          INTEGER,
    n_docs        INTEGER,
    n_files       INTEGER,
    best_config   TEXT,
    best_model    TEXT,
    best_ensemble TEXT,
    valid_obj     REAL,
    test_macro_f1 REAL,
    test_accuracy REAL,
    test_c_recall REAL,
    test_under    REAL,
    test_over     REAL,
    rules_only_f1 REAL,               -- 뉴럴 미사용 최적(빠른 경로) 참고
    per_model     TEXT,               -- {model: best_valid_obj}
    elapsed_s     REAL,
    created_at    TEXT
);
CREATE TABLE IF NOT EXISTS llm_predictions (
    doc_id        TEXT,
    fmt           TEXT,
    model         TEXT,
    true_grade    TEXT,
    llm_grade     TEXT,
    latency_ms    REAL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    raw           TEXT,
    PRIMARY KEY (doc_id, fmt, model)
);
"""


class DB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

    # ── 일반 헬퍼 ────────────────────────────────────────────────────────
    def upsert(self, table: str, row: Dict[str, Any]):
        cols = list(row.keys())
        ph = ",".join("?" for _ in cols)
        sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({ph})"
        self.conn.execute(sql, [self._enc(row[c]) for c in cols])

    def upsert_many(self, table: str, rows: Iterable[Dict[str, Any]]):
        rows = list(rows)
        if not rows:
            return
        cols = list(rows[0].keys())
        ph = ",".join("?" for _ in cols)
        sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({ph})"
        self.conn.executemany(sql, [[self._enc(r[c]) for c in cols] for r in rows])
        self.conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    @staticmethod
    def _enc(v):
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return v

    @staticmethod
    def jload(s: Optional[str], default=None):
        if not s:
            return default
        try:
            return json.loads(s)
        except Exception:
            return default
