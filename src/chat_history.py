"""사용자 대화 기록 SQLite 저장소."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "chat_history.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS chat_logs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id            TEXT    NOT NULL,
    timestamp          TEXT    NOT NULL,
    question           TEXT    NOT NULL,
    answer             TEXT    NOT NULL,
    legal_category     TEXT    NOT NULL DEFAULT '',
    step_reached       INTEGER NOT NULL DEFAULT 0,
    answer_reliability REAL,
    qafs               REAL,
    rrs                REAL
);
CREATE INDEX IF NOT EXISTS idx_user_category ON chat_logs (user_id, legal_category);
CREATE INDEX IF NOT EXISTS idx_user_time     ON chat_logs (user_id, timestamp DESC);
"""


class ChatHistoryStore:
    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_CREATE_SQL)
                conn.commit()
            finally:
                conn.close()

    def save(
        self,
        user_id: str,
        question: str,
        answer: str,
        legal_category: str = "",
        step_reached: int = 0,
        answer_reliability: Optional[float] = None,
        qafs: Optional[float] = None,
        rrs: Optional[float] = None,
    ) -> None:
        """대화 1건을 저장합니다. 실패해도 파이프라인에 영향을 주지 않습니다."""
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO chat_logs
                        (user_id, timestamp, question, answer, legal_category,
                         step_reached, answer_reliability, qafs, rrs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, timestamp, question, answer, legal_category,
                     step_reached, answer_reliability, qafs, rrs),
                )
                conn.commit()
            finally:
                conn.close()

    def get_recent_by_category(
        self,
        user_id: str,
        legal_category: str,
        limit: int = 5,
    ) -> list[dict]:
        """같은 카테고리의 최근 대화를 최신순으로 반환합니다 (2단계 히스토리 반영용)."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT timestamp, question, answer, legal_category,
                           answer_reliability, qafs, rrs
                    FROM   chat_logs
                    WHERE  user_id = ? AND legal_category = ?
                    ORDER  BY timestamp DESC
                    LIMIT  ?
                    """,
                    (user_id, legal_category, limit),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()


chat_history = ChatHistoryStore()
