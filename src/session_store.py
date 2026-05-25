"""
세션 저장소
카카오톡 user_id 기반으로 대화 상태를 관리한다.
인메모리를 기본으로 하고, SQLite 파일에 백업하여 서버 재시작 후에도 세션이 유지된다.
"""

import dataclasses
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import config

logger = logging.getLogger("lawsguard.session")

_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "sessions.db"


@dataclass
class ClarificationSession:
    original_question: str = ""
    accumulated_context: str = ""
    retry_count: int = 0
    last_score: float = 0.0
    is_complete: bool = False
    use_general_answer: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # remember last requery to prevent repeating identical follow-ups
    last_requery_message: str = ""
    last_missing_elements: list[str] = field(default_factory=list)
    last_requery_topic: str = ""


def _session_to_dict(s: ClarificationSession) -> dict:
    return dataclasses.asdict(s)


def _session_from_dict(d: dict) -> ClarificationSession:
    known = {f.name for f in dataclasses.fields(ClarificationSession)}
    return ClarificationSession(**{k: v for k, v in d.items() if k in known})


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, ClarificationSession] = {}
        self._lock = threading.Lock()
        self._ttl = config.session_ttl_sec
        self._db: Optional[sqlite3.Connection] = None
        self._init_db()
        self._load_from_db()

    # ── SQLite 초기화 ────────────────────────────────────────────────────────

    def _init_db(self):
        try:
            _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS sessions "
                "(user_id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at REAL NOT NULL)"
            )
            self._db.commit()
        except Exception as e:
            logger.warning("SQLite 세션 DB 초기화 실패 — 인메모리 전용: %s", e)
            self._db = None

    def _load_from_db(self):
        if self._db is None:
            return
        now = time.time()
        try:
            rows = self._db.execute(
                "SELECT user_id, data FROM sessions WHERE updated_at > ?",
                (now - self._ttl,),
            ).fetchall()
            for user_id, data in rows:
                try:
                    session = _session_from_dict(json.loads(data))
                    self._sessions[user_id] = session
                except Exception:
                    pass
            # 만료 행 삭제
            self._db.execute("DELETE FROM sessions WHERE updated_at <= ?", (now - self._ttl,))
            self._db.commit()
            if rows:
                logger.info("세션 DB에서 %d개 세션 복구", len(rows))
        except Exception as e:
            logger.warning("세션 DB 로드 실패: %s", e)

    def _save_to_db(self, user_id: str, session: ClarificationSession):
        if self._db is None:
            return
        try:
            data = json.dumps(_session_to_dict(session), ensure_ascii=False)
            self._db.execute(
                "INSERT OR REPLACE INTO sessions (user_id, data, updated_at) VALUES (?, ?, ?)",
                (user_id, data, session.updated_at),
            )
            self._db.commit()
        except Exception as e:
            logger.warning("세션 DB 저장 실패 (%s): %s", user_id, e)

    def _delete_from_db(self, user_id: str):
        if self._db is None:
            return
        try:
            self._db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            self._db.commit()
        except Exception as e:
            logger.warning("세션 DB 삭제 실패 (%s): %s", user_id, e)

    # ── 공개 인터페이스 ──────────────────────────────────────────────────────

    def get(self, user_id: str) -> Optional[ClarificationSession]:
        with self._lock:
            session = self._sessions.get(user_id)
            if session is None:
                return None
            if time.time() - session.updated_at > self._ttl:
                del self._sessions[user_id]
                self._delete_from_db(user_id)
                return None
            return session

    def set(self, user_id: str, session: ClarificationSession):
        session.updated_at = time.time()
        with self._lock:
            self._sessions[user_id] = session
        self._save_to_db(user_id, session)

    def create(self, user_id: str, question: str) -> ClarificationSession:
        session = ClarificationSession(original_question=question, accumulated_context=question)
        self.set(user_id, session)
        return session

    def delete(self, user_id: str):
        with self._lock:
            self._sessions.pop(user_id, None)
        self._delete_from_db(user_id)

    def cleanup_expired(self):
        now = time.time()
        with self._lock:
            expired = [uid for uid, s in self._sessions.items() if now - s.updated_at > self._ttl]
            for uid in expired:
                del self._sessions[uid]
        if expired and self._db is not None:
            try:
                self._db.execute("DELETE FROM sessions WHERE updated_at <= ?", (now - self._ttl,))
                self._db.commit()
            except Exception:
                pass


session_store = SessionStore()
