"""
세션 저장소
카카오톡 user_id 기반으로 대화 상태를 인메모리에서 관리
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from config import config


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


class SessionStore:
	def __init__(self):
		self._sessions: dict[str, ClarificationSession] = {}
		self._lock = threading.Lock()
		self._ttl = config.session_ttl_sec

	def get(self, user_id: str) -> Optional[ClarificationSession]:
		with self._lock:
			session = self._sessions.get(user_id)
			if session is None:
				return None
			if time.time() - session.updated_at > self._ttl:
				del self._sessions[user_id]
				return None
			return session

	def set(self, user_id: str, session: ClarificationSession):
		session.updated_at = time.time()
		with self._lock:
			self._sessions[user_id] = session

	def create(self, user_id: str, question: str) -> ClarificationSession:
		session = ClarificationSession(original_question=question, accumulated_context=question)
		self.set(user_id, session)
		return session

	def delete(self, user_id: str):
		with self._lock:
			self._sessions.pop(user_id, None)

	def cleanup_expired(self):
		now = time.time()
		with self._lock:
			expired = [uid for uid, s in self._sessions.items() if now - s.updated_at > self._ttl]
			for uid in expired:
				del self._sessions[uid]


session_store = SessionStore()
