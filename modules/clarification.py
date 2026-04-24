"""
명확화 대화 모듈 (Step 0)
"""

import json
from dataclasses import dataclass
from typing import Optional

from config import config
from modules.llm_client import llm_client
from session_store import ClarificationSession


EVAL_SYSTEM = """당신은 대한민국 법률 상담 AI의 질문 품질 평가 모듈입니다.
사용자의 질문을 분석하여 법률 자문에 필요한 정보가 충분한지 평가합니다.
반드시 JSON 형식으로만 응답하세요."""

EVAL_USER_TEMPLATE = """다음 사용자 질문을 분석하세요.

[사용자 질문]
{question}

[누적 대화 컨텍스트]
{context}

다음 기준으로 평가하고 JSON으로 응답하세요:

1. entity_check: 다음 항목의 존재 여부를 true/false로 표시
   - subject: 주체(나, 회사, 임대인 등)
   - timing: 시기(사건 발생일, 기간 등)
   - action: 행위(무슨 일이 발생했는지)
   - purpose: 목적(원하는 결과)

2. score: 정보 충실도 점수 (1.0~5.0)
   - 1~2: 매우 모호
   - 3~4: 상황은 있으나 세부 맥락 부족
   - 5: 매우 구체적

3. legal_category: 분류 가능한 법률 영역 (민사/형사/가사/노동/성범죄/기타/불명확)

4. missing_elements: 부족한 정보를 한국어로 구체적으로 기술 (배열, 없으면 빈 배열)

5. can_proceed: score >= 3.0이고 legal_category != "불명확"이면 true

응답 JSON 형식:
{{
  "entity_check": {{"subject": bool, "timing": bool, "action": bool, "purpose": bool}},
  "score": float,
  "legal_category": str,
  "missing_elements": [str, ...],
  "can_proceed": bool
}}"""

REQUERY_SYSTEM = """당신은 법률 상담 AI입니다. 사용자에게 친절하고 구체적인 재질문을 합니다.
질문은 반드시 하나만 하세요. 친근하고 간결하게 작성하세요."""

REQUERY_USER_TEMPLATE = """사용자가 법률 상담을 요청했지만 다음 정보가 부족합니다.

[사용자 원래 질문]: {question}
[부족한 정보]: {missing}
[현재 재질문 횟수]: {retry_count}/{max_retries}

재질문 하나만 생성하세요. 사용자가 쉽게 답변할 수 있도록 구체적인 예시를 포함하세요.
예: "언제 발생한 일인가요? (예: 2024년 3월경, 약 2달 전 등)"
"""


@dataclass
class EvalResult:
	score: float
	legal_category: str
	missing_elements: list[str]
	can_proceed: bool
	entity_check: dict


@dataclass
class ClarificationResult:
	final_question: str
	needs_requery: bool
	requery_message: str
	use_general: bool
	legal_category: str
	eval: Optional[EvalResult]


class ClarificationManager:
	def __init__(self):
		self._cfg = config.clarification

	async def evaluate(self, question: str, context: str) -> EvalResult:
		prompt = EVAL_USER_TEMPLATE.format(question=question, context=context)
		raw = await llm_client.complete(system_prompt=EVAL_SYSTEM, user_prompt=prompt, temperature=0.0, json_mode=True)

		try:
			data = json.loads(raw)
		except json.JSONDecodeError:
			return EvalResult(
				score=1.0,
				legal_category="불명확",
				missing_elements=["구체적인 상황 설명"],
				can_proceed=False,
				entity_check={},
			)

		return EvalResult(
			score=float(data.get("score", 1.0)),
			legal_category=data.get("legal_category", "불명확"),
			missing_elements=data.get("missing_elements", []),
			can_proceed=bool(data.get("can_proceed", False)),
			entity_check=data.get("entity_check", {}),
		)

	async def generate_requery(self, question: str, missing: list[str], retry_count: int) -> str:
		prompt = REQUERY_USER_TEMPLATE.format(
			question=question,
			missing=", ".join(missing) if missing else "구체적인 상황",
			retry_count=retry_count,
			max_retries=self._cfg.max_retries,
		)
		return await llm_client.complete(system_prompt=REQUERY_SYSTEM, user_prompt=prompt, temperature=0.3)

	async def process(self, session: ClarificationSession, user_input: str) -> ClarificationResult:
		if user_input != session.original_question:
			session.accumulated_context += f"\n[추가 정보]: {user_input}"

		if session.use_general_answer:
			return ClarificationResult(
				final_question=session.accumulated_context,
				needs_requery=False,
				requery_message="",
				use_general=True,
				legal_category="불명확",
				eval=None,
			)

		eval_result = await self.evaluate(question=user_input, context=session.accumulated_context)
		session.last_score = eval_result.score

		if eval_result.can_proceed:
			session.is_complete = True
			return ClarificationResult(
				final_question=session.accumulated_context,
				needs_requery=False,
				requery_message="",
				use_general=False,
				legal_category=eval_result.legal_category,
				eval=eval_result,
			)

		if session.retry_count >= self._cfg.max_retries:
			session.use_general_answer = True
			return ClarificationResult(
				final_question=session.accumulated_context,
				needs_requery=False,
				requery_message="",
				use_general=True,
				legal_category=eval_result.legal_category,
				eval=eval_result,
			)

		session.retry_count += 1
		requery_msg = await self.generate_requery(
			question=session.original_question,
			missing=eval_result.missing_elements,
			retry_count=session.retry_count,
		)

		remaining = self._cfg.max_retries - session.retry_count
		if remaining > 0:
			requery_msg += f"\n\n(추가 질문 {remaining}회 남았습니다)"

		return ClarificationResult(
			final_question="",
			needs_requery=True,
			requery_message=requery_msg,
			use_general=False,
			legal_category=eval_result.legal_category,
			eval=eval_result,
		)


clarification_manager = ClarificationManager()
