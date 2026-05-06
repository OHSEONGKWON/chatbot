"""
LawsGuard 메인 파이프라인 오케스트레이터
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from .modules.answer_formatter import answer_formatter
from .modules.clarification import ClarificationResult, clarification_manager
from .modules.consistency_checker import consistency_checker
from .modules.ner_checker import ner_checker
from .modules.rag import retriever
from .session_store import ClarificationSession, session_store
from .config import config


@dataclass
class PipelineResult:
    response_text: str
    needs_requery: bool
    session_ended: bool
    consistency_score: Optional[float] = None
    was_ner_corrected: bool = False
    legal_category: str = ""
    step_reached: int = 0


class LawsGuardPipeline:
    def __init__(self):
        self._clarify = clarification_manager
        self._consistency = consistency_checker
        self._ner = ner_checker
        self._formatter = answer_formatter
        self._retriever = retriever
        self._cfg = config

    def _build_consistency_requery(self, question: str, legal_category: str, avg_score: float) -> str:
        # Generic requery builder used as a fallback only.
        if legal_category == "불명확":
            return (
                "어떤 상황인지 조금만 더 자세히 알려주세요.\n"
                "언제, 어디서, 누가, 어떤 일이 있었는지 알려주시면 더 정확히 도와드릴 수 있어요."
            )

        if avg_score is not None and avg_score < self._cfg.hallucination.consistency_threshold:
            return (
                "조금 더 구체적으로 알려주실 수 있나요?\n"
                "필요한 정보(예: 언제/어디서/누구 등)를 한두 문장으로 알려주시면 재검토하겠습니다."
            )

        return "조금 더 구체적으로 알려주실 수 있나요?"

    async def process(self, user_id: str, user_input: str) -> PipelineResult:
        session = session_store.get(user_id)
        if session is None:
            session = session_store.create(user_id, user_input)

        clarify_result: ClarificationResult = await self._clarify.process(session=session, user_input=user_input)
        session_store.set(user_id, session)

        if clarify_result.needs_requery:
            return PipelineResult(
                response_text=clarify_result.requery_message,
                needs_requery=True,
                session_ended=False,
                legal_category=clarify_result.legal_category,
                step_reached=0,
            )

        general_prefix = ""
        if clarify_result.use_general:
            general_prefix = self._cfg.clarification.fallback_message + "\n\n"

        final_question = clarify_result.final_question
        legal_category = clarify_result.legal_category

        is_reliable, original_answer, avg_score, _ = await self._consistency.run(final_question)
        if not is_reliable:
            session.retry_count += 1
            # Use dynamic requery generation (avoid fixed sexual-case templates)
            try:
                eval_obj = clarify_result.eval
                missing = eval_obj.missing_elements if eval_obj else []
                entity_check = eval_obj.entity_check if eval_obj else None
                requery_message = await self._clarify.generate_requery(
                    question=final_question,
                    missing=missing,
                    retry_count=session.retry_count,
                    entity_check=entity_check,
                    context=session.accumulated_context,
                    session=session,
                )
            except Exception:
                # fallback to previous heuristic builder
                requery_message = self._build_consistency_requery(final_question, legal_category, avg_score)

            return PipelineResult(
                response_text=requery_message,
                needs_requery=True,
                session_ended=False,
                consistency_score=avg_score,
                legal_category=legal_category,
                step_reached=4,
            )

        rag_docs = await self._retriever.retrieve_async(final_question)
        ner_result = await self._ner.check_and_correct(answer=original_answer, rag_docs=rag_docs)
        corrected_answer = ner_result.corrected_answer

        final_response = await self._formatter.format(
            question=final_question,
            draft_answer=corrected_answer,
            rag_docs=rag_docs,
            legal_category=legal_category,
            is_general=clarify_result.use_general,
        )

        session_store.delete(user_id)
        return PipelineResult(
            response_text=general_prefix + final_response,
            needs_requery=False,
            session_ended=True,
            consistency_score=avg_score,
            was_ner_corrected=ner_result.was_corrected,
            legal_category=legal_category,
            step_reached=7,
        )


pipeline = LawsGuardPipeline()
