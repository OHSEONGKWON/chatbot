"""
LawsGuard 메인 파이프라인 오케스트레이터
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from modules.answer_formatter import answer_formatter
from modules.clarification import ClarificationResult, clarification_manager
from modules.consistency_checker import consistency_checker
from modules.ner_checker import ner_checker
from modules.rag import retriever
from session_store import ClarificationSession, session_store
from config import config


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
            session_store.delete(user_id)
            return PipelineResult(
                response_text=self._cfg.hallucination.answer_give_up_message,
                needs_requery=False,
                session_ended=True,
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
