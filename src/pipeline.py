"""LawsGuard 메인 파이프라인 오케스트레이터."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .modules.answer_formatter import answer_formatter
from .modules.case_frame_contract import (
    build_answer_contract,
    build_issue_plan,
    build_safe_contract_answer,
    check_answer_contract,
    extract_case_frame,
    should_answer_without_requery,
    should_force_answer,
)
from .modules.clarification import ClarificationResult, clarification_manager
from .modules.consistency_checker import consistency_checker
from .modules.legal_reasoning_validator import legal_reasoning_validator
from .modules.ner_checker import ner_checker
from .modules.rag import retriever
from .session_store import session_store
from .config import config


@dataclass
class PipelineResult:
    response_text: str
    needs_requery: bool
    session_ended: bool
    was_ner_corrected: bool = False
    legal_category: str = ""
    step_reached: int = 0
    answer_reliability: Optional[float] = None
    qafs: Optional[float] = None
    rrs: Optional[float] = None
    quality_report: dict[str, Any] | None = None
    similar_questions: list[str] | None = None
    similar_answers: list[str] | None = None
    original_consistency_answer: str = ""
    ner_mismatches: list[dict[str, Any]] | None = None
    ner_mid_confidence_warnings: list[dict[str, Any]] | None = None


class LawsGuardPipeline:
    def __init__(self):
        self._clarify = clarification_manager
        self._consistency = consistency_checker
        self._legal_reasoning = legal_reasoning_validator
        self._ner = ner_checker
        self._formatter = answer_formatter
        self._retriever = retriever
        self._cfg = config

    def _build_quality_report(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        consistency_report = getattr(self._consistency, "last_quality_report", {}) or {}
        report: dict[str, Any] = {}
        if extra:
            report.update(extra)
        if consistency_report:
            report.update(consistency_report)
        return report

    def _evidence_status_from_rag_docs(self, rag_docs: list[dict[str, Any]]) -> dict[str, Any]:
        if not rag_docs:
            return {"evidence_status": "FAIL", "evidence_action": "safe_fallback", "reason": "no_rag_docs"}
        scored = [float(doc.get("score") or 0.0) for doc in rag_docs if isinstance(doc, dict)]
        best_score = max(scored) if scored else 0.0
        if best_score >= 0.65 or len(rag_docs) >= 2:
            return {"evidence_status": "PASS", "evidence_action": "use_rag_normally", "reason": "rag_docs_available", "best_score": best_score, "doc_count": len(rag_docs)}
        return {"evidence_status": "WEAK", "evidence_action": "use_cautious_answer", "reason": "weak_rag_docs", "best_score": best_score, "doc_count": len(rag_docs)}

    def _apply_answer_guard(self, answer: str, question: str, legal_category: str, evidence_report: dict[str, Any]) -> str:
        guarded = (answer or "").strip()
        evidence_status = evidence_report.get("evidence_status", "WEAK")
        if evidence_status == "FAIL" and "현재 검색된 근거만으로" not in guarded:
            guarded = "현재 검색된 근거만으로 특정 조문을 단정하기는 어렵습니다. " + guarded
        elif evidence_status == "WEAK":
            guarded = guarded.replace("해당합니다", "해당할 수 있습니다").replace("성립합니다", "성립할 수 있습니다")
        return guarded

    def _effective_category(self, clarify_category: str, issue_plan: Any, case_frame: Any | None = None, user_text: str = "") -> str:
        primary = str(getattr(issue_plan, "primary_issue", "") or "")
        domain = str(getattr(issue_plan, "main_domain", "") or "")
        excluded = set(str(x) for x in (getattr(issue_plan, "excluded_issues", []) or []))
        text = user_text or ""
        unpaid_claim = bool(getattr(case_frame, "unpaid_claim", None)) if case_frame is not None else False
        condition_or_exchange = str(getattr(case_frame, "condition_or_exchange", "") or "") if case_frame is not None else ""
        labor_terms = ("임금", "알바비", "주휴수당", "최저임금", "월급", "급여", "시급", "근무시간", "해고", "사장", "점장", "상사", "알바", "회사", "직장")
        explicit_labor = unpaid_claim or any(t in text for t in labor_terms) or any(t in condition_or_exchange for t in labor_terms)

        # IssuePlan이 명시적으로 배제한 쟁점은 최종 분류에서 중심 분류로 쓰지 않는다.
        if domain == "성폭력" or "강제추행" in primary or "성희롱" in primary:
            if explicit_labor and "임금체불" not in excluded:
                return "성폭력/노동"
            return "성폭력"
        if domain == "노동" or "임금" in primary or "해고" in primary:
            return "노동"
        if domain:
            return domain
        return clarify_category

    async def process(self, user_id: str, user_input: str) -> PipelineResult:
        session = session_store.get(user_id)
        if session is None:
            session = session_store.create(user_id, user_input)

        clarify_result: ClarificationResult = await self._clarify.process(session=session, user_input=user_input)
        session_store.set(user_id, session)

        context_text = session.accumulated_context or session.original_question or user_input
        case_frame = extract_case_frame(user_input, context=context_text)
        issue_plan = build_issue_plan(case_frame, clarify_result.legal_category)
        answer_contract = build_answer_contract(case_frame, issue_plan)
        legal_category = self._effective_category(clarify_result.legal_category, issue_plan, case_frame=case_frame, user_text=context_text)

        force_answer = should_force_answer(user_input) or should_force_answer(context_text)
        can_answer_by_frame = should_answer_without_requery(case_frame, issue_plan)

        if clarify_result.needs_requery and not force_answer and not can_answer_by_frame:
            previous_requery = getattr(session, "last_requery_message", "")
            repeated_requery = bool(previous_requery and previous_requery.strip() == (clarify_result.requery_message or "").strip())
            # 반복 재질문이면 더 묻지 않고 조건부 답변으로 전환한다.
            if not repeated_requery:
                quality_report = self._build_quality_report(extra={
                    "CaseFrame": case_frame.to_dict(),
                    "IssuePlan": issue_plan.to_dict(),
                    "AnswerContract": answer_contract.to_dict(),
                })
                return PipelineResult(
                    response_text=clarify_result.requery_message,
                    needs_requery=True,
                    session_ended=False,
                    legal_category=legal_category,
                    step_reached=0,
                    quality_report=quality_report,
                )

        general_prefix = ""
        if clarify_result.use_general:
            general_prefix = self._cfg.clarification.fallback_message + "\n\n"

        final_question = clarify_result.final_question or context_text
        # CaseFrame/IssuePlan이 잡은 핵심 법령·초점을 검색어에 반영
        search_query = " ".join([
            final_question,
            str(issue_plan.primary_issue or ""),
            " ".join(issue_plan.needed_laws or []),
            " ".join(issue_plan.answer_focus or []),
        ]).strip()

        rag_docs = await self._retriever.retrieve_async(
            search_query,
            legal_category=legal_category if legal_category in {"노동", "성폭력"} else clarify_result.legal_category,
            case_frame=case_frame,
            issue_plan=issue_plan,
        )
        rrs_report = self._retriever.evaluate_retrieval(
            query=search_query,
            docs=rag_docs,
            legal_category=legal_category,
            case_frame=case_frame,
            issue_plan=issue_plan,
        )

        is_reliable, original_answer, avg_score, _ = await self._consistency.run(
            final_question,
            rag_docs=rag_docs,
            legal_category=legal_category,
            case_frame=case_frame,
            issue_plan=issue_plan,
            answer_contract=answer_contract,
        )
        quality_report = self._build_quality_report(extra={
            "RRS": rrs_report,
            "CaseFrame": case_frame.to_dict(),
            "IssuePlan": issue_plan.to_dict(),
            "AnswerContract": answer_contract.to_dict(),
        })
        sqqs = quality_report.get("SQQS") or 0.0
        answer_reliability = round(avg_score * min(1.0, sqqs / 0.5), 3)
        evidence_report = self._evidence_status_from_rag_docs(rag_docs)

        if answer_reliability < self._cfg.hallucination.consistency_threshold:
            if not (rrs_report.get("score", 0.0) >= 0.72 and issue_plan.confidence >= 0.70):
                session.retry_count += 1
                requery_message = self._cfg.clarification.fallback_message
                quality_report = self._build_quality_report(extra={"RRS": rrs_report, "CaseFrame": case_frame.to_dict(), "IssuePlan": issue_plan.to_dict(), "AnswerContract": answer_contract.to_dict()})
                return PipelineResult(
                    response_text=requery_message,
                    needs_requery=True,
                    session_ended=False,
                    legal_category=legal_category,
                    step_reached=4,
                    answer_reliability=answer_reliability,
                    rrs=rrs_report.get("score"),
                    quality_report=quality_report,
                    similar_questions=quality_report.get("similar_questions"),
                    similar_answers=quality_report.get("similar_answers"),
                    original_consistency_answer=quality_report.get("original_answer", ""),
                )

        initial_contract_check = check_answer_contract(original_answer, case_frame, issue_plan, answer_contract)
        if initial_contract_check.has_blocking_violation:
            original_answer = build_safe_contract_answer(case_frame, issue_plan)
            initial_contract_check = check_answer_contract(original_answer, case_frame, issue_plan, answer_contract)

        legal_reasoning_result = await self._legal_reasoning.validate(
            question=final_question,
            answer=original_answer,
            rag_docs=rag_docs,
            legal_category=legal_category,
        )

        route = getattr(clarify_result, "route", None)
        route_issue = route.get("issue") if isinstance(route, dict) else (getattr(route, "issue", None) if route is not None else None)
        ner_result = await self._ner.check_and_correct(
            answer=legal_reasoning_result.validated_answer,
            rag_docs=rag_docs,
            route=route,
            route_issue=route_issue,
        )
        corrected_answer = ner_result.corrected_answer
        guarded_answer = self._apply_answer_guard(
            corrected_answer,
            question=final_question,
            legal_category=legal_category,
            evidence_report=evidence_report,
        )

        final_contract_check = check_answer_contract(guarded_answer, case_frame, issue_plan, answer_contract)
        if final_contract_check.has_blocking_violation:
            guarded_answer = build_safe_contract_answer(case_frame, issue_plan)
            final_contract_check = check_answer_contract(guarded_answer, case_frame, issue_plan, answer_contract)

        # QAFS가 낮으면 최종 답변을 그대로 출력하지 않고 CaseFrame/IssuePlan 기반 안전 답변으로 재작성한다.
        # RRS가 낮은 경우 formatter가 참고 근거를 보수적으로 정리하도록 rrs_report를 함께 넘긴다.
        if final_contract_check.score < 0.65:
            guarded_answer = build_safe_contract_answer(case_frame, issue_plan)
            final_contract_check = check_answer_contract(guarded_answer, case_frame, issue_plan, answer_contract)

        quality_report.update({
            "ContractCheck": final_contract_check.to_dict(),
            "QAFS": final_contract_check.score,
        })

        final_response = await self._formatter.format(
            question=final_question,
            draft_answer=guarded_answer,
            rag_docs=rag_docs,
            legal_category=legal_category,
            is_general=clarify_result.use_general,
            case_frame=case_frame,
            issue_plan=issue_plan,
            answer_contract=answer_contract,
            rrs_report=rrs_report,
        )

        session_store.delete(user_id)
        return PipelineResult(
            response_text=general_prefix + final_response,
            needs_requery=False,
            session_ended=True,
            was_ner_corrected=ner_result.was_corrected,
            legal_category=legal_category,
            step_reached=7,
            answer_reliability=answer_reliability,
            qafs=final_contract_check.score,
            rrs=rrs_report.get("score"),
            quality_report=quality_report,
            similar_questions=quality_report.get("similar_questions"),
            similar_answers=quality_report.get("similar_answers"),
            original_consistency_answer=quality_report.get("original_answer", ""),
            ner_mismatches=getattr(ner_result, "mismatched_entities", None),
            ner_mid_confidence_warnings=getattr(ner_result, "mid_confidence_warnings", None),
        )


pipeline = LawsGuardPipeline()
