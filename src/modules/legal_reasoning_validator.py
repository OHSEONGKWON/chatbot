from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..config import config
from .llm_client import llm_client


@dataclass
class LegalReasoningIssue:
    kind: str
    message: str
    severity: str = "warning"
    suggestion: str = ""


@dataclass
class LegalReasoningResult:
    original_answer: str
    validated_answer: str
    score: float
    issues: list[LegalReasoningIssue] = field(default_factory=list)
    used_llm: bool = False


class LegalReasoningValidator:
    async def validate(
        self,
        question: str,
        answer: str,
        rag_docs: list[dict[str, Any]],
        legal_category: str,
    ) -> LegalReasoningResult:
        heuristic = self._heuristic_validate(question, answer, rag_docs, legal_category)
        if llm_client.available:
            llm_result = await self._llm_validate(question, heuristic.validated_answer, rag_docs, legal_category)
            if llm_result is not None:
                post_check = self._heuristic_validate(question, llm_result.validated_answer, rag_docs, legal_category)
                if post_check.issues or post_check.score < llm_result.score:
                    return LegalReasoningResult(
                        original_answer=answer,
                        validated_answer=post_check.validated_answer,
                        score=min(llm_result.score, post_check.score),
                        issues=llm_result.issues + post_check.issues,
                        used_llm=True,
                    )
                return llm_result
        return heuristic

    async def _llm_validate(
        self,
        question: str,
        answer: str,
        rag_docs: list[dict[str, Any]],
        legal_category: str,
    ) -> LegalReasoningResult | None:
        prompt = self._build_llm_prompt(question, answer, rag_docs, legal_category)
        raw = await llm_client.complete(prompt, system=self._system_prompt(), max_tokens=1800)
        data = self._parse_json(raw)
        if not data:
            return None

        issues = [
            LegalReasoningIssue(
                kind=str(item.get("kind", "llm_review")),
                message=str(item.get("message", "")),
                severity=str(item.get("severity", "warning")),
                suggestion=str(item.get("suggestion", "")),
            )
            for item in data.get("issues", [])
            if isinstance(item, dict) and item.get("message")
        ]
        revised = str(data.get("revised_answer") or "").strip()
        score = self._clamp_score(data.get("score"), default=0.75)
        if not revised:
            revised = self._apply_issue_appendix(answer, issues)

        return LegalReasoningResult(
            original_answer=answer,
            validated_answer=revised,
            score=score,
            issues=issues,
            used_llm=True,
        )

    def _heuristic_validate(
        self,
        question: str,
        answer: str,
        rag_docs: list[dict[str, Any]],
        legal_category: str,
    ) -> LegalReasoningResult:
        issues: list[LegalReasoningIssue] = []
        support_text = self._support_text(rag_docs)
        source_refs = self._source_ref_keys(rag_docs)

        cited_refs = self._extract_law_refs(answer)
        unsupported = [ref for ref in cited_refs if not self._ref_supported(ref, support_text, source_refs)]
        for ref in unsupported:
            issues.append(
                LegalReasoningIssue(
                    kind="unsupported_legal_reference",
                    severity="error",
                    message=f"답변의 '{ref}' 근거가 검색 문서에서 충분히 확인되지 않습니다.",
                    suggestion=f"'{ref}'를 단정 근거로 쓰지 말고, 검색된 근거 범위에서 설명하세요.",
                )
            )

        category_issue = self._category_support_issue(legal_category, support_text)
        if category_issue:
            issues.append(category_issue)

        for missing in self._missing_required_elements(question, answer, legal_category):
            issues.append(missing)

        if self._has_strong_conclusion(answer) and not self._has_sufficient_support(answer, support_text, legal_category):
            issues.append(
                LegalReasoningIssue(
                    kind="overconfident_application",
                    severity="warning",
                    message="사실관계 확정 없이 법적 결론을 강하게 단정하고 있습니다.",
                    suggestion="가능성 표현으로 낮추고, 필요한 추가 사실과 증거를 함께 안내하세요.",
                )
            )

        score = max(0.25, 1.0 - 0.18 * sum(1 for issue in issues if issue.severity == "error") - 0.08 * sum(1 for issue in issues if issue.severity != "error"))
        validated = self._mask_unsupported_refs(answer, unsupported)
        validated = self._soften_overclaims(validated) if issues else validated
        validated = self._apply_issue_appendix(validated, issues)
        return LegalReasoningResult(
            original_answer=answer,
            validated_answer=validated,
            score=round(score, 3),
            issues=issues,
            used_llm=False,
        )

    def _system_prompt(self) -> str:
        return (
            "너는 한국 법률 상담 답변의 법리 적용 타당성을 검토하는 감수자다. "
            "제공된 RAG 근거 안에서만 판단하고, 근거 밖의 조문·판례·형량을 만들지 않는다. "
            "사용자에게 유리한 결론도 사실관계가 부족하면 가능성으로 낮춰야 한다."
        )

    def _build_llm_prompt(self, question: str, answer: str, rag_docs: list[dict[str, Any]], legal_category: str) -> str:
        return (
            "아래 답변이 사용자 사실관계에 법리를 적절히 적용했는지 검토하세요.\n"
            "반드시 JSON만 출력하세요.\n\n"
            "JSON 스키마:\n"
            "{\n"
            '  "score": 0.0부터 1.0,\n'
            '  "issues": [{"kind": "...", "severity": "error|warning", "message": "...", "suggestion": "..."}],\n'
            '  "revised_answer": "근거와 사실관계에 맞게 보수적으로 수정한 전체 답변"\n'
            "}\n\n"
            "검토 기준:\n"
            "- 답변이 인용한 조문/판례/법률명이 RAG 근거에 실제로 있는가?\n"
            "- 해당 조문이 질문의 분야와 사실관계에 적용될 수 있는가?\n"
            "- 법률요건이 부족한데 결론을 단정하지 않았는가?\n"
            "- 필요한 추가 사실, 증거, 기관 안내가 빠지지 않았는가?\n\n"
            f"[분류]\n{legal_category}\n\n"
            f"[사용자 질문]\n{question}\n\n"
            f"[답변]\n{answer}\n\n"
            f"[RAG 근거]\n{self._format_docs(rag_docs)}"
        )

    def _format_docs(self, rag_docs: list[dict[str, Any]]) -> str:
        blocks = []
        for i, doc in enumerate(rag_docs[: config.rag.top_k], 1):
            metadata = doc.get("metadata") or {}
            title = metadata.get("law_name") or metadata.get("source_file") or metadata.get("manual_title") or "근거 문서"
            article = metadata.get("article_id") or metadata.get("article_title") or metadata.get("section_label") or ""
            text = re.sub(r"\s+", " ", doc.get("text") or "").strip()
            blocks.append(f"[{i}] {title} {article}\n{text[:1200]}")
        return "\n\n".join(blocks)

    def _parse_json(self, raw: str) -> dict | None:
        if not raw:
            return None
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    def _support_text(self, rag_docs: list[dict[str, Any]]) -> str:
        parts = []
        for doc in rag_docs:
            metadata = doc.get("metadata") or {}
            parts.append(doc.get("text") or "")
            parts.extend(str(metadata.get(key) or "") for key in ["law_name", "source_file", "article_id", "article_title", "section_label"])
        return "\n".join(parts)

    def _extract_law_refs(self, text: str) -> list[str]:
        refs = []
        known_laws = [
            "근로기준법",
            "최저임금법",
            "임금채권보장법",
            "근로자퇴직급여 보장법",
            "성폭력범죄의 처벌 등에 관한 특례법",
            "성폭력범죄의처벌등에관한특례법",
            "형법",
            "양성평등기본법",
            "남녀고용평등법",
            "민법",
        ]
        for law in known_laws:
            for match in re.finditer(rf"{re.escape(law)}(?:\s*제\s*\d+\s*조(?:의\s*\d+)?)?", text):
                refs.append(re.sub(r"\s+", " ", match.group(0)).strip())

        patterns = [
            r"[가-힣A-Za-z·\s]{2,30}법\s*제\s*\d+\s*조(?:의\s*\d+)?",
            r"제\s*\d+\s*조(?:의\s*\d+)?",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                ref = re.sub(r"\s+", " ", match.group(0)).strip()
                if self._looks_like_false_law_ref(ref):
                    continue
                refs.append(ref)
        return self._dedupe(refs)

    def _looks_like_false_law_ref(self, ref: str) -> bool:
        compact = re.sub(r"\s+", "", ref)
        false_fragments = ["관련법", "대한법", "법률구조", "법률상담", "법률정보", "법리"]
        return any(fragment in compact for fragment in false_fragments)

    def _source_ref_keys(self, rag_docs: list[dict[str, Any]]) -> set[str]:
        refs = set()
        for doc in rag_docs:
            metadata = doc.get("metadata") or {}
            law = metadata.get("law_name") or ""
            article = metadata.get("article_id") or metadata.get("article_title") or ""
            law_key = self._normalize_ref(str(law))
            article_match = re.search(r"제\d+조(?:의\d+)?", self._normalize_ref(str(article)))
            if law_key and article_match:
                refs.add(f"{law_key}{article_match.group(0)}")
            if law_key:
                refs.add(law_key)
        return refs

    def _ref_supported(self, ref: str, support_text: str, source_refs: set[str] | None = None) -> bool:
        norm_ref = self._normalize_ref(ref)
        norm_support = self._normalize_ref(support_text)
        source_refs = source_refs or set()
        if norm_ref in norm_support:
            if not source_refs:
                return True
        article_match = re.search(r"제\d+조(?:의\d+)?", norm_ref)
        law_name = re.sub(r"제\d+조(?:의\d+)?", "", norm_ref).strip()
        if article_match:
            article = article_match.group(0)
            if law_name:
                return f"{law_name}{article}" in source_refs
            return any(key.endswith(article) for key in source_refs)
        if source_refs and law_name:
            return law_name in source_refs
        return bool(law_name and law_name in norm_support)

    def _mask_unsupported_refs(self, answer: str, unsupported_refs: list[str]) -> str:
        masked = answer
        for ref in sorted(set(unsupported_refs), key=len, reverse=True):
            if not ref:
                continue
            masked = masked.replace(ref, "검색 근거에서 직접 확인되지 않은 조문")
        return masked

    def _normalize_ref(self, text: str) -> str:
        return re.sub(r"\s+", "", text or "")

    def _category_support_issue(self, category: str, support_text: str) -> LegalReasoningIssue | None:
        if category == "노동":
            hits = sum(term in support_text for term in ["근로기준법", "임금", "근로계약", "체불", "고용노동부"])
            wrong = sum(term in support_text for term in ["성폭력", "강제추행", "성희롱"])
            if hits < 2 or wrong > hits + 2:
                return LegalReasoningIssue(
                    kind="category_mismatch",
                    severity="error",
                    message="노동 사안인데 검색 근거가 노동 법리와 충분히 맞지 않습니다.",
                    suggestion="임금 지급, 근로계약, 체불 진정 등 노동 근거를 다시 검색해야 합니다.",
                )
        if category == "성폭력":
            hits = sum(term in support_text for term in ["성폭력", "성희롱", "강제추행", "피해자", "상담", "신고"])
            wrong = sum(term in support_text for term in ["임금", "근로계약", "퇴직금"])
            if hits < 2 or wrong > hits + 2:
                return LegalReasoningIssue(
                    kind="category_mismatch",
                    severity="error",
                    message="성폭력 사안인데 검색 근거가 성폭력/피해지원 법리와 충분히 맞지 않습니다.",
                    suggestion="동의, 신체접촉, 피해자 보호, 신고 절차 관련 근거를 다시 검색해야 합니다.",
                )
        return None

    def _missing_required_elements(self, question: str, answer: str, category: str) -> list[LegalReasoningIssue]:
        issues = []
        if category == "노동":
            required = {
                "근로 제공/근무시간": ["근무", "근로", "시간", "출퇴근"],
                "임금 약정/체불액": ["임금", "월급", "급여", "체불액", "지급"],
                "증거": ["카카오톡", "문자", "계좌", "근무표", "증거"],
            }
        elif category == "성폭력":
            required = {
                "동의 여부/구체적 행위": ["동의", "신체접촉", "행위", "강제", "거절"],
                "증거": ["카카오톡", "문자", "녹음", "목격", "진료", "증거"],
                "지원/신고": ["112", "1366", "상담", "신고", "보호"],
            }
        else:
            required = {}

        for name, terms in required.items():
            if not any(term in answer for term in terms):
                issues.append(
                    LegalReasoningIssue(
                        kind="missing_legal_element",
                        severity="warning",
                        message=f"법리 적용에 필요한 '{name}' 안내가 부족합니다.",
                        suggestion=f"'{name}'와 관련된 추가 사실 또는 증거를 확인하도록 안내하세요.",
                    )
                )
        return issues

    def _has_strong_conclusion(self, answer: str) -> bool:
        patterns = ["해당합니다", "성립합니다", "위반입니다", "청구할 수 있습니다", "처벌됩니다", "반드시"]
        return any(pattern in answer for pattern in patterns)

    def _has_sufficient_support(self, answer: str, support_text: str, category: str) -> bool:
        if category == "노동":
            return "근로기준법" in support_text and ("임금" in support_text or "체불" in support_text)
        if category == "성폭력":
            return ("강제추행" in support_text or "성폭력" in support_text or "성희롱" in support_text) and ("상담" in support_text or "피해자" in support_text or "신고" in support_text)
        return len(support_text) > 300

    def _soften_overclaims(self, answer: str) -> str:
        replacements = {
            "해당합니다": "해당할 가능성이 있습니다",
            "성립합니다": "성립할 가능성이 있습니다",
            "위반입니다": "위반으로 문제될 가능성이 있습니다",
            "처벌됩니다": "처벌 대상이 될 가능성이 있습니다",
            "반드시": "가능하면",
        }
        softened = answer
        for before, after in replacements.items():
            softened = softened.replace(before, after)
        return softened

    def _apply_issue_appendix(self, answer: str, issues: list[LegalReasoningIssue]) -> str:
        if not issues:
            return answer
        blocking = [issue for issue in issues if issue.severity == "error"]
        warnings = [issue for issue in issues if issue.severity != "error"]
        lines = []
        if blocking:
            lines.append("\n\n[법리 적용 검토]")
            lines.append("검색 근거와 답변 사이에 확인이 필요한 부분이 있어, 아래 사항을 전제로 보수적으로 봐야 합니다.")
            for issue in blocking[:3]:
                lines.append(f"- {issue.message}")
        elif warnings:
            lines.append("\n\n[법리 적용 검토]")
            for issue in warnings[:3]:
                lines.append(f"- {issue.message}")
        return answer.rstrip() + "\n".join(lines)

    def _clamp_score(self, value, default: float) -> float:
        try:
            number = float(value)
        except Exception:
            return default
        return max(0.0, min(1.0, number))

    def _dedupe(self, values: list[str]) -> list[str]:
        result = []
        seen = set()
        for value in values:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result


legal_reasoning_validator = LegalReasoningValidator()
