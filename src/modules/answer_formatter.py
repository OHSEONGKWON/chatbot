from __future__ import annotations

from typing import Any


class AnswerFormatter:
    async def format(
        self,
        question: str,
        draft_answer: str,
        rag_docs: list[dict[str, Any]],
        legal_category: str,
        is_general: bool = False,
    ) -> str:
        answer = draft_answer.strip()
        if not answer:
            answer = "답변을 생성하지 못했습니다. 상황을 조금 더 구체적으로 알려주세요."

        sources = self._source_lines(rag_docs)
        disclaimer = (
            "\n\n※ 이 답변은 법률정보 제공을 위한 일반 안내입니다. 긴급한 위험이 있거나 기한이 임박한 경우 "
            "전문기관 또는 변호사 상담을 우선 이용하세요."
        )
        category_line = f"[분류: {legal_category}]\n" if legal_category else ""
        source_block = f"\n\n[참고한 근거]\n{sources}" if sources else "\n\n[참고한 근거]\n직접 인용할 근거 문서를 충분히 찾지 못했습니다."
        return f"{category_line}{answer}{source_block}{disclaimer}"

    def _source_lines(self, rag_docs: list[dict[str, Any]]) -> str:
        lines = []
        seen = set()
        for doc in rag_docs:
            metadata = doc.get("metadata") or {}
            label = (
                metadata.get("law_name")
                or metadata.get("source_file")
                or metadata.get("manual_title")
                or metadata.get("case")
                or doc.get("chunk_id")
                or "근거 문서"
            )
            article = metadata.get("article_id") or metadata.get("article_title") or metadata.get("section_label") or ""
            key = (label, article)
            if key in seen:
                continue
            seen.add(key)
            suffix = f" / {article}" if article else ""
            lines.append(f"{len(lines) + 1}. {label}{suffix}")
            if len(lines) >= 3:
                break
        return "\n".join(lines)


answer_formatter = AnswerFormatter()
