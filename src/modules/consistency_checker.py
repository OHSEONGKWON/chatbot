from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from ..config import config
from .llm_client import llm_client


class ConsistencyChecker:
    async def run(self, question: str, rag_docs: list[dict[str, Any]] | None = None, legal_category: str = ""):
        rag_docs = rag_docs or []
        category = legal_category or self._classify(question)
        original_answer = await self._answer(question, rag_docs, category)

        if not llm_client.available:
            score = 0.75 if original_answer else 0.0
            return bool(original_answer), original_answer, score, {"mode": "offline_fallback"}

        variants = await self._generate_variants(question)
        if not variants:
            return True, original_answer, 0.85, {"mode": "single_answer"}

        variant_prompts = [self._answer_prompt(variant, rag_docs, category) for variant in variants]
        answers = await llm_client.complete_many(variant_prompts, system=self._system_prompt())
        answers = [answer for answer in answers if answer]
        if not answers:
            return True, original_answer, 0.70, {"mode": "variant_generation_failed"}

        scores = [self._text_similarity(original_answer, answer) for answer in answers]
        avg_score = sum(scores) / len(scores)
        return (
            avg_score >= config.hallucination.consistency_threshold,
            original_answer,
            avg_score,
            {"variants": variants, "scores": scores},
        )

    async def _generate_variants(self, question: str) -> list[str]:
        prompt = (
            f"다음 법률 상담 질문의 의미를 유지하되 표현만 다르게 {config.hallucination.num_similar_questions}개 작성하세요.\n"
            "각 문장은 한 줄에 하나씩만 출력하세요.\n\n"
            f"질문: {question}"
        )
        text = await llm_client.complete(prompt, system="너는 한국어 법률 상담 질문을 의미 보존 방식으로 재작성한다.", max_tokens=1024)
        variants = []
        for line in text.splitlines():
            line = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
            if line:
                variants.append(line)
        return variants[: config.hallucination.num_similar_questions]

    async def _answer(self, question: str, rag_docs: list[dict[str, Any]], category: str) -> str:
        if llm_client.available:
            answer = await llm_client.complete(self._answer_prompt(question, rag_docs, category), system=self._system_prompt())
            if answer:
                return answer
        return self._fallback_answer(question, rag_docs, category)

    def _system_prompt(self) -> str:
        return (
            "너는 대학생을 돕는 한국 법률 상담 챗봇이다. 성폭력 및 노동 문제에 대해 답한다. "
            "제공된 근거 문서 안에서만 단정하고, 근거가 부족하면 불확실하다고 말한다. "
            "변호사 선임이 필요한 사안, 긴급 위험, 신고/상담 기관 연결 필요성을 분명히 안내한다."
        )

    def _answer_prompt(self, question: str, rag_docs: list[dict[str, Any]], category: str = "") -> str:
        context = self._format_docs(rag_docs)
        return (
            "아래 근거 문서를 바탕으로 사용자 질문에 답하세요.\n"
            "답변 형식: 1) 상황 정리 2) 관련 법률 쟁점 3) 근거에서 확인한 내용 4) 지금 할 일 5) 주의사항.\n"
            "근거 문서에 없는 조문, 판례번호, 금액, 형량은 만들지 마세요.\n"
            "사용자에게 유리한 결론을 단정하지 말고, 필요한 추가 사실과 증거를 분리해서 말하세요.\n\n"
            f"[분류]\n{category or '미확정'}\n\n"
            f"[사용자 질문]\n{question}\n\n"
            f"[근거 문서]\n{context}"
        )

    def _format_docs(self, rag_docs: list[dict[str, Any]]) -> str:
        if not rag_docs:
            return "검색된 근거 문서가 없습니다."
        blocks = []
        for i, doc in enumerate(rag_docs[: config.rag.top_k], 1):
            metadata = doc.get("metadata") or {}
            title = metadata.get("law_name") or metadata.get("source_file") or metadata.get("manual_title") or "근거 문서"
            text = (doc.get("text") or "").strip()
            source_type = metadata.get("source_type") or "unknown"
            reason = doc.get("retrieval_reason") or ""
            blocks.append(f"[{i}] {title} ({source_type}) {reason}\n{text[:1200]}")
        return "\n\n".join(blocks)

    def _fallback_answer(self, question: str, rag_docs: list[dict[str, Any]], legal_category: str = "") -> str:
        if not rag_docs:
            return (
                "현재 질문과 직접 연결되는 근거 문서를 찾지 못했습니다. "
                "사실관계가 더 필요하므로 발생 시점, 장소/관계, 구체적 행위, 증거 유무를 알려주세요."
            )
        category = legal_category or self._classify(question)
        facts = self._summarize_user_facts(question, category)
        issue = self._issue_summary(category)
        evidence = self._evidence_guidance(category)
        actions = self._action_guidance(category)
        if category == "성폭력":
            risk = "상대방과의 접촉이 계속되거나 보복 위험이 있으면 안전 확보와 긴급 신고를 우선하세요."
        else:
            risk = "임금채권은 시간 경과에 따라 증거 확보가 어려워질 수 있으니 자료를 먼저 정리하세요."

        snippets = self._relevant_snippets(question, rag_docs, category, limit=3)
        basis = "\n".join(f"- {snippet}" for snippet in snippets) if snippets else "- 검색된 문서에서 직접 관련 근거를 충분히 추리지 못했습니다."

        return (
            "1) 상황 정리\n"
            f"{facts}\n\n"
            "2) 관련 법률 쟁점\n"
            f"{issue}\n\n"
            "3) 근거에서 확인한 내용\n"
            f"{basis}\n\n"
            "4) 지금 할 일\n"
            f"- {evidence}\n"
            f"- {actions}\n"
            f"- {risk}\n\n"
            "5) 주의사항\n"
            "이 답변은 입력한 사실관계와 검색 근거에 따른 일반 안내입니다. 실제 청구 가능성, 신고 전략, 소멸시효·고소기간 등은 "
            "구체 자료를 본 뒤 달라질 수 있습니다."
        )

    def _classify(self, text: str) -> str:
        if any(
            term in text
            for term in [
                "성폭력",
                "성추행",
                "성희롱",
                "강제추행",
                "강간",
                "불법촬영",
                "스토킹",
                "동의 없이",
                "동의없이",
                "몸을 만",
                "허리를 만",
                "가슴",
                "엉덩이",
                "거절했는데",
            ]
        ):
            return "성폭력"
        if any(term in text for term in ["임금", "월급", "알바", "아르바이트", "근로", "해고", "퇴직금", "최저임금"]):
            return "노동"
        return "일반"

    def _summarize_user_facts(self, question: str, category: str) -> str:
        pieces = []
        if category == "노동":
            if any(term in question for term in ["임금", "월급", "알바비", "급여"]):
                pieces.append("임금 또는 급여 미지급 문제가 핵심으로 보입니다")
            if "근로계약서" in question and any(term in question for term in ["없", "안", "미작성"]):
                pieces.append("근로계약서 미작성/미교부 가능성이 있습니다")
            if any(term in question for term in ["카톡", "문자", "대화", "녹음"]):
                pieces.append("대화 기록 등 증거가 일부 있는 것으로 보입니다")
        elif category == "성폭력":
            if any(term in question for term in ["성추행", "강제추행", "만졌", "접촉"]):
                pieces.append("동의 없는 신체접촉 여부가 핵심으로 보입니다")
            if any(term in question for term in ["카톡", "문자", "녹음", "사진", "목격"]):
                pieces.append("사후 대화나 기록 등 증거가 일부 있는 것으로 보입니다")

        if not pieces:
            return "말씀하신 내용은 법률 검토가 필요한 사안으로 보입니다."
        return "말씀하신 내용을 기준으로 보면, " + ", ".join(pieces) + "."

    def _issue_summary(self, category: str) -> str:
        if category == "노동":
            return (
                "근로 제공 사실, 약정 임금, 실제 지급 여부, 근무시간, 근로계약서 작성 여부가 중요합니다. "
                "근로계약서가 없어도 실제 사용자의 지휘·감독 아래 일했다면 임금청구와 임금체불 진정 가능성을 검토할 수 있습니다. "
                "다만 체불액은 약속한 시급/월급, 실제 근무일·시간, 이미 받은 돈을 대조해서 산정해야 합니다."
            )
        if category == "성폭력":
            return (
                "행위의 구체적 내용, 동의 여부, 관계와 장소, 반복성·강제성, 사건 직후 대화와 주변 진술이 중요합니다. "
                "동의 없는 신체접촉은 구체적 행위와 정황에 따라 강제추행 등 형사 문제와 학교 내 신고/보호조치가 동시에 문제될 수 있습니다."
            )
        return "사안의 법적 분류, 발생 시점, 상대방과의 관계, 증거 유무를 먼저 확인해야 합니다."

    def _evidence_guidance(self, category: str) -> str:
        if category == "노동":
            return "카카오톡/문자, 출퇴근 기록, 근무표, 계좌내역, 급여 약속 내용, 사업장 정보, 함께 일한 사람의 진술을 모아 체불액 표를 만들어두세요."
        if category == "성폭력":
            return "카카오톡/문자, 통화·녹음, 사진, 진료기록, 목격자, 사건 직후 작성한 메모를 원본 상태로 보존하세요."
        return "문자, 녹음, 계약서, 사진, 계좌내역처럼 사건을 뒷받침할 자료를 원본으로 보존하세요."

    def _action_guidance(self, category: str) -> str:
        if category == "노동":
            return "임금체불은 고용노동부 진정, 학교 노동상담/법률상담, 대한법률구조공단 132 상담을 검토하세요."
        if category == "성폭력":
            return "긴급하면 112, 피해 상담은 여성긴급전화 1366, 학교 인권센터/상담센터, 대한법률구조공단 132 상담을 검토하세요."
        return "학교 상담센터, 공공 상담기관, 대한법률구조공단 132 등에서 구체 자료를 바탕으로 상담을 받아보세요."

    def _relevant_snippets(self, question: str, rag_docs: list[dict[str, Any]], category: str, limit: int = 3) -> list[str]:
        query_terms = set(re.findall(r"[0-9A-Za-z가-힣]{2,}", question.lower()))
        if category == "노동":
            query_terms.update(["근로", "임금", "체불", "지급", "근로계약서", "고용노동부"])
        elif category == "성폭력":
            query_terms.update(["성폭력", "성희롱", "강제추행", "동의", "피해자", "상담", "신고"])
        snippets = []
        seen = set()
        for doc in rag_docs:
            metadata = doc.get("metadata") or {}
            title = self._clean_title(metadata.get("law_name") or metadata.get("source_file") or metadata.get("manual_title") or "근거 문서")
            text = re.sub(r"\s+", " ", (doc.get("text") or "")).strip()
            sentences = re.split(r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+", text)
            best = self._best_sentence(sentences, query_terms) or text[:240]
            best = best.strip()
            if len(best) > 260:
                best = best[:260].rstrip() + "..."
            key = (title, best)
            if not best or key in seen:
                continue
            seen.add(key)
            snippets.append(f"{title}: {best}")
            if len(snippets) >= limit:
                break
        return snippets

    def _clean_title(self, title: str) -> str:
        title = re.sub(r"\s+", " ", str(title or "")).strip()
        if len(title) > 80:
            title = title[:80].rstrip() + "..."
        return title

    def _best_sentence(self, sentences: list[str], query_terms: set[str]) -> str:
        best_sentence = ""
        best_score = 0
        for sentence in sentences:
            if len(sentence.strip()) < 25:
                continue
            lowered = sentence.lower()
            score = sum(1 for term in query_terms if term in lowered)
            if score > best_score:
                best_score = score
                best_sentence = sentence
        return best_sentence

    def _text_similarity(self, left: str, right: str) -> float:
        return SequenceMatcher(None, self._normalize(left), self._normalize(right)).ratio()

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()


consistency_checker = ConsistencyChecker()
