from __future__ import annotations

from dataclasses import dataclass, field

from ..config import config
from ..session_store import ClarificationSession


@dataclass
class ClarificationEval:
    score: float
    missing_elements: list[str] = field(default_factory=list)
    entity_check: dict | None = None


@dataclass
class ClarificationResult:
    needs_requery: bool
    requery_message: str
    final_question: str
    legal_category: str
    use_general: bool = False
    eval: ClarificationEval | None = None


class ClarificationManager:
    def process_sync(self, session: ClarificationSession, user_input: str) -> ClarificationResult:
        if user_input and user_input not in session.accumulated_context:
            session.accumulated_context = f"{session.accumulated_context}\n추가 답변: {user_input}".strip()

        context = session.accumulated_context.strip()
        category = self._classify(context)
        evaluation = self._evaluate(context, category)

        if evaluation.score < config.clarification.min_score_threshold and session.retry_count < config.clarification.max_retries:
            session.retry_count += 1
            return ClarificationResult(
                needs_requery=True,
                requery_message=self._build_requery(evaluation.missing_elements, category),
                final_question=context,
                legal_category=category,
                eval=evaluation,
            )

        return ClarificationResult(
            needs_requery=False,
            requery_message="",
            final_question=context,
            legal_category=category,
            use_general=evaluation.score < config.clarification.min_score_threshold,
            eval=evaluation,
        )

    async def process(self, session: ClarificationSession, user_input: str) -> ClarificationResult:
        return self.process_sync(session=session, user_input=user_input)

    async def generate_requery(
        self,
        question: str,
        missing: list[str],
        retry_count: int,
        entity_check=None,
        context: str = "",
        session: ClarificationSession | None = None,
    ) -> str:
        return self._build_requery(missing, self._classify(context or question))

    def _classify(self, text: str) -> str:
        sexual_terms = [
            "성폭력",
            "성추행",
            "강제추행",
            "성희롱",
            "강간",
            "스토킹",
            "불법촬영",
            "디지털성범죄",
            "동의 없이",
            "동의없이",
            "몸을 만",
            "허리를 만",
            "가슴",
            "엉덩이",
            "거절했는데",
        ]
        labor_terms = ["임금", "월급", "알바비", "급여", "알바", "아르바이트", "근로", "해고", "퇴직금", "최저임금", "노동", "주휴"]
        if any(term in text for term in sexual_terms):
            return "성폭력"
        if any(term in text for term in labor_terms):
            return "노동"
        return "불명확"

    def _evaluate(self, text: str, category: str) -> ClarificationEval:
        missing = []
        score = 0.0

        if len(text.strip()) >= 20:
            score += 1.0
        else:
            missing.append("상황 설명")

        if any(token in text for token in ["어제", "오늘", "작년", "지난", "월", "일", "년", "부터", "까지"]):
            score += 1.0
        else:
            missing.append("언제 발생했는지")

        if any(token in text for token in ["학교", "회사", "알바", "직장", "동아리", "술자리", "카톡", "온라인", "집"]):
            score += 1.0
        else:
            missing.append("어디서/어떤 관계에서 발생했는지")

        if category == "성폭력":
            if any(token in text for token in ["동의", "거절", "강제로", "만졌", "촬영", "협박", "반복"]):
                score += 1.0
            else:
                missing.append("동의 여부와 구체적 행위")
        elif category == "노동":
            if any(token in text for token in ["계약", "시급", "월급", "임금", "근무", "시간", "해고", "퇴직"]):
                score += 1.0
            else:
                missing.append("계약/근무/임금 조건")
        else:
            missing.append("성폭력 또는 노동 문제 중 어느 쪽인지")

        if any(token in text for token in ["증거", "문자", "카톡", "녹음", "계약서", "급여명세서", "목격자"]):
            score += 1.0
        else:
            missing.append("증거나 기록이 있는지")

        return ClarificationEval(score=score, missing_elements=missing[:4], entity_check={"category": category})

    def _build_requery(self, missing: list[str], category: str) -> str:
        if not missing:
            missing = ["언제", "어디서", "누가", "무엇을 했는지"]
        prefix = "더 정확한 법률 판단을 위해 몇 가지만 더 알려주세요."
        if category == "성폭력":
            prefix = "성폭력 관련 판단은 구체적 행위와 동의 여부가 중요합니다."
        elif category == "노동":
            prefix = "노동 문제는 근무 조건과 임금/계약 내용이 중요합니다."
        bullets = "\n".join(f"- {item}" for item in missing[:4])
        return f"{prefix}\n{bullets}\n\n아는 범위에서 한두 문장으로 답해주시면 됩니다."


clarification_manager = ClarificationManager()
