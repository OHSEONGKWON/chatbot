"""
명확화 대화 모듈 (Step 0)

사용자 질문이 법률 자문에 충분히 구체적인지 평가하고,
부족한 정보가 있으면 한 번에 하나의 재질문을 생성합니다.
재질문은 최대 5회까지만 허용하며, 한계에 도달하면 일반 기준 답변으로 전환합니다.
"""

from dataclasses import dataclass
from typing import Optional
import json
import re

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

REQUERY_SYSTEM = """당신은 법률 상담 AI입니다.
사용자에게 친절하고 구체적인 재질문을 합니다.
질문은 반드시 하나만 하세요.
친근하고 간결하게 작성하세요."""

REQUERY_USER_TEMPLATE = """사용자가 법률 상담을 요청했지만 다음 정보가 부족합니다.

[사용자 원래 질문]
{question}

[누적 대화 컨텍스트]
{context}

[부족한 정보]
{missing}

[현재 재질문 횟수]
{retry_count}/{max_retries}

아래 조건을 모두 지켜 재질문 한 문장만 작성하세요.
- 반드시 하나의 질문만 작성
- 부족한 정보 중 가장 핵심적인 것 1개만 물어볼 것
- 이미 컨텍스트에 나온 정보는 다시 묻지 말 것
- 사용자가 바로 답하기 쉽게 구체적인 예시를 넣을 것
- 성범죄, 민사, 형사 등 특정 사건 유형마다 고정 문구를 쓰지 말 것

예시:
"언제 발생한 일인지 알려주실 수 있나요? (예: 어제, 2024년 3월경 등)"
"누가 관련된 일인지 알려주실 수 있나요? (예: 선배, 임대인, 회사 등)"
"어떤 결과를 원하시는지 알려주실 수 있나요? (예: 고소, 반환, 합의 등)"
"""

GENERAL_FALLBACK_MESSAGE = "대답에 필요한 정보가 충분하지 않아 일반적인 기준으로 대답하겠습니다."


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


def _safe_json_loads(raw: str) -> dict:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _normalize_entity_check(entity_check: object) -> dict:
    if isinstance(entity_check, dict):
        return {
            "subject": bool(entity_check.get("subject", False)),
            "timing": bool(entity_check.get("timing", False)),
            "action": bool(entity_check.get("action", False)),
            "purpose": bool(entity_check.get("purpose", False)),
        }
    return {"subject": False, "timing": False, "action": False, "purpose": False}


def _normalize_missing_elements(missing: object) -> list[str]:
    if not isinstance(missing, list):
        return []
    cleaned = []
    for item in missing:
        if isinstance(item, str):
            text = item.strip()
            if text:
                cleaned.append(text)
    return cleaned


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


class ClarificationManager:
    def __init__(self):
        self._cfg = config.clarification

    async def evaluate(self, question: str, context: str) -> EvalResult:
        prompt = EVAL_USER_TEMPLATE.format(question=question, context=context)
        raw = await llm_client.complete(
            system_prompt=EVAL_SYSTEM,
            user_prompt=prompt,
            temperature=0.0,
            json_mode=True,
        )

        data = _safe_json_loads(raw)
        return EvalResult(
            score=float(data.get("score", 1.0)),
            legal_category=str(data.get("legal_category", "불명확")),
            missing_elements=_normalize_missing_elements(data.get("missing_elements", [])),
            can_proceed=bool(data.get("can_proceed", False)),
            entity_check=_normalize_entity_check(data.get("entity_check", {})),
        )

    def _has_context_signal(self, context: str, topic: str) -> bool:
        ctx = (context or "").lower()
        if topic == "subject":
            return _contains_any(ctx, ("선배", "동료", "임대인", "회사", "채무자", "상대방", "가해자", "아내", "남편", "친구"))
        if topic == "timing":
            return bool(re.search(r"\b\d{4}[-./]\d{1,2}[-./]\d{1,2}\b", ctx)) or _contains_any(
                ctx, ("어제", "오늘", "그제", "지난", "방금", "전", "개월", "년", "월", "일", "시", "분", "경")
            )
        if topic == "action":
            return _contains_any(ctx, ("추행", "폭행", "사기", "해고", "임금", "보증금", "계약", "협박", "명예훼손", "성희롱", "횡령"))
        if topic == "purpose":
            return _contains_any(ctx, ("고소", "고발", "합의", "반환", "손해배상", "처벌", "해지", "차단", "상담", "소송"))
        return False

    def _topic_from_missing_text(self, text: str) -> str:
        t = text.lower()
        if _contains_any(t, ("누구", "누가", "주체", "subject", "대상", "행위자")):
            return "subject"
        if _contains_any(t, ("언제", "시기", "timing", "날짜", "시간")):
            return "timing"
        if _contains_any(t, ("무엇", "행위", "action", "어떤 일", "상황")):
            return "action"
        if _contains_any(t, ("목적", "원하", "purpose", "원하는 결과", "무엇을 원")):
            return "purpose"
        if _contains_any(t, ("민사", "형사", "가사", "노동", "성범죄", "기타", "불명확", "쟁점")):
            return "legal_issue"
        return "general"

    def _topic_to_message(self, topic: str, variant_seed: int = 0) -> tuple[str, str]:
        variant = variant_seed % 2
        if topic == "subject":
            if variant == 0:
                return (
                    "누가 관련된 일인지 알려주실 수 있나요?",
                    "예: 선배, 임대인, 회사, 채무자 등",
                )
            return (
                "상대방이 누구인지 조금만 더 알려주실 수 있나요?",
                "예: 선배, 임대인, 회사, 채무자 등",
            )
        if topic == "timing":
            if variant == 0:
                return (
                    "언제 발생한 일인지 알려주실 수 있나요?",
                    "예: 어제, 2024년 3월경, 계약일 무렵 등",
                )
            return (
                "사건이 언제 있었는지 알려주실 수 있나요?",
                "예: 어제, 2024년 3월경, 계약일 무렵 등",
            )
        if topic == "action":
            if variant == 0:
                return (
                    "어떤 일이 있었는지 조금만 더 알려주실 수 있나요?",
                    "예: 추행을 당함, 돈을 못 받음, 계약을 해지함 등",
                )
            return (
                "무슨 일이 있었는지 간단히 말씀해주실 수 있나요?",
                "예: 추행을 당함, 돈을 못 받음, 계약을 해지함 등",
            )
        if topic == "purpose":
            if variant == 0:
                return (
                    "어떤 결과를 원하시는지 알려주실 수 있나요?",
                    "예: 고소, 반환, 손해배상, 처벌 수위 확인 등",
                )
            return (
                "가장 원하시는 해결 방향이 무엇인지 알려주실 수 있나요?",
                "예: 고소, 반환, 손해배상, 처벌 수위 확인 등",
            )
        if topic == "legal_issue":
            if variant == 0:
                return (
                    "민사/형사/가사/노동 중 어떤 쟁점에 가까운지 조금만 더 알려주실 수 있나요?",
                    "예: 임대차 보증금 반환, 폭행 사건, 해고 문제 등",
                )
            return (
                "어떤 법률 문제에 가까운지 한두 단어로 알려주실 수 있나요?",
                "예: 임대차 보증금 반환, 폭행 사건, 해고 문제 등",
            )
        return (
            "조금만 더 자세히 알려주실 수 있나요?",
            "예: 언제, 어디서, 누가, 어떤 일이 있었는지 등",
        )

    def _pick_next_topic(self, eval_result: EvalResult, context: str, last_missing: list[str], last_topic: str = "") -> str:
        ordered_topics = ["subject", "timing", "action", "purpose"]
        candidate_topics = []

        for item in eval_result.missing_elements:
            topic = self._topic_from_missing_text(item)
            if topic not in candidate_topics:
                candidate_topics.append(topic)

        if not candidate_topics:
            for topic, present in eval_result.entity_check.items():
                if not present:
                    candidate_topics.append(topic)

        if not candidate_topics and eval_result.legal_category == "불명확":
            candidate_topics.append("legal_issue")

        # deterministic priority and context filtering
        ctx = context or ""
        filtered_topics = []
        for topic in ordered_topics + ["legal_issue"]:
            if topic in candidate_topics and not self._has_context_signal(ctx, topic):
                filtered_topics.append(topic)

        if not filtered_topics:
            return "general"

        # Prefer a topic that is different from the last asked one.
        for topic in filtered_topics:
            if topic != last_topic:
                return topic

        # If everything repeats, fall back to a broader prompt instead of repeating the same exact question.
        if "legal_issue" not in filtered_topics:
            return "general"

        return filtered_topics[0]

    async def generate_requery(
        self,
        question: str,
        missing: list[str],
        retry_count: int,
        entity_check: dict | None = None,
        context: str = "",
        session: ClarificationSession | None = None,
    ) -> str:
        if retry_count >= self._cfg.max_retries:
            return GENERAL_FALLBACK_MESSAGE

        normalized_entity_check = _normalize_entity_check(entity_check or {})
        eval_stub = EvalResult(
            score=0.0,
            legal_category="불명확",
            missing_elements=_normalize_missing_elements(missing),
            can_proceed=False,
            entity_check=normalized_entity_check,
        )

        last_missing = list(getattr(session, "last_missing_elements", []) or []) if session else []
        last_topic = getattr(session, "last_requery_topic", "") if session else ""

        topic = self._pick_next_topic(eval_stub, context, last_missing, last_topic=last_topic)
        message, example = self._topic_to_message(topic, variant_seed=retry_count)

        # When we have multiple missing hints, expose only one priority topic in the user-facing question.
        if message == "조금만 더 자세히 알려주실 수 있나요?":
            if _contains_any((context or "").lower(), ("성추행", "폭행", "사기", "해고", "보증금", "임금")):
                message = "조금만 더 자세히 알려주실 수 있나요?"
                example = "예: 언제, 어디서, 누가, 어떤 일이 있었는지 등"

        result = f"{message} ({example})"
        remaining = self._cfg.max_retries - retry_count
        if remaining > 0:
            result += f"\n\n(추가 질문 {remaining}회 남았습니다)"

        if session is not None:
            session.last_requery_topic = topic
        return result

    def _should_requery(self, eval_result: EvalResult) -> bool:
        entity_check = _normalize_entity_check(eval_result.entity_check)
        if eval_result.score < self._cfg.min_score_threshold:
            return True
        if eval_result.legal_category == "불명확":
            return True
        if not eval_result.can_proceed:
            return True
        if not all(entity_check.get(key, False) for key in ("subject", "timing", "action")):
            return True
        return False

    async def process(self, session: ClarificationSession, user_input: str) -> ClarificationResult:
        if not session.original_question:
            session.original_question = user_input

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

        if not self._should_requery(eval_result):
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
        requery_message = await self.generate_requery(
            question=session.original_question,
            missing=eval_result.missing_elements,
            retry_count=session.retry_count,
            entity_check=eval_result.entity_check,
            context=session.accumulated_context,
            session=session,
        )

        # Store the latest requery snapshot to help avoid repeated follow-ups.
        session.last_requery_message = requery_message
        session.last_missing_elements = list(eval_result.missing_elements or [])

        return ClarificationResult(
            final_question="",
            needs_requery=True,
            requery_message=requery_message,
            use_general=False,
            legal_category=eval_result.legal_category,
            eval=eval_result,
        )


clarification_manager = ClarificationManager()