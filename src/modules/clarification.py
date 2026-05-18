from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..config import config
from .llm_client import llm_client
from ..session_store import ClarificationSession


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
   - subject: 주체(나, 회사, 임대인, 교수, 사장 등)
   - timing: 시기(사건 발생일, 기간 등)
   - action: 행위(무슨 일이 발생했는지)
   - purpose: 목적(원하는 결과: 성립 여부 확인, 신고, 돈 받기, 대처 방법 문의 등)

2. score: 정보 충실도 점수 (1.0~5.0)
   - 1~2: 매우 모호
   - 3~4: 상황은 있으나 세부 맥락 부족
   - 5: 매우 구체적

3. legal_category: 분류 가능한 법률 영역 (민사/형사/가사/노동/성범죄/성희롱/인권/기타/불명확)

4. missing_elements: 부족한 정보를 한국어로 구체적으로 기술 (배열, 없으면 빈 배열)

5. can_proceed:
   - 일반 사안: score >= 3.0이고 legal_category != "불명확"이면 true
   - 성폭력/강간/성추행/강제추행: 행위(무슨 일이 있었는지)와 주체(가해자 또는 관계)가 확인되면 timing 없어도 true
   - 언어적 성희롱/교육기관 성희롱: 주체, 행위, 원하는 판단 또는 조치가 있으면 timing이 없어도 true 가능
   - 임금체불: 미지급 임금, 상대방, 받고 싶은 목적이 있으면 timing이 없어도 true 가능
   - 심각한 피해 사안(강간, 성폭행, 성추행, 불법촬영)은 핵심 사실만 있어도 가능한 한 true로 판단하세요.
   - '어떻게 해야 하나요?', '~인가요?', '신고할 수 있나요?', '처벌되나요?', '받을 수 있나요?' 같은 표현은 purpose=true로 보세요.

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
            model=llm_client.clarify_model,
        )
        data = _safe_json_loads(raw)
        if not data:
            # 파인튜닝 모델 실패 시 기본 모델로 재시도
            raw = await llm_client.complete(
                system_prompt=EVAL_SYSTEM,
                user_prompt=prompt,
                temperature=0.0,
                json_mode=True,
                model="gpt-4o-mini",
            )
            data = _safe_json_loads(raw)
        if not data:
            return self._heuristic_eval(context or question)

        result = EvalResult(
            score=float(data.get("score", 1.0)),
            legal_category=str(data.get("legal_category", "불명확")),
            missing_elements=_normalize_missing_elements(data.get("missing_elements", [])),
            can_proceed=bool(data.get("can_proceed", False)),
            entity_check=_normalize_entity_check(data.get("entity_check", {})),
        )
        return self._repair_eval_result(result, context or question)

    def _heuristic_eval(self, context: str) -> EvalResult:
        entity_check = {
            "subject": self._has_context_signal(context, "subject"),
            "timing": self._has_context_signal(context, "timing"),
            "action": self._has_context_signal(context, "action"),
            "purpose": self._has_context_signal(context, "purpose") or self._has_obvious_purpose(context),
        }
        issue = self._infer_issue(context)
        legal_category = self._category_from_issue(issue)

        score = 1.0
        score += 0.8 if entity_check["subject"] else 0.0
        score += 1.1 if entity_check["action"] else 0.0
        score += 1.0 if entity_check["purpose"] else 0.0
        score += 0.6 if entity_check["timing"] else 0.0
        if issue != "일반":
            score += 0.5
        score = min(5.0, round(score, 1))

        result = EvalResult(
            score=score,
            legal_category=legal_category,
            missing_elements=self._missing_from_entity_check(entity_check),
            can_proceed=False,
            entity_check=entity_check,
        )
        result.can_proceed = not self._should_requery(result, context=context)
        return result

    def _repair_eval_result(self, result: EvalResult, context: str) -> EvalResult:
        heuristic = self._heuristic_eval(context)
        repaired_entity = _normalize_entity_check(result.entity_check)

        for key, value in heuristic.entity_check.items():
            if value:
                repaired_entity[key] = True

        issue = self._infer_issue(context, result.legal_category)
        repaired_category = self._normalize_legal_category(result.legal_category, issue)
        if repaired_category == "불명확" and heuristic.legal_category != "불명확":
            repaired_category = heuristic.legal_category

        repaired = EvalResult(
            score=max(float(result.score or 0.0), heuristic.score),
            legal_category=repaired_category,
            missing_elements=result.missing_elements,
            can_proceed=bool(result.can_proceed),
            entity_check=repaired_entity,
        )
        repaired.missing_elements = self._missing_from_entity_check(repaired_entity)
        # 최종 진행 가능 여부는 휴리스틱 기준을 우선한다. LLM이 과하게 낙관적인 경우를 막기 위함이다.
        repaired.can_proceed = not self._should_requery(heuristic, context=context)
        return repaired

    def _missing_from_entity_check(self, entity_check: dict) -> list[str]:
        labels = {
            "subject": "관련 당사자 또는 상대방 정보",
            "timing": "사건 발생 시기",
            "action": "구체적으로 발생한 일",
            "purpose": "원하는 대응 방향",
        }
        return [label for key, label in labels.items() if not entity_check.get(key, False)]

    def _category_from_issue(self, issue: str) -> str:
        if issue in {"임금체불", "부당해고"}:
            return "노동"
        if issue in {"교육기관 언어적 성희롱", "언어적 성희롱", "신체접촉형 강제추행", "불법촬영", "강간"}:
            return "성폭력"
        return "불명확"

    def _normalize_legal_category(self, legal_category: str, issue: str) -> str:
        if legal_category in {"민사", "형사", "가사", "노동", "성폭력", "성희롱", "인권", "기타"}:
            if legal_category in {"성희롱", "형사"} and issue in {
                "교육기관 언어적 성희롱",
                "언어적 성희롱",
                "신체접촉형 강제추행",
                "불법촬영",
            }:
                return "성폭력"
            return legal_category
        return self._category_from_issue(issue)

    def _infer_issue(self, text: str, legal_category: str = "") -> str:
        t = text or ""
        if _contains_any(t, ("강간", "성폭행", "강제성교", "성적 피해", "성범죄")):
            return "강간"

        if _contains_any(t, ("교수", "선생님", "교사", "수업", "강의", "학생", "학교", "대학")) and _contains_any(
            t, ("성희롱", "성적 발언", "여자답", "남자답", "가슴이 커야", "가슴이 작", "다리가 굵", "치마", "몸매", "외모")
        ):
            return "교육기관 언어적 성희롱"

        if _contains_any(t, ("성희롱", "성적 발언", "음담패설", "외모 평가", "몸매")) and not _contains_any(
            t, ("만졌", "신체 접촉", "가슴을 만", "엉덩이를 만", "허벅지를 만")
        ):
            return "언어적 성희롱"

        if _contains_any(t, ("가슴", "엉덩이", "허벅지", "신체 접촉", "만졌", "성추행", "추행")):
            return "신체접촉형 강제추행"

        if _contains_any(t, ("몰카", "불법촬영", "도촬", "찍힌", "찍혔", "촬영당", "카메라", "촬영물")):
            return "불법촬영"

        if _contains_any(t, ("알바비", "월급", "급여", "시급", "임금체불", "임금 체불", "덜 줬")):
            return "임금체불"

        if _contains_any(t, ("해고", "잘렸", "그만 나오", "부당해고", "권고사직")):
            return "부당해고"

        if "성희롱" in legal_category:
            return "언어적 성희롱"
        return "일반"

    def _has_context_signal(self, context: str, topic: str) -> bool:
        ctx = context or ""
        if topic == "subject":
            return _contains_any(ctx, ("나", "저는", "제가", "저에게", "본인", "교수", "선생님", "사장", "점장", "회사", "상사", "동료", "엄마", "아빠", "상대방", "선배", "후배", "친구", "남자친구", "여자친구", "지인", "연인", "파트너", "이웃", "모르는", "아는 사람", "남성", "여성", "남자", "여자", "오빠", "형", "언니", "누나", "아저씨", "직장상사", "직장 상사"))
        if topic == "timing":
            return bool(re.search(r"\b\d{4}[-./]\d{1,2}[-./]\d{1,2}\b", ctx)) or _contains_any(
                ctx, ("어제", "오늘", "그제", "지난", "방금", "전", "개월", "년", "월", "일", "시", "분", "경", "수업 중", "회식")
            )
        if topic == "action":
            return _contains_any(ctx, ("추행", "성희롱", "발언", "말했", "만졌", "해고", "그만 나오", "나오지 말", "임금", "월급", "급여", "알바비", "못 받", "미지급", "일했", "근무", "근로계약서", "촬영", "몰카", "찍힌", "찍혔", "유포", "괴롭힘", "강간", "성폭행", "성폭력", "강제성교", "성적 피해", "성범죄", "협박", "동영상"))
        if topic == "purpose":
            return _contains_any(ctx, ("고소", "신고", "진정", "합의", "반환", "손해배상", "처벌", "상담", "성립", "받고", "알고 싶", "구제", "어떻게 해야", "어떻게 하나", "어떡", "가능", "할 수 있", "인가요", "되나요", "되나요?", "해야 하나요"))
        if topic == "evidence":
            return _contains_any(ctx.lower(), ("증거", "녹음", "문자", "카톡", "목격자", "cctv", "출근기록", "계좌", "급여명세서", "영상", "사진", "목격", "블랙박스", "녹화"))
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
        if _contains_any(t, ("증거", "목격", "녹음", "자료")):
            return "evidence"
        if _contains_any(t, ("민사", "형사", "가사", "노동", "성범죄", "기타", "불명확", "쟁점")):
            return "legal_issue"
        return "general"

    def _pick_next_topic(
        self,
        eval_result: EvalResult,
        context: str,
        last_missing: list[str] | None = None,
        last_topic: str = "",
    ) -> str:
        entity_check = _normalize_entity_check(eval_result.entity_check)
        issue = self._infer_issue(context, eval_result.legal_category)

        if issue in ("교육기관 언어적 성희롱", "언어적 성희롱"):
            priority = ("purpose", "evidence", "action", "subject")
        elif issue == "임금체불":
            priority = ("purpose", "evidence", "action", "subject", "timing")
        elif issue == "신체접촉형 강제추행":
            priority = ("purpose", "evidence", "timing", "action", "subject")
        elif issue == "강간":
            priority = ("subject", "evidence", "timing", "purpose")
        else:
            priority = ("subject", "action", "purpose", "timing", "evidence")

        for topic in priority:
            if topic in ("evidence",):
                if not self._has_context_signal(context, topic):
                    return topic
                continue

            if not entity_check.get(topic, False) and not self._has_context_signal(context, topic):
                if topic == last_topic:
                    continue
                return topic

        for item in eval_result.missing_elements:
            topic = self._topic_from_missing_text(item)
            if topic and topic != last_topic and topic != "timing":
                return topic

        return "purpose" if issue in ("교육기관 언어적 성희롱", "언어적 성희롱") else "general"

    def _legal_topic_to_message(self, legal_category: str, topic: str, retry_count: int, context: str = "") -> tuple[str, str]:
        issue = self._infer_issue(context, legal_category)

        if issue in ("교육기관 언어적 성희롱", "언어적 성희롱"):
            mapping = {
                "purpose": (
                    "원하시는 조치가 무엇인지 알려주실 수 있나요?",
                    "예: 학교에 신고, 사과 요구, 재발 방지, 국가인권위원회 진정, 법적 대응 가능성 확인 등",
                ),
                "evidence": (
                    "그 발언을 들은 사람이나 남아 있는 증거가 있는지 알려주실 수 있나요?",
                    "예: 같이 들은 학생, 녹음, 수업자료, 문자·공지, 당시 메모 등",
                ),
                "action": (
                    "상대방이 어떤 말을 했는지 최대한 정확히 알려주실 수 있나요?",
                    "예: 실제 표현, 공개적으로 말했는지, 반복했는지 등",
                ),
                "subject": (
                    "상대방과의 관계를 알려주실 수 있나요?",
                    "예: 교수-학생, 선생님-학생, 조교-학생, 동아리 선배 등",
                ),
            }
            return mapping.get(topic, mapping["purpose"])

        if issue == "임금체불":
            mapping = {
                "purpose": (
                    "원하시는 대응 방향이 있으면 알려주세요. 이미 대처 방법이나 성립 여부를 묻는 질문이라면 바로 일반 기준으로 안내하겠습니다.",
                    "예: 미지급 임금 받기, 고용노동부 신고, 지급명령·소액소송 검토 등",
                ),
                "evidence": (
                    "임금과 근무 사실을 입증할 자료가 있는지 알려주실 수 있나요?",
                    "예: 근무표, 문자·카톡, 계좌 입금내역, 급여명세서, 출퇴근 기록 등",
                ),
                "action": (
                    "얼마를 못 받았고 어떤 임금인지 알려주실 수 있나요?",
                    "예: 월급 10만 원, 주휴수당, 야간수당, 퇴직금 등",
                ),
                "subject": (
                    "근무 형태와 상대방을 알려주실 수 있나요?",
                    "예: 알바 사장, 회사, 점장, 프랜차이즈 매장 등",
                ),
                "timing": (
                    "언제 일한 임금인지 알려주실 수 있나요?",
                    "예: 4월 월급, 퇴사 후 남은 임금, 지난주 근무분 등",
                ),
            }
            return mapping.get(topic, mapping["purpose"])

        if issue == "신체접촉형 강제추행":
            mapping = {
                "purpose": (
                    "원하시는 대응 방향이 있으면 알려주세요. 이미 대처 방법이나 성립 여부를 묻는 질문이라면 바로 일반 기준으로 안내하겠습니다.",
                    "예: 경찰 신고, 고소, 합의, 증거 확보, 처벌 가능성 확인 등",
                ),
                "evidence": (
                    "당시 상황을 입증할 자료나 목격자가 있는지 알려주실 수 있나요?",
                    "예: 목격자, CCTV, 카톡, 통화녹음, 직후 보낸 메시지 등",
                ),
                "timing": (
                    "언제 어디서 발생한 일인지 알려주실 수 있나요?",
                    "예: 어제 회식 중, 지난주 수업 후, 2024년 3월경 등",
                ),
                "action": (
                    "어떤 신체 접촉이 있었는지 구체적으로 알려주실 수 있나요?",
                    "예: 가슴을 만짐, 엉덩이를 만짐, 손목을 잡고 끌어당김 등",
                ),
                "subject": (
                    "상대방과의 관계를 알려주실 수 있나요?",
                    "예: 사장, 교수, 동료, 선배, 가족, 모르는 사람 등",
                ),
            }
            return mapping.get(topic, mapping["purpose"])

        if issue == "불법촬영":
            mapping = {
                "purpose": (
                    "원하시는 대응 방향이 무엇인지 알려주실 수 있나요?",
                    "예: 경찰 신고, 촬영물 삭제 지원, 증거 보존, 처벌 가능성 확인 등",
                ),
                "evidence": (
                    "촬영 정황이나 남아 있는 증거가 있는지 알려주실 수 있나요?",
                    "예: 카메라 발견, 사진·영상, 메시지, 목격자, 장소 CCTV 등",
                ),
                "action": (
                    "어떤 방식으로 촬영된 것 같은지 알려주실 수 있나요?",
                    "예: 탈의실, 화장실, 숙박업소, 휴대폰 카메라, 몰래 설치된 카메라 등",
                ),
                "subject": (
                    "상대방이나 장소를 알 수 있는지 알려주실 수 있나요?",
                    "예: 지인, 모르는 사람, 업주, 학교, 직장, 숙박업소 등",
                ),
            }
            return mapping.get(topic, mapping["purpose"])

        generic = {
            "subject": ("누가 관련된 일인지 알려주실 수 있나요?", "예: 회사, 사장, 교수, 임대인, 친구, 상대방 등"),
            "timing": ("언제 발생한 일인지 알려주실 수 있나요?", "예: 어제, 지난달, 계약일 무렵, 퇴사 후 등"),
            "action": ("어떤 일이 있었는지 조금만 더 구체적으로 알려주실 수 있나요?", "예: 돈을 못 받음, 해고당함, 성적 발언을 들음, 신체 접촉이 있었음 등"),
            "purpose": ("어떤 결과를 원하시는지 알려주실 수 있나요?", "예: 신고, 돈 반환, 손해배상, 성립 여부 확인, 상담 등"),
            "evidence": ("관련 증거나 자료가 있는지 알려주실 수 있나요?", "예: 문자, 녹음, 목격자, 계약서, 급여내역 등"),
        }
        return generic.get(topic, generic["purpose"])

    def _has_obvious_purpose(self, context: str) -> bool:
        ctx = context or ""
        return _contains_any(
            ctx,
            (
                "어떻게 해야",
                "어떻게 하나",
                "어떡",
                "신고하고 싶",
                "신고할 수",
                "고소할 수",
                "성립하",
                "인가요",
                "처벌되",
                "받을 수",
                "구제받",
                "가능한가요",
                "가능할까요",
                "해야 하나요",
                "대처",
            ),
        )

    def _has_strong_issue_signal(self, context: str) -> bool:
        ctx = context or ""
        return _contains_any(
            ctx,
            (
                "가슴", "엉덩이", "허벅지", "만졌", "성추행", "강제추행",
                "강간", "성폭행", "성폭력", "강제성교", "성적 피해", "성범죄",
                "몰카", "불법촬영", "도촬", "찍힌", "찍혔", "촬영당", "몰래 찍",
                "유포", "협박", "통매음", "성기 사진", "음란 메시지",
                "스토킹", "계속 연락", "계속 따라",
                "성희롱", "몸매", "외모 평가", "성적 발언", "교수", "수업",
                "알바비", "월급", "급여", "임금", "임금체불", "최저임금", "주휴수당",
                "해고", "부당해고", "내일부터 나오지", "그만 나오", "출근하지 말",
                "근로계약서", "퇴직금", "산재", "직장 내 괴롭힘",
            ),
        )

    def _should_requery(self, eval_result: EvalResult, context: str = "") -> bool:
        entity_check = _normalize_entity_check(eval_result.entity_check)
        issue = self._infer_issue(context, eval_result.legal_category)

        has_action = entity_check.get("action") or self._has_context_signal(context, "action")
        has_purpose = (
            entity_check.get("purpose")
            or self._has_context_signal(context, "purpose")
            or self._has_obvious_purpose(context)
        )

        # 신체접촉형 강제추행: 더 엄격한 기준 적용
        if issue == "신체접촉형 강제추행":
            # 신체 부위가 구체적으로 명시되어야 함
            has_specific_body_part = _contains_any(context, (
                "가슴", "엉덩이", "허벅지", "음부", "젖가슴",
                "가슴을 만", "엉덩이를 만", "허벅지를 만",
                "음부를 만", "사타구니"
            ))
            # 타이밍이 구체적으로 명시되어야 함
            has_specific_timing = bool(re.search(r"\b\d{4}[-./]\d{1,2}[-./]\d{1,2}\b", context)) or _contains_any(
                context, ("어제", "지난주", "지난달", "오늘", "수업 중", "회식 중", "길가에서", "직장에서", "숙박업소에서", "술자리", "신청", "신청이")
            )
            # 상대방과의 관계가 명시되어야 함 (구체성 강화)
            has_specific_subject = _contains_any(context, (
                "교수", "선생님", "사장", "점장", "동료", "선배", "후배", 
                "지인", "남자친구", "여자친구", "모르는 사람", "아는 사람"
            ))
            
            # 신체접촉 강제추행은 세 가지 조건을 모두 만족해야만 진행 가능
            # (하나라도 부족하면 반드시 재질문)
            if not (has_specific_body_part and has_specific_timing and has_specific_subject and has_purpose):
                return True
            # 모두 있으면 재질문 안 함
            return False

        # 강간: 주체 + 행위만 있으면 진행
        if issue == "강간":
            has_subject = entity_check.get("subject") or self._has_context_signal(context, "subject")
            return not (has_action and has_subject)

        # 다른 강제추행/성범죄: 기본 기준
        if issue in (
            "준강간",
            "준강제추행",
            "불법촬영",
            "촬영물 유포·협박",
            "통신매체이용음란",
            "스토킹",
        ):
            return not (has_action and has_purpose)

        # 성희롱 관련: 약간 더 엄격
        if issue in (
            "교육기관 언어적 성희롱",
            "언어적 성희롱",
            "직장 내 성희롱",
        ):
            # 성희롱은 action + purpose만으로는 부족, 상대방 정보도 필요
            has_subject = entity_check.get("subject") or self._has_context_signal(context, "subject")
            return not (has_action and has_purpose and (has_subject or issue != "교육기관 언어적 성희롱"))

        # 노동 관련 이슈: 기본 기준
        if issue in (
            "임금체불",
            "최저임금",
            "주휴수당",
            "연장·야간·휴일수당",
            "부당해고",
            "근로계약서 미작성",
            "퇴직금",
            "직장 내 괴롭힘",
            "산재",
        ):
            has_subject = entity_check.get("subject") or self._has_context_signal(context, "subject")
            has_timing = entity_check.get("timing") or self._has_context_signal(context, "timing")
            has_evidence = self._has_context_signal(context, "evidence")
            return not (has_action and (has_purpose or has_subject or has_timing or has_evidence))

        strong_issue = self._has_strong_issue_signal(context)

        if strong_issue and has_action and has_purpose:
            return False

        if eval_result.legal_category == "불명확" and not has_action:
            return True

        if eval_result.score < self._cfg.min_score_threshold:
            if has_action and has_purpose:
                return False
            return True

        if not has_action:
            return True

        if not has_purpose:
            return True

        return False

    async def generate_requery(
        self,
        question: str,
        missing: list[str],
        retry_count: int,
        entity_check: dict | None = None,
        context: str = "",
        session: ClarificationSession | None = None,
        legal_category: str = "불명확",
    ) -> str:
        if retry_count > self._cfg.max_retries:
            return GENERAL_FALLBACK_MESSAGE

        normalized_entity_check = _normalize_entity_check(entity_check or {})
        normalized_missing = _normalize_missing_elements(missing)

        eval_stub = EvalResult(
            score=0.0,
            legal_category=legal_category or "불명확",
            missing_elements=normalized_missing,
            can_proceed=False,
            entity_check=normalized_entity_check,
        )

        last_missing = list(getattr(session, "last_missing_elements", []) or []) if session else []
        last_topic = getattr(session, "last_requery_topic", "") if session else ""

        topic = self._pick_next_topic(eval_stub, context, last_missing, last_topic=last_topic)

        message, example = self._legal_topic_to_message(
            legal_category=legal_category,
            topic=topic,
            retry_count=retry_count,
            context=context,
        )

        result = f"{message}\n({example})"

        remaining = self._cfg.max_retries - retry_count
        if remaining > 0:
            result += f"\n\n추가 질문 {remaining}회 남았습니다."

        if session is not None:
            session.last_requery_topic = topic
            session.last_requery_message = result
            session.last_missing_elements = normalized_missing

        return result

    async def process(self, session: ClarificationSession, user_input: str) -> ClarificationResult:
        if user_input and user_input.strip():
            if session.accumulated_context:
                if user_input.strip() not in session.accumulated_context:
                    session.accumulated_context += "\n" + user_input.strip()
            else:
                session.accumulated_context = user_input.strip()

        context = session.accumulated_context or user_input
        eval_result = await self.evaluate(question=user_input, context=context)

        if session.retry_count >= self._cfg.max_retries:
            return ClarificationResult(
                final_question=context,
                needs_requery=False,
                requery_message="",
                use_general=True,
                legal_category=eval_result.legal_category,
                eval=eval_result,
            )

        if not eval_result.can_proceed:
            session.retry_count += 1
            requery_message = await self.generate_requery(
                question=session.original_question,
                missing=eval_result.missing_elements,
                retry_count=session.retry_count,
                entity_check=eval_result.entity_check,
                context=context,
                session=session,
                legal_category=eval_result.legal_category,
            )
            return ClarificationResult(
                final_question=context,
                needs_requery=True,
                requery_message=requery_message,
                use_general=False,
                legal_category=eval_result.legal_category,
                eval=eval_result,
            )

        return ClarificationResult(
            final_question=context,
            needs_requery=False,
            requery_message="",
            use_general=False,
            legal_category=eval_result.legal_category,
            eval=eval_result,
        )


clarification_manager = ClarificationManager()
