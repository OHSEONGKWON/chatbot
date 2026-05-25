from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


WORK_RELATION_TERMS = (
    "알바", "아르바이트", "사장", "사장님", "점장", "상사", "매니저", "회사", "직장", "근무", "시급", "월급", "급여"
)
SCHOOL_RELATION_TERMS = (
    "교수", "조교", "선배", "후배", "동기", "학교", "대학", "강의", "수업", "학점", "추천서", "동아리", "학생회"
)
SEXUAL_REQUEST_TERMS = (
    "성관계", "잠자리", "자자", "섹스", "성적 요구", "만나주면", "조건만남", "스폰", "술자리 같이", "사적으로 만나"
)
SEXUAL_CONTACT_TERMS = (
    "만졌", "만져", "키스", "입맞춤", "껴안", "가슴", "엉덩이", "허벅지", "성기", "몸을 만"
)
BENEFIT_TERMS = (
    "시급", "월급", "급여", "돈", "수당", "근무시간", "근무표", "채용", "합격", "승진", "추천서", "학점", "A+", "에이쁠", "점수", "통과"
)
UNPAID_TERMS = (
    "못 받", "못받", "안 줬", "안줬", "미지급", "체불", "지급일", "월급일", "임금체불", "알바비", "수당 안", "돈을 안", "돈 안"
)
EXCHANGE_PATTERNS = (
    r"(.{0,18})(하면|해주면|만나주면|자면|성관계하면|성관계를 하면|잠자리하면)(.{0,30})(올려|주겠|준대|준다고|더 줄|두 배|두배|3배|세 배|합격|A\+|에이쁠|추천서)",
    r"(.{0,18})(올려|주겠|준대|준다고|더 줄|두 배|두배|3배|세 배)(.{0,30})(하면|해주면|만나주면|자면|성관계)",
)
FORCE_ANSWER_PATTERNS = (
    "대답해줘", "답변해줘", "그냥 답", "그냥 대답", "왜 계속", "계속 같은", "계속 물어", "몰라", "모르겠", "바로 답", "일단 답"
)


@dataclass
class CaseFrame:
    actor: str | None = None
    target: str | None = None
    relationship: str | None = None
    requested_act: str | None = None
    performed_act: str | None = None
    condition_or_exchange: str | None = None
    threat_or_disadvantage: str | None = None
    unpaid_claim: str | None = None
    user_question: str | None = None
    not_claimed: tuple[str, ...] = ()
    body_part: str | None = None
    evidence: str | None = None
    confidence: float = 0.5

    @property
    def has_condition_exchange(self) -> bool:
        return bool(self.condition_or_exchange)

    @property
    def has_unpaid_claim(self) -> bool:
        return bool(self.unpaid_claim)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor,
            "target": self.target,
            "relationship": self.relationship,
            "requested_act": self.requested_act,
            "performed_act": self.performed_act,
            "condition_or_exchange": self.condition_or_exchange,
            "threat_or_disadvantage": self.threat_or_disadvantage,
            "unpaid_claim": self.unpaid_claim,
            "user_question": self.user_question,
            "not_claimed": list(self.not_claimed),
            "body_part": self.body_part,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass
class IssuePlan:
    primary_issue: str = "일반"
    main_domain: str = "기타"
    secondary_issues: list[str] = field(default_factory=list)
    excluded_issues: list[str] = field(default_factory=list)
    fact_slots: dict[str, Any] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    needed_laws: list[str] = field(default_factory=list)
    answer_focus: list[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_issue": self.primary_issue,
            "main_domain": self.main_domain,
            "secondary_issues": self.secondary_issues,
            "excluded_issues": self.excluded_issues,
            "fact_slots": self.fact_slots,
            "missing_slots": self.missing_slots,
            "needed_laws": self.needed_laws,
            "answer_focus": self.answer_focus,
            "confidence": self.confidence,
        }


@dataclass
class AnswerContract:
    must_include: list[str] = field(default_factory=list)
    must_not_frame_as: list[str] = field(default_factory=list)
    allowed_secondary: list[str] = field(default_factory=list)
    required_user_question_answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "must_include": self.must_include,
            "must_not_frame_as": self.must_not_frame_as,
            "allowed_secondary": self.allowed_secondary,
            "required_user_question_answer": self.required_user_question_answer,
        }


@dataclass
class ContractCheckResult:
    has_blocking_violation: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_blocking_violation": self.has_blocking_violation,
            "violations": self.violations,
            "warnings": self.warnings,
            "score": self.score,
        }


def _contains_any(text: str, terms: tuple[str, ...] | list[str]) -> bool:
    return any(term and term in text for term in terms)


def _extract_actor(text: str) -> str | None:
    for term in ("사장님", "사장", "점장", "상사", "교수님", "교수", "조교", "선배", "남편", "아내", "와이프"):
        if term in text:
            return term
    return None


def _extract_body_part(text: str) -> str | None:
    for part in ("가슴", "엉덩이", "허벅지", "성기", "입술", "입", "허리", "어깨", "손", "팔", "다리"):
        if part in text:
            return part
    return None


def _extract_requested_act(text: str) -> str | None:
    if _contains_any(text, ("성관계", "잠자리", "자자", "섹스")):
        return "성관계 요구"
    if _contains_any(text, ("만나주면", "사적으로 만나", "밥 먹어주면", "술자리 같이", "술자리에 같이")):
        return "사적 만남 또는 요구"
    return None


def _extract_condition_or_exchange(text: str) -> str | None:
    # 돈/학점/근무조건 등이 '~하면 주겠다/올려주겠다' 구조에 있으면 대가·조건으로 본다.
    for pattern in EXCHANGE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            raw = re.sub(r"\s+", " ", m.group(0)).strip()
            return raw[:80]
    if _contains_any(text, SEXUAL_REQUEST_TERMS) and _contains_any(text, BENEFIT_TERMS) and _contains_any(text, ("올려", "주겠", "준대", "더 줄", "두 배", "두배", "3배", "세 배")):
        return "성적 요구와 고용상 이익 또는 조건이 연결됨"
    return None


def _extract_unpaid_claim(text: str) -> str | None:
    if _contains_any(text, UNPAID_TERMS):
        # condition_or_exchange가 명확하면 unpaid_claim으로 오인하지 않는다.
        if _extract_condition_or_exchange(text):
            return None
        return "임금 또는 금품 미지급 주장"
    return None


def extract_case_frame(question: str, context: str = "") -> CaseFrame:
    text = re.sub(r"\s+", " ", f"{context}\n{question}".strip())
    compact = text.replace(" ", "")
    actor = _extract_actor(text)
    body_part = _extract_body_part(text)
    requested_act = _extract_requested_act(text)
    condition_or_exchange = _extract_condition_or_exchange(text)
    unpaid_claim = _extract_unpaid_claim(text)

    relationship = None
    if _contains_any(text, WORK_RELATION_TERMS):
        relationship = "고용관계/알바 관계"
    elif _contains_any(text, SCHOOL_RELATION_TERMS):
        relationship = "학교/교육 관계"
    elif _contains_any(text, ("남편", "아내", "와이프", "배우자")):
        relationship = "배우자 관계"

    performed_act = None
    if _contains_any(text, SEXUAL_CONTACT_TERMS):
        performed_act = f"{body_part or '신체'} 접촉"
    elif _contains_any(text, ("야근", "연장근무", "추가 근무")):
        performed_act = "근로 제공"

    threat = None
    if _contains_any(text, ("해고", "그만 나오", "근무 빼", "시급 깎", "불이익", "탈퇴", "소문")):
        threat = "불이익 또는 압박 가능성"

    user_question = None
    if _contains_any(text, ("해도 되", "동의하면", "범죄", "성추행", "신고", "처벌", "받을 수")):
        user_question = "위법성 또는 대응 가능성 문의"

    not_claimed: list[str] = []
    if condition_or_exchange and not unpaid_claim:
        not_claimed.extend(["임금체불", "이미 일한 임금 미지급", "주휴수당 미지급", "급여일 경과"])
    if requested_act and not performed_act:
        not_claimed.extend(["신체접촉", "강제추행 중심 사안", "불법촬영", "스토킹"])
    if unpaid_claim and not requested_act:
        not_claimed.extend(["성관계 요구", "성희롱 중심 사안"])

    confidence = 0.5
    if relationship:
        confidence += 0.15
    if requested_act or performed_act:
        confidence += 0.15
    if condition_or_exchange or unpaid_claim:
        confidence += 0.15
    if user_question:
        confidence += 0.05
    confidence = min(1.0, confidence)

    return CaseFrame(
        actor=actor,
        target="사용자" if any(x in text for x in ("나", "내", "저", "제")) else None,
        relationship=relationship,
        requested_act=requested_act,
        performed_act=performed_act,
        condition_or_exchange=condition_or_exchange,
        threat_or_disadvantage=threat,
        unpaid_claim=unpaid_claim,
        user_question=user_question,
        not_claimed=tuple(dict.fromkeys(not_claimed)),
        body_part=body_part,
        evidence="증거 언급" if _contains_any(text, ("카톡", "문자", "녹음", "CCTV", "목격자", "증거")) else None,
        confidence=round(confidence, 3),
    )


def build_issue_plan(case_frame: CaseFrame, legal_category: str = "") -> IssuePlan:
    slots = case_frame.to_dict()

    if case_frame.relationship in {"고용관계/알바 관계", "학교/교육 관계"} and case_frame.requested_act and case_frame.condition_or_exchange:
        return IssuePlan(
            primary_issue="직장 내 성희롱" if case_frame.relationship == "고용관계/알바 관계" else "교육기관 성희롱/권력관계 성적 요구",
            main_domain="성폭력/노동" if case_frame.relationship == "고용관계/알바 관계" else "성폭력/인권",
            secondary_issues=["고용상 이익 제공 조건", "거절 후 불이익 가능성"],
            excluded_issues=["임금체불", "강제추행", "불법촬영", "스토킹"],
            fact_slots=slots,
            missing_slots=[s for s in ("threat_or_disadvantage", "evidence") if not getattr(case_frame, s)],
            needed_laws=["남녀고용평등법 제2조", "남녀고용평등법 제14조", "양성평등기본법 제3조"],
            answer_focus=[
                "성적 요구가 고용·교육 관계에서 이루어졌는지",
                "시급·학점·추천서 등 이익과 성적 요구가 조건으로 연결되었는지",
                "동의 여부와 무관하게 자유로운 의사결정이 침해될 수 있는지",
                "거절 후 불이익과 증거 확보 필요성",
            ],
            confidence=max(0.75, case_frame.confidence),
        )

    if case_frame.unpaid_claim:
        return IssuePlan(
            primary_issue="임금체불",
            main_domain="노동",
            secondary_issues=["금품청산", "노동청 진정"],
            excluded_issues=["성희롱", "강제추행", "불법촬영"],
            fact_slots=slots,
            missing_slots=["금액", "지급일", "근무기간"],
            needed_laws=["근로기준법 제43조", "근로기준법 제36조"],
            answer_focus=["약정 금품인지", "미지급 금액", "지급기일 도과", "증거 확보"],
            confidence=max(0.7, case_frame.confidence),
        )

    if case_frame.performed_act and (case_frame.body_part or "접촉" in case_frame.performed_act):
        return IssuePlan(
            primary_issue="강제추행 검토",
            main_domain="성폭력",
            secondary_issues=["동의 범위", "거부 후 계속 여부", "항거곤란"],
            excluded_issues=["임금체불", "불법촬영", "스토킹"],
            fact_slots=slots,
            missing_slots=["동의 또는 거부 의사", "거부 후 계속 여부"],
            needed_laws=["형법 제298조", "형법 제299조"],
            answer_focus=["원치 않는 신체접촉", "동의 범위", "강제성 또는 기습성"],
            confidence=max(0.65, case_frame.confidence),
        )

    return IssuePlan(
        primary_issue=legal_category or "일반 법률상담",
        main_domain=legal_category or "기타",
        fact_slots=slots,
        missing_slots=[],
        needed_laws=[],
        answer_focus=[],
        confidence=case_frame.confidence,
    )


def build_answer_contract(case_frame: CaseFrame, issue_plan: IssuePlan) -> AnswerContract:
    must_include: list[str] = []
    must_not: list[str] = []
    allowed: list[str] = []

    if case_frame.relationship:
        must_include.append(case_frame.relationship)
    if case_frame.requested_act:
        must_include.append(case_frame.requested_act)
    if case_frame.condition_or_exchange:
        must_include.append("대가 또는 조건")
        must_include.append(case_frame.condition_or_exchange)
        must_not.extend(["임금체불", "이미 일한 임금 미지급", "주휴수당", "급여일 경과"])
    if case_frame.unpaid_claim:
        must_include.append("미지급")
        must_not.extend(["성관계 요구", "성희롱 중심 사안"])
    if issue_plan.primary_issue:
        must_include.append(issue_plan.primary_issue)

    must_not.extend(issue_plan.excluded_issues or [])
    allowed.extend(issue_plan.secondary_issues or [])
    if case_frame.threat_or_disadvantage:
        allowed.append("불이익 가능성")
    if case_frame.evidence:
        allowed.append("증거 확보")

    return AnswerContract(
        must_include=list(dict.fromkeys([x for x in must_include if x])),
        must_not_frame_as=list(dict.fromkeys([x for x in must_not if x])),
        allowed_secondary=list(dict.fromkeys([x for x in allowed if x])),
        required_user_question_answer=case_frame.user_question or "사용자의 질문에 직접 답변",
    )


def _normalized_contains(text: str, phrase: str) -> bool:
    if not phrase:
        return True
    compact_text = re.sub(r"\s+", "", text)
    compact_phrase = re.sub(r"\s+", "", phrase)
    if compact_phrase in compact_text:
        return True
    # 긴 원문 구절은 핵심 토큰 일부로도 인정한다.
    tokens = [t for t in re.findall(r"[0-9A-Za-z가-힣+]{2,}", phrase) if t not in {"또는", "그리고", "가능성"}]
    if not tokens:
        return True
    matched = sum(1 for t in tokens if t in text)
    return matched >= max(1, min(2, len(tokens)))


def check_answer_contract(answer: str, case_frame: CaseFrame, issue_plan: IssuePlan, contract: AnswerContract) -> ContractCheckResult:
    text = answer or ""
    violations: list[str] = []
    warnings: list[str] = []

    # 필수 사실 누락
    for item in contract.must_include:
        if item and not _normalized_contains(text, item):
            # 관계는 동의어로 완화
            if item == "고용관계/알바 관계" and any(x in text for x in ("알바", "사장", "직장", "고용", "근무")):
                continue
            if item == "성관계 요구" and any(x in text for x in ("성관계", "성적 요구", "잠자리")):
                continue
            warnings.append(f"must_include_missing:{item}")

    # 금지 프레임이 중심 쟁점으로 들어오면 blocking
    for item in contract.must_not_frame_as:
        if not item:
            continue
        if item in text:
            window = text[max(0, text.find(item) - 30): text.find(item) + 60]
            if item in {"임금체불", "이미 일한 임금 미지급", "주휴수당", "급여일 경과"} and case_frame.condition_or_exchange and not case_frame.unpaid_claim:
                # 부정 문맥("임금체불이 아니라...", "임금체불 문제가 없는...")은 위반 아님
                if "아니" not in window and "없" not in window and "단정" not in window:
                    violations.append(f"wrong_frame:{item}")
            elif item in {"강제추행", "불법촬영", "스토킹"} and item not in contract.allowed_secondary:
                if "단정" not in window and "없" not in window:
                    violations.append(f"wrong_frame:{item}")

    if case_frame.condition_or_exchange and case_frame.unpaid_claim is None:
        if any(x in text for x in ("임금체불", "주휴수당", "급여일")) and not any(x in text for x in ("성적 요구", "성관계", "대가", "조건", "직장 내 성희롱")):
            violations.append("condition_exchange_framed_as_unpaid_claim")

    if case_frame.requested_act and not any(x in text for x in ("성관계", "성적 요구", "잠자리", "성희롱")):
        violations.append("requested_act_missing")

    _ISSUE_KEYWORDS: dict[str, tuple[str, ...]] = {
        "직장 내 성희롱": ("직장 내 성희롱", "성희롱", "성적 요구", "고용상"),
        "교육기관 성희롱/권력관계 성적 요구": ("성희롱", "성적 요구", "교육기관"),
        "임금체불": ("임금", "임금체불", "미지급", "체불"),
        "강제추행 검토": ("강제추행", "추행", "신체접촉"),
    }
    if issue_plan.primary_issue and issue_plan.primary_issue not in {"일반", "일반 법률상담"}:
        keywords = _ISSUE_KEYWORDS.get(issue_plan.primary_issue)
        if keywords and not any(x in text for x in keywords):
            violations.append("primary_issue_missing")

    score = 1.0 - 0.25 * len(violations) - 0.08 * len(warnings)
    score = round(max(0.0, min(1.0, score)), 3)
    return ContractCheckResult(
        has_blocking_violation=bool(violations),
        violations=violations,
        warnings=warnings,
        score=score,
    )


def should_force_answer(text: str) -> bool:
    return _contains_any(text or "", FORCE_ANSWER_PATTERNS)


def should_answer_without_requery(case_frame: CaseFrame, issue_plan: IssuePlan) -> bool:
    if case_frame.condition_or_exchange and case_frame.requested_act and case_frame.relationship:
        return True
    if case_frame.unpaid_claim and case_frame.relationship:
        return True
    # 신체 접촉 행위는 관계 + 신체 부위 둘 다 확인되어야 우회 허용 (하나만으로는 부족)
    if case_frame.performed_act and case_frame.relationship and case_frame.body_part:
        return True
    return issue_plan.confidence >= 0.85 and bool(issue_plan.primary_issue)


def build_contract_context(case_frame: CaseFrame, issue_plan: IssuePlan, contract: AnswerContract) -> str:
    return (
        "[CaseFrame]\n"
        f"{case_frame.to_dict()}\n\n"
        "[IssuePlan]\n"
        f"{issue_plan.to_dict()}\n\n"
        "[AnswerContract]\n"
        f"{contract.to_dict()}\n"
        "위 CaseFrame의 사실 역할을 유지하세요. must_not_frame_as 항목을 중심 쟁점으로 답하지 마세요. "
        "None이거나 확인되지 않은 사실은 추가 확인사항으로만 다루고 단정하지 마세요."
    )


def build_safe_contract_answer(case_frame: CaseFrame, issue_plan: IssuePlan) -> str:
    # LLM이 계약을 반복적으로 위반할 때 사용하는 최소 안전 답변.
    if case_frame.condition_or_exchange and case_frame.requested_act:
        return (
            "상황 정리\n"
            f"{case_frame.actor or '상대방'}이 {case_frame.requested_act}를 조건으로 {case_frame.condition_or_exchange}을/를 제시한 사안입니다. "
            "이는 이미 일한 임금이 미지급된 문제라기보다, 성적 요구가 고용상 이익이나 조건과 연결된 문제로 보아야 합니다.\n\n"
            "법적 판단\n"
            "사업주나 상급자가 성관계나 성적 만남을 임금·시급·근무조건과 연결해 제안했다면 직장 내 성희롱 또는 고용상 지위를 이용한 성적 요구로 문제될 수 있습니다. "
            "사용자가 동의한다고 하더라도 사장과 알바생이라는 지위 차이 때문에 자유로운 동의였는지, 압박이나 위력이 있었는지가 문제될 수 있습니다. "
            "다만 형사범죄 성립 여부는 협박, 강요, 실제 신체접촉, 거절 후 불이익, 반복 요구 등 추가 사정에 따라 달라집니다.\n\n"
            "추가로 확인할 사항과 법적 의미\n"
            "사장님이 실제로 어떤 표현으로 제안했는지 → 성적 언동·요구 여부 판단에 중요합니다.\n\n"
            "시급 인상 등 고용상 이익과 성관계 요구가 연결되었는지 → 직장 내 성희롱 판단에 중요합니다.\n\n"
            "거절 후 근무시간 축소, 해고, 시급 삭감 같은 불이익이 있었는지 → 불리한 처우나 강요 문제와 연결될 수 있습니다.\n\n"
            "지금 할 일\n"
            "- 카톡, 문자, 녹음, 근무표 등 제안 내용을 확인할 자료를 보관하세요.\n"
            "- 원치 않는다면 거절 의사를 문자 등 기록이 남는 방식으로 남기는 것을 고려하세요.\n"
            "- 불이익이나 압박이 있으면 고용노동부 1350 또는 관할 노동청에 상담·진정을 문의할 수 있습니다.\n"
            "- 신변 위협이나 강제적 접촉이 있으면 112에 신고하세요.\n\n"
            "도움을 받을 수 있는 곳\n"
            "- 고용노동부 고객상담센터: 1350\n"
            "- 관할 노동청: 직장 내 성희롱·불리한 처우 진정 문의\n"
            "- 여성긴급전화: 1366\n"
            "- 경찰 긴급신고: 112"
        )
    return (
        "상황 정리\n"
        "제공된 사실관계만으로 확정적 판단은 어렵지만, 입력된 사실을 기준으로 관련 법적 쟁점을 검토할 수 있습니다.\n\n"
        "법적 판단\n"
        f"주된 쟁점은 {issue_plan.primary_issue}로 보입니다. 부족한 사실은 추가 확인사항으로 두고, 현재 확인된 사실을 기준으로 보수적으로 판단해야 합니다.\n\n"
        "지금 할 일\n"
        "- 관련 대화, 문자, 카톡, 녹음, 계약서, 근무표 등 자료를 보관하세요.\n"
        "- 긴급 위험이 있거나 불이익이 예상되면 관련 기관 상담을 이용하세요."
    )
