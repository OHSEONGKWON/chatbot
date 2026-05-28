from __future__ import annotations

import re
from typing import Any


class AnswerFormatter:
    async def format(
        self,
        question: str,
        draft_answer: str,
        rag_docs: list[dict[str, Any]],
        legal_category: str,
        is_general: bool = False,
        *,
        case_frame: Any | None = None,
        issue_plan: Any | None = None,
        answer_contract: Any | None = None,
        rrs_report: dict[str, Any] | None = None,
    ) -> str:
        """
        최종 답변을 계약 기반 구조화 형식으로 출력한다.

        핵심 원칙
        1. AnswerContract.must_not_frame_as를 가장 먼저 반영한다.
        2. IssuePlan.main_domain / primary_issue를 템플릿 선택의 1차 기준으로 쓴다.
        3. RAG 문서나 draft_answer의 키워드는 보조 신호로만 사용한다.
        4. 성폭력/노동 복합 템플릿은 질문 또는 CaseFrame에 노동 사실이 명시된 경우에만 허용한다.
        5. 참고 근거는 법령명-조문번호 매칭이 안전한 경우만 출력한다.
        """
        question = question or ""
        answer = self._fix_known_contacts(self._clean_answer((draft_answer or "").strip()))
        category = self._select_category(
            question=question,
            draft_answer=answer,
            legal_category=legal_category,
            case_frame=case_frame,
            issue_plan=issue_plan,
            answer_contract=answer_contract,
        )
        sources = self._source_lines(
            rag_docs,
            category=category,
            issue_plan=issue_plan,
            rrs_report=rrs_report,
        )
        return self._structured_answer(
            question=question,
            draft_answer=answer,
            legal_category=category,
            sources=sources,
            is_general=is_general,
            case_frame=case_frame,
            issue_plan=issue_plan,
            answer_contract=answer_contract,
        )

    # ------------------------------------------------------------------
    # 객체/딕셔너리 접근 유틸
    # ------------------------------------------------------------------

    def _korean_particle(self, word: str, consonant_form: str, vowel_form: str) -> str:
        if not word:
            return vowel_form
        code = ord(word[-1])
        if 0xAC00 <= code <= 0xD7A3:
            return consonant_form if (code - 0xAC00) % 28 != 0 else vowel_form
        return vowel_form

    def _as_dict(self, obj: Any) -> dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "to_dict"):
            try:
                data = obj.to_dict()
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        data: dict[str, Any] = {}
        for name in (
            "actor", "target", "relationship", "requested_act", "performed_act", "condition_or_exchange",
            "threat_or_disadvantage", "unpaid_claim", "body_part", "evidence", "primary_issue",
            "main_domain", "secondary_issues", "excluded_issues", "fact_slots", "missing_slots",
            "needed_laws", "answer_focus", "confidence", "must_include", "must_not_frame_as", "allowed_secondary",
        ):
            if hasattr(obj, name):
                data[name] = getattr(obj, name)
        return data

    def _field(self, obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _list_field(self, obj: Any, key: str) -> list[str]:
        value = self._field(obj, key, [])
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value if str(v).strip()]
        return [str(value)] if str(value).strip() else []

    # ------------------------------------------------------------------
    # 템플릿 선택: 금지 규칙 → IssuePlan → 명시적 사실 → legal_category 순서
    # ------------------------------------------------------------------

    def _select_category(
        self,
        question: str,
        draft_answer: str,
        legal_category: str,
        case_frame: Any | None,
        issue_plan: Any | None,
        answer_contract: Any | None,
    ) -> str:
        contract = self._as_dict(answer_contract)
        plan = self._as_dict(issue_plan)
        frame = self._as_dict(case_frame)

        blocked = set(str(x) for x in contract.get("must_not_frame_as", []) or [])
        primary = str(plan.get("primary_issue") or "")
        domain = str(plan.get("main_domain") or "")
        raw = (legal_category or "").strip()

        explicit_labor = self._has_explicit_labor_fact(question, frame)
        explicit_sex = self._has_explicit_sex_fact(question, frame) or domain == "성폭력" or "강제추행" in primary or "성희롱" in primary
        school_status = self._has_school_discipline_signal(question)

        # 금지 쟁점이 최우선이다. RAG나 draft에 노동 단어가 섞여도 여기서 차단한다.
        labor_blocked = any(x in blocked for x in ("임금체불", "노동", "부당해고", "직장 내 괴롭힘"))
        sex_blocked = any(x in blocked for x in ("강제추행", "성폭력", "성희롱", "불법촬영"))

        if school_status and not explicit_sex and not explicit_labor:
            return "학교/생활관"

        if explicit_sex and explicit_labor and not labor_blocked and not sex_blocked:
            return "성폭력/노동"

        if explicit_sex and not sex_blocked:
            return "성폭력"

        if (domain == "노동" or raw == "노동" or explicit_labor) and not labor_blocked:
            return "노동"

        # raw가 성범죄/성희롱이면 성폭력으로 정규화한다.
        if raw in {"성범죄", "성희롱", "성폭력"}:
            return "성폭력"
        if raw:
            return raw
        return "일반 법률"

    def _has_explicit_labor_fact(self, question: str, frame: dict[str, Any]) -> bool:
        text = question or ""
        if frame.get("unpaid_claim"):
            return True
        if frame.get("condition_or_exchange") and any(k in str(frame.get("condition_or_exchange")) for k in ("시급", "월급", "급여", "돈", "근무", "채용", "평가", "학점", "추천서")):
            return True
        labor_terms = (
            "임금", "알바비", "주휴수당", "최저임금", "급여", "월급", "시급", "퇴직금", "근로계약",
            "해고", "권고사직", "근무시간", "출퇴근", "사업주", "사장", "점장", "상사", "직장", "회사", "알바",
        )
        return any(k in text for k in labor_terms)

    def _has_explicit_sex_fact(self, question: str, frame: dict[str, Any]) -> bool:
        text = " ".join(str(x or "") for x in (question, frame.get("performed_act"), frame.get("body_part"), frame.get("requested_act")))
        sex_terms = (
            "성폭력", "성범죄", "성희롱", "성추행", "강제추행", "준강제추행", "강간", "준강간",
            "불법촬영", "몰카", "촬영", "유포", "허벅지", "가슴", "엉덩이", "키스", "입맞춤",
            "만졌", "신체접촉", "성관계", "성적",
        )
        return any(k in text for k in sex_terms)

    def _has_school_discipline_signal(self, question: str) -> bool:
        return any(k in (question or "") for k in ("생활관", "기숙사", "벌점", "입실 제한", "퇴실", "학칙", "징계"))

    # ------------------------------------------------------------------
    # 구조화 답변 생성
    # ------------------------------------------------------------------

    def _structured_answer(
        self,
        question: str,
        draft_answer: str,
        legal_category: str,
        sources: str,
        is_general: bool = False,
        *,
        case_frame: Any | None = None,
        issue_plan: Any | None = None,
        answer_contract: Any | None = None,
    ) -> str:
        category = legal_category or "일반 법률"

        if category == "성폭력":
            _frame = self._as_dict(case_frame)
            questioner_role = str(_frame.get("questioner_role") or "피해자")

            if questioner_role == "피의자":
                situation = self._sex_situation_suspect(question, case_frame)
                legal_judgment = self._sex_legal_judgment_suspect(question, draft_answer, issue_plan)
                check_items = self._sex_check_items_suspect(question, case_frame)
                if self._is_photo_upload_case(question):
                    actions = [
                        "SNS·메신저에 올린 게시물을 지금 바로 삭제하세요.",
                        "여자친구(상대방)와 나눈 카카오톡·문자 대화를 캡처해 보관하세요 (동의 여부 확인용).",
                        "사진을 게시하게 된 경위와 대화 내용을 시간순으로 메모하세요.",
                        "상대방이 실제로 신고·고소한다면 즉시 변호사 상담을 받으세요.",
                    ]
                else:
                    actions = [
                        "당시 상황과 경위를 기억나는 대로 시간순으로 메모하세요.",
                        "CCTV, 목격자 등 상황을 객관적으로 확인할 수 있는 자료를 확보하세요.",
                        "상대방으로부터 연락이 오거나 경찰 조사 통보를 받으면 혼자 대응하지 말고 변호사 상담을 먼저 받으세요.",
                        "경찰 조사 시 진술 전 반드시 변호사 조력을 요청하세요.",
                    ]
                help_places = [
                    "대한법률구조공단: 132 (무료 법률상담)",
                    "대한변호사협회 법률상담센터: 1566-0500",
                    "경찰 민원 안내: 182",
                ]
            else:
                situation = self._sex_situation(question, case_frame)
                legal_judgment = self._sex_legal_judgment(question, draft_answer, issue_plan)
                check_items = self._sex_check_items(question, issue_plan, case_frame)
                _relationship = str(_frame.get("relationship") or "")
                _is_school = any(k in _relationship for k in ("학교", "교육", "선배", "교수", "동아리")) or \
                             any(k in question for k in ("학교", "동아리", "선배", "교수", "캠퍼스"))
                actions = [
                    "사건 직후 기억나는 내용을 시간순으로 메모하세요.",
                    "함께 있던 사람, 카카오톡 대화, 사과 메시지, CCTV, 장소 정보를 정리하세요.",
                ]
                if _is_school:
                    actions.append("학교 사건이면 학교 인권센터·성평등센터·학생상담센터에 상담 또는 신고를 요청할 수 있습니다.")
                actions.append("긴급하거나 신변 위협이 있으면 즉시 경찰에 신고하세요.")
                help_places = [
                    "경찰 긴급신고: 112",
                    "여성긴급전화: 1366",
                    "해바라기센터: 성폭력 피해 상담·의료·수사 연계 지원",
                    "학교 인권센터·성평등센터: 학내 상담·신고·보호조치 문의",
                ]
        elif category == "노동":
            situation = self._labor_situation(question)
            legal_judgment = self._labor_legal_judgment(question, draft_answer)
            check_items = self._labor_check_items(question)
            actions = [
                "근무일, 근무시간, 지급받은 금액과 미지급 금액을 표로 정리하세요.",
                "근로계약서, 출퇴근 기록, 급여 입금 내역, 문자·카카오톡 대화를 보관하세요.",
                "사업주에게 미지급 임금 지급 또는 부당한 조치의 시정을 요구하고, 해결되지 않으면 고용노동부 진정이나 노동위원회 구제절차를 검토하세요.",
            ]
            help_places = [
                "고용노동부 고객상담센터: 1350",
                "고용노동부 노동포털: 임금체불·근로조건 진정",
                "대한법률구조공단: 132",
            ]
        elif category == "성폭력/노동":
            situation = (
                "성적 요구나 원치 않는 신체접촉이 임금·근무조건·채용·평가 등 노동관계상 불이익과 명시적으로 연결된 사안입니다. "
                "성폭력·성희롱 쟁점과 노동관계상 불이익 또는 보복 문제를 분리하되, 연결된 사실관계로 함께 검토해야 합니다."
            )
            legal_judgment = self._mixed_legal_judgment(question, draft_answer)
            check_items = [
                ("성적 요구나 신체접촉이 있었는지", "성희롱·강제추행 등 성폭력 쟁점 판단에 중요합니다."),
                ("임금·시급·근무조건·채용·평가 등과 연결된 요구였는지", "노동관계상 불이익 또는 조건부 성적 요구인지 판단하는 데 중요합니다."),
                ("거부 후 불이익이 있었는지", "보복성 조치, 직장 내 성희롱 후속 조치 의무, 손해배상 가능성과 관련됩니다."),
                ("증거가 무엇인지", "카카오톡, 통화녹음, 급여내역, 근무표, 목격자 진술이 핵심 자료가 됩니다."),
            ]
            actions = [
                "성적 요구·접촉과 임금·근무조건 관련 대화를 분리해 시간순으로 정리하세요.",
                "카카오톡, 문자, 통화녹음, 급여명세, 근무표, 출퇴근 기록을 보관하세요.",
                "성폭력 피해지원기관과 고용노동부 상담을 함께 검토하세요.",
            ]
            help_places = [
                "경찰 긴급신고: 112",
                "여성긴급전화: 1366",
                "고용노동부 고객상담센터: 1350",
                "학교 인권센터·성평등센터 또는 직장 내 고충처리 창구",
            ]
        elif category == "학교/생활관":
            situation = "학교 또는 생활관에서 벌점, 퇴실, 입실 제한, 징계와 같은 처분을 받은 사안입니다. 내부 규정 위반 여부뿐 아니라 처분 절차와 불복 가능성을 함께 봐야 합니다."
            legal_judgment = "현재 입력만으로 처분의 위법성을 단정할 수는 없지만, 처분 주체, 규정 근거, 사전 통지와 의견 제출 기회, 불복 기간 안내가 부족했다면 절차상 문제가 될 수 있습니다."
            check_items = [
                ("처분 주체가 누구인지", "생활관장, 학생처, 징계위원회 등 권한 있는 주체인지 판단하는 데 필요합니다."),
                ("어떤 처분을 받았는지", "벌점, 퇴실, 다음 학기 입실 제한 등 처분의 불이익 정도를 판단합니다."),
                ("근거 규정이 제시됐는지", "생활관 규정 또는 학칙에 근거가 있는지 확인해야 합니다."),
                ("의견 제출 기회가 있었는지", "절차적 정당성 판단에 중요합니다."),
            ]
            actions = [
                "처분 통지서, 벌점 내역, 생활관 규정, 안내문을 모아두세요.",
                "처분 날짜와 이의신청 기간을 확인하세요.",
                "학교 내부 이의신청, 학생처 상담, 필요 시 행정심판·행정소송 가능성을 검토하세요.",
            ]
            help_places = ["학교 학생처·생활관 행정실", "학교 인권센터 또는 학생상담센터", "대한법률구조공단: 132"]
        else:
            situation = "현재 질문은 구체적인 사실관계에 따라 법적 판단이 달라질 수 있는 사안입니다."
            legal_judgment = self._general_legal_judgment(draft_answer)
            check_items = [
                ("상대방이 누구인지", "법적 책임 주체와 적용 절차를 판단하는 데 필요합니다."),
                ("어떤 행위와 피해가 있었는지", "민사·형사·행정 절차 중 어떤 방향인지 판단하는 데 중요합니다."),
                ("증거가 있는지", "주장 입증 가능성을 판단하는 데 중요합니다."),
                ("원하는 해결 방향이 무엇인지", "신고, 합의, 손해배상, 이의신청 등 대응 절차를 정하는 데 필요합니다."),
            ]
            actions = ["사건 경위를 시간순으로 정리하세요.", "문자, 카카오톡, 사진, 녹음, 계약서 등 관련 자료를 보관하세요.", "기한이 있는 절차인지 확인하고 필요하면 전문기관 상담을 이용하세요."]
            help_places = ["대한법률구조공단: 132", "정부민원안내콜센터: 110", "긴급 위험이 있으면 경찰: 112"]

        check_text = "\n\n".join(f"{item} → {meaning}" for item, meaning in check_items)
        action_text = "\n".join(f"- {item}" for item in actions)
        help_text = "\n".join(f"- {item}" for item in help_places)
        source_block = sources or "직접 인용할 근거 문서를 충분히 찾지 못했습니다."
        general_note = "제공된 정보가 부족하여 일반적인 기준 중심으로 안내합니다.\n\n" if is_general else ""

        text = f"""[분류: {category}]

1. 상황 정리
{general_note}{situation}

2. 법적 판단
{legal_judgment}

3. 추가로 확인할 사항과 법적 의미

{check_text}

4. 지금 할 일

{action_text}

5. 도움을 받을 수 있는 곳

{help_text}

※ 이 답변은 제공된 사실관계만을 바탕으로 한 일반 안내입니다.

[참고한 근거]

{source_block}

※ 이 답변은 법률정보 제공을 위한 일반 안내입니다. 긴급한 위험이 있거나 기한이 임박한 경우 전문기관 또는 변호사 상담을 우선 이용하세요."""
        return self._fix_known_contacts(re.sub(r"\n{3,}", "\n\n", text).strip())

    # ------------------------------------------------------------------
    # 사안별 문장 생성
    # ------------------------------------------------------------------

    @staticmethod
    def _is_photo_upload_case(question: str) -> bool:
        return any(k in question for k in ("사진", "영상", "동영상", "올렸", "올린", "게시", "유포", "공유", "인스타", "sns", "SNS", "카톡"))

    def _sex_situation_suspect(self, question: str, case_frame: Any | None = None) -> str:
        if self._is_photo_upload_case(question):
            return (
                "질문자가 상대방(여자친구 등)의 사진·영상을 SNS나 메신저 등에 게시·공유하였고, "
                "상대방이 이를 음란물 유포 또는 초상권·사생활 침해로 신고하겠다고 한 상황입니다. "
                "사전 동의 여부와 사진의 성적 성격에 따라 법적 결론이 크게 달라집니다."
            )

        frame = self._as_dict(case_frame)
        body_part = frame.get("body_part") or ""
        if not body_part:
            for part in ("가슴", "엉덩이", "허벅지", "신체"):
                if part in question:
                    body_part = part
                    break
            else:
                body_part = "신체"

        accidental = any(k in question for k in ("실수로", "우연히", "넘어지면서", "넘어지다가", "부딪히"))
        if accidental:
            return (
                f"질문자가 우연한 사고 또는 실수로 상대방의 {body_part}에 접촉한 사안입니다. "
                "형법상 강제추행 성립 여부는 고의성과 성적 수치심 유발 가능성을 함께 판단하므로, "
                "실수임을 뒷받침하는 경위와 객관적 증거가 중요합니다."
            )
        return (
            f"질문자가 상대방의 {body_part}에 접촉한 사안으로, 고소·처벌 가능성에 대해 문의하고 있습니다. "
            "성추행(강제추행) 성립 여부는 행위의 고의성, 상대방 의사, 당시 상황의 성적 성격 등에 따라 달라집니다."
        )

    def _sex_legal_judgment_suspect(self, question: str, draft_answer: str, issue_plan: Any | None = None) -> str:
        if self._is_photo_upload_case(question):
            is_swimsuit = any(k in question for k in ("수영복", "비키니", "수영"))
            if is_swimsuit:
                swimsuit_note = (
                    "수영복 사진은 통상적인 법적 기준상 음란물에 해당하기 어렵습니다. "
                    "음란물은 사회통념상 성욕을 자극하고 수치심을 일으키는 노골적 성적 표현을 요건으로 하는데, "
                    "일반적인 수영복 착용 사진은 이 기준을 충족하지 않는 경우가 대부분입니다. "
                )
            else:
                swimsuit_note = ""
            return (
                f"{swimsuit_note}"
                "다만 상대방의 동의 없이 사진을 게시했다면 민법상 초상권·사생활 침해로 손해배상 청구 대상이 될 수 있고, "
                "사진의 성적 성격이 강하거나 상대방이 성적 수치심을 느낀다고 주장하면 "
                "정보통신망법상 사생활 침해(제44조의7) 쟁점이 될 수 있습니다. "
                "처벌 가능성은 동의 여부, 사진 내용, 게시 목적에 따라 달라지므로 "
                "즉시 게시물을 삭제하고 변호사 상담을 받는 것이 우선입니다."
            )

        accidental = any(k in question for k in ("실수로", "우연히", "넘어지면서", "넘어지다가", "부딪히"))
        if accidental:
            return (
                "형법상 강제추행(제298조)은 고의로 성적 수치심을 유발하는 신체접촉을 요건으로 합니다. "
                "완전히 우발적인 사고(넘어지면서의 접촉 등)는 고의성 부재를 이유로 강제추행이 성립하지 않을 수 있으나, "
                "상대방의 진술과 당시 상황에 따라 수사기관이 다르게 판단할 수 있습니다. "
                "고의성 여부를 객관적으로 입증할 수 있는 자료(CCTV, 목격자 진술 등)가 핵심입니다."
            )
        return (
            "형법상 강제추행(제298조)은 폭행·협박을 수반하거나 상대방 의사에 반하는 신체접촉으로 "
            "성적 수치심을 유발하는 행위를 말합니다. "
            "고소 가능성은 행위의 고의성, 방법, 상대방 반응과 진술, 증거에 따라 달라지므로 "
            "지금 단계에서 확정하기 어렵습니다. 형사 전문 변호사 상담을 우선하세요."
        )

    def _sex_check_items_suspect(self, question: str, case_frame: Any | None = None) -> list[tuple[str, str]]:
        if self._is_photo_upload_case(question):
            return [
                ("사진·영상 게시 전 상대방 동의를 받았는지", "사전 동의 여부가 처벌 가능성을 결정하는 핵심 요소입니다."),
                ("사진의 성적 성격 — 단순 수영복인지, 신체 노출이 과도한지", "성폭력처벌법 적용 기준인 '성적 수치심 유발 여부'를 판단합니다."),
                ("게시물이 아직 남아 있는지, 즉시 삭제 가능한지", "즉각 삭제는 피해 최소화 의사를 보여줘 수사에 유리하게 작용할 수 있습니다."),
                ("이미 경찰 신고나 고소가 접수됐는지", "신고 접수 여부에 따라 대응 시급성이 달라집니다."),
            ]

        items: list[tuple[str, str]] = [
            ("접촉이 우연·사고로 발생했는지, 고의적이었는지", "강제추행 고의성 판단에 가장 중요한 요소입니다."),
            ("당시 상황을 입증할 CCTV나 목격자가 있는지", "객관적 증거가 고의성 부재 주장을 뒷받침합니다."),
            ("상대방이 어떤 반응을 보였는지", "피해자 진술의 방향과 행위의 성격을 판단하는 데 영향을 줍니다."),
            ("이미 경찰 조사나 고소 통보를 받았는지", "대응 시급성과 절차가 달라집니다."),
        ]
        if any(k in question for k in ("술", "취해", "만취")):
            items.insert(1, ("당시 음주 상태", "본인과 상대방의 음주 상태는 준강제추행(제299조) 쟁점에 영향을 줄 수 있습니다."))
        return items[:4]

    def _sex_situation(self, question: str, case_frame: Any | None = None) -> str:
        frame = self._as_dict(case_frame)
        actor = frame.get("actor") or "상대방"
        relationship = frame.get("relationship") or ""
        performed = frame.get("performed_act") or ""
        body_part = frame.get("body_part") or ""

        if "술" in question or "술자리" in question or "회식" in question:
            place = "술자리에서"
        elif any(k in question for k in ("기숙사", "원룸", "방", "집")):
            place = "기숙사·원룸·방 등 사적 공간에서"
        elif any(k in question for k in ("학교", "동아리", "선배", "교수")) or "학교" in relationship:
            place = "학교 또는 동아리 관계에서"
        else:
            place = "상대방과의 관계에서"

        if not body_part:
            if "허벅지" in question:
                body_part = "허벅지"
            elif "가슴" in question:
                body_part = "가슴"
            elif "엉덩이" in question:
                body_part = "엉덩이"
            elif "키스" in question or "입맞춤" in question:
                body_part = "입맞춤"
            else:
                body_part = "신체"

        act = performed or f"{body_part} 접촉"
        actor_p = self._korean_particle(actor, "이", "가")
        act_p = self._korean_particle(act, "을", "를")
        return (
            f"{place} {actor}{actor_p} 사용자의 {act}{act_p} 한 사안입니다. "
            f"그 접촉이 원치 않는 신체접촉이었다면 단순한 불쾌감 문제를 넘어 성적 자기결정권 침해로 문제될 수 있습니다."
        )

    def _sex_legal_judgment(self, question: str, draft_answer: str, issue_plan: Any | None = None) -> str:
        needed_laws = " ".join(self._list_field(issue_plan, "needed_laws"))
        has_alcohol = any(k in question for k in ("술", "취해", "만취", "회식", "술자리"))
        if has_alcohol:
            base = (
                "현재 입력만으로 죄명을 단정할 수는 없지만, 원치 않는 신체접촉이었다면 "
                "형법상 강제추행 사안으로 검토될 가능성이 있습니다. 특히 술에 취해 의식이 흐리거나 "
                "항거하기 어려운 상태를 이용했다면 준강제추행 또는 준강간 쟁점이 함께 검토될 수 있습니다."
            )
        else:
            base = (
                "현재 입력만으로 죄명을 단정할 수는 없지만, 원치 않는 신체접촉이었다면 "
                "형법상 강제추행 사안으로 검토될 가능성이 있습니다."
            )
        if "제299조" in needed_laws and "준강" not in base:
            base += " 항거곤란 상태가 문제되면 형법 제299조 쟁점도 함께 보아야 합니다."
        if any(k in question for k in ("촬영", "영상", "사진", "몰카", "유포")):
            base += " 동의 없는 촬영이나 유포가 있었다면 성폭력처벌법상 카메라등이용촬영죄 쟁점도 별도로 검토해야 합니다."
        if "성희롱" in question and not any(k in question for k in ("만졌", "접촉", "가슴", "허벅지", "엉덩이")):
            base = (
                "현재 입력만으로 형사범죄 성립을 단정할 수는 없지만, 성적 발언이나 외모 평가, 음담패설 등으로 "
                "성적 굴욕감이나 혐오감을 느꼈다면 성희롱 사안으로 검토될 수 있습니다. 학교나 직장 관계라면 기관의 조사·보호조치 의무도 함께 문제될 수 있습니다."
            )
        return base

    def _sex_check_items(self, question: str, issue_plan: Any | None = None, case_frame: Any | None = None) -> list[tuple[str, str]]:
        frame = self._as_dict(case_frame)
        q = question or ""
        has_alcohol = any(k in q for k in ("술", "취해", "만취", "회식", "술자리"))
        has_private_space = any(k in q for k in ("원룸", "기숙사", "방", "집", "숙소"))
        has_threat = bool(frame.get("threat_or_disadvantage")) or any(k in q for k in ("협박", "흉기", "위험한 물건", "여러 명", "같이 있던"))

        items: list[tuple[str, str]] = []

        # missing_slots에 동의/거부가 있으면 최우선 항목으로 추가 (중복 방지를 위해 아래 거부 항목은 skip)
        missing = " ".join(self._list_field(issue_plan, "missing_slots"))
        include_consent_first = "동의" in missing or "거부" in missing
        if include_consent_first:
            items.append(("동의 또는 거부 의사가 있었는지", "강제추행·준강제추행 여부와 피해자 진술의 핵심 쟁점이 됩니다."))

        items.append(("접촉이 우연이었는지, 의도적·반복적이었는지", "의도적이고 성적 수치심을 일으킬 수 있는 접촉이면 강제추행 판단에 중요합니다."))

        # 동의/거부를 앞에 이미 추가했으면 아래 유사 항목은 생략해 중복 방지
        if not include_consent_first:
            items.append(("당시 거부 의사를 표현했는지, 또는 표현하기 어려운 상황이었는지", "동의 없는 접촉인지, 기습적 추행 또는 폭행·협박을 수반한 추행인지 판단하는 데 중요합니다."))

        if has_alcohol:
            items.append(("술에 취해 의식이 흐리거나 항거하기 어려운 상태였는지", "그런 상태를 이용한 경우 준강제추행 또는 준강간 쟁점이 될 수 있습니다."))

        if has_threat:
            items.append(("여러 명이 함께 했거나 흉기·위험한 물건·협박이 있었는지", "공동범행 또는 위험한 물건 사용이 있으면 특수강제추행·특수강간 쟁점이 될 수 있습니다."))

        if has_private_space:
            items.append(("원룸·기숙사·방 등 주거 공간에 허락 없이 들어온 뒤 접촉이 있었는지", "주거침입과 성범죄가 결합된 사안으로 더 중하게 평가될 수 있습니다."))

        return items[:6]

    def _labor_situation(self, question: str) -> str:
        if any(k in question for k in ("알바비", "임금", "월급", "급여", "주휴수당", "최저임금", "퇴직금")):
            return "아르바이트나 근로 과정에서 임금, 주휴수당, 최저임금, 퇴직금 등이 지급되지 않았다면 단순한 약속 불이행이 아니라 임금체불 문제로 다뤄질 수 있습니다."
        if any(k in question for k in ("해고", "잘렸", "그만 나오", "권고사직")):
            return "근무 중 갑자기 해고 통보를 받았거나 그만 나오라는 말을 들었다면, 해고 사유와 통보 방식에 따라 부당해고 또는 해고예고수당 문제가 될 수 있습니다."
        if any(k in question for k in ("괴롭힘", "폭언", "갑질", "따돌림")):
            return "직장이나 아르바이트 현장에서 폭언, 따돌림, 부당한 업무지시가 있었다면 직장 내 괴롭힘 또는 인격권 침해 문제가 될 수 있습니다."
        return "근로관계에서 임금, 근로조건, 해고, 괴롭힘 등과 관련된 불이익이 발생한 사안입니다."

    def _labor_legal_judgment(self, question: str, draft_answer: str) -> str:
        if any(k in question for k in ("알바비", "임금", "월급", "급여", "주휴수당", "최저임금", "퇴직금")):
            return "현재 입력만으로 체불 금액을 확정할 수는 없지만, 실제 근로를 제공했는데 약정 임금이나 법정수당을 받지 못했다면 근로기준법상 임금체불로 검토될 수 있습니다. 근로계약서가 없더라도 출퇴근 기록, 문자, 급여 입금 내역 등으로 실제 근로를 입증할 수 있습니다."
        if any(k in question for k in ("해고", "잘렸", "그만 나오", "권고사직")):
            return "현재 입력만으로 부당해고를 단정할 수는 없지만, 정당한 이유 없이 해고했거나 해고 사유와 시기를 서면으로 통지하지 않았다면 근로기준법상 해고 제한 또는 해고 서면통지 위반 문제가 될 수 있습니다."
        if any(k in question for k in ("괴롭힘", "폭언", "갑질", "따돌림")):
            return "현재 입력만으로 직장 내 괴롭힘 성립을 단정할 수는 없지만, 업무상 지위나 관계의 우위를 이용해 업무상 적정 범위를 넘는 행위로 신체적·정신적 고통이나 근무환경 악화가 발생했다면 직장 내 괴롭힘으로 검토될 수 있습니다."
        return self._general_legal_judgment(draft_answer)

    def _labor_check_items(self, question: str) -> list[tuple[str, str]]:
        if any(k in question for k in ("해고", "잘렸", "그만 나오", "권고사직")):
            return [
                ("해고 통보를 언제 어떤 방식으로 받았는지", "해고예고수당과 해고 서면통지 위반 여부를 판단하는 데 중요합니다."),
                ("해고 사유가 무엇인지", "정당한 이유가 있는 해고인지 판단하는 핵심 자료입니다."),
                ("근무기간과 사업장 인원", "부당해고 구제신청 가능성과 적용 법령 판단에 필요합니다."),
                ("문자, 카카오톡, 통화녹음 등 통보 자료", "해고 사실과 통보 방식을 입증하는 자료가 됩니다."),
            ]
        if any(k in question for k in ("괴롭힘", "폭언", "갑질", "따돌림")):
            return [
                ("상대방의 지위와 관계", "업무상 우위가 있었는지 판단하는 데 중요합니다."),
                ("행위가 반복되었는지", "단순 갈등인지 괴롭힘인지 구분하는 데 필요합니다."),
                ("업무상 필요성을 넘었는지", "업무상 적정 범위 초과 여부 판단에 중요합니다."),
                ("피해 내용과 증거", "정신적 고통, 근무환경 악화, 녹음·메시지·목격자 자료가 중요합니다."),
            ]
        return [
            ("근무기간과 실제 근무시간", "임금, 주휴수당, 퇴직금 산정의 기초가 됩니다."),
            ("약정 시급·월급과 실제 지급액", "체불 금액을 계산하는 핵심 자료입니다."),
            ("근로계약서 작성 여부", "계약 내용과 사업주의 의무를 확인하는 데 중요합니다."),
            ("출퇴근 기록, 카카오톡, 급여 입금 내역", "실제 근로와 미지급 사실을 입증하는 자료가 됩니다."),
        ]

    def _mixed_legal_judgment(self, question: str, draft_answer: str) -> str:
        return "현재 입력만으로 하나의 죄명이나 노동관계 위반을 단정할 수는 없지만, 성적 요구나 원치 않는 접촉이 있었다면 성희롱·강제추행 쟁점이 될 수 있고, 그 요구가 임금, 시급, 근무시간, 채용, 평가 등과 연결되었다면 노동관계상 불이익 또는 보복 조치 문제도 함께 검토될 수 있습니다."

    def _general_legal_judgment(self, draft_answer: str) -> str:
        cleaned = self._clean_answer(draft_answer)
        if cleaned:
            parts = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
            candidate = parts[0] if parts else cleaned
            if len(candidate) > 350:
                candidate = candidate[:350].rstrip() + "..."
            return candidate
        return "현재 입력만으로 결론을 단정하기는 어렵지만, 상대방의 행위, 피해 내용, 증거, 관련 규정 또는 계약관계에 따라 법적 대응 가능성이 달라질 수 있습니다."

    # ------------------------------------------------------------------
    # 정리/출처
    # ------------------------------------------------------------------

    def _clean_answer(self, text: str) -> str:
        text = text or ""
        text = re.sub(r"\[법리 적용 검토\].*?(?=\n\[참고한 근거\]|\Z)", "", text, flags=re.S)
        text = re.sub(r"\[참고한 근거\].*?\Z", "", text, flags=re.S)
        text = re.sub(r"^\s*\[분류:[^\]]+\]\s*", "", text)
        text = re.sub(r"※\s*이 답변은 제공된 사실관계만을 바탕으로 한 일반 안내입니다\.?,?", "", text)
        text = re.sub(r"※\s*이 답변은 법률정보 제공을 위한 일반 안내입니다.*?(?=\n|\Z)", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _fix_known_contacts(self, text: str) -> str:
        replacements = {
            "경찰 긴급신고: 1366": "경찰 긴급신고: 112",
            "여성긴급전화: 112": "여성긴급전화: 1366",
            "고용노동부 고객상담센터: 132": "고용노동부 고객상담센터: 1350",
            "대한법률구조공단: 1350": "대한법률구조공단: 132",
            "고용노동부: 132": "고용노동부: 1350",
            "법률구조공단: 1350": "법률구조공단: 132",
        }
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        return text

    def _source_lines(self, rag_docs: list[dict[str, Any]], category: str = "", issue_plan: Any | None = None, rrs_report: dict[str, Any] | None = None) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        rrs_score = None
        if isinstance(rrs_report, dict):
            try:
                rrs_score = float(rrs_report.get("score"))
            except Exception:
                rrs_score = None

        # RRS가 낮으면 안전한 기본 근거와 확실한 법령명+조문만 사용한다.
        conservative = rrs_score is not None and rrs_score < 0.55

        for doc in rag_docs or []:
            ref = self._safe_reference_from_doc(doc, category=category, conservative=conservative)
            if not ref or ref in seen:
                continue
            seen.add(ref)
            lines.append(f"{len(lines) + 1}. {ref}")
            if len(lines) >= 3:
                break

        if conservative or not lines:
            defaults = self._default_sources(category)
            for src in defaults:
                if src not in seen:
                    lines.append(f"{len(lines) + 1}. {src}")
                    seen.add(src)
                if len(lines) >= 3:
                    break

        return "\n".join(lines[:3])

    def _safe_reference_from_doc(self, doc: Any, category: str = "", conservative: bool = False) -> str:
        if isinstance(doc, dict):
            metadata = doc.get("metadata") or {}
            text = str(doc.get("text") or doc.get("content") or doc.get("page_content") or "")
            chunk_id = str(doc.get("chunk_id") or "")
        else:
            metadata = getattr(doc, "metadata", {}) or {}
            text = str(getattr(doc, "text", "") or getattr(doc, "page_content", "") or getattr(doc, "content", "") or "")
            chunk_id = str(getattr(doc, "chunk_id", "") or "")

        label = str(
            metadata.get("law_name")
            or metadata.get("source_file")
            or metadata.get("manual_title")
            or metadata.get("case")
            or metadata.get("title")
            or chunk_id
            or self._guess_source_label(text)
            or ""
        ).strip()
        article = str(
            metadata.get("article_id")
            or metadata.get("article_title")
            or metadata.get("section_label")
            or metadata.get("article")
            or self._guess_article(text)
            or ""
        ).strip()

        label = self._normalize_source_label(label)
        article = self._normalize_source_label(article)

        # 법령명-조문번호 안전 결합. 제298/299조는 형법과 함께 확인될 때만 출력한다.
        joined_text = f"{label} {article} {text}"
        if article in {"제298조", "298조"}:
            if "형법" in label or re.search(r"형법\s*제?\s*298조", joined_text):
                return "형법 / 제298조"
            return ""
        if article in {"제299조", "299조"}:
            if "형법" in label or re.search(r"형법\s*제?\s*299조", joined_text):
                return "형법 / 제299조"
            return ""
        if article in {"제14조", "14조"}:
            if "성폭력" in label or "카메라" in joined_text or "촬영" in joined_text:
                return "성폭력처벌법 / 제14조"
            return ""
        if article in {"제43조", "43조"}:
            if "근로기준법" in label or "임금" in joined_text:
                return "근로기준법 / 제43조"
            return ""

        if conservative and not self._is_high_confidence_source(label, text, category):
            return ""

        if not label:
            return ""
        if article and article not in label:
            return f"{label} / {article}"
        return label

    def _is_high_confidence_source(self, label: str, text: str, category: str) -> bool:
        blob = f"{label} {text}"
        if category == "성폭력":
            return any(k in blob for k in ("형법", "성폭력", "성희롱", "강제추행", "공공부문"))
        if category == "노동":
            return any(k in blob for k in ("근로기준법", "고용노동부", "임금", "해고", "괴롭힘"))
        return bool(label)

    def _guess_source_label(self, text: str) -> str:
        if "공공부문" in text and ("성희롱" in text or "성폭력" in text):
            return "공공부문 성희롱·성폭력 사건 처리 매뉴얼"
        if "형법" in text:
            return "형법"
        if "근로기준법" in text:
            return "근로기준법"
        if "남녀고용평등" in text:
            return "남녀고용평등법"
        if "양성평등기본법" in text:
            return "양성평등기본법"
        return ""

    def _guess_article(self, text: str) -> str:
        if re.search(r"형법\s*제?\s*298조", text):
            return "제298조"
        if re.search(r"형법\s*제?\s*299조", text):
            return "제299조"
        if "제14조" in text and "촬영" in text:
            return "제14조"
        if "근로기준법" in text and "제43조" in text:
            return "제43조"
        if "근로기준법" in text and "제23조" in text:
            return "제23조"
        if "근로기준법" in text and "제27조" in text:
            return "제27조"
        return ""

    def _normalize_source_label(self, label: str) -> str:
        label = re.sub(r"\s+", " ", label or "").strip(" /")
        for law in ("형법", "근로기준법", "근로기준법 시행령", "남녀고용평등법", "양성평등기본법"):
            label = label.replace(f"{law} {law}", law)
        label = label.replace("성폭력 성폭력", "성폭력")
        return label.strip(" /")

    def _default_sources(self, category: str) -> list[str]:
        if category == "성폭력":
            return [
                "형법 / 제298조",
                "형법 / 제299조",
                "공공부문 성희롱·성폭력 사건 처리 매뉴얼 / 상담 및 신고 절차",
            ]
        if category == "노동":
            return ["근로기준법 / 임금·근로조건 관련 규정", "고용노동부 노동관계 상담·진정 안내"]
        if category == "성폭력/노동":
            return ["형법 / 제298조", "공공부문 성희롱·성폭력 사건 처리 매뉴얼 / 상담 및 신고 절차", "근로기준법 / 임금·근로조건 관련 규정"]
        return ["관련 법령 및 공공기관 안내자료"]


answer_formatter = AnswerFormatter()
