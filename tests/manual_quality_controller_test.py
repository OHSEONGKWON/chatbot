# -*- coding: utf-8 -*-
"""
manual_quality_controller_test.py

lawsguard_quality_controller.py 독립 실행 테스트.

실행:
cd /d C:\chatbot
C:\Programs\Python\Python312\python.exe tests\manual_quality_controller_test.py

테스트 내용
1. 운영 모드
   - 빠른 Guard 중심
   - SQQS/LCS 비활성화
   - 실시간 카카오톡 응답용

2. 평가 모드
   - LRQS/SQQS/LCS 전체 계산
   - 유사 질문/답변은 stub 사용

3. 디버그 모드
   - 전체 점수, 질문, 답변, 유사 질문, 유사 질문 답변 출력

실제 pipeline 연결 시:
- answer_question_stub → 기존 pipeline 답변 생성 함수로 교체
- generate_similar_questions_stub → 기존 consistency_checker 유사 질문 생성 함수로 교체
- rag_docs → rag.py 검색 결과 전달
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import List


def _load_controller():
    try:
        from src.modules import lawsguard_quality_controller as qc
        return qc
    except ModuleNotFoundError:
        here = Path(__file__).resolve()
        candidates = [
            here.parents[1] / "src" / "modules" / "lawsguard_quality_controller.py",
            Path("C:/chatbot/src/modules/lawsguard_quality_controller.py"),
            Path.cwd() / "src" / "modules" / "lawsguard_quality_controller.py",
            here.parents[1] / "modules" / "lawsguard_quality_controller.py",
            Path.cwd() / "modules" / "lawsguard_quality_controller.py",
        ]
        for path in candidates:
            if path.exists():
                spec = importlib.util.spec_from_file_location("lawsguard_quality_controller", path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules["lawsguard_quality_controller"] = module
                spec.loader.exec_module(module)
                return module
        raise FileNotFoundError(
            "lawsguard_quality_controller.py를 찾을 수 없습니다.\n"
            "C:\\chatbot\\src\\modules\\lawsguard_quality_controller.py 위치에 배치하세요."
        )


qc = _load_controller()


def generate_similar_questions_stub(question: str) -> List[str]:
    if "허벅지" in question or "성추행" in question or "만졌" in question:
        return [
            "학교 술자리에서 선배가 제 허벅지를 만졌는데 성추행으로 볼 수 있나요?",
            "동아리 선배가 동의 없이 신체를 만진 경우 신고할 수 있나요?",
            "술자리에서 원치 않는 신체접촉을 당했는데 증거가 부족하면 어떻게 해야 하나요?",
            "선배가 장난이라고 하면서 몸을 만졌는데 법적으로 문제가 되나요?",
            "대학 선배의 원치 않는 신체접촉에 대해 학교와 경찰에 어떻게 대응해야 하나요?",
        ]

    if "알바비" in question or "임금" in question or "주휴수당" in question:
        return [
            "알바비 일부를 받지 못했는데 노동청에 진정할 수 있나요?",
            "주휴수당을 안 주는 사장에게 어떤 자료를 모아야 하나요?",
            "근로계약서가 없어도 임금체불을 주장할 수 있나요?",
            "최저임금보다 적게 받은 경우 어떻게 대응해야 하나요?",
            "퇴사 후에도 밀린 임금을 청구할 수 있나요?",
        ]

    return [
        question + " 이 경우 법적으로 어떤 쟁점이 있나요?",
        question + " 관련해서 어떤 증거가 필요할까요?",
        question + " 어떻게 대응하는 것이 좋을까요?",
    ]


def answer_question_stub(question: str) -> str:
    if "허벅지" in question or "성추행" in question or "신체" in question or "만졌" in question:
        return (
            "원치 않는 신체접촉이 있었다면 강제추행 또는 성희롱 문제가 될 가능성이 있습니다. "
            "다만 최종 판단은 접촉 부위, 행위 당시 상황, 동의 여부, 행위자와의 관계, 반복성, "
            "피해 직후의 메시지·목격자·CCTV 등 증거에 따라 달라집니다. "
            "가능하면 당시 상황을 시간순으로 기록하고, 카카오톡 메시지나 주변 목격자 진술을 확보한 뒤 "
            "학교 인권센터·상담센터 또는 경찰 상담을 검토할 수 있습니다."
        )

    if "임금" in question or "알바비" in question or "주휴수당" in question or "최저임금" in question:
        return (
            "임금이나 주휴수당이 지급되지 않았다면 임금체불 문제가 될 수 있습니다. "
            "근무기간, 실제 근무시간, 약정 시급, 지급 내역, 출퇴근 기록, 근로계약서 또는 문자 내역을 "
            "확인해야 합니다. 자료를 정리한 뒤 사업주에게 지급을 요구하고, 해결되지 않으면 "
            "고용노동부 진정을 검토할 수 있습니다."
        )

    return (
        "현재 질문만으로는 단정적인 결론을 내리기 어렵습니다. "
        "사실관계, 상대방, 피해 내용, 증거, 원하는 해결 방향을 기준으로 추가 검토가 필요합니다."
    )


def bad_answer_stub(question: str) -> str:
    # Answer Guard가 보정해야 하는 나쁜 답변 예시
    return "무조건 성립합니다. 바로 처벌됩니다."


def run_case(title: str, question: str, mode: str, latest_user_text: str | None = None, use_bad_answer: bool = False) -> None:
    print("\n" + "#" * 80)
    print(title)
    print("#" * 80)

    original_answer = bad_answer_stub(question) if use_bad_answer else answer_question_stub(question)

    rag_docs = [
        {
            "content": (
                "성희롱·강제추행 판단에서는 신체접촉의 내용, 동의 여부, 행위자와 피해자의 관계, "
                "당시 상황, 피해 직후 메시지, 목격자, CCTV 등 증거가 중요하다. "
                "학교 사안에서는 인권센터, 상담센터, 징계 절차 등도 검토될 수 있다."
            )
        },
        {
            "content": (
                "임금체불 사안에서는 근무기간, 근무시간, 약정 임금, 실제 지급액, "
                "근로계약서, 출퇴근 기록, 급여 입금 내역 등이 핵심 자료가 된다. "
                "해결되지 않으면 고용노동부 진정을 검토할 수 있다."
            )
        },
    ]

    result = qc.evaluate_quality(
        question=question,
        original_answer=original_answer,
        mode=mode,
        rag_docs=rag_docs,
        latest_user_text=latest_user_text,
        asked_history=[
            "카톡, 문자, CCTV, 목격자, 직후 기록처럼 확인 가능한 자료가 있는지 알려주실 수 있나요?"
        ],
        requery_count=1,
        generate_similar_questions=generate_similar_questions_stub,
        answer_question=answer_question_stub,
    )

    qc.print_quality_summary(result)

    if mode in {"evaluation", "debug"}:
        qc.print_questions_and_answers(result)

    print("\n[최종 decision]")
    print(result.decision.value)
    print("\n[최종 답변]")
    print(result.final_answer)


if __name__ == "__main__":
    # 1. 운영 모드: 빠른 Guard 중심
    run_case(
        title="[1] 운영 모드: 실시간 응답용",
        question="선배가 술자리에서 내 허벅지를 만졌어요. 이거 성추행인가요?",
        mode="operation",
    )

    # 2. 평가 모드: LRQS/SQQS/LCS 전체 계산
    run_case(
        title="[2] 평가 모드: LRQS/SQQS/LCS 전체 측정",
        question="선배가 술자리에서 내 허벅지를 만졌어요. 이거 성추행인가요?",
        mode="evaluation",
    )

    # 3. 디버그 모드: 전체 출력
    run_case(
        title="[3] 디버그 모드: 전체 출력",
        question="알바비랑 주휴수당을 못 받았어요. 어떻게 해야 하나요?",
        mode="debug",
    )

    # 4. 재질문 반복 방지: 사용자가 증거 없다고 답한 경우
    run_case(
        title="[4] 재질문 반복 방지: 사용자가 '없어'라고 답한 경우",
        question="선배가 술자리에서 내 허벅지를 만졌어요. 이거 성추행인가요?",
        mode="operation",
        latest_user_text="증거는 없어",
    )

    # 5. 나쁜 답변 보정: 단정 표현 완화
    run_case(
        title="[5] Answer Guard 보정: 단정 답변 완화",
        question="선배가 술자리에서 내 허벅지를 만졌어요. 이거 성추행인가요?",
        mode="operation",
        use_bad_answer=True,
    )
