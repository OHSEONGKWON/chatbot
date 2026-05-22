import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modules.answer_formatter import answer_formatter
from src.modules.case_frame_contract import extract_case_frame, build_issue_plan, build_answer_contract


async def main():
    question = "술자리에서 선배가 내 허벅지 만졌어요"
    case_frame = extract_case_frame(question, context=question)
    issue_plan = build_issue_plan(case_frame, "성폭력")
    answer_contract = build_answer_contract(case_frame, issue_plan)
    rag_docs = [
        {"text": "형법 제298조 강제추행 관련 근거", "metadata": {"law_name": "형법", "article_id": "제298조"}},
        {"text": "형법 제299조 준강제추행 관련 근거", "metadata": {"law_name": "형법", "article_id": "제299조"}},
        {"text": "공공부문 성희롱 성폭력 사건 처리 매뉴얼 상담 및 신고 접수", "metadata": {"source_file": "공공부문 성희롱 성폭력 사건 처리 매뉴얼.pdf", "section_label": "상담 및 신고 접수"}},
    ]
    result = await answer_formatter.format(
        question=question,
        draft_answer="주된 쟁점은 강제추행 검토로 보입니다.",
        rag_docs=rag_docs,
        legal_category="성폭력",
        case_frame=case_frame,
        issue_plan=issue_plan,
        answer_contract=answer_contract,
        rrs_report={"score": 0.54},
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
