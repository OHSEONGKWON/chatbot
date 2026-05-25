import asyncio
import uuid
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import pipeline
from src.session_store import session_store


async def main() -> None:
    user_id = f"manual-{uuid.uuid4().hex[:8]}"
    print("LawsGuard manual test")
    print("명령어: 'new' = 새 대화 시작 (세션 초기화), 'exit' = 종료")
    print(f"현재 세션 ID: {user_id}")

    while True:
        question = input("\nQ> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Bye")
            break
        if question.lower() in {"new", "reset", "새로"}:
            session_store.delete(user_id)
            user_id = f"manual-{uuid.uuid4().hex[:8]}"
            print(f"새 세션 시작: {user_id}")
            continue

        try:
            result = await pipeline.process(user_id=user_id, user_input=question)
            print(f"  step_reached    : {result.step_reached}")
            print(f"  needs_requery   : {result.needs_requery}")
            print(f"  legal_category  : {result.legal_category}")
            print(f"  answer_reliability: {result.answer_reliability}")
            print(f"  qafs            : {result.qafs}")
            print(f"  was_ner_corrected: {result.was_ner_corrected}")
            print("A>", result.response_text)
            if result.needs_requery:
                print("  → 재질문 중. 위 질문에 답변 후 계속 입력하세요.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("ERROR:", e)


if __name__ == "__main__":
    asyncio.run(main())
