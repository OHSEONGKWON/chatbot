import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import pipeline


async def main() -> None:
    user_id = "manual-user"
    print("LawsGuard manual test")
    print("Type your legal question. Type 'exit' to quit.")

    while True:
        question = input("\nQ> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Bye")
            break

        try:
            result = await pipeline.process(user_id=user_id, user_input=question)
            print(f"step_reached: {result.step_reached}")
            print(f"needs_requery: {result.needs_requery}")
            print(f"consistency_score: {result.consistency_score}")
            print(f"legal_reasoning_score: {result.legal_reasoning_score}")
            print(f"was_ner_corrected: {result.was_ner_corrected}")
            print("A>", result.response_text)
        except Exception as e:
            print("ERROR:", e)


if __name__ == "__main__":
    asyncio.run(main())
