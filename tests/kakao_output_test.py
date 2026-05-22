"""파이프라인 전체(명확화→RAG→일관성→NER→포맷)를 실행하고 카카오톡 출력 형태로 확인하는 스크립트."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import pipeline
from src.modules.kakao_response import default_quick_replies, split_for_kakao


def print_kakao_preview(result) -> None:
    print("\n" + "=" * 60)
    print("[ 카카오톡 출력 미리보기 ]")
    print("=" * 60)

    chunks = split_for_kakao(result.response_text)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[말풍선 {i}/{len(chunks)}]")
        print("-" * 40)
        print(chunk)

    qr = default_quick_replies(
        needs_requery=result.needs_requery,
        category=result.legal_category,
    )
    if qr:
        print("\n[빠른 답장 버튼]")
        for btn in qr:
            print(f"  ▶ {btn['label']} → \"{btn['messageText']}\"")

    print("\n" + "-" * 60)
    print(f"step_reached     : {result.step_reached}  (0=재질문, 4=일관성실패, 7=정상완료)")
    print(f"needs_requery    : {result.needs_requery}")
    print(f"legal_category   : {result.legal_category}")
    print(f"answer_reliability: {result.answer_reliability}")
    print(f"qafs             : {result.qafs}")
    print(f"rrs              : {result.rrs}")
    print(f"was_ner_corrected: {result.was_ner_corrected}")
    print("=" * 60)


async def main() -> None:
    user_id = "kakao-test-user"
    print("LawsGuard 카카오톡 전체 파이프라인 테스트")
    print("  질문 입력 → 명확화 → RAG → 일관성검증 → NER/환각교정 → 최종답변 → 카카오톡 포맷 출력")
    print("  종료: exit\n")

    while True:
        question = input("Q> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("종료합니다.")
            break

        print("\n처리 중 (일관성 검증에 30~60초 소요될 수 있습니다)...\n")
        try:
            result = await pipeline.process(user_id=user_id, user_input=question)
            print_kakao_preview(result)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
