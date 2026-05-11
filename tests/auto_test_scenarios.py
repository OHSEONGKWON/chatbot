"""자동화된 시나리오 테스트 - 병합된 코드 전체 흐름 검증"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import pipeline


class TestScenario:
    def __init__(self, name: str, question: str, expected_step: int, expected_needs_requery: bool):
        self.name = name
        self.question = question
        self.expected_step = expected_step
        self.expected_needs_requery = expected_needs_requery

    async def run(self, user_id: str):
        print(f"\n{'='*80}")
        print(f"📋 테스트: {self.name}")
        print(f"{'='*80}")
        print(f"💬 질문: {self.question}")
        print(f"📌 예상: step={self.expected_step}, needs_requery={self.expected_needs_requery}")
        print("-" * 80)

        try:
            result = await pipeline.process(user_id=user_id, user_input=self.question)

            print(f"✅ 실행 완료!")
            print(f"   step_reached: {result.step_reached}")
            print(f"   needs_requery: {result.needs_requery}")
            print(f"   legal_category: {result.legal_category}")
            print(f"   consistency_score: {result.consistency_score}")
            print(f"   legal_reasoning_score: {result.legal_reasoning_score}")
            print(f"   was_ner_corrected: {result.was_ner_corrected}")
            print(f"\n📝 답변:")
            print(f"   {result.response_text[:200]}..." if len(result.response_text) > 200 else f"   {result.response_text}")

            # 검증
            step_match = result.step_reached == self.expected_step
            requery_match = result.needs_requery == self.expected_needs_requery

            status = "✓ PASS" if (step_match and requery_match) else "✗ FAIL"
            print(f"\n{status}")
            if not step_match:
                print(f"   ⚠️ step 불일치: 예상={self.expected_step}, 실제={result.step_reached}")
            if not requery_match:
                print(f"   ⚠️ requery 불일치: 예상={self.expected_needs_requery}, 실제={result.needs_requery}")

            return step_match and requery_match

        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    print("\n🧪 LawsGuard 병합 코드 자동 테스트")
    print("=" * 80)

    scenarios = [
        # 시나리오 1: 모호한 질문 → 재질문 나와야 함
        TestScenario(
            name="시나리오 1: 모호한 성범죄 질문 (재질문 예상)",
            question="술자리에서 후배를 만졌는데 처벌받을 수 있나요?",
            expected_step=0,
            expected_needs_requery=True,
        ),

        # 시나리오 2: 구체적인 노동 관련 질문 → 최종 답변
        TestScenario(
            name="시나리오 2: 구체적인 임금 체불 질문 (최종 답변 예상)",
            question="편의점 알바를 2026년 4월부터 5월까지 주 20시간 했는데 월급 80만원을 못 받았습니다. 이 경우 어떻게 해야 하나요?",
            expected_step=7,
            expected_needs_requery=False,
        ),

        # 시나리오 3: 희소한 질문 → 재질문
        TestScenario(
            name="시나리오 3: 희소한 퇴직금 질문 (재질문 예상)",
            question="퇴직금 못 받았어요",
            expected_step=0,
            expected_needs_requery=True,
        ),

        # 시나리오 4: 법률 관련 일반 질문 → 최종 답변
        TestScenario(
            name="시나리오 4: 교육기관 성희롱 질문 (최종 답변 예상)",
            question="대학교에서 교수가 학생에게 부적절한 신체 접촉을 했습니다. 이것이 성희롱에 해당하나요?",
            expected_step=7,
            expected_needs_requery=False,
        ),

        # 시나리오 5: 모호한 계약 관련 질문 → 재질문
        TestScenario(
            name="시나리오 5: 불명확한 계약 질문 (재질문 예상)",
            question="계약을 취소하고 싶어요",
            expected_step=0,
            expected_needs_requery=True,
        ),
    ]

    results = []
    for i, scenario in enumerate(scenarios, 1):
        user_id = f"auto-test-user-{i}"
        passed = await scenario.run(user_id)
        results.append((scenario.name, passed))

    # 최종 결과
    print(f"\n\n{'='*80}")
    print("📊 최종 테스트 결과")
    print("=" * 80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")

    print(f"\n총 {passed_count}/{total_count} 테스트 통과")

    if passed_count == total_count:
        print("🎉 모든 테스트가 통과했습니다!")
    else:
        print(f"⚠️ {total_count - passed_count}개 테스트 실패")

    return passed_count == total_count


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
