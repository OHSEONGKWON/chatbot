"""C단계: 병합된 코드 전체 통합 테스트"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import pipeline


async def test_scenario_1():
    """시나리오 1: 모호한 성범죄 질문 → 재질문 시도"""
    print("\n" + "="*80)
    print("🧪 시나리오 1: 모호한 성범죄 질문")
    print("="*80)
    print("💬 사용자 질문: 술자리에서 후배를 만졌는데 이거 처벌 받을 수도 있나?")
    
    try:
        result = await pipeline.process(user_id="test-user-1", user_input="술자리에서 후배를 만졌는데 이거 처벌 받을 수도 있나?")
        
        print(f"\n📊 결과:")
        print(f"   step_reached: {result.step_reached}")
        print(f"   needs_requery: {result.needs_requery}")
        print(f"   legal_category: {result.legal_category}")
        print(f"   response_text: {result.response_text[:150]}...")
        
        return True
    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_scenario_2():
    """시나리오 2: 구체적인 임금체불 질문 → 최종 답변"""
    print("\n" + "="*80)
    print("🧪 시나리오 2: 구체적인 임금체불 질문")
    print("="*80)
    print("💬 사용자 질문: 편의점 알바를 했는데 2026년 4월부터 5월까지 주 20시간 일했고 월급 80만원을 못 받았어요")
    
    try:
        result = await pipeline.process(
            user_id="test-user-2",
            user_input="편의점 알바를 했는데 2026년 4월부터 5월까지 주 20시간 일했고 월급 80만원을 못 받았어요"
        )
        
        print(f"\n📊 결과:")
        print(f"   step_reached: {result.step_reached}")
        print(f"   needs_requery: {result.needs_requery}")
        print(f"   legal_category: {result.legal_category}")
        print(f"   consistency_score: {result.consistency_score}")
        print(f"   legal_reasoning_score: {result.legal_reasoning_score}")
        print(f"   was_ner_corrected: {result.was_ner_corrected}")
        print(f"   response_text: {result.response_text[:200]}...")
        
        # 검증
        step_ok = result.step_reached == 7
        requery_ok = result.needs_requery == False
        
        if step_ok and requery_ok:
            print(f"\n✅ 검증 통과: 최종 답변 단계 도달!")
            return True
        else:
            print(f"\n⚠️ 검증 실패:")
            if not step_ok:
                print(f"   - step: 예상=7, 실제={result.step_reached}")
            if not requery_ok:
                print(f"   - needs_requery: 예상=False, 실제={result.needs_requery}")
            return False
        
    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_scenario_3():
    """시나리오 3: 희소한 퇴직금 질문 → 재질문"""
    print("\n" + "="*80)
    print("🧪 시나리오 3: 희소한 퇴직금 질문")
    print("="*80)
    print("💬 사용자 질문: 퇴직금을 못 받았어요")
    
    try:
        result = await pipeline.process(
            user_id="test-user-3",
            user_input="퇴직금을 못 받았어요"
        )
        
        print(f"\n📊 결과:")
        print(f"   step_reached: {result.step_reached}")
        print(f"   needs_requery: {result.needs_requery}")
        print(f"   legal_category: {result.legal_category}")
        print(f"   response_text: {result.response_text[:150]}...")
        
        # 검증
        requery_ok = result.needs_requery == True
        
        if requery_ok:
            print(f"\n✅ 검증 통과: 재질문이 생성됨!")
            return True
        else:
            print(f"\n⚠️ 검증 실패: 재질문이 필요하지만 생성되지 않음")
            return False
            
    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_scenario_4():
    """시나리오 4: 일반 법률 질문"""
    print("\n" + "="*80)
    print("🧪 시나리오 4: 일반 법률 상담")
    print("="*80)
    print("💬 사용자 질문: 계약서 없이 일한 경우 어떻게 해야 하나요?")
    
    try:
        result = await pipeline.process(
            user_id="test-user-4",
            user_input="계약서 없이 일한 경우 어떻게 해야 하나요?"
        )
        
        print(f"\n📊 결과:")
        print(f"   step_reached: {result.step_reached}")
        print(f"   needs_requery: {result.needs_requery}")
        print(f"   legal_category: {result.legal_category}")
        print(f"   consistency_score: {result.consistency_score}")
        print(f"   response_text: {result.response_text[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("\n" + "🎯 C단계: 병합된 코드 통합 테스트")
    print("=" * 80)
    
    results = {}
    
    print("\n⏳ 테스트 1 실행 중...")
    results["시나리오 1: 모호한 질문"] = await test_scenario_1()
    
    print("\n⏳ 테스트 2 실행 중...")
    results["시나리오 2: 구체적 질문"] = await test_scenario_2()
    
    print("\n⏳ 테스트 3 실행 중...")
    results["시나리오 3: 희소한 질문"] = await test_scenario_3()
    
    print("\n⏳ 테스트 4 실행 중...")
    results["시나리오 4: 일반 상담"] = await test_scenario_4()
    
    # 최종 결과
    print(f"\n\n{'='*80}")
    print("📊 최종 통합 테스트 결과")
    print("=" * 80)
    
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
    
    passed_count = sum(1 for p in results.values() if p)
    total_count = len(results)
    
    print(f"\n총 {passed_count}/{total_count} 시나리오 성공")
    
    if passed_count == total_count:
        print("🎉 모든 통합 테스트가 성공했습니다!")
    else:
        print(f"⚠️ {total_count - passed_count}개 시나리오에서 예상과 다른 결과")


if __name__ == "__main__":
    asyncio.run(main())
