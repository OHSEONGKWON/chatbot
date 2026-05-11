"""단계별 모듈 테스트 - 각 단계가 제대로 작동하는지 검증"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modules.clarification import clarification_manager
from src.modules.consistency_checker import consistency_checker
from src.modules.ner_checker import ner_checker
from src.modules.rag import retriever
from src.session_store import session_store


async def test_clarification():
    """Step 1: 재질문 로직 테스트"""
    print("\n" + "="*80)
    print("🔍 테스트 1: 재질문(Clarification) 로직")
    print("="*80)
    
    question = "술자리에서 후배를 만졌는데 처벌받을 수 있나요?"
    print(f"💬 입력: {question}")
    
    try:
        user_id = "test-clarify"
        session = session_store.create(user_id, question)
        
        print("⏳ clarification_manager.process() 실행 중...")
        result = await clarification_manager.process(session=session, user_input=question)
        
        print(f"✅ 완료!")
        print(f"   needs_requery: {result.needs_requery}")
        print(f"   legal_category: {result.legal_category}")
        print(f"   final_question: {result.final_question[:100]}...")
        if result.needs_requery:
            print(f"   requery_message: {result.requery_message}")
        
        return True
        
    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_rag_retrieval():
    """Step 2: RAG 검색 테스트"""
    print("\n" + "="*80)
    print("🔍 테스트 2: RAG 검색")
    print("="*80)
    
    question = "교수가 학생에게 신체 접촉을 했을 때 성희롱인가요?"
    print(f"💬 입력: {question}")
    
    try:
        print("⏳ retriever.retrieve_async() 실행 중...")
        docs = await retriever.retrieve_async(question, legal_category="성희롱")
        
        print(f"✅ 완료!")
        print(f"   검색된 문서 수: {len(docs)}")
        if docs:
            print(f"   첫 번째 문서: {docs[0]['text'][:100]}...")
        
        return len(docs) > 0
        
    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_consistency_checking():
    """Step 3: 일관성 검증 테스트"""
    print("\n" + "="*80)
    print("🔍 테스트 3: 일관성 검증 (Consistency Checking)")
    print("="*80)
    
    question = "알바비를 못 받았어요. 2개월간 주 20시간 일했고 월급 80만원을 아직 못 받았습니다."
    print(f"💬 입력: {question}")
    
    try:
        print("⏳ consistency_checker.run() 실행 중...")
        docs = await retriever.retrieve_async(question, legal_category="노동")
        is_reliable, original_answer, score, similar_answers = await consistency_checker.run(
            question, 
            rag_docs=docs, 
            legal_category="노동"
        )
        
        print(f"✅ 완료!")
        print(f"   is_reliable: {is_reliable}")
        print(f"   consistency_score: {score:.3f}")
        print(f"   original_answer: {original_answer[:100]}...")
        print(f"   유사 질문 개수: {len(similar_answers)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_ner_checking():
    """Step 4: NER 환각 탐지 테스트"""
    print("\n" + "="*80)
    print("🔍 테스트 4: NER 환각 탐지 (NER Checking)")
    print("="*80)
    
    # 더미 답변
    answer = "가출 후 72시간 이상 귀가하지 않으면 체포될 수 있습니다. 아동은 보호관찰 대상이 될 수 있습니다."
    docs = [
        {
            "content": "가출은 형법 제 258조의 이탈죄에 해당할 수 있으나, 미성년자의 경우는 보호관찰로 처리되는 경우가 많습니다.",
            "source": "법률 자료"
        }
    ]
    
    print(f"💬 입력 답변: {answer[:80]}...")
    
    try:
        print("⏳ ner_checker.check_and_correct() 실행 중...")
        result = await ner_checker.check_and_correct(answer=answer, rag_docs=docs)
        
        print(f"✅ 완료!")
        print(f"   was_corrected: {result.was_corrected}")
        print(f"   corrected_answer: {result.corrected_answer[:100]}...")
        if result.mismatched_entities:
            print(f"   탐지된 환각:")
            for h in result.mismatched_entities[:3]:
                print(f"     - {h.get('label')}: {h.get('wrong_word')} → {h.get('correct_word')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("\n🧪 LawsGuard 단계별 모듈 테스트")
    print("=" * 80)
    
    results = {
        "재질문(Clarification)": await test_clarification(),
        "RAG 검색": await test_rag_retrieval(),
        "일관성 검증": await test_consistency_checking(),
        "NER 환각 탐지": await test_ner_checking(),
    }
    
    # 최종 결과
    print(f"\n\n{'='*80}")
    print("📊 최종 테스트 결과")
    print("=" * 80)
    
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
    
    passed_count = sum(1 for p in results.values() if p)
    total_count = len(results)
    
    print(f"\n총 {passed_count}/{total_count} 테스트 통과")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
