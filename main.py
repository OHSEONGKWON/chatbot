from modules.ner_checker import NERFactChecker
from modules.corrector import AnswerCorrector

class LawsGuardPipeline:
    def __init__(self):
        self.checker = NERFactChecker()
        self.corrector = AnswerCorrector()
        print("[System] 파이프라인 준비 완료.\n")

    def run(self, llm_answer, rag_chunks):
        # 1. 탐지 (Detection)
        hallucinations = self.checker.find_hallucinations(llm_answer, rag_chunks)
        
        # 2. 통과 (Pass)
        if not hallucinations:
            return {"status": "PASS", "final_answer": llm_answer, "is_modified": False, "logs": []}
            
        # 3. 교정 (Correction)
        corrected_answer = self.corrector.fix_answer(llm_answer, hallucinations)
        
        return {
            "status": "CORRECTED",
            "final_answer": corrected_answer,
            "is_modified": True,
            "logs": hallucinations
        }

# ==========================================
# 🧪 규격 테스트용 실행부
# ==========================================
if __name__ == "__main__":
    pipeline = LawsGuardPipeline()
    
    # [시뮬레이션] 나중에 팀원이 넘겨줄 RAG 데이터 규격 (껍데기만 진짜, 내용은 임의 작성)
    mock_rag_chunks = [
        {
            "chunk_id": "f4b057319cc181f1_제5조_0", 
            "text": "제5조(부당해고등의 구제신청) 근로자는 부당해고등의 구제 신청서를 관할 중앙노동위원회에 30일 이내에 제출하여야 한다.", 
            "metadata": {"law_name": "근로기준법 시행규칙"}
        }
    ]
    
    # [시뮬레이션] 팀원이 1차 검사 후 넘겨준 답변 (일부러 날짜를 '60일'로 환각 처리함)
    mock_llm_answer = "근로기준법 시행규칙 제5조에 따라, 부당해고 구제 신청서는 60일 이내에 제출해야 합니다."
    
    print(f"🔹 1차 답변 (환각 포함): {mock_llm_answer}")
    
    # 파이프라인 가동!
    result = pipeline.run(mock_llm_answer, mock_rag_chunks)
    
    print("\n--- [카카오톡 최종 출력 결과] ---")
    print(f"✅ 최종 답변: {result['final_answer']}")
    print(f"🔄 수정 여부: {result['is_modified']}")
    if result['is_modified']:
        print(f"📋 수정 로그: {result['logs']}")