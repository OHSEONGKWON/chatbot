from transformers import pipeline

class NERFactChecker:
    def __init__(self, model_path="klue/roberta-base"):
        print("[System] NER 팩트체커 모델 로딩 중...")
        # 향후 법률 특화 파인튜닝 모델로 교체 예정
        self.ner_pipeline = pipeline("ner", model=model_path, aggregation_strategy="simple")
        self.target_labels = ['LAW', 'PENALTY', 'AMOUNT', 'DATE', 'ORG', 'CRIME']

    def extract_entities(self, text):
        entities = self.ner_pipeline(text)
        return [ent for ent in entities if ent['entity_group'] in self.target_labels]

    def find_hallucinations(self, llm_answer, rag_chunks):
        """
        llm_answer: 팀원이 일관성 검사 후 넘겨준 LLM 답변 (string)
        rag_chunks: 팀원이 검색해온 RAG 원문 데이터 리스트 (규격: [{"chunk_id":..., "text":..., "metadata":...}, ...])
        """
        # 1. LLM 답변에서 개체명 추출
        ans_entities = self.extract_entities(llm_answer)
        
        # 2. RAG 데이터 규격에 맞춰 "text" 필드만 하나로 병합하여 정답지 생성
        rag_text_combined = " ".join([chunk.get("text", "") for chunk in rag_chunks])
        
        # 3. 정답지에서도 개체명 추출 (교정용 단어 탐색을 위해)
        rag_entities = self.extract_entities(rag_text_combined)
        
        hallucinations = []
        
        # 4. 팩트 체크 대조 로직
        for ans_ent in ans_entities:
            label = ans_ent['entity_group']
            word = ans_ent['word']
            
            # 답변의 개체가 RAG 정답지 텍스트에 문자열 그대로 존재하지 않으면 환각!
            if word not in rag_text_combined:
                # 같은 범주(label)를 가진 RAG 개체를 찾아서 교정 후보로 삼음
                correct_candidate = None
                for rag_ent in rag_entities:
                    if rag_ent['entity_group'] == label:
                        correct_candidate = rag_ent['word']
                        break
                
                hallucinations.append({
                    "label": label,
                    "wrong_word": word,
                    "correct_word": correct_candidate
                })
                
        return hallucinations