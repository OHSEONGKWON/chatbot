# src/ - 핵심 소스 코드

## 파일 설명

- **main.py**: FastAPI 애플리케이션 진입점, 카카오톡 연동 엔드포인트
- **pipeline.py**: 전체 파이프라인 오케스트레이터, 각 모듈 조율
- **session_store.py**: 사용자 세션 관리 (카카오톡 user_id 기반)
- **config.py**: 모든 설정값 (RAG, NER, LLM, 명확화 등)

## modules/ 내부

각 모듈은 독립적으로 테스트 가능하며, pipeline.py에서 조율됩니다.

- **clarification.py** (팀원 개발): 재질문 로직
- **consistency_checker.py**: 답변 일관성 검증 (NLI)
- **ner_checker.py** (사용자 개발): NER 환각 탐지 (하이브리드 매칭)
- **rag.py**: RAG 검색 (ChromaDB)
- **llm_client.py**: LLM API 호출 추상화
- **answer_formatter.py**: 최종 답변 생성
- **corrector.py**: 환각 교정
