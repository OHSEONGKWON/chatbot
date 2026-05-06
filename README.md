# LawsGuard - 법률 AI 상담 챗봇

RAG + NER + 일관성 검증 기반의 한국 법률 상담 AI 챗봇입니다.

## 프로젝트 구조

```
chatbot/
├── src/                          # 핵심 소스 코드
│   ├── main.py                   # FastAPI 진입점
│   ├── pipeline.py               # 메인 파이프라인 오케스트레이터
│   ├── session_store.py          # 사용자 세션 관리
│   ├── config.py                 # 프로젝트 설정
│   └── modules/                  # 핵심 기능 모듈
│       ├── clarification.py      # 재질문 로직 (부족 정보 입력 요청)
│       ├── consistency_checker.py # 일관성 검증 (NLI 기반)
│       ├── ner_checker.py        # NER 환각 탐지 (하이브리드 매칭)
│       ├── rag.py                # RAG 검색 (ChromaDB)
│       ├── llm_client.py         # LLM API 클라이언트 (OpenAI)
│       ├── answer_formatter.py   # 최종 답변 포맷팅
│       └── corrector.py          # 답변 텍스트 교정
│
├── tests/                         # 테스트 및 개발용 스크립트
│   ├── manual_pipeline_test.py   # 수동 파이프라인 테스트
│   └── test_long_answer.py       # 긴 답변 테스트
│
├── scripts/                       # 운영 유틸리티
│   └── tune_ner_hybrid_thresholds.py  # NER 하이브리드 임계값 튜닝
│
├── data/                          # 데이터 (RAG 벡터DB 등)
│   └── RAG_data/
│       └── chroma_db/            # ChromaDB 벡터 데이터베이스
│
├── outputs/                       # 학습된 모델들
│   └── legal-ner-lawsguard-v2-30k/  # NER 모델
│
├── requirements.txt               # Python 의존성
├── config.py                      # (src/config.py로 이동됨)
└── .env                           # 환경 변수 (API 키 등)
```

## 실행 방법

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
# .env 파일 생성 및 설정
OPENAI_API_KEY=your-api-key
LAWSGUARD_NER_MODEL=outputs/legal-ner-lawsguard-v2-30k
```

### 3. 서버 실행
```bash
# FastAPI 서버 시작
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 테스트 실행
```bash
cd tests/
python -m pytest manual_pipeline_test.py
```

## 파이프라인 흐름

1. **명확화 (Clarification)** - `src/modules/clarification.py`
   - 사용자 질문 평가 (부족 정보 확인)
   - 필요시 재질문 (최대 5회)

2. **일관성 검증 (Consistency)** - `src/modules/consistency_checker.py`
   - 비슷한 질문들 생성 (LLM)
   - 일관된 답변 확인 (NLI + 임베딩)
   - 신뢰도 점수 계산

3. **RAG 검색 (Retrieval)** - `src/modules/rag.py`
   - 관련 법률 문서 검색 (ChromaDB)
   - 답변 생성 컨텍스트 구성

4. **NER 환각 탐지 (NER Hallucination Detection)** - `src/modules/ner_checker.py`
   - 개체명 인식 (NER 모델)
   - RAG 문서와 대조 (하이브리드 매칭: 정확/퍼지/의미)
   - 환각 수정

5. **답변 최종화 (Formatting)** - `src/modules/answer_formatter.py`
   - 최종 답변 생성 (구조화된 형식)

## 주요 기능

### NER 환각 탐지 (Hybrid Matching)
- **정규화**: 날짜/금액 숫자 표준화, 별칭 처리
- **세 단계 매칭**:
  1. 정확 매칭 (Exact)
  2. 퍼지 매칭 (Fuzzy) - 90% 이상
  3. 의미적 유사도 (Semantic) - 임베딩 기반
- **보수적 정책**: 숫자형(DATE/AMOUNT)은 strict numeric만 허용

### 재질문 로직
- LLM + 컨텍스트 휴리스틱 기반
- 이미 답한 정보는 재질문하지 않음
- 최대 5회 제한 + 동일 문구 반복 방지

## 설정 파일

`src/config.py` - 모든 설정값 중앙 관리:
- RAG: ChromaDB 경로, 임베딩 모델
- NER: 모델 경로, 환각 감지 임계값
- LLM: OpenAI API, 모델명, 파라미터
- 명확화: 최대 재질문 수, 점수 임계값

## 팀원 협력

- **재질문 로직**: `src/modules/clarification.py` (팀원이 수정 중)
- **NER 환각 탐지**: `src/modules/ner_checker.py` (사용자가 보완 중)

변경 사항 병합 시 import 경로가 통일되어 있으므로 의존성 충돌이 최소화됩니다.
