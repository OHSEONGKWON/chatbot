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
│       ├── legal_reasoning_validator.py # 조문/법리 적용 타당성 검증
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

### 1. 가상환경 및 의존성 설치
```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
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
.\.venv\Scripts\python.exe -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 테스트 실행
```bash
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tests\manual_pipeline_test.py
.\.venv\Scripts\python.exe scripts\evaluate_rag_quality.py
.\.venv\Scripts\python.exe scripts\evaluate_ner_factcheck.py
```

> 현재 저장소 기준 가상환경 폴더는 `.venv`입니다. `venv`를 새로 만들 수도 있지만, 팀 전체가 하나의 이름으로 통일하는 편이 좋습니다.

## 파이프라인 흐름

1. **명확화 (Clarification)** - `src/modules/clarification.py`
   - 사용자 질문 평가 (부족 정보 확인)
   - 필요시 재질문 (최대 5회)

2. **RAG 검색 (Retrieval)** - `src/modules/rag.py`
   - 기본값: JSONL 기반 BM25 + 분야별 재랭킹 (네트워크 없이 동작)
   - 선택값: `LAWSGUARD_ENABLE_VECTOR_RAG=1` 설정 시 ChromaDB + 로컬 임베딩 검색 사용
   - `scripts/evaluate_rag_quality.py`로 대표 질의 검색 품질 점검

3. **일관성 검증 (Consistency)** - `src/modules/consistency_checker.py`
   - LLM 사용 가능 시 유사 질문 생성 및 답변 일관성 점검
   - LLM/API 키가 없으면 RAG 근거 기반 fallback 답변 생성

4. **법리 적용 검증 (Legal Reasoning Validation)** - `src/modules/legal_reasoning_validator.py`
   - 답변이 인용한 조문/법률명이 RAG 근거에 실제로 있는지 확인
   - 노동/성폭력 분야와 검색 근거의 적합성 확인
   - 법률요건이 부족한데 결론을 단정하는 표현을 보수적으로 완화
   - LLM 사용 가능 시 근거-사실관계-답변을 JSON 형식으로 추가 감수

5. **NER 환각 탐지 (NER Hallucination Detection)** - `src/modules/ner_checker.py`
   - 개체명 인식 (NER 모델)
   - RAG 문서와 대조 (정확/퍼지/의미 매칭)
   - 법률명과 조문 번호를 함께 검증 (`근로기준법 제43조`와 `제37조`를 구분)
   - high-risk 조문 오류만 보수적으로 자동 수정

6. **답변 최종화 (Formatting)** - `src/modules/answer_formatter.py`
   - 최종 답변 생성 (구조화된 형식)

## 주요 기능

### NER 환각 탐지 (Hybrid Matching)
- **정규화**: 날짜/금액 숫자 표준화, 법률 별칭 처리, 조문 번호 보존
- **세 단계 매칭**:
  1. 정확 매칭 (Exact)
  2. 퍼지 매칭 (Fuzzy) - 90% 이상
  3. 의미적 유사도 (Semantic) - 임베딩 기반
- **보수적 정책**: 숫자형(DATE/AMOUNT)은 strict numeric만 허용하고, 기관명(ORG)은 자동 수정 대상에서 제외
- **검증 지표**: `scripts/evaluate_ner_factcheck.py`로 답변-근거 쌍 기준 precision/recall/F1을 확인

### NER 모델 재학습
```bash
.\.venv\Scripts\python.exe scripts\train_legal_ner.py --base-model outputs\legal-ner-lawsguard-v2-30k --output-dir outputs\legal-ner-lawsguard-v3 --epochs 3 --batch-size 8
```
- 학습 데이터: `data/real_bio_data/**`와 `data/hallucination_data/**`의 BIO JSONL
- 산출 지표: token F1과 entity-span F1을 `outputs/<model>/metrics.json`에 저장
- 새 모델 적용: `.env`의 `LAWSGUARD_NER_MODEL`을 새 output 경로로 변경

### 재질문 로직
- LLM + 컨텍스트 휴리스틱 기반
- 이미 답한 정보는 재질문하지 않음
- 최대 5회 제한 + 동일 문구 반복 방지

## 카카오톡 챗봇 준비

- 엔드포인트: `POST /webhook/kakao`
- 헬스체크: `GET /health`
- 기본 응답은 Kakao Skill 응답 포맷 `version: "2.0"`을 사용합니다.
- 긴 답변은 `simpleText` 여러 개로 자동 분할합니다.
- `callbackUrl`이 들어오면 즉시 `useCallback: true`를 반환하고, 백그라운드에서 최종 답변을 콜백으로 전송합니다.
- 응답에는 상황에 맞는 quickReplies가 포함됩니다.

로컬 확인:
```bash
.\.venv\Scripts\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

카카오 개발자/오픈빌더에 연결할 때는 로컬 서버를 HTTPS로 외부 노출해야 하므로 ngrok, Cloudflare Tunnel, 배포 서버 중 하나를 사용하세요.

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
