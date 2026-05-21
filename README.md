# LawsGuard

한국어 법률 상담 챗봇 백엔드입니다. 현재 프로젝트는 **FastAPI 기반 카카오톡 챗봇 Skill 서버**와 **법률 답변 생성 파이프라인**이 구현되어 있으며, 카카오톡 챗봇 UI/배포 담당 팀원이 이 저장소를 받아 엔드포인트를 연결하고 코드를 병합하는 상황을 기준으로 정리했습니다.

## 현재 구현 상태

- FastAPI 서버 진입점: `src/main.py`
- 카카오톡 Skill webhook: `POST /webhook/kakao`
- 상태 확인 API: `GET /health`
- 메인 상담 파이프라인: `src/pipeline.py`
- 카카오 응답 포맷 유틸: `src/modules/kakao_response.py`
- 세션 저장소: `src/session_store.py`
- 설정 중앙 관리: `src/config.py`
- 테스트: `pytest` 기준 14개 통과 확인

현재 파이프라인은 다음 순서로 동작합니다.

1. 사용자 질문 세션 생성 또는 기존 세션 조회
2. 질문 명확성 판단 및 추가 질문 생성
3. RAG 근거 문서 검색
4. 답변 일관성 검증
5. 법률 추론 검증
6. NER 기반 환각 탐지 및 교정
7. 카카오톡 응답용 최종 답변 포맷팅

## 프로젝트 구조

```text
chatbot/
├─ src/
│  ├─ main.py                         # FastAPI 앱, 카카오 webhook, callback 처리
│  ├─ pipeline.py                     # LawsGuard 메인 파이프라인
│  ├─ session_store.py                # 사용자별 대화 세션 관리
│  ├─ config.py                       # 환경변수 및 기본 설정
│  └─ modules/
│     ├─ answer_formatter.py          # 최종 답변 포맷팅
│     ├─ clarification.py             # 질문 명확성 판단 및 재질문
│     ├─ consistency_checker.py       # 답변 일관성 검증
│     ├─ corrector.py                 # 환각 탐지 결과 기반 답변 교정
│     ├─ kakao_response.py            # 카카오 Skill 응답 JSON 생성
│     ├─ legal_reasoning_validator.py # 법률 근거와 답변의 정합성 검증
│     ├─ llm_client.py                # OpenAI 호환 LLM 클라이언트
│     ├─ ner_checker.py               # NER 기반 법률 개체 환각 탐지
│     └─ rag.py                       # JSONL/BM25 및 선택적 ChromaDB 검색
├─ tests/                             # 자동 테스트 및 수동 테스트 스크립트
├─ scripts/                           # 평가, 학습, 점검용 스크립트
├─ data/                              # RAG 및 학습/평가 데이터
├─ outputs/                           # 학습된 NER 모델
├─ requirements.txt
├─ pytest.ini
└─ .env                               # 로컬 환경변수, Git 추적 제외
```

## 카카오톡 챗봇 연동 정보

### Webhook

```text
POST /webhook/kakao
```

카카오 i 오픈빌더 Skill 서버의 엔드포인트로 사용합니다. 서버가 외부에서 HTTPS로 접근 가능해야 하므로 로컬 개발 시에는 ngrok, Cloudflare Tunnel, 배포 서버 중 하나를 사용해야 합니다.

예상 입력은 카카오 Skill 요청 형식입니다. 코드에서는 아래 값을 사용합니다.

```json
{
  "userRequest": {
    "utterance": "질문 내용",
    "callbackUrl": "카카오 callback URL",
    "user": {
      "id": "사용자 ID"
    }
  }
}
```

### 응답 방식

`src/main.py`는 `callbackUrl`이 있으면 카카오 callback 모드를 우선 사용합니다.

- 즉시 응답: `{"version": "2.0", "useCallback": true, ...}`
- 백그라운드 처리: `pipeline.process(...)`
- 최종 응답 전송: `callbackUrl`로 `simpleText` 응답 POST

`callbackUrl`이 없으면 `config.kakao.response_timeout_sec` 안에 동기 응답을 시도합니다. 기본값은 4.5초입니다.

### 카카오 응답 포맷

응답 생성은 `src/modules/kakao_response.py`에서 담당합니다.

- `build_simple_text(...)`: `version: "2.0"` 및 `template.outputs.simpleText` 생성
- 긴 답변은 900자 단위로 최대 3개 `simpleText` 블록으로 분리
- `default_quick_replies(...)`: 상황별 quickReplies 생성
- `build_callback_response(...)`: callback 대기 응답 생성

## 실행 방법

### 1. 가상환경 생성

현재 저장소에는 `venv/`가 있을 수 있지만, 팀원별 로컬 환경에서는 새로 만드는 것을 권장합니다.

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 환경변수 설정

루트에 `.env` 파일을 만들고 필요한 값을 설정합니다.

```env
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini

# 선택 사항
OPENAI_BASE_URL=
LAWSGUARD_LLM_TEMPERATURE=0.2
LAWSGUARD_LLM_MAX_TOKENS=2048
LAWSGUARD_LLM_TIMEOUT=60

# RAG
LAWSGUARD_ENABLE_VECTOR_RAG=0

# NER
NER_MODEL_ID=legal-ner-v3
LAWSGUARD_USE_MODEL_NER=1
LAWSGUARD_NER_MIN_CONFIDENCE=0.70
LAWSGUARD_NER_MAX_LENGTH=510

# 검증/재질문
LAWSGUARD_CONSISTENCY_THRESHOLD=0.75
LAWSGUARD_CLARIFICATION_MIN_SCORE=3.5
```

`NER_MODEL_ID`는 절대경로, `outputs/...` 상대경로, 또는 `outputs/` 아래 모델 디렉터리명으로 사용할 수 있습니다. 기본값은 `outputs/legal-ner-v3`입니다.

### 3. 서버 실행

```powershell
venv\Scripts\python.exe -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

상태 확인:

```powershell
curl http://127.0.0.1:8000/health
```

정상 응답:

```json
{"status":"ok","service":"LawsGuard"}
```

## 테스트

전체 자동 테스트:

```powershell
venv\Scripts\python.exe -m pytest
```

수동 파이프라인 테스트:

```powershell
venv\Scripts\python.exe tests\manual_pipeline_test.py
```

교정 모듈 단독 확인:

```powershell
venv\Scripts\python.exe scripts\test_corrector.py
```

최근 확인 결과:

```text
14 passed
```

## 데이터와 모델

현재 구현은 로컬 데이터와 모델 아티팩트를 사용합니다.

- RAG JSONL 데이터: `data/real_data/New_Dataset/*.jsonl`
- 선택적 ChromaDB 벡터 DB: `data/RAG_data/chroma_db/`
- NER 학습 모델: `outputs/legal-ner-v3/`
- NER 평가/학습 데이터: `data/real_bio_data/`, `data/hallucination_data/`, `data/mock_data/`

기본 RAG는 JSONL 기반 검색으로 동작합니다. `LAWSGUARD_ENABLE_VECTOR_RAG=1`을 설정하면 ChromaDB와 임베딩 기반 검색을 먼저 시도하고, 실패 시 JSONL 검색으로 fallback합니다.

## 병합 시 주의할 점

- 카카오톡 챗봇 구현팀은 `POST /webhook/kakao`만 연결하면 됩니다.
- 카카오 응답 JSON을 직접 만들지 말고 `src/modules/kakao_response.py`의 유틸을 재사용하는 것이 안전합니다.
- `pipeline.process(user_id, user_input)`은 비동기 함수이며 `PipelineResult`를 반환합니다.
- 사용자별 맥락은 `session_store`가 메모리에서 관리합니다. 서버 재시작 시 세션은 유지되지 않습니다.
- `.env`, `venv/`, `__pycache__/`, `.pytest_cache/`는 Git에 올리지 않습니다.
- `data/`, `outputs/`는 용량이 큽니다. Git LFS 또는 별도 아티팩트 전달 방식이 필요합니다.
- 현재 일부 소스 주석과 문자열에 인코딩이 깨진 텍스트가 남아 있습니다. 동작 테스트는 통과하지만, 사용자 노출 문구는 병합 전 별도 점검이 필요합니다.

## 주요 설정 위치

설정은 `src/config.py`에 모여 있습니다.

- `RAGConfig`: ChromaDB 경로, 컬렉션명, 임베딩 모델, JSONL 경로
- `NERConfig`: NER 모델 경로, 사용 여부, confidence, max length
- `ClarificationConfig`: 재질문 최대 횟수, 명확성 기준 점수
- `HallucinationConfig`: 일관성 기준 점수, 모델명
- `LLMConfig`: OpenAI API 키, 모델명, base URL, timeout
- `KakaoConfig`: callback 사용 여부, 동기 응답 timeout, 서버 host/port

## 평가 및 학습 스크립트

```powershell
# RAG 검색 품질 평가
venv\Scripts\python.exe scripts\evaluate_rag_quality.py

# NER 환각 탐지 평가
venv\Scripts\python.exe scripts\evaluate_ner_factcheck.py

# NER hybrid threshold 튜닝
venv\Scripts\python.exe scripts\tune_ner_hybrid_thresholds.py

# NER 모델 추가 학습
venv\Scripts\python.exe scripts\train_legal_ner.py --base-model outputs\legal-ner-v3 --output-dir outputs\legal-ner-lawsguard-v3 --epochs 3 --batch-size 8
```

## 팀원 인수인계 요약

카카오톡 챗봇 쪽에서 필요한 핵심은 다음 네 가지입니다.

1. FastAPI 서버를 실행합니다.
2. 외부 HTTPS 주소를 카카오 Skill 서버 URL로 등록합니다.
3. 카카오 요청은 `POST /webhook/kakao`로 보냅니다.
4. 응답은 이미 카카오 `version: "2.0"` 포맷으로 반환됩니다.

병합 후 최소 확인 순서는 `GET /health`, `POST /webhook/kakao` 샘플 요청, `pytest` 순서로 보면 됩니다.
