# 성능 평가 작업 인수인계 가이드

> 작성일: 2026-05-21  
> 목적: 논문용 성능 평가 3종(NER / RAG / 환각 탐지) 완료를 위한 인수인계

---

## 1. 프로젝트 개요

한국 법률 AI 챗봇(`c:\GitHub\chatbot`)의 성능을 외부 모델과 비교 평가하는 실험.  
`data/external_raw/legal_pdfs/` 의 실제 법령·판례 PDF를 기반으로 평가 데이터를 생성한 후,  
우리 모델과 베이스라인 모델 3개를 **동일한 데이터**로 평가하여 논문용 표를 생성한다.

---

## 2. 현재 작업 상태 (2026-05-21 기준)

### ✅ 완료
- `data/external_processed/rag_chunks/rag_corpus.jsonl` — PDF 4,973청크 전처리 완료
- `scripts/` 내 7개 스크립트 전부 작성 및 버그 수정 완료
- `.venv` 패키지 설치 완료 (`pdfminer.six`, `rank-bm25` 포함)

### 🔄 진행 중
- **NER 평가 데이터 생성** (`data/evaluation/ner_eval.jsonl`)
  - 현재 약 55/500 청크 완료 (백그라운드 실행 중, PID 13328)
  - 완료까지 약 10분 더 소요 예정

### ❌ 미완료 (순서대로 실행 필요)
1. RAG 평가 데이터 생성 → `data/evaluation/rag_eval.jsonl`
2. 환각 탐지 평가 데이터 생성 → `data/evaluation/hallu_eval.jsonl`
3. NER 성능 평가 실행 → `data/evaluation/results/ner_results.json`
4. RAG 성능 평가 실행 → `data/evaluation/results/rag_results.json`
5. 환각 탐지 성능 평가 실행 → `data/evaluation/results/hallucination_results.json`
6. 논문용 표 생성 → `data/evaluation/results/`

---

## 3. 환경 설정

```
작업 디렉토리: c:\GitHub\chatbot
Python 가상환경: .venv\Scripts\python.exe
환경 변수: .env (OPENAI_API_KEY 설정됨)
```

### 패키지 확인
```powershell
.venv\Scripts\python.exe -c "import openai, pdfminer, transformers, sentence_transformers, rank_bm25, sklearn; print('OK')"
```
모두 OK가 나와야 함. 없는 패키지는:
```powershell
.venv\Scripts\pip.exe install pdfminer.six rank-bm25
```

---

## 4. 실행 순서 (반드시 이 순서를 지킬 것)

### Step 1 — NER 평가 데이터 생성 (현재 실행 중)

실행 중인 프로세스가 있는지 먼저 확인:
```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, CPU
(Get-Content data\evaluation\ner_eval.jsonl | Measure-Object -Line).Lines  # 500이면 완료
```

**500줄 미만이고 Python 프로세스가 없으면** 이어서 실행:
```powershell
cd c:\GitHub\chatbot
.venv\Scripts\python.exe scripts/generate_ner_eval_data.py --max-pdfs 60 --max-chunks 500
```

완료 기준: `data/evaluation/ner_eval.jsonl` 가 500줄

---

### Step 2 — RAG 평가 데이터 생성

NER 완료 후 실행:
```powershell
cd c:\GitHub\chatbot
.venv\Scripts\python.exe scripts/generate_rag_eval_data.py --n-queries 150
```

완료 기준: `data/evaluation/rag_eval.jsonl` 가 150줄  
소요 시간: 약 5~8분

---

### Step 3 — 환각 탐지 평가 데이터 생성

RAG 완료 후 실행:
```powershell
cd c:\GitHub\chatbot
.venv\Scripts\python.exe scripts/generate_hallu_eval_data.py --n-positive 150 --n-negative 150
```

완료 기준: `data/evaluation/hallu_eval.jsonl` 가 300줄  
소요 시간: 약 5~10분 (OpenAI API로 positive 샘플 추가 생성)

---

### Step 4 — NER 성능 평가

평가 데이터 3개 모두 완료 후 실행:
```powershell
cd c:\GitHub\chatbot
.venv\Scripts\python.exe scripts/evaluate_ner.py
```

- 우리 모델 + 베이스라인 3개를 `ner_eval.jsonl` 로 평가
- 출력: `data/evaluation/results/ner_results.json`, `ner_table.csv`
- 소요 시간: 모델 로딩 포함 약 10~20분 (GPU 없으면 더 걸림)

---

### Step 5 — RAG 성능 평가

```powershell
cd c:\GitHub\chatbot
.venv\Scripts\python.exe scripts/evaluate_rag.py
```

- BM25 + Dense 임베딩 4종 + Hybrid 비교
- 출력: `data/evaluation/results/rag_results.json`, `rag_table.csv`
- 소요 시간: 임베딩 계산 포함 약 20~40분

---

### Step 6 — 환각 탐지 성능 평가

```powershell
cd c:\GitHub\chatbot
.venv\Scripts\python.exe scripts/evaluate_hallucination.py
```

- NLI 단독 / NER 단독 / 3계층 앙상블 / Vectara HEM 비교
- 출력: `data/evaluation/results/hallucination_results.json`, `hallucination_table.csv`
- 소요 시간: 모델 로딩 포함 약 15~30분

---

### Step 7 — 논문용 표 생성

```powershell
cd c:\GitHub\chatbot
.venv\Scripts\python.exe scripts/generate_paper_tables.py
```

---

## 5. 디렉토리 구조 (평가 관련)

```
c:\GitHub\chatbot\
├── scripts/
│   ├── generate_ner_eval_data.py      # Step 1 스크립트
│   ├── generate_rag_eval_data.py      # Step 2 스크립트
│   ├── generate_hallu_eval_data.py    # Step 3 스크립트
│   ├── evaluate_ner.py                # Step 4 스크립트
│   ├── evaluate_rag.py                # Step 5 스크립트
│   ├── evaluate_hallucination.py      # Step 6 스크립트
│   └── generate_paper_tables.py       # Step 7 스크립트
│
├── data/
│   ├── external_raw/legal_pdfs/
│   │   ├── cases/   (~120개 판례 PDF)
│   │   └── laws/    (14개 법령 PDF)
│   ├── external_processed/rag_chunks/
│   │   └── rag_corpus.jsonl           # ✅ 이미 완료 (4,973청크)
│   ├── evaluation/
│   │   ├── ner_eval.jsonl             # 🔄 생성 중 (~55/500)
│   │   ├── rag_eval.jsonl             # ❌ 미생성
│   │   ├── hallu_eval.jsonl           # ❌ 미생성
│   │   └── results/                   # 평가 결과 저장 위치
│   ├── mock_data/                     # 환각 탐지용 기존 데이터
│   └── real_data/New_Dataset/         # RAG용 정상 청크
│
└── outputs/
    ├── legal-ner-lawsguard-v2-30k/    # ✅ 우리 NER 모델
    └── baselines/
        ├── Babelscape_wikineural-multilingual-ner/   # ✅ 베이스라인 1
        ├── Leo97_KoELECTRA-small-v3-modu-ner/        # ✅ 베이스라인 2
        └── monologg_koelectra-base-v3-naver-ner/     # ✅ 베이스라인 3
```

---

## 6. 데이터 형식

### NER 평가 데이터 (`ner_eval.jsonl`)
```json
{
  "chunk_id": "abc123_eval_chunk_0001",
  "tokens": ["대", "법", "원", " ", "2", "0", "2", "4", ...],
  "ner_tags": ["B-ORG", "I-ORG", "I-ORG", "O", "B-DATE", ...],
  "metadata": {"source_file": "대법원 2024.pdf", "chunk_index": 0, "entity_count": 3}
}
```
- 문자 단위(character-level) BIO 태깅
- 엔티티 종류: LAW, ORG, DATE, AMOUNT, CRIME, PENALTY

### RAG 평가 데이터 (`rag_eval.jsonl`)
```json
{
  "query_id": "rag_eval_0001",
  "query": "근로기준법에서 임금 지급 원칙은 무엇인가?",
  "relevant_chunk_ids": ["chunk_id_1", "chunk_id_2"],
  "source_chunk_id": "chunk_id_1",
  "difficulty": "easy"
}
```

### 환각 탐지 데이터 (`hallu_eval.jsonl`)
```json
{
  "eval_id": "hallu_eval_0001",
  "answer": "근로기준법 제37조에 따라...",
  "rag_docs": [{"text": "근로기준법 제43조는...", "metadata": {}}],
  "is_hallucination": true,
  "hallucination_type": "article_num",
  "expected_mismatches": [{"label": "LAW", "wrong_word": "제37조", "correct_word": "제43조"}]
}
```

---

## 7. 비교 모델 구성 (논문용)

### NER
| 키 | 모델명 | 경로 |
|----|--------|------|
| `our_model` | LawsGuard-NER-v2 (우리) | `outputs/legal-ner-lawsguard-v2-30k/` |
| `babelscape` | WikiNEural-Multilingual | `outputs/baselines/Babelscape_wikineural-multilingual-ner/` |
| `koelectra_modu` | KoELECTRA-modu-NER | `outputs/baselines/Leo97_KoELECTRA-small-v3-modu-ner/` |
| `koelectra_naver` | KoELECTRA-naver-NER | `outputs/baselines/monologg_koelectra-base-v3-naver-ner/` |

### RAG
BM25 / Dense(E5-large, ko-SRoBERTa, KoSimCSE, KURE-RoBERTa) / Hybrid(E5+BM25)

### 환각 탐지
NLI 단독(klue/roberta-large) / NER 단독 / 3계층 앙상블(우리 시스템) / Vectara HEM

---

## 8. 알려진 이슈 및 해결책

### pdfminer CPU 점유 문제
- **증상**: PDF 추출 중 CPU 200~300% 점유, 스크립트 멈춤
- **원인**: pdfminer가 특정 복잡한 PDF에서 무한 루프에 가까운 처리
- **해결**: `generate_ner_eval_data.py`를 `multiprocessing` 방식으로 수정 완료  
  → 30초 타임아웃 시 서브프로세스 강제 종료 (`p.terminate()` → `p.kill()`)

### OpenAI API 키 오류
- **증상**: `Error code: 401 - Incorrect API key`
- **해결**: `.env` 파일의 `OPENAI_API_KEY` 값을 최신 키로 업데이트

### 두 Python 프로세스 동시 실행
- **증상**: 동일 파일에 두 프로세스가 동시 쓰기 → 데이터 오염
- **해결**: 새 스크립트 실행 전 반드시 기존 프로세스 확인 및 종료
  ```powershell
  Get-Process python | Select-Object Id, CPU
  Stop-Process -Id <PID> -Force  # 필요 시
  ```

---

## 9. 진행 상황 모니터링 방법

```powershell
# NER 진행률 확인
(Get-Content c:\GitHub\chatbot\data\evaluation\ner_eval.jsonl | Measure-Object -Line).Lines

# RAG 진행률 확인
(Get-Content c:\GitHub\chatbot\data\evaluation\rag_eval.jsonl | Measure-Object -Line).Lines

# 환각 진행률 확인
(Get-Content c:\GitHub\chatbot\data\evaluation\hallu_eval.jsonl | Measure-Object -Line).Lines

# Python 프로세스 상태
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, @{N='CPU%';E={[math]::Round($_.CPU,1)}}, @{N='MB';E={[math]::Round($_.WorkingSet/1MB,1)}}
```

---

## 10. 최종 산출물

모든 단계 완료 후 `data/evaluation/results/` 에 다음 파일이 생성됨:

| 파일 | 내용 |
|------|------|
| `ner_results.json` | NER 모델별 P/R/F1 상세 |
| `ner_table.csv` | 논문용 NER 표 |
| `rag_results.json` | RAG 시스템별 Hit@k, MRR, NDCG 상세 |
| `rag_table.csv` | 논문용 RAG 표 |
| `hallucination_results.json` | 환각 탐지 Acc/P/R/F1/AUC 상세 |
| `hallucination_table.csv` | 논문용 환각 탐지 표 |
