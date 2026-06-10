# RAG 성능 평가 스크립트

BM25, Dense (BGE-M3), Hybrid 검색 방법을 비교하여 법률 상담 챗봇의 RAG 성능을 평가합니다.

## 설치

```bash
# 기본 라이브러리
pip install -r requirements.txt

# 평가용 추가 라이브러리
pip install -r scripts/requirements_eval.txt

# GPU 사용 (권장)
# PyTorch CUDA 버전 설치 (CUDA 11.8 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### GPU 사용 여부 확인

```bash
python -c "import torch; print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"
```

**✅ GPU 사용 시 장점:**
- 평가 속도 **2-3배 빠름**
- Phase 1: 60분 → 20분
- Phase 2: 180분 → 90분

자세한 내용: [GPU_USAGE.md](GPU_USAGE.md)

## 실행 순서

### 1단계: 평가 데이터셋 생성 (200쌍)

```bash
python scripts/generate_dataset.py
```

**출력:**
- `data/evaluation/generated_dataset_200.jsonl`: 노동 100쌍 + 성폭력 100쌍

**소요 시간:** 약 30-60분 (GPT API 호출 포함)

---

### 2단계: Phase 1 - Retrieval 성능 비교

```bash
python scripts/compare_retrieval.py
```

**비교 대상:**
- BM25 (현재 사용 중)
- Dense Retrieval (BGE-M3)
- Hybrid (BM25 + Dense, 가중치 최적화)

**평가 지표:**
- Precision@K (K=1,3,5)
- Recall@K
- MRR (Mean Reciprocal Rank)
- NDCG@5 (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)

**출력:**
- `results/phase1_retrieval_comparison.json`: 상세 결과
- 콘솔: 결과 표

**소요 시간:** 약 20-40분 (BGE-M3 임베딩 계산 포함)

---

### 3단계: Phase 2 - RAG End-to-End 평가 (RAGAS)

```bash
python scripts/evaluate_rag_end_to_end.py
```

**비교 대상:**
- BM25-RAG (전체 파이프라인)
- Dense-RAG (전체 파이프라인)
- Hybrid-RAG (전체 파이프라인)

**평가 지표 (RAGAS):**
- **Faithfulness**: 답변이 검색 문서에 충실한가? (환각 측정 포함)
  - 0.92 = 답변의 92%가 검색 문서에 근거 (환각 8%)
- **Answer Relevancy**: 답변이 질문에 적절한가?
- **Context Precision**: 검색 문서가 정밀한가?
- **Context Recall**: 필요한 정보를 다 찾았는가?

**법률 도메인 메트릭:**
- **Citation Accuracy**: 정답 법률 조문 언급 정확도

**출력:**
- `results/phase2_ragas_evaluation.json`: 상세 결과
- 콘솔: 결과 표

**소요 시간:** 약 2-3시간 (LLM 평가 포함, 전체 200개)

---

### 4단계: Phase 3 - 결과 분석 및 시각화

```bash
python scripts/analyze_results.py
```

**기능:**
- Phase 1 + Phase 2 종합 결과 표
- 카테고리별 (노동 vs 성폭력) 성능 분석
- 통계적 유의성 검정 (t-test)
- 시각화 그래프 생성
- Excel 파일 내보내기

**출력:**
- 콘솔: 종합 결과 표
- `results/visualizations/phase1_retrieval_comparison.png`: 검색 성능 비교 그래프
- `results/visualizations/phase2_ragas_comparison.png`: RAGAS 레이더 차트
- `results/rag_evaluation_summary.xlsx`: Excel 요약

**소요 시간:** 약 1-2분

---

## 순차 실행 (하나씩 실행하면서 오류 확인)

각 단계를 **개별적으로 실행**하여 오류가 생기면 바로 확인하고 수정하세요.

```bash
# 1. 데이터셋 생성
python scripts/generate_dataset.py

# 2. Phase 1 실행
python scripts/compare_retrieval.py

# 3. Phase 2 실행 (시간 많이 걸림)
python scripts/evaluate_rag_end_to_end.py

# 4. Phase 3 실행
python scripts/analyze_results.py
```

**💡 Tip:** 각 단계가 완료될 때까지 기다린 후 다음 단계로 진행하세요.

---

## 결과 예시

### Phase 1: Retrieval 성능 비교

```
╔═══════════════════╦══════════╦══════════╦══════════╦══════════╦══════════╦══════════╗
║ 방법              ║ P@1      ║ P@3      ║ P@5      ║ MRR      ║ NDCG@5   ║ MAP      ║
╠═══════════════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════════╣
║ BM25              ║ 0.8438   ║ 0.9167   ║ 0.9375   ║ 0.9219   ║ 0.8470   ║ 0.8123   ║
║ Dense (BGE-M3)    ║ 0.7250   ║ 0.8500   ║ 0.8750   ║ 0.8345   ║ 0.7621   ║ 0.7234   ║
║ Hybrid (α=0.7)    ║ 0.8125   ║ 0.9000   ║ 0.9250   ║ 0.8987   ║ 0.8234   ║ 0.7891   ║
╚═══════════════════╩══════════╩══════════╩══════════╩══════════╩══════════╩══════════╝
```

### Phase 2: RAG End-to-End 평가

```
╔═══════════════════╦═══════════════╦═══════════════╦═══════════════╦═══════════════╗
║ 방법              ║ Faithfulness  ║ Answer Rel.   ║ Context Prec. ║ Citation Acc. ║
╠═══════════════════╬═══════════════╬═══════════════╬═══════════════╬═══════════════╣
║ BM25-RAG          ║ 0.9200        ║ 0.8800        ║ 0.8500        ║ 0.9100        ║
║ Dense-RAG         ║ 0.8500        ║ 0.8200        ║ 0.7600        ║ 0.8300        ║
║ Hybrid-RAG (α=0.7)║ 0.8900        ║ 0.8500        ║ 0.8200        ║ 0.8800        ║
╚═══════════════════╩═══════════════╩═══════════════╩═══════════════╩═══════════════╝

※ Faithfulness 0.92 = 답변의 92%가 검색 문서에 근거 (환각 8%)
```

---

## 문제 해결

### BGE-M3 모델 다운로드 실패

```bash
# HuggingFace 캐시 디렉토리 확인
echo $HF_HOME

# 수동 다운로드
huggingface-cli download BAAI/bge-m3
```

### GPU 메모리 부족

BGE-M3 임베딩 시 GPU 메모리가 부족하면:

1. `compare_retrieval.py`에서 배치 크기 조정:
```python
self.doc_embeddings = self.model.encode(
    texts,
    batch_size=16,  # 32 → 16으로 줄이기
    ...
)
```

2. CPU 모드로 강제:
```python
self.device = "cpu"  # "cuda" 대신
```

### OpenAI API Rate Limit

데이터셋 생성 시 Rate Limit이 걸리면:

1. 배치 크기 조정:
```python
per_issue = labor_count // len(LABOR_ISSUES)
# 또는 sleep 추가
await asyncio.sleep(1)
```

---

## 참고 문헌

- **RAGAS**: [https://arxiv.org/abs/2309.15217](https://arxiv.org/abs/2309.15217)
- **BM25**: Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009)
- **BGE-M3**: [https://arxiv.org/abs/2402.03216](https://arxiv.org/abs/2402.03216)
- **BEIR Benchmark**: [https://arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663)
