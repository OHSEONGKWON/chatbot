# 환각 탐지 성능평가 계획

## 📋 평가 목표

**우리 시스템의 3단계 환각 탐지와 기존 방법들의 성능을 공평하게 비교**

---

## 🔍 비교 대상

### 1. **우리 시스템 (Ours)**
**3단계 파이프라인:**
1. **일관성 평가 (Consistency Check)**
   - 유사 질문 10개 생성
   - 각 질문에 대한 답변 생성
   - 답변 간 의미적 유사도 측정
   - 유사도 < threshold → 환각 의심

2. **NER 체크 (Named Entity Recognition)**
   - 답변에서 법률 엔티티 추출 (조문, 판례, 형량, 금액 등)
   - RAG 문서에서 엔티티 존재 여부 확인
   - 존재하지 않는 엔티티 → 환각

3. **NLI 의미적 모순검사 (Natural Language Inference)**
   - 답변을 문장 단위로 분해
   - 각 문장과 RAG 문서 간 NLI 수행
   - Contradiction (모순) → 환각

**특징:**
- ✅ 법률 도메인 특화
- ✅ 3단계로 세밀한 검증
- ❌ 복잡하고 느림 (3단계 모두 수행)

---

### 2. **SelfCheckGPT**
**원리:** 자기 일관성 기반 (Self-Consistency)

**방법:**
1. 동일한 질문에 대해 LLM이 여러 답변 생성 (sampling)
2. 각 답변의 각 문장에 대해:
   - 다른 답변들과 일관성 측정
   - 일관성이 낮은 문장 → 환각 가능성 높음

**변형:**
- **SelfCheckGPT-BERTScore**: BERT 임베딩 유사도
- **SelfCheckGPT-NLI**: NLI 모델로 일관성 체크
- **SelfCheckGPT-Prompt**: LLM에게 직접 일관성 평가 요청

**특징:**
- ✅ 간단하고 빠름
- ✅ 도메인 독립적
- ❌ RAG 문서를 직접 사용하지 않음
- ❌ 법률 조문 등 정확성 검증 약함

**출처:** 
- Paper: "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models" (Manakul et al., 2023)
- GitHub: https://github.com/potsawee/selfcheckgpt

---

### 3. **MetaQA** (Meta-Question Answering)
**원리:** 메타 질문을 통한 검증

**방법:**
1. 원래 답변 생성
2. 답변에서 주요 사실(fact) 추출
3. 각 사실에 대해 "메타 질문" 생성
   - 예: "원래 답변에서 X라고 했는데, 이게 맞습니까?"
4. LLM에게 메타 질문 → 답변이 inconsistent하면 환각

**특징:**
- ✅ 사실 단위로 검증
- ✅ 해석 가능성 높음
- ❌ 메타 질문 생성 품질에 의존
- ❌ LLM 호출 많음 (느림)

**출처:**
- Paper: "Chain-of-Verification Reduces Hallucination in Large Language Models" (Dhuliawala et al., 2023)
- Related: "Self-Refine: Iterative Refinement with Self-Feedback" (Madaan et al., 2023)

---

## 📊 공평한 평가 설계

### 1. **데이터셋**

#### Option A: 기존 데이터셋 재사용
- `data/evaluation/generated_dataset_200.jsonl` (198개)
- 노동법 100개 + 성폭력법 98개
- **문제:** 환각 여부가 라벨링되지 않음!

#### Option B: 환각 데이터셋 생성 ⭐ **추천**
**구성:**
- **정상 답변 (Non-Hallucinated)**: 100개
  - RAG 문서에 근거한 올바른 답변
  
- **환각 답변 (Hallucinated)**: 100개
  - **조문 환각**: 존재하지 않는 법률 조문 (예: "근로기준법 제999조")
  - **판례 환각**: 존재하지 않는 판례
  - **형량 환각**: 잘못된 형량/벌금
  - **사실 왜곡**: RAG 문서와 모순되는 내용

**생성 방법:**
1. RAG로 정상 답변 생성
2. LLM에게 "환각이 포함된 답변" 생성 요청
3. 수동으로 검증 및 라벨링

---

### 2. **평가 메트릭**

#### 이진 분류 (환각 vs 정상)

| 메트릭 | 설명 |
|--------|------|
| **Precision** | 환각이라고 판단한 것 중 실제 환각 비율 |
| **Recall** | 실제 환각 중 찾아낸 비율 |
| **F1-Score** | Precision과 Recall의 조화평균 |
| **Accuracy** | 전체 정확도 |
| **AUC-ROC** | 임계값에 무관한 성능 (확률 점수 필요) |

#### 문장 단위 평가 (세밀한 분석)

환각이 포함된 특정 문장을 찾아내는 능력:
- **Sentence-level Precision/Recall/F1**

---

### 3. **실험 설정**

#### 공통 조건:
- **데이터셋**: 동일 (200개)
- **LLM**: 동일 (GPT-4 또는 현재 사용 중인 모델)
- **RAG 검색**: 동일 (BM25, top_k=5)
- **평가 메트릭**: 동일

#### 각 방법별 최적 설정:

**Ours (3-stage):**
- 일관성: 유사 질문 10개, threshold=0.8
- NER: 법률 엔티티 추출기
- NLI: 한국어 NLI 모델 (klue/roberta-large-nli)

**SelfCheckGPT:**
- 샘플링: 5-10개 답변
- 변형: BERTScore, NLI, Prompt 모두 테스트
- NLI 모델: 동일 (공평성)

**MetaQA:**
- 메타 질문: fact당 1-2개
- LLM: 동일

---

### 4. **비교 차원**

| 차원 | 설명 |
|------|------|
| **정확도** | Precision, Recall, F1 |
| **속도** | 처리 시간 (초/케이스) |
| **비용** | LLM API 호출 횟수 |
| **도메인 적합성** | 법률 특화 vs 범용 |
| **해석 가능성** | 왜 환각으로 판단했는지 설명 가능? |

---

## 🎯 예상 결과 가설

### 정확도 (F1-Score):
1. **Ours**: 높음 (3단계로 세밀하게 검증)
2. **SelfCheckGPT-NLI**: 중간 (일관성 기반)
3. **MetaQA**: 중상 (사실 검증)

### 속도:
1. **SelfCheckGPT-BERTScore**: 빠름
2. **Ours**: 느림 (3단계)
3. **MetaQA**: 매우 느림 (메타 질문 생성)

### 법률 도메인 적합성:
1. **Ours**: 최고 (법률 엔티티 검증)
2. **MetaQA**: 중간
3. **SelfCheckGPT**: 낮음 (범용)

---

## 📁 결과 보고 형식

### 종합 비교 표:

| 방법 | Precision | Recall | F1 | 속도 (초) | LLM 호출 | 해석가능성 |
|------|-----------|--------|----|-----------|-----------|-----------| 
| Ours (3-stage) | 0.92 | 0.88 | 0.90 | 15.3 | 11 | ⭐⭐⭐ |
| SelfCheckGPT-NLI | 0.85 | 0.82 | 0.83 | 3.2 | 6 | ⭐⭐ |
| SelfCheckGPT-BERTScore | 0.78 | 0.75 | 0.76 | 1.5 | 5 | ⭐ |
| MetaQA | 0.88 | 0.85 | 0.86 | 22.1 | 15 | ⭐⭐⭐ |

### 세부 분석:
- **환각 유형별 성능**: 조문 환각, 판례 환각, 사실 왜곡 각각에 대한 Recall
- **문장 단위 분석**: 환각 문장을 정확히 찾아내는 능력
- **실패 사례 분석**: 각 방법이 놓친 환각 케이스

---

## 🛠️ 구현 계획

### Phase 1: 데이터셋 생성 (1-2일)
- `generate_hallucination_dataset.py`
- 정상 100개 + 환각 100개
- JSON 형식으로 저장

### Phase 2: 각 방법 구현 (2-3일)
- `evaluate_ours.py` - 우리 시스템 (기존 코드 활용)
- `evaluate_selfcheckgpt.py` - SelfCheckGPT 3가지 변형
- `evaluate_metaqa.py` - MetaQA

### Phase 3: 평가 및 분석 (1일)
- `compare_methods.py` - 종합 비교
- `analyze_results.py` - 세부 분석, 표 생성

---

## ❓ 논의 사항

### 1. 데이터셋 크기
- 200개 (정상 100 + 환각 100)로 충분한가?
- 더 많이 필요한가? (400개? 600개?)

### 2. 환각 유형 비율
- 조문 환각: 30%
- 판례 환각: 20%
- 형량 환각: 20%
- 사실 왜곡: 30%
- 이 비율이 적절한가?

### 3. SelfCheckGPT 샘플링 수
- 5개? 10개? 20개?
- 많을수록 정확하지만 느림

### 4. 평가 기준
- F1-Score를 주요 메트릭으로 하는 것이 맞는가?
- 속도와 정확도의 trade-off를 어떻게 평가할 것인가?

### 5. 기존 벤치마크 사용
- TruthfulQA, HaluEval 같은 기존 환각 평가 데이터셋 사용?
- 아니면 법률 도메인 특화 데이터셋만 사용?

---

## 📚 참고 문헌

1. **SelfCheckGPT**
   - Manakul, P., Liusie, A., & Gales, M. J. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. arXiv preprint arXiv:2303.08896.

2. **Chain-of-Verification (MetaQA 관련)**
   - Dhuliawala, S., et al. (2023). Chain-of-Verification Reduces Hallucination in Large Language Models. arXiv preprint arXiv:2309.11495.

3. **환각 평가 벤치마크**
   - Lin, S., et al. (2021). TruthfulQA: Measuring How Models Mimic Human Falsehoods. arXiv preprint arXiv:2109.07958.
   - Li, J., et al. (2023). HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models. arXiv preprint arXiv:2305.11747.

4. **NLI 모델 (한국어)**
   - Park, S., et al. (2021). KLUE: Korean Language Understanding Evaluation. arXiv preprint arXiv:2105.09680.

---

**이 계획에 대해 의견을 주시면 수정 후 코드 작성을 시작하겠습니다!**
