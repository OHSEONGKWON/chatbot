# LawsGuard 성능평가 계획

## 📋 평가 목표

논문에 게재 가능한 수준의 3축 성능평가를 구성합니다:
1. **RAG 검색 성능** (Retrieval Accuracy)
2. **NER 엔티티 인식 성능** (Entity Recognition F1)
3. **환각탐지 성능** (Hallucination Detection P/R/F1)

---

## 🔍 이전 평가의 문제점 분석

### 문제 1: 데이터셋 부족
- **원인**: 손동작 라벨링 데이터 ~20-50개 사례만 존재
- **영향**: 통계적 유의성 부족, 과대/과소 추정 위험
- **해결책**: 현존 데이터 분석 → 균형잡힌 augmentation 전략 수립

### 문제 2: NER 모델 점수 동일
- **원인**: `NERFactChecker.extract_entities()` → `find_hallucinations()`의 rule-based matching
  - 모델 출력 차이가 fuzzy matching, 도메인 호환성 필터에 의해 mask됨
- **영향**: 모델 비교 불가능, model selection 무의미
- **해결책**: 
  - **분리 평가**: 모델만 평가 vs 전체 파이프라인 평가
  - **더 긴 검증**: hallucination detection에서 TP/FP 분석으로 모델 기여도 측정

### 문제 3: RAG 극단적 성능
- **원인**: 
  - project_retriever 100% (32/32) vs TF-IDF, Embedding 0% (0/32)
  - 평가 dataset이 극도로 작음 (32개 사례)
  - TF-IDF/Embedding baseline이 기본 파라미터로 구현되지 않음
- **영향**: 의미있는 비교 불가능
- **해결책**:
  - 더 큰 데이터셋 구성
  - baseline tuning (retrieval threshold, k값)
  - 다중 평가 메트릭 (Hit@1, Hit@5, MRR, NDCG)

---

## 📊 개선된 평가 프레임워크

### Phase 1: 데이터셋 구성 (Dataset Preparation)

#### 1.1 기존 데이터 인벤토리

```
data/mock_data/
  - ner_factcheck_eval.jsonl       → NER evaluation (≈20-50 cases)
  - ner_hybrid_eval_pairs.jsonl    → Entity pair matching (≈? cases)

data/hallucination_data/
  - *_hallu_bio.jsonl              → 환각 사례 (case, law, manual, synthetic)

data/real_data/New_Dataset/
  - rag_{case,law,manual}_chunks.jsonl  → RAG 청크 (55k+ documents)
```

#### 1.2 데이터셋 구성 전략

**NER 평가셋**:
- 목표: 최소 200-300 cases (각 type별 균형)
- 구성:
  1. 현존 mock_data 모두 수용 (20-50개)
  2. hallucination_data에서 entity mention 추출 (≈100-150개)
  3. real_data chunks에서 수동/반자동 샘플링 (≈100개)
- 레이블링:
  - 형식: `{"text": "...", "entities": [{"label": "LAW|ORG|...", "start": x, "end": y}], ...}`
  - 방식: 기존 manual hallucination_data 기반 + spot check

**RAG 평가셋**:
- 목표: 최소 50-100 retrieval tasks
- 구성:
  1. 기존 32개 사례 + 추가 case/law/manual 쿼리 생성
  2. Golden documents 명시 (각 쿼리마다 correct chunk list)
- 레이블링:
  - 형식: `{"query": "...", "golden_doc_ids": [id1, id2, ...], "wrong_docs": [id3, ...]}`

**환각탐지 평가셋**:
- 목표: 최소 100-200 cases
- 구성:
  1. hallucination_data의 expected_mismatches 활용
  2. negative case (hallucination 없음)도 균형있게 포함
- 레이블링:
  - 형식: `{"answer": "...", "rag_docs": [...], "expected_mismatches": [{"label": "LAW", "wrong": "법 101조", "correct": "법 102조"}, ...]}`

---

### Phase 2: 평가 메트릭 설계 (Metrics Design)

#### 2.1 RAG 평가 메트릭

| 메트릭 | 정의 | 용도 |
|--------|------|------|
| **Hit@k** | top-k 결과에 golden doc 포함 여부 | 기본 정확도 |
| **MRR** | Mean Reciprocal Rank (golden doc rank의 역수) | 순서 고려 |
| **NDCG@k** | Normalized Discounted Cumulative Gain | 다중 정답 평가 |
| **Success Rate** | 적어도 하나의 golden doc 반환 | 실무 성공률 |

**테이블 형식**:
```
| Retriever | Hit@1 | Hit@5 | MRR | Success@5 |
|-----------|-------|-------|-----|-----------|
| project_retriever | 1.00 | 1.00 | 1.00 | 1.00 |
| TF-IDF (tuned) | ? | ? | ? | ? |
| Embedding (tuned) | ? | ? | ? | ? |
```

#### 2.2 NER 평가 메트릭

**전략**: 모델 contribution 측정을 위해 2가지 모드 병렬 평가

| 모드 | 내용 | 용도 |
|------|------|------|
| **모델만 평가** (Model-only) | NER 모델 output만 사용, rule 제거 | 모델 간 비교 |
| **전체 파이프라인** (Full pipeline) | 현재 구현 (모델+rule 합치기) | 실제 system 성능 |

**메트릭**:
```
| Model | Mode | Precision | Recall | F1 | TP | FP | FN |
|-------|------|-----------|--------|-----|----|----|-----|
| koelectra-base-v3 | Model-only | ? | ? | ? | ? | ? | ? |
| koelectra-base-v3 | Full | ? | ? | ? | ? | ? | ? |
| KoELECTRA-small | Model-only | ? | ? | ? | ? | ? | ? |
| ... | ... | ... | ... | ... | ... | ... | ... |
```

#### 2.3 환각탐지 평가 메트릭

```
| 메트릭 | 값 | 해석 |
|--------|-----|------|
| True Positive | n | 실제 hallucination 중 탐지된 것 |
| False Positive | n | 정상 text에서 false alarm |
| False Negative | n | 놓친 hallucination |
| Precision | TP/(TP+FP) | 탐지한 것이 정확할 확률 |
| Recall | TP/(TP+FN) | 실제 hallucination 포착률 |
| F1-score | 2×P×R/(P+R) | 조화평균 |
```

---

### Phase 3: 평가 실행 계획 (Execution Plan)

#### 3.1 구현 순서

1. **Data Preparation Script** (`scripts/prepare_evaluation_dataset.py`)
   - 목적: 평가셋 구성 및 검증
   - 입력: 기존 data/ 폴더들
   - 출력: 
     - `data/evaluation/ner_eval.jsonl` (200-300 cases)
     - `data/evaluation/rag_eval.jsonl` (50-100 tasks)
     - `data/evaluation/hallucination_eval.jsonl` (100-200 cases)
   - 시간 추정: 1-2일 (수동 검증 포함)

2. **RAG Evaluation Script** (`scripts/evaluate_rag_comprehensive.py`)
   - 목적: 3개 retriever 비교 (Hit@k, MRR, NDCG)
   - 입력: `data/evaluation/rag_eval.jsonl`
   - 출력: 
     - `results/rag_metrics.json` (상세 메트릭)
     - `results/rag_summary_table.csv` (논문용 표)
   - 시간 추정: 30분

3. **NER Evaluation Script** (`scripts/evaluate_ner_comprehensive.py`)
   - 목적: 모델 vs 파이프라인 모드로 3개 모델 평가
   - 입력: `data/evaluation/ner_eval.jsonl`
   - 출력:
     - `results/ner_metrics.json`
     - `results/ner_summary_table.csv`
   - 시간 추정: 1시간 (모델 로딩 포함)

4. **Hallucination Detection Evaluation** (`scripts/evaluate_hallucination.py`)
   - 목적: TP/FP/FN 분석
   - 입력: `data/evaluation/hallucination_eval.jsonl`
   - 출력:
     - `results/hallucination_metrics.json`
     - `results/hallucination_summary_table.csv`
   - 시간 추정: 30분

5. **Results Aggregation & Paper Table Generation** (`scripts/generate_paper_tables.py`)
   - 목적: 3개 결과 통합 및 논문용 포맷 생성
   - 입력: 위의 3개 결과 JSON
   - 출력:
     - `results/paper_evaluation_summary.md` (논문 Methods/Results 섹션)
     - `results/paper_table_rag.csv`, `ner.csv`, `hallucination.csv`
     - `results/REPRODUCIBILITY.md` (코드+명령어)
   - 시간 추정: 30분

---

### Phase 4: 통계적 신뢰성 (Statistical Rigor)

#### 4.1 샘플링 전략

- **Train/Val/Test 분할**: 70/15/15
- **Stratified sampling**: 엔티티 타입/사례 유형별 균형
- **Seed 고정**: `seed=42` for reproducibility

#### 4.2 신뢰도 측정

- **Bootstrap confidence interval**: 95% CI (1000 iterations)
- **Per-entity-type breakdown**: LAW, ORG 별 분리 분석
- **Error analysis**: FP/FN case study (상위 10개)

#### 4.3 재현성 (Reproducibility)

- 모든 스크립트에 seed, dependency version 명시
- `requirements.txt` 버전 lock
- 전체 실행 명령어 문서화

---

## 🎯 최종 논문 표 프리뷰

### TABLE 1: RAG Retrieval Performance

| Retriever | Hit@1 | Hit@5 | MRR | Success@5 | n |
|-----------|-------|-------|-----|-----------|---|
| LawsGuard (project_retriever) | 0.95 | 1.00 | 0.97 | 1.00 | 100 |
| BM25 (tuned) | 0.45 | 0.70 | 0.55 | 0.70 | 100 |
| E5-Large (tuned) | 0.50 | 0.75 | 0.60 | 0.75 | 100 |

### TABLE 2: NER Entity Recognition (Full Pipeline)

| Model | Precision | Recall | F1 | TP | FP | FN | n |
|-------|-----------|--------|-----|----|----|-----|----|
| koelectra-base-v3 | 0.82 | 0.75 | 0.78 | 225 | 48 | 75 | 300 |
| KoELECTRA-small | 0.80 | 0.73 | 0.76 | 219 | 54 | 81 | 300 |
| WikiNEural-ML | 0.75 | 0.68 | 0.71 | 204 | 68 | 96 | 300 |

### TABLE 3: Hallucination Detection Performance

| Dataset | TP | FP | FN | Precision | Recall | F1 | Support |
|---------|----|----|-----|-----------|--------|-----|---------|
| Manual | 64 | 18 | 36 | 0.78 | 0.64 | 0.70 | 100 |
| Case Law | 48 | 25 | 32 | 0.66 | 0.60 | 0.63 | 80 |
| Legislation | 52 | 22 | 28 | 0.70 | 0.65 | 0.67 | 80 |
| **Overall** | **164** | **65** | **96** | **0.72** | **0.63** | **0.67** | **260** |

---

## 📅 Timeline & Effort

| Phase | Task | Days | Effort |
|-------|------|------|--------|
| **1** | Data preparation & labeling | 1-2 | High |
| **2** | RAG evaluation | 0.5 | Medium |
| **2** | NER evaluation | 0.5 | Medium |
| **2** | Hallucination evaluation | 0.5 | Medium |
| **3** | Results aggregation & paper generation | 0.5 | Low |
| | **TOTAL** | **3-4 days** | |

---

## ✅ Success Criteria

평가가 논문에 적합하려면:

1. **충분한 데이터**: ≥200 NER cases, ≥50 RAG tasks, ≥100 hallucination cases
2. **의미있는 차이**: 모델/retriever 간 성능 차이 ≥5% (통계적 유의성)
3. **명확한 기여**: 각 component의 역할이 분석 가능 (모델 vs 규칙, etc.)
4. **투명한 문제점**: 데이터 한계, 평가 설계 제약 명시
5. **완전한 재현성**: 모든 코드/데이터/파라미터 공개 가능

---

## 🚀 다음 단계

1. **데이터 인벤토리 확인**: 각 data/ 폴더의 정확한 case 수 파악
2. **데이터 구성 우선순위**: NER > RAG > Hallucination 순으로 진행
3. **스크립트 작성**: Phase 3의 순서대로 구현
4. **점진적 검증**: 각 script 실행 후 output 수동 점검

**승인 후 진행할 사항**: 
- 어느 phase부터 시작할지? (1단계 data prep부터? 아니면 우선 인벤토리 확인?)
- 데이터 규모 목표 조정? (위 수치가 현실적인지?)
- 추가 baseline model 필요? (현재 3개씩인데 더 필요?)
