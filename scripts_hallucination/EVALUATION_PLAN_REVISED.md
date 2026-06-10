# 환각 탐지 성능평가 계획 (수정본)

## 📋 평가 목표

**우리 시스템의 3단계 환각 탐지와 SelfCheckGPT, MetaQA의 성능을 공평하게 비교**

---

## 📊 데이터셋

### 기존 데이터 사용: `data/evaluation/hallu_eval.jsonl`

**총 800개 케이스:**
- **정상 (Normal)**: 400개
- **환각 (Hallucinated)**: 400개

**환각 유형 분포:**

| 환각 유형 | 개수 | 설명 |
|-----------|------|------|
| `article_number_error` | 100개 | 존재하지 않는 조문 번호 |
| `contact_mismatch` | 100개 | 연락처/정보 불일치 |
| `forbidden_law_injection` | 100개 | 부적절한 법률 주입 |
| `semantic_error` | 100개 | 의미적 오류 |

**데이터 형식:**
```json
{
  "hallu_id": "hallu_0001",
  "question": "질문 내용",
  "answer": "답변 내용 (환각 포함 가능)",
  "label": "normal" 또는 "normal" (hallu_type으로 구분),
  "hallu_type": "none" 또는 환각 유형
}
```

**실제 라벨:**
- `hallu_type == "none"` → 정상
- `hallu_type != "none"` → 환각

---

## 🔍 비교 대상 (3가지)

### 1. **우리 시스템 (Ours)**
**3단계 파이프라인:**

```
답변 → [1] 일관성 평가 → [2] NER 체크 → [3] NLI 모순검사 → 환각 판정
```

**각 단계:**

1. **일관성 평가 (Consistency Check)**
   - 유사 질문 **10개** 생성
   - 각 질문에 대한 답변 생성
   - 답변 간 의미적 유사도 측정 (임베딩)
   - 유사도 < 0.8 → 환각 의심

2. **NER 체크 (Named Entity Recognition)**
   - 답변에서 법률 엔티티 추출
     - 조문 번호 (예: "근로기준법 제26조")
     - 판례 번호
     - 형량/벌금
   - RAG 문서에 존재 여부 확인
   - 미존재 엔티티 → 환각

3. **NLI 의미적 모순검사**
   - 답변을 문장 단위로 분해
   - 각 문장 vs RAG 문서: NLI 수행
   - Contradiction → 환각

**최종 판정:**
- 3단계 중 **하나라도 환각 판정** → 환각
- 모두 통과 → 정상

---

### 2. **SelfCheckGPT**

**원리:** 자기 일관성 기반 (Self-Consistency)

**방법:**
1. 동일 질문에 대해 **10개 답변** 생성 (temperature=1.0)
2. 각 문장에 대해:
   - 다른 답변들과의 일관성 측정
   - 일관성 낮음 → 환각 가능성

**3가지 변형 모두 테스트:**

#### (a) SelfCheckGPT-BERTScore
- 문장 임베딩 유사도 (BERTScore)
- 빠름

#### (b) SelfCheckGPT-NLI
- NLI 모델로 일관성 체크
- **우리와 동일한 NLI 모델 사용** (공평성)
- 중간 속도

#### (c) SelfCheckGPT-Prompt
- LLM에게 직접 일관성 평가 요청
- 느림

**샘플링 수:** 10개 (우리 시스템과 동일)

---

### 3. **MetaQA** (Meta-Question Answering)

**원리:** 메타 질문을 통한 검증

**방법:**
1. 원래 답변 생성
2. 답변에서 주요 사실(fact) 추출
   - LLM에게 "이 답변에서 주요 사실을 추출하세요" 요청
3. 각 사실에 대해 메타 질문 생성
   - 예: "답변에서 '근로기준법 제26조'라고 했는데 맞습니까?"
4. LLM에게 메타 질문 → 일관성 체크
5. 불일치 발견 → 환각

**구현:**
- Fact 추출: LLM
- 메타 질문 생성: LLM
- 검증: LLM

---

## 📊 평가 메트릭

### 이진 분류 (환각 vs 정상)

| 메트릭 | 수식 | 의미 |
|--------|------|------|
| **Precision** | TP / (TP + FP) | 환각이라고 판단한 것 중 실제 환각 비율 |
| **Recall** | TP / (TP + FN) | 실제 환각 중 찾아낸 비율 |
| **F1-Score** | 2 × (P × R) / (P + R) | Precision과 Recall의 조화평균 |

**용어:**
- TP (True Positive): 환각을 환각으로 판정 ✅
- FP (False Positive): 정상을 환각으로 오판 ❌
- FN (False Negative): 환각을 정상으로 놓침 ❌
- TN (True Negative): 정상을 정상으로 판정 ✅

---

## 🎯 실험 설정

### 공통 조건 (공평성 보장)

| 항목 | 설정 |
|------|------|
| **데이터셋** | hallu_eval.jsonl (800개) |
| **LLM** | GPT-4 (현재 프로젝트 사용) |
| **임베딩 모델** | multilingual-e5-large (현재 사용) |
| **NLI 모델** | klue/roberta-large-nli (한국어) |
| **평가 메트릭** | Precision, Recall, F1 |

### 각 방법별 설정

**Ours:**
- 일관성: 10개 유사 질문, threshold=0.8
- NER: 법률 엔티티 추출
- NLI: klue/roberta-large-nli

**SelfCheckGPT:**
- 샘플링: **10개** (우리와 동일)
- Temperature: 1.0
- 변형: BERTScore, NLI, Prompt

**MetaQA:**
- Fact 추출: LLM
- 메타 질문: fact당 1-2개

---

## 📈 예상 결과

### 정확도 (F1-Score 예측):

| 순위 | 방법 | 예상 F1 | 이유 |
|------|------|---------|------|
| 🥇 | **Ours** | 0.88-0.92 | 3단계 세밀 검증, 법률 특화 |
| 🥈 | **MetaQA** | 0.83-0.87 | 사실 검증 효과적 |
| 🥉 | **SelfCheckGPT-NLI** | 0.78-0.82 | 일관성만 체크 |
| 4 | **SelfCheckGPT-Prompt** | 0.75-0.80 | LLM 의존 |
| 5 | **SelfCheckGPT-BERTScore** | 0.70-0.75 | 임베딩 유사도만 |

### 속도 (초/케이스 예측):

| 방법 | 예상 시간 | LLM 호출 |
|------|-----------|----------|
| SelfCheckGPT-BERTScore | ~2초 | 10회 |
| **Ours** | ~12초 | 11회 |
| SelfCheckGPT-NLI | ~15초 | 10회 |
| MetaQA | ~20초 | 15-20회 |
| SelfCheckGPT-Prompt | ~25초 | 20회 |

### 환각 유형별 성능:

**우리 시스템 강점:**
- ✅ `article_number_error`: NER로 정확히 탐지
- ✅ `semantic_error`: NLI로 탐지

**SelfCheckGPT 강점:**
- ✅ 일관성 없는 답변 (샘플링으로 발견)

**MetaQA 강점:**
- ✅ 사실 오류 (메타 질문으로 검증)

---

## 📁 출력 형식

### 1. 종합 비교 표

```
+----------------------+-------+-------+-------+----------+----------+
| 방법                 | Prec. | Rec.  | F1    | 시간(초) | LLM 호출 |
+======================+=======+=======+=======+==========+==========+
| Ours (3-stage)       | 0.91  | 0.89  | 0.90  | 12.3     | 11       |
| SelfCheckGPT-NLI     | 0.80  | 0.78  | 0.79  | 15.1     | 10       |
| SelfCheckGPT-BERT    | 0.73  | 0.71  | 0.72  | 2.1      | 10       |
| SelfCheckGPT-Prompt  | 0.77  | 0.75  | 0.76  | 24.8     | 20       |
| MetaQA               | 0.85  | 0.83  | 0.84  | 19.5     | 17       |
+----------------------+-------+-------+-------+----------+----------+
```

### 2. 환각 유형별 Recall

```
+----------------------+---------------+------------+----------+--------------+
| 방법                 | 조문번호오류  | 연락처불일치 | 법률주입 | 의미적오류   |
+======================+===============+============+==========+==============+
| Ours                 | 0.95          | 0.82       | 0.88     | 0.91         |
| SelfCheckGPT-NLI     | 0.65          | 0.78       | 0.80     | 0.82         |
| MetaQA               | 0.80          | 0.85       | 0.83     | 0.84         |
+----------------------+---------------+------------+----------+--------------+
```

### 3. Confusion Matrix (각 방법)

```
Ours (3-stage):
              예측 정상  예측 환각
실제 정상      380        20
실제 환각       45       355
```

---

## 🛠️ 구현 계획

### 파일 구조

```
scripts_hallucination/
├── EVALUATION_PLAN_REVISED.md        # 이 문서
├── load_dataset.py                   # 데이터셋 로더
├── evaluate_ours.py                  # 우리 시스템 평가
├── evaluate_selfcheckgpt.py          # SelfCheckGPT 3가지 평가
├── evaluate_metaqa.py                # MetaQA 평가
├── compare_all.py                    # 종합 비교 및 분석
└── results/
    ├── ours_results.json
    ├── selfcheckgpt_results.json
    ├── metaqa_results.json
    └── comparison_report.md
```

### Phase 1: 데이터 로더 (30분)
- `load_dataset.py`
- hallu_eval.jsonl 파싱
- 정상/환각 라벨링

### Phase 2: 각 방법 구현 (2-3일)

**Day 1:**
- `evaluate_ours.py` (기존 시스템 활용)

**Day 2:**
- `evaluate_selfcheckgpt.py` (3가지 변형)

**Day 3:**
- `evaluate_metaqa.py`

### Phase 3: 비교 분석 (1일)
- `compare_all.py`
- 종합 표, 환각 유형별 분석
- 실패 케이스 분석

---

## 🎯 핵심 원칙

### 공평성 보장:

1. ✅ **동일 데이터**: 800개 모두 동일
2. ✅ **동일 LLM**: GPT-4
3. ✅ **동일 샘플링 수**: 10개 (Ours, SelfCheckGPT)
4. ✅ **동일 NLI 모델**: klue/roberta-large-nli
5. ✅ **동일 평가 메트릭**: Precision, Recall, F1

### 각 방법의 장점 최대화:

- **Ours**: 3단계 모두 활용
- **SelfCheckGPT**: 3가지 변형 모두 테스트
- **MetaQA**: 충분한 메타 질문 생성

---

## 📊 예상 소요 시간

### 개발:
- Day 1: 데이터 로더 + Ours 구현 (8시간)
- Day 2: SelfCheckGPT 구현 (8시간)
- Day 3: MetaQA 구현 (8시간)
- Day 4: 비교 분석 (4시간)

### 실행:
- Ours: 800개 × 12초 = **2.7시간**
- SelfCheckGPT-BERTScore: 800개 × 2초 = **0.4시간**
- SelfCheckGPT-NLI: 800개 × 15초 = **3.3시간**
- SelfCheckGPT-Prompt: 800개 × 25초 = **5.6시간**
- MetaQA: 800개 × 20초 = **4.4시간**

**총 실행 시간: 약 16시간** (병렬 실행 가능)

---

## ✅ 확인 완료 사항

- [x] 데이터셋: hallu_eval.jsonl (800개)
  - 정상: 400개
  - 환각: 400개 (4가지 유형)
- [x] 샘플링 수: 10개
- [x] 메트릭: Precision, Recall, F1
- [x] 도메인: 법률 전용
- [x] 환각 유형 분포 확인

---

**이 계획대로 코드 작성을 시작하시겠습니까?**
