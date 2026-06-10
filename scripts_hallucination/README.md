# 환각 탐지 성능평가

## 📋 비교 대상 (4가지)

1. **Ours** - 우리 시스템 (일관성 → NER → NLI 3단계)
2. **SelfCheckGPT-BERTScore** - 임베딩 유사도 기반
3. **SelfCheckGPT-NLI** - 자연어 추론 기반
4. **MetaQA** - 메타 질문 기반

## 📊 데이터셋

- **파일**: `data/evaluation/hallu_eval.jsonl`
- **총 800개**: 정상 400개 + 환각 400개
- **환각 유형**:
  - 조문 번호 오류: 100개
  - 연락처 불일치: 100개
  - 금지된 법률 주입: 100개
  - 의미적 오류: 100개

## 🚀 실행 방법

### 1단계: 데이터 확인
```bash
cd c:/GitHub/chatbot
python scripts_hallucination/load_dataset.py
```

### 2단계: 평가 실행 (병렬 가능)

```bash
# 1. 우리 시스템 (~2.5시간)
python scripts_hallucination/evaluate_ours.py

# 2. SelfCheckGPT - BERTScore (~0.5시간)
python scripts_hallucination/evaluate_selfcheckgpt.py bertscore

# 3. SelfCheckGPT - NLI (~3.5시간)
python scripts_hallucination/evaluate_selfcheckgpt.py nli

# 4. MetaQA (~4.5시간)
python scripts_hallucination/evaluate_metaqa.py
```

### 3단계: 종합 비교
```bash
python scripts_hallucination/compare_all.py
```

## ⏱️ 예상 소요 시간

- **총 실행 시간**: ~11시간
- **병렬 실행 시**: ~4.5시간

## 📁 결과 파일

```
scripts_hallucination/results/
├── ours_results.json
├── selfcheckgpt_bertscore_results.json
├── selfcheckgpt_nli_results.json
└── metaqa_results.json
```

## 📊 평가 메트릭

- **Precision**: 환각으로 판단한 것 중 실제 환각 비율
- **Recall**: 실제 환각 중 찾아낸 비율
- **F1-Score**: Precision과 Recall의 조화평균

## 🎯 예상 결과

| 방법 | 예상 F1 |
|------|---------|
| Ours | ~0.90 |
| SelfCheck-NLI | ~0.79 |
| SelfCheck-BERT | ~0.72 |
| MetaQA | ~0.84 |
