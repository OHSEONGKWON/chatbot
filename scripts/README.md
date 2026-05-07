# scripts/

운영, 평가, 재학습용 스크립트입니다.

## NER

```bash
# 답변-근거 쌍 기준 NER 환각 탐지 평가
python scripts/evaluate_ner_factcheck.py

# 하이브리드 매칭 임계값 튜닝
python scripts/tune_ner_hybrid_thresholds.py

# 기존 모델 이어 학습
python scripts/train_legal_ner.py --base-model outputs/legal-ner-lawsguard-v2-30k --output-dir outputs/legal-ner-lawsguard-v3 --epochs 3 --batch-size 8
```

- `evaluate_ner_factcheck.py`: 실제 답변과 RAG 근거 문서를 비교해 precision/recall/F1을 계산합니다.
- `train_legal_ner.py`: BIO JSONL 데이터로 NER 모델을 학습하고 token F1, entity-span F1을 저장합니다.
- `tune_ner_hybrid_thresholds.py`: 개체 매칭 임계값을 실험합니다.

## RAG

```bash
python scripts/evaluate_rag_quality.py
```

대표 질의별로 검색된 문서가 기대 법령/매뉴얼을 포함하는지 확인합니다.
