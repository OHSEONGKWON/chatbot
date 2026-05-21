# LawsGuard 성능 평가 결과

## TABLE 1: NER 개체명 인식 성능 (Entity-level Span F1)

| 모델 | LAW | ORG | DATE | AMOUNT | CRIME | PENALTY | Macro F1 | Overall F1 |
|------|-----|-----|------|--------|-------|---------|----------|------------|
| our_model | 0.015 | 0.020 | 0.000 | 0.000 | 0.059 | 0.020 | 0.019 | 0.014 |
| babelscape | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| koelectra_modu | 0.022 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.004 | 0.010 |
| koelectra_naver | 0.007 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.003 |
| klue_bert | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

*F1: span-level exact match, macro: 6개 엔티티 평균*

## TABLE 2: RAG 검색 성능

| 시스템 | Hit@1 | Hit@5 | Hit@10 | MRR@10 | NDCG@10 | N |
|--------|-------|-------|--------|--------|---------|---|
| BM25 | 0.087 | 0.193 | 0.300 | 0.124 | 0.084 | 150 |
| Dense (multilingual-E5-large) | 0.220 | 0.440 | 0.467 | 0.311 | 0.254 | 150 |
| Hybrid (multilingual-E5-large) | 0.333 | 0.420 | 0.467 | 0.374 | 0.205 | 150 |
| Dense (ko-SRoBERTa-multitask) | 0.240 | 0.400 | 0.420 | 0.309 | 0.280 | 150 |
| Hybrid (ko-SRoBERTa-multitask) | 0.227 | 0.333 | 0.420 | 0.283 | 0.195 | 150 |
| Dense (KoSimCSE-RoBERTa) | 0.133 | 0.347 | 0.360 | 0.208 | 0.159 | 150 |
| Hybrid (KoSimCSE-RoBERTa) | 0.280 | 0.327 | 0.407 | 0.304 | 0.153 | 150 |

*Hit@k: top-k 내 정답 문서 포함 비율, MRR: Mean Reciprocal Rank, NDCG: Normalized DCG*

## TABLE 3: 환각 탐지 성능

| 시스템 | Accuracy | Precision | Recall | F1 | AUC-ROC | N |
|--------|----------|-----------|--------|----|---------|----|
| NLI 단독 (klue/roberta-large) | 0.500 | 0.000 | 0.000 | 0.000 | 0.500 | 300 |
| NER 단독 (우리 모델) | 0.580 | 0.688 | 0.293 | 0.411 | 0.571 | 300 |
| 3계층 앙상블 (우리 시스템) | 0.487 | 0.389 | 0.047 | 0.083 | 0.572 | 300 |

*binary classification: is_hallucination (positive=환각)*
