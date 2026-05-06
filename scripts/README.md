# scripts/ - 운영 유틸리티 및 튜닝 스크립트

## 파일 설명

- **tune_ner_hybrid_thresholds.py**: NER 하이브리드 매칭 임계값 튜닝
  - 평가 샘플셋으로 precision/recall/f1 측정
  - 최적 fuzzy_threshold, semantic_threshold 찾기

## 실행 방법

```bash
# NER 하이브리드 임계값 튜닝
python tune_ner_hybrid_thresholds.py
```

출력:
- 각 임계값 조합별 성능 (precision, recall, f1)
- 최고 성능 설정값
- 오류 케이스 분석

## 결과 적용

튜닝 결과에서 최적 설정값을 발견하면:
- `src/modules/ner_checker.py` 의 다음 값을 업데이트:
  - `self.fuzzy_threshold`
  - `self.semantic_threshold`
