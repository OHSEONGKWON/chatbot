# tests/ - 테스트 및 개발용 스크립트

## 파일 설명

- **manual_pipeline_test.py**: 수동으로 파이프라인을 테스트하는 스크립트
  - 사용자 질문 입력 → 전체 파이프라인 실행 → 결과 확인
  - 개발/디버깅 용도

- **test_long_answer.py**: 긴 답변에 대한 테스트
  - 답변 길이 및 구조 검증용

## 실행 방법

```bash
# manual_pipeline_test 실행
python manual_pipeline_test.py

# test_long_answer 실행
python test_long_answer.py
```

## 주의사항

- 이 폴더의 스크립트들은 개발/테스트 용도입니다
- 운영 환경에서는 실행하지 않습니다
- src/ 내 모듈 변경 시 여기서 즉시 테스트할 수 있습니다
