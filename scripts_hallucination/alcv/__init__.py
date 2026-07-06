"""ALCV — Atomic Legal-Claim Verification (deterministic version).

법률 도메인 LLM 답변의 환각을 LLM/NLI 없이 규칙·법령 DB·수치비교로 탐지한다.
판정에 LLM이 개입하지 않으므로 'LLM이 LLM을 검사한다'는 순환 문제에서 자유롭고,
모든 판정이 사람이 검증 가능(auditable)하다.

파이프라인:  답변 → extract(NER) → verify(규칙, 근거=RAG) → aggregate(판정)
  LAW            → 법령 DB 조문 존재 + 사안-법령 도메인 정합성 (하드)
  PENALTY/AMOUNT → RAG 근거 값과 수치/방향 정합성 (하드)
  ORG/DATE       → RAG 근거 등장 여부 (소프트)
"""
