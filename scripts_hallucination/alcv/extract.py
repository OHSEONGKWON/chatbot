"""답변에서 법률 개체 추출 (기존 NER 모델 사용; LLM 미사용).

NERFactChecker(legal-ner-v3 + 규칙 폴백)로 6종 개체를 추출한다:
LAW / PENALTY / AMOUNT / DATE / ORG / CRIME.
모델이 없으면 규칙 기반으로 폴백한다(결과는 동일 형식).
"""
from src.modules.ner_checker import NERFactChecker


class Extractor:
    def __init__(self):
        self.ner = NERFactChecker()

    async def warmup(self):
        await self.ner.warmup()

    def extract(self, answer: str) -> list:
        """답변 → 개체 리스트 [{entity_group, word, start, end, score}, ...]."""
        if not answer or not answer.strip():
            return []
        try:
            return self.ner.extract_entities(answer)
        except Exception as e:
            print(f"[WARNING] 개체 추출 실패: {e}")
            return []
