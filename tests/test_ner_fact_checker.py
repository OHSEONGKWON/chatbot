from src.modules.ner_checker import NERFactChecker


def test_law_article_mismatch_is_flagged_as_high_risk():
    checker = NERFactChecker()
    rag_docs = [
        {
            "text": "근로기준법 제43조는 임금 지급 원칙을 정한다.",
            "metadata": {"law_name": "근로기준법", "article_id": "제43조"},
        }
    ]
    entities = [{"entity_group": "LAW", "word": "근로기준법 제37조", "start": 0, "end": 10}]

    mismatches = checker.find_hallucinations("근로기준법 제37조", rag_docs, found_entities=entities)

    assert len(mismatches) == 1
    assert mismatches[0]["reason_code"] == "same_law_article_mismatch"
    assert mismatches[0]["risk_level"] == "high"
    assert mismatches[0]["confidence"] >= 0.95
    assert mismatches[0]["correct_word"] == "근로기준법 제43조"


def test_same_law_article_is_supported():
    checker = NERFactChecker()

    result = checker.debug_match_entity(
        label="LAW",
        word="근로기준법 제43조",
        candidates=["근로기준법 제43조"],
    )

    assert result["is_supported"] is True
    assert result["method"] == "exact"


def test_same_law_name_without_article_does_not_support_article_claim():
    checker = NERFactChecker()

    result = checker.debug_match_entity(
        label="LAW",
        word="근로기준법 제43조",
        candidates=["근로기준법"],
    )

    assert result["is_supported"] is False
    assert result["method"] == "law_name_only"


def test_org_entities_are_not_used_for_hallucination_correction():
    checker = NERFactChecker()
    rag_docs = [{"text": "대법원 판례", "metadata": {"organization": "대법원"}}]
    entities = [{"entity_group": "ORG", "word": "대한법률구조공단", "start": 0, "end": 8}]

    mismatches = checker.find_hallucinations("대한법률구조공단", rag_docs, found_entities=entities)

    assert mismatches == []
