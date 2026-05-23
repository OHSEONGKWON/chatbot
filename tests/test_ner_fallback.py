import pytest

from src.modules.ner_checker import NERFactChecker


def test_penalty_without_rag_support_is_flagged():
    """RAG에 PENALTY 후보가 없을 때 형량 엔티티가 log_only로 기록된다."""
    checker = NERFactChecker()
    found_entities = [
        {"entity_group": "PENALTY", "label": "PENALTY", "word": "5년 이하의 징역", "start": 0, "end": 8}
    ]
    candidates_empty = {label: [] for label in checker.target_labels}

    hallucinations = checker._validate_penalty_claims(found_entities, candidates_empty)
    assert isinstance(hallucinations, list)
    assert len(hallucinations) == 1
    h = hallucinations[0]
    assert h["label"] == "PENALTY"
    assert h["reason_code"] == "penalty_unverified_by_rag"
    assert h["action"] == "log_only"
    assert h["confidence"] == 0.70


def test_penalty_with_rag_support_not_flagged():
    """RAG에 PENALTY 후보가 있으면 _validate_penalty_claims는 빈 리스트를 반환한다."""
    checker = NERFactChecker()
    found_entities = [
        {"entity_group": "PENALTY", "label": "PENALTY", "word": "5년 이하의 징역", "start": 0, "end": 8}
    ]
    candidates_with_penalty = {label: [] for label in checker.target_labels}
    candidates_with_penalty["PENALTY"] = ["10년 이하의 징역"]

    hallucinations = checker._validate_penalty_claims(found_entities, candidates_with_penalty)
    assert hallucinations == []


def test_wrong_law_article_for_crime_is_flagged():
    """범죄명에 대해 잘못된 조문 번호가 인용되면 Layer 3에서 탐지된다."""
    checker = NERFactChecker()
    # 강제추행인데 강간 조항(제297조)을 인용한 케이스
    answer = "강제추행은 형법 제297조에 해당합니다."
    entities = checker._extract_entities_with_rules(answer)
    hallucinations = checker._validate_law_articles(answer, entities)

    assert any(h["reason_code"] == "wrong_article_for_crime" for h in hallucinations), (
        f"wrong_article_for_crime 탐지 실패. hallucinations={hallucinations}"
    )
    flagged = next(h for h in hallucinations if h["reason_code"] == "wrong_article_for_crime")
    assert flagged["action"] == "replace"
    assert "제298조" in flagged["correct_word"]


def test_correct_law_article_for_crime_not_flagged():
    """올바른 법조항을 인용하면 Layer 3에서 플래그되지 않는다."""
    checker = NERFactChecker()
    answer = "강제추행은 형법 제298조에 해당합니다."
    entities = checker._extract_entities_with_rules(answer)
    hallucinations = checker._validate_law_articles(answer, entities)

    assert not any(h["reason_code"] == "wrong_article_for_crime" for h in hallucinations)


def test_claim_consistency_no_rag_returns_empty():
    """RAG 문서가 없으면 Layer 2.5는 빈 리스트를 반환한다."""
    checker = NERFactChecker()
    found_entities = [
        {"entity_group": "LAW", "label": "LAW", "word": "근로기준법 제23조", "start": 0, "end": 9}
    ]
    result = checker._validate_claim_consistency(
        "근로기준법 제23조에 의해 3개월 근무하면 부당해고 성립합니다.",
        found_entities,
        [],  # RAG 없음
    )
    assert result == []


def test_claim_consistency_skipped_when_semantic_disabled():
    """enable_semantic_match=False이면 Layer 2.5를 건너뛴다."""
    checker = NERFactChecker()
    checker.enable_semantic_match = False
    found_entities = [
        {"entity_group": "PENALTY", "label": "PENALTY", "word": "10년 이하의 징역", "start": 0, "end": 8}
    ]
    rag_docs = [{"text": "형법 제297조 강간 - 3년 이상의 유기징역에 처한다."}]
    result = checker._validate_claim_consistency(
        "강간죄의 형량은 10년 이하의 징역입니다.", found_entities, rag_docs
    )
    assert result == []
