import pytest

from src.modules.ner_checker import NERFactChecker


def test_crime_without_penalty_generates_log_warning():
    checker = NERFactChecker()
    answer = "피고인은 강제추행을 저질렀다."
    found_entities = [
        {"entity_group": "CRIME", "label": "CRIME", "word": "강제추행", "start": 6, "end": 10}
    ]

    hallucinations = checker._crime_penalty_fallback_hallucinations(found_entities, answer)
    assert isinstance(hallucinations, list)
    assert len(hallucinations) == 1
    h = hallucinations[0]
    assert h["label"] == "CRIME"
    assert h["reason_code"] == "sparse_rag_crime_unverified"
    assert h["action"] == "log_only"


def test_crime_with_penalty_in_same_sentence_no_warning():
    checker = NERFactChecker()
    answer = "피고인은 강제추행을 저질렀고 1년의 징역을 선고받았다."
    found_entities = [
        {"entity_group": "CRIME", "label": "CRIME", "word": "강제추행", "start": 6, "end": 10},
        {"entity_group": "PENALTY", "label": "PENALTY", "word": "1년의 징역", "start": 18, "end": 25},
    ]

    hallucinations = checker._crime_penalty_fallback_hallucinations(found_entities, answer)
    assert isinstance(hallucinations, list)
    assert len(hallucinations) == 0
