"""Evaluate NER-based hallucination detection on answer/RAG pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.ner_checker import NERFactChecker


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def simplify(mismatches: list[dict]) -> set[tuple[str, str, str]]:
    rows = set()
    for item in mismatches:
        rows.add(
            (
                str(item.get("label") or ""),
                str(item.get("wrong_word") or ""),
                str(item.get("correct_word") or ""),
            )
        )
    return rows


def expected_set(case: dict) -> set[tuple[str, str, str]]:
    return {
        (
            str(item.get("label") or ""),
            str(item.get("wrong_word") or ""),
            str(item.get("correct_word") or ""),
        )
        for item in case.get("expected_mismatches", [])
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate NER fact-checking behavior.")
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "mock_data" / "ner_factcheck_eval.jsonl"))
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    checker = NERFactChecker(model_path=args.model) if args.model else NERFactChecker()
    cases = load_cases(Path(args.data))
    tp = fp = fn = 0
    failures = []

    for case in cases:
        result = checker.check_and_correct_sync(case["answer"], case.get("rag_docs", []))
        predicted = simplify(result.mismatched_entities)
        expected = expected_set(case)
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        if predicted != expected:
            failures.append(
                {
                    "id": case.get("id"),
                    "expected": sorted(expected),
                    "predicted": sorted(predicted),
                    "found_entities": result.found_entities,
                    "mismatches": result.mismatched_entities,
                }
            )

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "cases": len(cases),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "failures": failures,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
