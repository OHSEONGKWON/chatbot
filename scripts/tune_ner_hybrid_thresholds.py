import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.ner_checker import NERFactChecker


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_div(n, d):
    return n / d if d else 0.0


def evaluate(checker: NERFactChecker, rows, fuzzy_threshold: float, semantic_threshold: float):
    checker.fuzzy_threshold = fuzzy_threshold
    checker.semantic_threshold = semantic_threshold

    tp = fp = fn = tn = 0
    details = []
    for row in rows:
        result = checker.debug_match_entity(
            label=row["label"],
            word=row["word"],
            candidates=row["candidates"],
        )
        pred = bool(result.get("is_supported", False))
        gold = bool(row["expected_supported"])

        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1

        details.append({
            "label": row["label"],
            "word": row["word"],
            "pred": pred,
            "gold": gold,
            "method": result.get("method"),
            "score": round(float(result.get("score", 0.0)), 4),
            "candidate": result.get("candidate"),
        })

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "fuzzy_threshold": fuzzy_threshold,
        "semantic_threshold": semantic_threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "details": details,
    }


def main():
    data_path = REPO_ROOT / "data" / "mock_data" / "ner_hybrid_eval_pairs.jsonl"
    rows = load_jsonl(data_path)

    checker = NERFactChecker()

    fuzzy_grid = [0.85, 0.90, 0.93]
    semantic_grid = [0.78, 0.82, 0.86]

    best = None
    all_results = []
    for fuzzy in fuzzy_grid:
        for semantic in semantic_grid:
            result = evaluate(checker, rows, fuzzy, semantic)
            all_results.append(result)
            if best is None or result["f1"] > best["f1"]:
                best = result

    print("=== Hybrid Matcher Threshold Tuning ===")
    for r in all_results:
        print(
            f"fuzzy={r['fuzzy_threshold']:.2f}, semantic={r['semantic_threshold']:.2f} "
            f"| precision={r['precision']:.3f}, recall={r['recall']:.3f}, f1={r['f1']:.3f} "
            f"| tp={r['tp']} fp={r['fp']} fn={r['fn']} tn={r['tn']}"
        )

    print("\n=== Best ===")
    print(
        f"fuzzy={best['fuzzy_threshold']:.2f}, semantic={best['semantic_threshold']:.2f}, "
        f"precision={best['precision']:.3f}, recall={best['recall']:.3f}, f1={best['f1']:.3f}"
    )

    print("\n=== Error Cases (best setting) ===")
    for d in best["details"]:
        if d["pred"] != d["gold"]:
            print(
                f"label={d['label']}, word={d['word']}, pred={d['pred']}, gold={d['gold']}, "
                f"method={d['method']}, score={d['score']}, candidate={d['candidate']}"
            )


if __name__ == "__main__":
    main()
