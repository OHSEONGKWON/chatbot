import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from modules.ner_checker import NERFactChecker

TARGET_LABELS = ["LAW", "PENALTY", "AMOUNT", "DATE", "ORG", "CRIME"]


def find_bio_files(data_dirs: List[str]) -> List[str]:
    files: List[str] = []
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            continue
        for root, _, names in os.walk(data_dir):
            for name in sorted(names):
                if name.endswith("_bio.jsonl"):
                    files.append(os.path.join(root, name))
    if not files:
        raise FileNotFoundError(f"No *_bio.jsonl files in {data_dirs}")
    return files


def load_file_samples(path: str) -> List[Tuple[str, str, List[str]]]:
    rows: List[Tuple[str, str, List[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            chunk_id = str(obj.get("chunk_id", "")).strip()
            tokens = obj.get("tokens")
            tags = obj.get("ner_tags")
            if not chunk_id or not isinstance(tokens, list) or not isinstance(tags, list):
                continue
            if len(tokens) != len(tags) or not tokens:
                continue
            rows.append((chunk_id, "".join(tokens), tags))
    return rows


def load_samples_parallel(files: List[str], max_samples: int, file_workers: int) -> List[Tuple[str, str, List[str]]]:
    all_rows: List[Tuple[str, str, List[str]]] = []
    workers = max(1, min(file_workers, len(files)))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(load_file_samples, p): p for p in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading files"):
            rows = fut.result()
            all_rows.extend(rows)
            if max_samples > 0 and len(all_rows) >= max_samples:
                all_rows = all_rows[:max_samples]
                break

    if max_samples > 0:
        return all_rows[:max_samples]
    return all_rows


def gold_spans_from_bio(text: str, tags: List[str]) -> List[Tuple[str, int, int, str]]:
    spans: List[Tuple[str, int, int, str]] = []
    i = 0
    n = min(len(text), len(tags))

    while i < n:
        tag = tags[i]
        if not isinstance(tag, str) or tag == "O":
            i += 1
            continue

        if tag.startswith("B-"):
            label = tag[2:]
            start = i
            i += 1
            while i < n and isinstance(tags[i], str) and tags[i] == f"I-{label}":
                i += 1
            end = i
            if label in TARGET_LABELS and start < end:
                spans.append((label, start, end, text[start:end]))
            continue

        if tag.startswith("I-"):
            label = tag[2:]
            start = i
            i += 1
            while i < n and isinstance(tags[i], str) and tags[i] == f"I-{label}":
                i += 1
            end = i
            if label in TARGET_LABELS and start < end:
                spans.append((label, start, end, text[start:end]))
            continue

        i += 1

    return spans


def pred_spans(checker: NERFactChecker, text: str) -> List[Tuple[str, int, int, str]]:
    spans: List[Tuple[str, int, int, str]] = []
    for ent in checker.extract_entities(text):
        label = ent.get("entity_group")
        start = ent.get("start")
        end = ent.get("end")
        word = ent.get("word")
        if label not in TARGET_LABELS:
            continue
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            continue
        surface = word if isinstance(word, str) and word else text[start:end]
        spans.append((label, start, end, surface))
    return spans


def compute_scores(counts: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    report: Dict[str, Dict[str, float]] = {}

    total_tp = total_fp = total_fn = total_support = 0
    for label in TARGET_LABELS:
        tp = counts[label]["tp"]
        fp = counts[label]["fp"]
        fn = counts[label]["fn"]
        support = counts[label]["support"]

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_support += support

    micro_p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0

    report["micro"] = {
        "precision": micro_p,
        "recall": micro_r,
        "f1": micro_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "support": total_support,
    }

    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Entity-level precision/recall/F1 report for legal NER")
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=["data/hallucination_data", "data/real_bio_data"],
        help="Directories containing *_bio.jsonl files",
    )
    parser.add_argument("--model-path", default=None, help="Optional model path. If omitted, checker auto-resolves.")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--file-workers", type=int, default=4)
    parser.add_argument("--output-json", default="outputs/legal-ner-lawsguard-v1-gpu/entity_report.json")
    return parser.parse_args()


def main():
    args = parse_args()

    files = find_bio_files(args.data_dirs)
    samples = load_samples_parallel(files, args.max_samples, args.file_workers)
    if not samples:
        raise ValueError("No valid samples loaded")

    checker = NERFactChecker(model_path=args.model_path)

    counts = {
        label: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for label in TARGET_LABELS
    }

    for _, text, tags in tqdm(samples, desc="Evaluating samples"):
        gold = set(gold_spans_from_bio(text, tags))
        pred = set(pred_spans(checker, text))

        for label in TARGET_LABELS:
            gold_l = {x for x in gold if x[0] == label}
            pred_l = {x for x in pred if x[0] == label}
            tp = len(gold_l & pred_l)
            fp = len(pred_l - gold_l)
            fn = len(gold_l - pred_l)

            counts[label]["tp"] += tp
            counts[label]["fp"] += fp
            counts[label]["fn"] += fn
            counts[label]["support"] += len(gold_l)

    report = compute_scores(counts)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": args.model_path or os.getenv("LAWSGUARD_NER_MODEL") or "auto",
        "num_samples": len(samples),
        "data_dirs": args.data_dirs,
        "labels": report,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n=== Entity-level Report ===")
    for label in TARGET_LABELS + ["micro"]:
        row = report[label]
        print(
            f"{label:8} P={row['precision']:.4f} R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} TP={row['tp']} FP={row['fp']} FN={row['fn']}"
        )

    print(f"\n[Done] Saved entity report to {args.output_json}")


if __name__ == "__main__":
    main()
