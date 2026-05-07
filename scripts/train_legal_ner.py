"""Train or continue-train the LawsGuard legal NER model.

The script uses the repository's BIO JSONL files directly, so it does not
depend on HuggingFace datasets or seqeval. It reports both token-level and
entity-span F1 because legal answer verification needs exact entity spans,
not just high token accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIO_FILES = [
    REPO_ROOT / "data" / "real_bio_data" / "New_Dataset" / "rag_law_chunks_bio.jsonl",
    REPO_ROOT / "data" / "real_bio_data" / "New_Dataset" / "rag_manual_chunks_bio.jsonl",
    REPO_ROOT / "data" / "real_bio_data" / "New_Dataset" / "rag_case_chunks_bio.jsonl",
    REPO_ROOT / "data" / "hallucination_data" / "synthetic_hallu_bio.jsonl",
    REPO_ROOT / "data" / "hallucination_data" / "law_hallu_bio.jsonl",
]

LABELS = [
    "O",
    "B-LAW",
    "I-LAW",
    "B-PENALTY",
    "I-PENALTY",
    "B-AMOUNT",
    "I-AMOUNT",
    "B-DATE",
    "I-DATE",
    "B-ORG",
    "I-ORG",
    "B-CRIME",
    "I-CRIME",
]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def load_bio_rows(paths: Iterable[Path], limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("tokens") and row.get("ner_tags"):
                    rows.append(row)
                    if limit and len(rows) >= limit:
                        return rows
    return rows


def bio_to_char_labels(tokens: list[str], tags: list[str]) -> tuple[str, list[str]]:
    text = "".join(tokens)
    char_labels = ["O"] * len(text)
    cursor = 0
    for token, tag in zip(tokens, tags):
        end = cursor + len(token)
        if tag != "O":
            for idx in range(cursor, end):
                char_labels[idx] = tag if idx == cursor else tag.replace("B-", "I-")
        cursor = end
    return text, char_labels


class LegalNERDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.features = []
        for row in rows:
            text, char_labels = bio_to_char_labels(row["tokens"], row["ner_tags"])
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_offsets_mapping=True,
            )
            labels = []
            for start, end in encoded.pop("offset_mapping"):
                if start == end:
                    labels.append(-100)
                    continue
                span_tags = [tag for tag in char_labels[start:end] if tag != "O"]
                if not span_tags:
                    labels.append(LABEL2ID["O"])
                    continue
                first = span_tags[0]
                labels.append(LABEL2ID.get(first, LABEL2ID["O"]))
            encoded["labels"] = labels
            self.features.append({key: torch.tensor(value) for key, value in encoded.items()})

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        return self.features[idx]


def ids_to_entities(label_ids: list[int]) -> set[tuple[str, int, int]]:
    entities = set()
    start = None
    current_label = None
    for idx, label_id in enumerate(label_ids + [LABEL2ID["O"]]):
        label = ID2LABEL.get(int(label_id), "O") if label_id != -100 else "O"
        if label == "O":
            if current_label is not None:
                entities.add((current_label, start, idx))
            current_label = None
            start = None
            continue
        prefix, _, entity_label = label.partition("-")
        if prefix == "B" or current_label != entity_label:
            if current_label is not None:
                entities.add((current_label, start, idx))
            current_label = entity_label
            start = idx
    return entities


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    tp = fp = fn = 0
    entity_tp = entity_fp = entity_fn = 0

    for pred_row, label_row in zip(preds, labels):
        valid_preds = []
        valid_labels = []
        for pred_id, label_id in zip(pred_row, label_row):
            if label_id == -100:
                continue
            valid_preds.append(int(pred_id))
            valid_labels.append(int(label_id))
            pred_is_entity = ID2LABEL[int(pred_id)] != "O"
            gold_is_entity = ID2LABEL[int(label_id)] != "O"
            if pred_is_entity and gold_is_entity:
                tp += 1
            elif pred_is_entity and not gold_is_entity:
                fp += 1
            elif gold_is_entity and not pred_is_entity:
                fn += 1

        pred_entities = ids_to_entities(valid_preds)
        gold_entities = ids_to_entities(valid_labels)
        entity_tp += len(pred_entities & gold_entities)
        entity_fp += len(pred_entities - gold_entities)
        entity_fn += len(gold_entities - pred_entities)

    token_precision = tp / (tp + fp) if tp + fp else 0.0
    token_recall = tp / (tp + fn) if tp + fn else 0.0
    token_f1 = 2 * token_precision * token_recall / (token_precision + token_recall) if token_precision + token_recall else 0.0
    entity_precision = entity_tp / (entity_tp + entity_fp) if entity_tp + entity_fp else 0.0
    entity_recall = entity_tp / (entity_tp + entity_fn) if entity_tp + entity_fn else 0.0
    entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall) if entity_precision + entity_recall else 0.0
    return {
        "token_precision": token_precision,
        "token_recall": token_recall,
        "token_f1": token_f1,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
    }


def split_rows(rows: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LawsGuard legal NER model.")
    parser.add_argument("--base-model", default=str(REPO_ROOT / "outputs" / "legal-ner-lawsguard-v2-30k"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "legal-ner-lawsguard-v3"))
    parser.add_argument("--data", nargs="*", default=[str(path) for path in DEFAULT_BIO_FILES])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=510)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [Path(path) for path in args.data]
    rows = load_bio_rows(paths, limit=args.limit)
    if not rows:
        print("No BIO rows found.", file=sys.stderr)
        return 1

    train_rows, val_rows, test_rows = split_rows(rows, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        local_files_only=True,
    )

    train_dataset = LegalNERDataset(train_rows, tokenizer, args.max_length)
    val_dataset = LegalNERDataset(val_rows, tokenizer, args.max_length)
    test_dataset = LegalNERDataset(test_rows, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = {
        "rows": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "val": trainer.evaluate(val_dataset),
        "test": trainer.evaluate(test_dataset),
    }
    metrics_path = Path(args.output_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
