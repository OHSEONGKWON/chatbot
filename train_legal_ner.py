import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

LABELS = [
    "O",
    "B-LAW", "I-LAW",
    "B-PENALTY", "I-PENALTY",
    "B-AMOUNT", "I-AMOUNT",
    "B-DATE", "I-DATE",
    "B-ORG", "I-ORG",
    "B-CRIME", "I-CRIME",
]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


@dataclass
class Sample:
    chunk_id: str
    text: str
    char_tags: List[str]


class LegalNERDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[List[int]]], labels: List[List[int]]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def parse_args():
    parser = argparse.ArgumentParser(description="Legal NER fine-tuning")
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=["data/hallucination_data", "data/real_bio_data"],
        help="One or more directories containing *_bio.jsonl files",
    )
    parser.add_argument("--model-name", default="klue/roberta-base")
    parser.add_argument("--output-dir", default="outputs/legal-ner-klue")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--drop-no-entity", action="store_true")
    parser.add_argument("--eval-strategy", choices=["no", "epoch"], default="epoch")
    parser.add_argument("--save-strategy", choices=["no", "epoch"], default="epoch")
    parser.add_argument("--dataloader-num-workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument(
        "--group-by-length",
        action="store_true",
        help="Deprecated in current transformers version; kept for backward compatibility.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_files(data_dirs: List[str]) -> List[str]:
    files = []
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            continue
        for root, _, filenames in os.walk(data_dir):
            for name in sorted(filenames):
                if name.endswith("_bio.jsonl"):
                    files.append(os.path.join(root, name))
    if not files:
        raise FileNotFoundError(f"No *_bio.jsonl files in: {data_dirs}")
    return files


def load_samples(files: List[str], drop_no_entity: bool) -> List[Sample]:
    samples: List[Sample] = []
    seen = set()

    for path in files:
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
                if len(tokens) == 0 or len(tokens) != len(tags):
                    continue
                if any(tag not in LABEL_TO_ID for tag in tags):
                    continue
                if chunk_id in seen:
                    continue
                if drop_no_entity and all(tag == "O" for tag in tags):
                    continue

                seen.add(chunk_id)
                samples.append(Sample(chunk_id=chunk_id, text="".join(tokens), char_tags=tags))

    if len(samples) < 50:
        raise ValueError(f"Too few samples: {len(samples)}")

    return samples


def split_data(samples: List[Sample], train_ratio: float, val_ratio: float, seed: int):
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")
    random.Random(seed).shuffle(samples)
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]
    if not train or not val or not test:
        raise ValueError("Split produced empty set. Adjust ratios.")
    return train, val, test


def char_to_token_labels(offsets, char_tags):
    labels = []
    for start, end in offsets:
        if start == end:
            labels.append(-100)
            continue
        span = char_tags[start:end]
        non_o = [x for x in span if x != "O"]
        if not non_o:
            labels.append(LABEL_TO_ID["O"])
            continue
        tag = non_o[0]
        if tag.startswith("I-"):
            tag = "B-" + tag[2:]
        labels.append(LABEL_TO_ID.get(tag, LABEL_TO_ID["O"]))
    return labels


def encode(tokenizer, samples: List[Sample], max_length: int):
    texts = [s.text for s in samples]
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
    )

    labels = []
    for i, offsets in enumerate(enc["offset_mapping"]):
        labels.append(char_to_token_labels(offsets, samples[i].char_tags))

    enc.pop("offset_mapping")
    return enc, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    tp = fp = fn = 0
    correct = total = 0

    for p_row, l_row in zip(preds, labels):
        for p, l in zip(p_row, l_row):
            if l == -100:
                continue
            total += 1
            if p == l:
                correct += 1

            p_name = ID_TO_LABEL[int(p)]
            l_name = ID_TO_LABEL[int(l)]

            p_pos = p_name != "O"
            l_pos = l_name != "O"

            if p_pos and l_pos and p_name == l_name:
                tp += 1
            elif p_pos and (not l_pos or p_name != l_name):
                fp += 1
            elif (not p_pos) and l_pos:
                fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = correct / total if total else 0.0

    return {
        "token_accuracy": acc,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
        print(f"[Perf] torch_num_threads={args.torch_num_threads}")
    if args.group_by_length:
        print("[Perf] --group-by-length is ignored on this transformers version.")

    files = find_files(args.data_dirs)
    print("[Data] Files:")
    for f in files:
        print(" -", f)

    samples = load_samples(files, args.drop_no_entity)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    train_samples, val_samples, test_samples = split_data(samples, args.train_ratio, args.val_ratio, args.seed)
    print(f"[Data] train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    train_enc, train_labels = encode(tokenizer, train_samples, args.max_length)
    val_enc, val_labels = encode(tokenizer, val_samples, args.max_length)
    test_enc, test_labels = encode(tokenizer, test_samples, args.max_length)

    train_ds = LegalNERDataset(train_enc, train_labels)
    val_ds = LegalNERDataset(val_enc, val_labels)
    test_ds = LegalNERDataset(test_enc, test_labels)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    use_best_model = args.eval_strategy != "no" and args.save_strategy != "no"
    training_kwargs = {
        "output_dir": args.output_dir,
        "eval_strategy": args.eval_strategy,
        "save_strategy": args.save_strategy,
        "logging_steps": 50,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "load_best_model_at_end": use_best_model,
        "report_to": "none",
        "save_total_limit": 2,
        "fp16": torch.cuda.is_available(),
        "seed": args.seed,
        "dataloader_num_workers": args.dataloader_num_workers,
    }
    if use_best_model:
        training_kwargs["metric_for_best_model"] = "token_f1"
        training_kwargs["greater_is_better"] = True

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    test_metrics = trainer.evaluate(eval_dataset=test_ds)

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, ensure_ascii=False, indent=2)

    print("[Done] Saved model to", args.output_dir)


if __name__ == "__main__":
    main()
