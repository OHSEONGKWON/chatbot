"""
NER 모델 재학습 스크립트

베이스 모델: klue/roberta-large
학습 데이터: 기존 BIO 태깅 데이터 전체 (~90k)
개선 사항:
  - 30k → 90k 데이터 확장
  - roberta-base → roberta-large
  - epoch 2 → 4
  - LR 3e-5 → 2e-5
  - gradient accumulation으로 유효 배치 크기 확대

사용법:
  python scripts/train_ner.py
  python scripts/train_ner.py --output-dir outputs/ner-v3 --epochs 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Windows UTF-8 출력 설정
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── 레이블 정의 ──────────────────────────────────────────
LABELS = [
    "O",
    "B-LAW", "I-LAW",
    "B-ORG", "I-ORG",
    "B-DATE", "I-DATE",
    "B-AMOUNT", "I-AMOUNT",
    "B-CRIME", "I-CRIME",
    "B-PENALTY", "I-PENALTY",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# 학습 데이터 경로
BIO_FILES = [
    REPO_ROOT / "data" / "hallucination_data" / "synthetic_hallu_bio.jsonl",
    REPO_ROOT / "data" / "hallucination_data" / "law_hallu_bio.jsonl",
    REPO_ROOT / "data" / "hallucination_data" / "case_hallu_bio.jsonl",
    REPO_ROOT / "data" / "hallucination_data" / "manual_hallu_bio.jsonl",
    REPO_ROOT / "data" / "real_bio_data" / "New_Dataset" / "rag_law_chunks_bio.jsonl",
    REPO_ROOT / "data" / "real_bio_data" / "New_Dataset" / "rag_case_chunks_bio.jsonl",
    REPO_ROOT / "data" / "real_bio_data" / "New_Dataset" / "rag_manual_chunks_bio.jsonl",
]

BASE_MODEL = "klue/roberta-large"


# ── 데이터 로딩 ──────────────────────────────────────────

def load_all_bio_data(seed: int = 42) -> tuple[list, list, list]:
    """모든 BIO 파일을 로드하고 train/val/test로 분할 (8:1:1)."""
    records = []
    for path in BIO_FILES:
        if not path.exists():
            print(f"[WARN] 파일 없음: {path.name}")
            continue
        with path.open("r", encoding="utf-8") as f:
            file_recs = [json.loads(l) for l in f if l.strip()]
        records.extend(file_recs)
        print(f"  {path.name}: {len(file_recs)}개 로드")

    print(f"\n전체 데이터: {len(records)}개")

    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    n_val = n_test = int(n * 0.1)
    n_train = n - n_val - n_test

    train = records[:n_train]
    val   = records[n_train:n_train + n_val]
    test  = records[n_train + n_val:]

    print(f"train: {len(train)} / val: {len(val)} / test: {len(test)}")
    return train, val, test


# ── 토크나이저 정렬 ───────────────────────────────────────

def tokenize_and_align(examples: list[dict], tokenizer, max_length: int = 512) -> dict:
    """
    문자 단위 BIO 태그를 서브워드 토큰에 정렬합니다.
    첫 번째 서브워드에 레이블 부여, 나머지는 -100 (무시).
    """
    all_input_ids, all_attention_mask, all_labels = [], [], []

    for rec in examples:
        char_tokens = rec["tokens"]        # 문자 리스트
        char_tags   = rec["ner_tags"]      # 문자별 BIO 태그

        # 문자 리스트를 단어(어절) 단위로 그룹핑
        # → 공백을 기준으로 나누면 서브워드 정렬이 더 자연스러움
        text = "".join(char_tokens)
        encoding = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        offset_mapping = encoding["offset_mapping"]   # (start, end) per subword
        input_ids      = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # 각 서브워드에 대응하는 레이블 결정
        token_labels = []
        for (start, end) in offset_mapping:
            if start == end:
                # [CLS], [SEP], padding → -100
                token_labels.append(-100)
            else:
                # 서브워드가 커버하는 문자 범위의 첫 문자 태그 사용
                # (문자 인덱스가 max_length를 넘을 수 있으므로 clamp)
                char_idx = min(start, len(char_tags) - 1)
                tag = char_tags[char_idx]
                token_labels.append(LABEL2ID.get(tag, LABEL2ID["O"]))

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(token_labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


# ── Dataset ───────────────────────────────────────────────

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict):
        self.input_ids      = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels         = encodings["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids":      torch.tensor(self.input_ids[idx],      dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels":         torch.tensor(self.labels[idx],         dtype=torch.long),
        }


# ── 메트릭 ────────────────────────────────────────────────

def compute_span_f1(pred_labels: list, gold_labels: list) -> dict:
    """Span-level Precision/Recall/F1 계산."""
    def extract_spans(seq: list[str]) -> set[tuple]:
        spans = set()
        i = 0
        while i < len(seq):
            if seq[i].startswith("B-"):
                label = seq[i][2:]
                j = i + 1
                while j < len(seq) and seq[j] == f"I-{label}":
                    j += 1
                spans.add((i, j, label))
                i = j
            else:
                i += 1
        return spans

    tp = fp = fn = 0
    per_type: dict[str, dict] = {}

    for gold_seq, pred_seq in zip(gold_labels, pred_labels):
        gold_spans = extract_spans(gold_seq)
        pred_spans = extract_spans(pred_seq)

        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)

        for s in gold_spans & pred_spans:
            per_type.setdefault(s[2], {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
        for s in pred_spans - gold_spans:
            per_type.setdefault(s[2], {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
        for s in gold_spans - pred_spans:
            per_type.setdefault(s[2], {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    p, r, f = prf(tp, fp, fn)
    result = {"precision": p, "recall": r, "f1": f}

    for etype, c in per_type.items():
        ep, er, ef = prf(c["tp"], c["fp"], c["fn"])
        result[f"f1_{etype}"] = ef

    entity_types = ["LAW", "ORG", "DATE", "AMOUNT", "CRIME", "PENALTY"]
    macro = sum(result.get(f"f1_{t}", 0.0) for t in entity_types) / len(entity_types)
    result["macro_f1"] = round(macro, 4)

    return result


def make_compute_metrics(id2label: dict):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels, pred_labels = [], []
        for pred_row, label_row in zip(predictions, labels):
            true_seq, pred_seq = [], []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                true_seq.append(id2label[l])
                pred_seq.append(id2label.get(p, "O"))
            true_labels.append(true_seq)
            pred_labels.append(pred_seq)

        metrics = compute_span_f1(pred_labels, true_labels)
        return {
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "f1":        metrics["f1"],
            "macro_f1":  metrics["macro_f1"],
            **{k: v for k, v in metrics.items() if k.startswith("f1_")},
        }

    return compute_metrics


# ── 메인 ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NER 모델 재학습")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "legal-ner-v3"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)   # 유효 배치 = 4×4 = 16
    parser.add_argument("--max-length", type=int, default=256)  # 메모리 절약
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=0,    help="디버그용 최대 학습 샘플 수 (0=전체)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification,
    )

    print(f"\n=== NER 재학습 시작 ===")
    print(f"베이스 모델  : {args.base_model}")
    print(f"출력 경로    : {args.output_dir}")
    print(f"GPU          : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Epoch        : {args.epochs}")
    print(f"LR           : {args.lr}")
    print(f"배치         : {args.batch_size} × grad_accum {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"max_length   : {args.max_length}")

    # 1. 데이터 로드
    print("\n[1] 데이터 로딩...")
    train_raw, val_raw, test_raw = load_all_bio_data(args.seed)

    if args.max_train > 0:
        train_raw = train_raw[:args.max_train]
        print(f"[DEBUG] 학습 데이터 {args.max_train}개로 제한")

    # 2. 토크나이저 로드
    print(f"\n[2] 토크나이저 로딩: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 3. 인코딩
    print("\n[3] 토큰화 및 레이블 정렬 중...")
    BATCH = 1000  # 메모리 절약을 위한 배치 처리

    def encode_in_batches(raw_list: list, desc: str) -> dict:
        all_ids, all_masks, all_lbls = [], [], []
        for i in range(0, len(raw_list), BATCH):
            if i % 5000 == 0:
                print(f"  {desc}: {i}/{len(raw_list)}")
            batch = raw_list[i:i + BATCH]
            enc = tokenize_and_align(batch, tokenizer, args.max_length)
            all_ids.extend(enc["input_ids"])
            all_masks.extend(enc["attention_mask"])
            all_lbls.extend(enc["labels"])
        return {"input_ids": all_ids, "attention_mask": all_masks, "labels": all_lbls}

    train_enc = encode_in_batches(train_raw, "train")
    val_enc   = encode_in_batches(val_raw,   "val")
    test_enc  = encode_in_batches(test_raw,  "test")

    train_dataset = NERDataset(train_enc)
    val_dataset   = NERDataset(val_enc)
    test_dataset  = NERDataset(test_enc)

    print(f"  Dataset 크기 — train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    # 4. 모델 로드
    print(f"\n[4] 모델 로딩: {args.base_model}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # 5. 학습 설정
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # warmup_steps = 전체 스텝의 10%
    total_steps = (len(train_dataset) // (args.batch_size * args.grad_accum)) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=200,
        save_total_limit=2,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=0,   # Windows 호환
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,   # transformers 5.x API
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(ID2LABEL),
    )

    # 6. 학습
    print("\n[5] 학습 시작...")
    trainer.train()

    # 7. 테스트셋 최종 평가
    print("\n[6] 테스트셋 최종 평가...")
    test_result = trainer.evaluate(test_dataset)
    print("\n=== 테스트 결과 ===")
    for k, v in test_result.items():
        if "f1" in k or "precision" in k or "recall" in k:
            print(f"  {k}: {v:.4f}")

    # 8. 결과 저장
    metrics = {
        "val":  trainer.evaluate(val_dataset),
        "test": test_result,
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 최종 모델 저장
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n모델 저장 완료: {output_dir}")

    # 9. 기존 모델과 성능 비교
    print("\n=== 기존 모델 vs 새 모델 비교 ===")
    old_metrics_path = REPO_ROOT / "outputs" / "legal-ner-lawsguard-v2-30k" / "metrics.json"
    if old_metrics_path.exists():
        with old_metrics_path.open(encoding="utf-8") as f:
            old = json.load(f)
        old_f1 = old.get("test", {}).get("eval_token_f1", 0)
        new_f1 = test_result.get("eval_f1", 0)
        print(f"  기존 (v2, 30k): Token F1 = {old_f1:.4f}")
        print(f"  신규 (v3, 90k): Span F1  = {new_f1:.4f}")
        print(f"  (주의: 메트릭 기준이 다름 — 신규는 span-level, 기존은 token-level)")


if __name__ == "__main__":
    main()
