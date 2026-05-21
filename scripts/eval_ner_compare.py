"""
NER Span F1 비교 평가: 우리 모델 vs 베이스라인 3개

사용법:
  python scripts/eval_ner_compare.py
  python scripts/eval_ner_compare.py --models outputs/legal-ner-v3 outputs/baselines/...
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── 베이스라인 레이블 → 우리 스키마 매핑 ─────────────────────────────────────
# 각 모델이 사용하는 레이블을 우리 6개 엔티티 (LAW/ORG/DATE/AMOUNT/CRIME/PENALTY) 로 변환.
# 매핑되지 않는 레이블(PER, LOC 등)은 "O"로 처리.
LABEL_MAPPING: dict[str, dict[str, str]] = {
    # Babelscape/wikineural-multilingual-ner: B-ORG/I-ORG → ORG
    "Babelscape_wikineural-multilingual-ner": {
        "B-ORG": "B-ORG", "I-ORG": "I-ORG",
    },
    # Leo97/KoELECTRA-small-v3-modu-ner: OG→ORG, DT→DATE, QT→AMOUNT
    "Leo97_KoELECTRA-small-v3-modu-ner": {
        "B-OG": "B-ORG", "I-OG": "I-ORG",
        "B-DT": "B-DATE", "I-DT": "I-DATE",
        "B-QT": "B-AMOUNT", "I-QT": "I-AMOUNT",
    },
    # monologg/koelectra-base-v3-naver-ner: IOB 순서 반대 (ORG-B → B-ORG)
    "monologg_koelectra-base-v3-naver-ner": {
        "ORG-B": "B-ORG", "ORG-I": "I-ORG",
        "DAT-B": "B-DATE", "DAT-I": "I-DATE",
        "NUM-B": "B-AMOUNT", "NUM-I": "I-AMOUNT",
    },
}


def map_label(label: str, model_name: str) -> str:
    """베이스라인 레이블을 우리 스키마로 변환. 매핑 없으면 O 반환."""
    mapping = LABEL_MAPPING.get(model_name, {})
    if not mapping:
        return label  # 우리 모델은 매핑 불필요
    return mapping.get(label, "O")

BIO_FILES = [
    REPO_ROOT / "data" / "hallucination_data" / "synthetic_hallu_bio.jsonl",
    REPO_ROOT / "data" / "hallucination_data" / "law_hallu_bio.jsonl",
    REPO_ROOT / "data" / "hallucination_data" / "case_hallu_bio.jsonl",
    REPO_ROOT / "data" / "hallucination_data" / "manual_hallu_bio.jsonl",
    REPO_ROOT / "data" / "real_bio_data" / "New_Dataset" / "rag_law_chunks_bio.jsonl",
    REPO_ROOT / "data" / "real_bio_data" / "New_Dataset" / "rag_case_chunks_bio.jsonl",
    REPO_ROOT / "data" / "real_bio_data" / "New_Dataset" / "rag_manual_chunks_bio.jsonl",
]

ENTITY_TYPES = ["LAW", "ORG", "DATE", "AMOUNT", "CRIME", "PENALTY"]


def load_test_split(seed: int = 42) -> list:
    records = []
    for path in BIO_FILES:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            records.extend(json.loads(l) for l in f if l.strip())

    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    n_val = n_test = int(n * 0.1)
    n_train = n - n_val - n_test
    test = records[n_train + n_val:]
    print(f"전체 {n}개 → 테스트셋 {len(test)}개 사용")
    return test


def tokenize(records: list, tokenizer, max_length: int, label2id: dict) -> list:
    dataset = []
    for rec in records:
        text = "".join(rec["tokens"])
        char_tags = rec["ner_tags"]

        enc = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        token_labels = []
        for start, end in enc["offset_mapping"]:
            if start == end:
                token_labels.append(-100)
            else:
                char_idx = min(start, len(char_tags) - 1)
                tag = char_tags[char_idx]
                token_labels.append(label2id.get(tag, label2id.get("O", 0)))

        dataset.append({
            "input_ids":      torch.tensor(enc["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(token_labels,          dtype=torch.long),
        })
    return dataset


def collate(batch):
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        l = x["input_ids"].size(0)
        input_ids[i, :l] = x["input_ids"]
        attention_mask[i, :l] = x["attention_mask"]
        labels[i, :l] = x["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def extract_spans(seq: list[str]) -> set[tuple]:
    spans, i = set(), 0
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


def span_f1(gold_list: list, pred_list: list) -> dict:
    tp = fp = fn = 0
    per_type: dict[str, dict] = {}

    for gold_seq, pred_seq in zip(gold_list, pred_list):
        gs = extract_spans(gold_seq)
        ps = extract_spans(pred_seq)
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
        for s in gs & ps:
            per_type.setdefault(s[2], {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
        for s in ps - gs:
            per_type.setdefault(s[2], {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
        for s in gs - ps:
            per_type.setdefault(s[2], {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    p, r, f = prf(tp, fp, fn)
    result = {"precision": p, "recall": r, "f1": f}
    for etype, c in per_type.items():
        _, _, ef = prf(c["tp"], c["fp"], c["fn"])
        result[f"f1_{etype}"] = ef
    macro = sum(result.get(f"f1_{t}", 0.0) for t in ENTITY_TYPES) / len(ENTITY_TYPES)
    result["macro_f1"] = round(macro, 4)
    return result


# 우리 모델의 고정 스키마 (gold 레이블 인코딩에 사용)
OUR_LABELS = [
    "O",
    "B-LAW", "I-LAW",
    "B-ORG", "I-ORG",
    "B-DATE", "I-DATE",
    "B-AMOUNT", "I-AMOUNT",
    "B-CRIME", "I-CRIME",
    "B-PENALTY", "I-PENALTY",
]
OUR_LABEL2ID = {l: i for i, l in enumerate(OUR_LABELS)}
OUR_ID2LABEL = {i: l for i, l in enumerate(OUR_LABELS)}


def evaluate_model(model_dir: str, test_records: list, device: torch.device, batch_size: int = 16) -> dict:
    model_path = Path(model_dir)
    model_name = model_path.name
    print(f"\n[평가] {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForTokenClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    pred_id2label = model.config.id2label   # 예측 디코딩용 (각 모델 고유)
    is_baseline = model_name in LABEL_MAPPING
    max_length = 256

    if is_baseline:
        # 핵심 수정: gold 레이블은 항상 우리 고정 스키마로 인코딩.
        # 베이스라인의 label2id에 B-ORG 등이 없어 전부 O로 매핑되는 버그 방지.
        # 예측 레이블은 베이스라인 id2label로 디코딩 후 map_label()로 우리 스키마로 변환.
        print(f"  레이블 매핑 적용: {model_name}")
        gold_label2id = OUR_LABEL2ID
        gold_id2label = OUR_ID2LABEL
    else:
        # 우리 모델: gold와 pred 모두 동일한 id2label 사용
        gold_label2id = {v: k for k, v in pred_id2label.items()}
        gold_id2label = pred_id2label

    print(f"  토크나이징 중... (총 {len(test_records)}개)")
    dataset = tokenize(test_records, tokenizer, max_length, gold_label2id)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)

    all_gold, all_pred = [], []
    print(f"  추론 중...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % 50 == 0:
                print(f"    {i * batch_size}/{len(dataset)}")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu()

            for pred_row, label_row in zip(preds, labels):
                gold_seq, pred_seq = [], []
                for p, l in zip(pred_row.tolist(), label_row.tolist()):
                    if l == -100:
                        continue
                    gold_label = gold_id2label.get(l, "O")
                    pred_label = pred_id2label.get(p, "O")
                    if is_baseline:
                        pred_label = map_label(pred_label, model_name)
                    gold_seq.append(gold_label)
                    pred_seq.append(pred_label)
                all_gold.append(gold_seq)
                all_pred.append(pred_seq)

    return span_f1(all_gold, all_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[
        "outputs/legal-ner-v3",
        "outputs/baselines/Babelscape_wikineural-multilingual-ner",
        "outputs/baselines/Leo97_KoELECTRA-small-v3-modu-ner",
        "outputs/baselines/monologg_koelectra-base-v3-naver-ner",
    ])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_records = load_test_split()

    results = {}
    for model_dir in args.models:
        path = REPO_ROOT / model_dir
        if not path.exists():
            print(f"\n[SKIP] 모델 없음: {model_dir}")
            continue
        results[path.name] = evaluate_model(str(path), test_records, device, args.batch_size)

    # 결과 출력
    print("\n" + "=" * 60)
    print("=== Span F1 비교 결과 ===")
    print("=" * 60)
    header = f"{'모델':<35} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Macro F1':>10}"
    print(header)
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<35} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['macro_f1']:>10.4f}")

    print("\n=== 엔티티별 F1 ===")
    header2 = f"{'모델':<35}" + "".join(f" {t:>8}" for t in ENTITY_TYPES)
    print(header2)
    print("-" * (35 + 9 * len(ENTITY_TYPES)))
    for name, r in results.items():
        row = f"{name:<35}" + "".join(f" {r.get(f'f1_{t}', 0.0):>8.4f}" for t in ENTITY_TYPES)
        print(row)

    # JSON 저장
    out_path = REPO_ROOT / "outputs" / "ner_compare_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
