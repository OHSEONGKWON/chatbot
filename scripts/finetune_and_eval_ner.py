"""공정 NER 비교 평가: 동일 데이터로 파인튜닝 후 span-F1 비교

베이스라인 3개 모델을 법률 도메인 BIO 데이터로 파인튜닝하고,
우리 모델(legal-ner-v3)과 동일 테스트 셋에서 span-level F1을 비교합니다.

공정성 보장:
  - 데이터 분할 : seed=42, 80/10/10 (train_legal_ner.py 동일)
  - 하이퍼파라미터: lr=2e-5, batch=8, epochs=3, max_length=510
  - 분류 헤드   : ignore_mismatched_sizes=True (헤드 교체, 도메인 적응)

사용법:
  python scripts/finetune_and_eval_ner.py              # 전체 파인튜닝 + 평가
  python scripts/finetune_and_eval_ner.py --dry-run    # 데이터/모델 확인만
  python scripts/finetune_and_eval_ner.py --skip-train # 기훈련 모델로 평가만
  python scripts/finetune_and_eval_ner.py --epochs 1 --limit 500  # 빠른 검증
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]

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
    "B-LAW", "I-LAW",
    "B-PENALTY", "I-PENALTY",
    "B-AMOUNT", "I-AMOUNT",
    "B-DATE", "I-DATE",
    "B-ORG", "I-ORG",
    "B-CRIME", "I-CRIME",
]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
ENTITY_TYPES = ["LAW", "PENALTY", "AMOUNT", "DATE", "ORG", "CRIME"]

OUR_MODEL_PATH = REPO_ROOT / "outputs" / "legal-ner-v3"
BASELINES_DIR = REPO_ROOT / "outputs" / "baselines"

BASELINE_MODELS = [
    {
        "name": "Babelscape/wikineural",
        "base": "Babelscape/wikineural-multilingual-ner",
        "output": BASELINES_DIR / "wikineural-finetuned",
    },
    {
        "name": "KoELECTRA-small-modu",
        "base": "Leo97/KoELECTRA-small-v3-modu-ner",
        "output": BASELINES_DIR / "koelectra-small-finetuned",
    },
    {
        "name": "KoELECTRA-base-naver",
        "base": "monologg/koelectra-base-v3-naver-ner",
        "output": BASELINES_DIR / "koelectra-base-finetuned",
    },
]


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────

def load_bio_rows(paths: Iterable[Path], limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            print(f"  [WARN] 데이터 파일 없음: {path}")
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


def split_rows(rows: list[dict], seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


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
                labels.append(LABEL2ID.get(span_tags[0], LABEL2ID["O"]))
            encoded["labels"] = labels
            self.features.append({k: torch.tensor(v) for k, v in encoded.items()})

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        return self.features[idx]


# ── 메트릭 계산 ───────────────────────────────────────────────────────────────

def ids_to_entities(label_ids: list[int]) -> set[tuple[str, int, int]]:
    entities: set[tuple[str, int, int]] = set()
    start = None
    current = None
    for idx, label_id in enumerate(label_ids + [LABEL2ID["O"]]):
        label = ID2LABEL.get(int(label_id), "O") if label_id != -100 else "O"
        if label == "O":
            if current is not None:
                entities.add((current, start, idx))
            current, start = None, None
            continue
        prefix, _, entity_type = label.partition("-")
        if prefix == "B" or current != entity_type:
            if current is not None:
                entities.add((current, start, idx))
            current, start = entity_type, idx
    return entities


def compute_metrics_callback(eval_pred):
    """Trainer 콜백: entity_f1 반환 (best model 선택 기준)."""
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    entity_tp = entity_fp = entity_fn = 0
    for pred_row, label_row in zip(preds, labels):
        valid_preds = [int(p) for p, l in zip(pred_row, label_row) if l != -100]
        valid_labels = [int(l) for l in label_row if l != -100]
        pred_ents = ids_to_entities(valid_preds)
        gold_ents = ids_to_entities(valid_labels)
        entity_tp += len(pred_ents & gold_ents)
        entity_fp += len(pred_ents - gold_ents)
        entity_fn += len(gold_ents - pred_ents)
    p = entity_tp / (entity_tp + entity_fp) if entity_tp + entity_fp else 0.0
    r = entity_tp / (entity_tp + entity_fn) if entity_tp + entity_fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "entity_f1": round(f1, 4),
        "entity_precision": round(p, 4),
        "entity_recall": round(r, 4),
    }


def compute_span_f1_by_type(preds: list[list[int]], labels: list[list[int]]) -> dict:
    """테스트셋 span-level F1: 엔티티 유형별 + macro + micro."""
    tp = {t: 0 for t in ENTITY_TYPES}
    fp = {t: 0 for t in ENTITY_TYPES}
    fn = {t: 0 for t in ENTITY_TYPES}

    for pred_row, label_row in zip(preds, labels):
        valid_preds = [int(p) for p, l in zip(pred_row, label_row) if l != -100]
        valid_labels = [int(l) for l in label_row if l != -100]
        pred_ents = ids_to_entities(valid_preds)
        gold_ents = ids_to_entities(valid_labels)
        for t in ENTITY_TYPES:
            p_t = {e for e in pred_ents if e[0] == t}
            g_t = {e for e in gold_ents if e[0] == t}
            tp[t] += len(p_t & g_t)
            fp[t] += len(p_t - g_t)
            fn[t] += len(g_t - p_t)

    results: dict = {}
    f1_scores = []
    for t in ENTITY_TYPES:
        prec = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else 0.0
        rec = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        results[t] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
        f1_scores.append(f1)
    results["macro_f1"] = round(sum(f1_scores) / len(f1_scores), 4)

    # micro 평균 (전체 엔티티 합산 기반)
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    results["precision"] = round(micro_p, 4)
    results["recall"] = round(micro_r, 4)
    results["f1"] = round(micro_f1, 4)
    return results


def _flatten_for_paper(metrics: dict) -> dict:
    """compute_span_f1_by_type 결과를 generate_paper_tables.py 형식으로 변환.

    형식: {precision, recall, f1, macro_f1, f1_LAW, f1_CRIME, ...}
    """
    flat: dict = {
        "precision": metrics.get("precision", 0.0),
        "recall":    metrics.get("recall", 0.0),
        "f1":        metrics.get("f1", 0.0),
        "macro_f1":  metrics.get("macro_f1", 0.0),
    }
    for t in ENTITY_TYPES:
        flat[f"f1_{t}"] = metrics.get(t, {}).get("f1", 0.0)
    return flat


# ── 훈련 / 평가 ───────────────────────────────────────────────────────────────

def _run_predict(model, tokenizer, test_dataset, batch_size, seed, output_dir) -> dict:
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=batch_size,
        seed=seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_callback,
    )
    pred_output = trainer.predict(test_dataset)
    if pred_output.label_ids is None:
        print("  [ERROR] label_ids가 None입니다. test_dataset에 labels 필드가 있는지 확인하세요.")
        return {}
    preds = pred_output.predictions.argmax(-1).tolist()
    labels_list = pred_output.label_ids.tolist()
    raw = compute_span_f1_by_type(preds, labels_list)
    model.cpu()
    torch.cuda.empty_cache()
    return _flatten_for_paper(raw)


def finetune_and_evaluate(
    model_name: str,
    base_model: str,
    output_dir: Path,
    tokenizer,
    train_dataset: LegalNERDataset,
    val_dataset: LegalNERDataset,
    test_dataset: LegalNERDataset,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    seed: int,
    local_files_only: bool = False,
) -> dict:
    print(f"\n{'─' * 65}")
    print(f"[파인튜닝] {model_name}")
    print(f"  base: {base_model}")
    print(f"  out : {output_dir}")
    print(f"{'─' * 65}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_callback,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return _run_predict(trainer.model, tokenizer, test_dataset, batch_size, seed, output_dir)


def evaluate_only(
    model_name: str,
    model_path: Path,
    tokenizer,
    test_dataset: LegalNERDataset,
    batch_size: int,
    seed: int,
    local_files_only: bool = True,
) -> dict | None:
    print(f"\n{'─' * 65}")
    print(f"[평가 전용] {model_name}")
    print(f"  path: {model_path}")
    print(f"{'─' * 65}")

    if not model_path.exists():
        print(f"  [SKIP] 모델 경로 없음: {model_path}")
        return None

    model = AutoModelForTokenClassification.from_pretrained(
        str(model_path), local_files_only=local_files_only
    )
    return _run_predict(model, tokenizer, test_dataset, batch_size, seed, model_path)


def _load_tokenizer(source: str, local: bool) -> AutoTokenizer:
    try:
        tok = AutoTokenizer.from_pretrained(source, use_fast=True, local_files_only=local)
    except Exception:
        tok = AutoTokenizer.from_pretrained(source, use_fast=False, local_files_only=local)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "[PAD]"
    return tok


# ── 결과 출력 ─────────────────────────────────────────────────────────────────

def print_results(all_results: dict[str, dict]) -> None:
    W = 90
    col_w = 10
    print("\n" + "=" * W)
    print("=== NER 공정 비교 평가 결과 (span-level F1, 동일 데이터·하이퍼파라미터로 파인튜닝) ===")
    print("=" * W)
    header = f"{'모델':<30}" + "".join(f"{t:>{col_w}}" for t in ENTITY_TYPES) + f"{'Macro-F1':>{col_w + 2}}"
    print(header)
    print("-" * W)
    for model_name, metrics in all_results.items():
        row = f"{model_name:<30}"
        for t in ENTITY_TYPES:
            row += f"{metrics.get(t, {}).get('f1', 0.0):>{col_w}.4f}"
        row += f"{metrics.get('macro_f1', 0.0):>{col_w + 2}.4f}"
        print(row)
    print("=" * W)

    print("\n=== 엔티티 유형별 상세 (Precision / Recall / F1) ===")
    for model_name, metrics in all_results.items():
        print(f"\n  [{model_name}]")
        for t in ENTITY_TYPES:
            m = metrics.get(t, {})
            p, r, f = m.get("precision", 0), m.get("recall", 0), m.get("f1", 0)
            print(f"    {t:<10}  P={p:.4f}  R={r:.4f}  F1={f:.4f}")
        print(f"    {'Macro-F1':<10}  {metrics.get('macro_f1', 0):.4f}")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="공정 NER 비교 평가")
    parser.add_argument("--dry-run", action="store_true", help="데이터·모델 확인만 (훈련·평가 생략)")
    parser.add_argument("--skip-train", action="store_true", help="파인튜닝 생략, 기훈련 모델 평가만")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=510)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="데이터 행 수 제한 (디버깅용)")
    parser.add_argument("--data", nargs="*", default=[str(p) for p in DEFAULT_BIO_FILES])
    parser.add_argument("--our-model", default=str(OUR_MODEL_PATH), help="우리 모델 경로")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ── 데이터 로드 ────────────────────────────────────────────────────────────
    paths = [Path(p) for p in args.data]
    print(f"BIO 데이터 로드 중... ({len(paths)}개 파일)")
    rows = load_bio_rows(paths, limit=args.limit)
    if not rows:
        print("[ERROR] BIO 데이터가 없습니다.", file=sys.stderr)
        return 1
    train_rows, val_rows, test_rows = split_rows(rows, seed=args.seed)
    print(f"  분할 완료: train={len(train_rows):,}, val={len(val_rows):,}, test={len(test_rows):,}")

    # ── dry-run: 모델 접근성만 확인 ─────────────────────────────────────────────
    if not HAS_TORCH and not args.dry_run:
        print("[ERROR] torch/transformers가 설치되지 않았습니다. GPU 환경에서 실행하세요.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n[--dry-run] 모델 접근성 확인:")
        our_path = Path(args.our_model)
        print(f"  우리 모델: {'OK' if our_path.exists() else 'NOT FOUND'} ({our_path})")
        for m in BASELINE_MODELS:
            status = "기훈련 있음" if m["output"].exists() else "다운로드 예정"
            print(f"  {m['name']}: {status}  (base={m['base']})")
        print("[--dry-run] 완료.")
        return 0

    all_results: dict[str, dict] = {}

    # ── 우리 모델: 파인튜닝 없이 평가만 ────────────────────────────────────────
    our_path = Path(args.our_model)
    if our_path.exists():
        print(f"\n우리 모델 tokenizer 로드: {our_path.name}")
        our_tok = _load_tokenizer(str(our_path), local=True)
        our_test_ds = LegalNERDataset(test_rows, our_tok, args.max_length)
        result = evaluate_only(
            "Ours(legal-ner-v3)", our_path, our_tok, our_test_ds,
            args.batch_size, args.seed,
        )
        if result:
            all_results["Ours(legal-ner-v3)"] = result
            print(f"  → Macro-F1: {result['macro_f1']:.4f}")
    else:
        print(f"[SKIP] 우리 모델 없음: {our_path}")

    # ── 베이스라인: 파인튜닝 후 평가 ────────────────────────────────────────────
    for m in BASELINE_MODELS:
        model_name: str = m["name"]
        base: str = m["base"]
        output_dir: Path = m["output"]

        # --skip-train: fine-tuned 모델이 없으면 건너뜀
        if args.skip_train and not output_dir.exists():
            print(f"\n[SKIP] {model_name}: 기훈련 모델 없음 ({output_dir})")
            continue

        # tokenizer: skip_train이고 저장된 tokenizer가 있으면 거기서 로드
        if args.skip_train and output_dir.exists():
            tok = _load_tokenizer(str(output_dir), local=True)
        else:
            tok = _load_tokenizer(base, local=False)

        bl_test_ds = LegalNERDataset(test_rows, tok, args.max_length)

        if args.skip_train:
            result = evaluate_only(model_name, output_dir, tok, bl_test_ds, args.batch_size, args.seed)
        else:
            bl_train_ds = LegalNERDataset(train_rows, tok, args.max_length)
            bl_val_ds = LegalNERDataset(val_rows, tok, args.max_length)
            result = finetune_and_evaluate(
                model_name=model_name,
                base_model=base,
                output_dir=output_dir,
                tokenizer=tok,
                train_dataset=bl_train_ds,
                val_dataset=bl_val_ds,
                test_dataset=bl_test_ds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
            )

        if result:
            all_results[model_name] = result
            print(f"  → {model_name} Macro-F1: {result['macro_f1']:.4f}")

    if not all_results:
        print("[ERROR] 평가된 모델이 없습니다.", file=sys.stderr)
        return 1

    print_results(all_results)

    out_path = REPO_ROOT / "outputs" / "ner_compare_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
