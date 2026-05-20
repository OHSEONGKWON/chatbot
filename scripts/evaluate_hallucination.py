"""
환각 탐지 성능 평가 스크립트

NLI 단독, NER 단독, 3계층 앙상블, 외부 모델을 비교합니다.

비교 시스템:
  1. NLI 단독: klue/roberta-large (NLI)
  2. NER 단독: 우리 NER 모델
  3. 3계층 앙상블: 현재 시스템 (NER + NLI + 법적추론)
  4. 외부 모델: vectara/hallucination_evaluation_model

평가 메트릭: Accuracy, Precision, Recall, F1, AUC-ROC

사용법:
  python scripts/evaluate_hallucination.py
  python scripts/evaluate_hallucination.py --eval-data data/evaluation/hallu_eval.jsonl
  python scripts/evaluate_hallucination.py --systems nli_only ner_only pipeline
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EVAL_DATA_PATH = REPO_ROOT / "data" / "evaluation" / "hallu_eval.jsonl"
OUT_DIR = REPO_ROOT / "data" / "evaluation" / "results"

NER_MODEL_PATH = str(REPO_ROOT / "outputs" / "legal-ner-lawsguard-v2-30k")
NLI_MODEL_ID = "klue/roberta-large"
VECTARA_MODEL_ID = "vectara/hallucination_evaluation_model"


def load_eval_data(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ──────────────────────────────
# 1. NLI 단독 시스템
# ──────────────────────────────

def build_nli_detector(model_id: str = NLI_MODEL_ID):
    from transformers import pipeline as hf_pipeline

    device = 0 if torch.cuda.is_available() else -1
    print(f"  NLI 모델 로딩: {model_id}")
    nli = hf_pipeline("text-classification", model=model_id, device=device, truncation=True, max_length=512)

    def detect(record: dict) -> tuple[bool, float]:
        answer = record.get("answer", "")
        rag_docs = record.get("rag_docs", [])
        if not rag_docs or not answer:
            return False, 0.5

        scores = []
        for doc in rag_docs[:3]:
            premise = doc.get("text", "")[:400]
            hypothesis = answer[:200]
            try:
                result = nli(f"{premise} [SEP] {hypothesis}")
                # KLUE RoBERTa NLI: CONTRADICTION → 환각
                label = result[0]["label"]
                score = result[0]["score"]
                if "contradiction" in label.lower():
                    scores.append(score)
                elif "entailment" in label.lower():
                    scores.append(1 - score)
                else:
                    scores.append(0.5)
            except Exception:
                scores.append(0.5)

        avg_score = sum(scores) / len(scores) if scores else 0.5
        return avg_score > 0.5, avg_score

    return detect


# ──────────────────────────────
# 2. NER 단독 시스템
# ──────────────────────────────

def build_ner_detector(model_path: str = NER_MODEL_PATH):
    if not Path(model_path).exists():
        print(f"  [WARN] NER 모델 없음: {model_path}", file=sys.stderr)
        return None

    from src.modules.ner_checker import NERFactChecker

    print(f"  NER 모델 로딩: {model_path}")
    checker = NERFactChecker(model_path=model_path)

    def detect(record: dict) -> tuple[bool, float]:
        answer = record.get("answer", "")
        rag_docs = record.get("rag_docs", [])
        if not answer or not rag_docs:
            return False, 0.0

        mismatches = checker.find_hallucinations(answer, rag_docs)
        has_hallu = len(mismatches) > 0
        score = min(1.0, len(mismatches) * 0.3)
        return has_hallu, score

    return detect


# ──────────────────────────────
# 3. 3계층 파이프라인 (현재 시스템)
# ──────────────────────────────

def build_pipeline_detector():
    try:
        from src.modules.ner_checker import NERFactChecker
        from src.modules.consistency_checker import ConsistencyChecker

        ner_checker = NERFactChecker()
        consistency_checker = ConsistencyChecker()

        def detect(record: dict) -> tuple[bool, float]:
            answer = record.get("answer", "")
            rag_docs = record.get("rag_docs", [])
            if not answer:
                return False, 0.0

            scores = []

            # Layer 1: NER
            try:
                mismatches = ner_checker.find_hallucinations(answer, rag_docs)
                ner_score = min(1.0, len(mismatches) * 0.4)
                scores.append(ner_score)
            except Exception:
                scores.append(0.0)

            # Layer 2: NLI 일관성 (ConsistencyChecker)
            try:
                is_consistent, cons_score = consistency_checker.check(answer, rag_docs)
                inconsistency_score = 1.0 - cons_score
                scores.append(inconsistency_score)
            except Exception:
                scores.append(0.0)

            avg = sum(scores) / len(scores) if scores else 0.0
            return avg > 0.4, avg

        return detect

    except Exception as e:
        print(f"  [WARN] 파이프라인 구성 실패: {e}", file=sys.stderr)
        return None


# ──────────────────────────────
# 4. Vectara HEM
# ──────────────────────────────

def build_vectara_detector():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_id = VECTARA_MODEL_ID
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Vectara 모델 로딩: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
        model.eval()

        def detect(record: dict) -> tuple[bool, float]:
            answer = record.get("answer", "")
            rag_docs = record.get("rag_docs", [])
            if not rag_docs or not answer:
                return False, 0.5

            scores = []
            for doc in rag_docs[:2]:
                source = doc.get("text", "")[:512]
                try:
                    inputs = tokenizer(
                        source,
                        answer[:256],
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    # Vectara HEM: label 0 = hallucinated, 1 = factual
                    probs = torch.softmax(outputs.logits, dim=-1)
                    hallu_prob = probs[0][0].item()
                    scores.append(hallu_prob)
                except Exception:
                    scores.append(0.5)

            avg = sum(scores) / len(scores) if scores else 0.5
            return avg > 0.5, avg

        return detect

    except Exception as e:
        print(f"  [WARN] Vectara 모델 로딩 실패: {e}", file=sys.stderr)
        return None


# ──────────────────────────────
# 메트릭 계산
# ──────────────────────────────

def compute_metrics(labels: list[bool], preds: list[bool], scores: list[float]) -> dict:
    tp = fp = tn = fn = 0
    for gold, pred in zip(labels, preds):
        if gold and pred:
            tp += 1
        elif not gold and pred:
            fp += 1
        elif not gold and not pred:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0

    # AUC-ROC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc_roc": round(auc, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": len(labels),
    }


def print_table(all_results: dict[str, dict]) -> None:
    print("\n" + "=" * 80)
    print("환각 탐지 성능 평가 결과")
    print("=" * 80)
    header = f"{'시스템':<30} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'N':>5}"
    print(header)
    print("-" * 80)

    for sys_name, metrics in all_results.items():
        row = (
            f"{sys_name:<30}"
            f" {metrics.get('accuracy', 0.0):>6.3f}"
            f" {metrics.get('precision', 0.0):>6.3f}"
            f" {metrics.get('recall', 0.0):>6.3f}"
            f" {metrics.get('f1', 0.0):>6.3f}"
            f" {metrics.get('auc_roc', 0.0):>6.3f}"
            f" {metrics.get('support', 0):>5}"
        )
        print(row)

    print("=" * 80)


def save_csv(all_results: dict[str, dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["시스템", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "TP", "FP", "TN", "FN", "N"])
        for sys_name, metrics in all_results.items():
            writer.writerow([
                sys_name,
                metrics.get("accuracy", 0.0),
                metrics.get("precision", 0.0),
                metrics.get("recall", 0.0),
                metrics.get("f1", 0.0),
                metrics.get("auc_roc", 0.0),
                metrics.get("tp", 0),
                metrics.get("fp", 0),
                metrics.get("tn", 0),
                metrics.get("fn", 0),
                metrics.get("support", 0),
            ])
    print(f"CSV 저장: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="환각 탐지 성능 비교 평가")
    parser.add_argument("--eval-data", type=Path, default=EVAL_DATA_PATH)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["all"],
        help="평가할 시스템 (nli_only, ner_only, pipeline, vectara) 또는 all",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if not args.eval_data.exists():
        print(f"[ERROR] 평가 데이터 없음: {args.eval_data}", file=sys.stderr)
        print("먼저 generate_hallu_eval_data.py 를 실행하세요.", file=sys.stderr)
        sys.exit(1)

    records = load_eval_data(args.eval_data)
    labels = [rec["is_hallucination"] for rec in records]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"평가 데이터: {len(records)}개 (positive={n_pos}, negative={n_neg})")

    run_all = "all" in args.systems
    all_results = {}

    systems_to_run = {
        "nli_only": ("NLI 단독 (klue/roberta-large)", build_nli_detector),
        "ner_only": ("NER 단독 (우리 모델)", build_ner_detector),
        "pipeline": ("3계층 앙상블 (우리 시스템)", build_pipeline_detector),
        "vectara": ("Vectara HEM", build_vectara_detector),
    }

    for sys_key, (sys_name, builder) in systems_to_run.items():
        if not run_all and sys_key not in args.systems:
            continue

        print(f"\n--- {sys_name} 평가 중 ---")
        try:
            detector = builder()
            if detector is None:
                print(f"  [SKIP] 빌드 실패")
                continue

            preds = []
            scores = []
            for i, rec in enumerate(records):
                if i % 50 == 0:
                    print(f"  진행: {i}/{len(records)}")
                is_hallu, score = detector(rec)
                preds.append(is_hallu)
                scores.append(score)

            metrics = compute_metrics(labels, preds, scores)
            all_results[sys_name] = metrics
            print(
                f"  Acc={metrics['accuracy']:.3f} P={metrics['precision']:.3f} "
                f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f} AUC={metrics['auc_roc']:.3f}"
            )

        except Exception as e:
            print(f"  [ERROR] {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    if not all_results:
        print("[ERROR] 평가된 시스템이 없습니다.", file=sys.stderr)
        sys.exit(1)

    print_table(all_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "hallucination_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 저장: {json_path}")

    save_csv(all_results, args.output_dir / "hallucination_table.csv")


if __name__ == "__main__":
    main()
