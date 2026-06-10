"""종합 비교 및 분석"""
import json
from pathlib import Path
from collections import defaultdict


def calculate_metrics(results):
    """Precision, Recall, F1 계산"""
    tp = sum(1 for r in results if r["true_label"] and r["predicted"])
    fp = sum(1 for r in results if not r["true_label"] and r["predicted"])
    fn = sum(1 for r in results if r["true_label"] and not r["predicted"])
    tn = sum(1 for r in results if not r["true_label"] and not r["predicted"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


def calculate_by_type(results):
    """환각 유형별 Recall"""
    by_type = defaultdict(lambda: {"tp": 0, "fn": 0})

    for r in results:
        if r["true_label"]:  # 실제 환각인 경우만
            h_type = r["hallu_type"]

            if r["predicted"]:
                by_type[h_type]["tp"] += 1
            else:
                by_type[h_type]["fn"] += 1

    # Recall 계산
    type_recall = {}
    for h_type, counts in by_type.items():
        total = counts["tp"] + counts["fn"]
        type_recall[h_type] = counts["tp"] / total if total > 0 else 0

    return type_recall


def print_comparison():
    """종합 비교 표 출력"""
    results_dir = Path("scripts_hallucination/results")

    # 결과 로드
    methods = {}

    files = {
        "우리 모델 (BERTScore+NER+NLI)": "ours_results.json",
        "SelfCheckGPT-NLI": "selfcheck_nli_results.json",
        "MetaQA": "metaqa_results.json"
    }

    for name, filename in files.items():
        path = results_dir / filename
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                results = json.load(f)
                methods[name] = calculate_metrics(results)

    if not methods:
        print("[ERROR] 결과 파일 없음")
        return

    # 종합 표
    print("\n" + "="*80)
    print("환각 탐지 성능 비교")
    print("="*80 + "\n")

    try:
        from tabulate import tabulate

        table = []
        for name, metrics in methods.items():
            table.append([
                name,
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}"
            ])

        headers = ["방법", "Precision", "Recall", "F1-Score"]
        print(tabulate(table, headers=headers, tablefmt="grid"))

    except ImportError:
        for name, metrics in methods.items():
            print(f"{name}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print()

    print("="*80)

    # 환각 유형별 분석 (Ours만)
    ours_path = results_dir / "ours_results.json"
    if ours_path.exists():
        with ours_path.open("r", encoding="utf-8") as f:
            ours_results = json.load(f)

        type_recall = calculate_by_type(ours_results)

        print("\n환각 유형별 Recall (Ours):")
        for h_type, recall in type_recall.items():
            print(f"  {h_type}: {recall:.4f}")


if __name__ == "__main__":
    print_comparison()
