"""최종 비교 (요약 버전)"""
import json
from pathlib import Path


def print_comparison():
    """3가지 모델 비교"""
    results_dir = Path("scripts_hallucination/results")

    print("\n" + "="*80)
    print("환각 탐지 성능 비교")
    print("="*80 + "\n")

    # SelfCheckGPT-NLI (요약)
    nli_path = results_dir / "selfcheck_nli_summary.json"
    if nli_path.exists():
        with nli_path.open("r", encoding="utf-8") as f:
            nli_data = json.load(f)
            print("SelfCheckGPT-NLI:")
            print(f"  Precision: {nli_data['metrics']['precision']:.4f}")
            print(f"  Recall: {nli_data['metrics']['recall']:.4f}")
            print(f"  F1-Score: {nli_data['metrics']['f1']:.4f}")
            print()

    # 우리 모델
    ours_path = results_dir / "ours_results.json"
    if ours_path.exists():
        with ours_path.open("r", encoding="utf-8") as f:
            ours_results = json.load(f)

            tp = sum(1 for r in ours_results if r["true_label"] and r["predicted"])
            fp = sum(1 for r in ours_results if not r["true_label"] and r["predicted"])
            fn = sum(1 for r in ours_results if r["true_label"] and not r["predicted"])

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print("우리 모델 (BERTScore + NER + NLI):")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print()

            # 단계별 기여도
            consistency = sum(1 for r in ours_results if r["true_label"] and r["is_inconsistent"])
            ner = sum(1 for r in ours_results if r["true_label"] and r["ner_hallucinations"] > 0)

            print("  단계별 기여도:")
            print(f"    BERTScore: {consistency}/{tp} ({consistency/tp*100 if tp > 0 else 0:.1f}%)")
            print(f"    NER/NLI: {ner}/{tp} ({ner/tp*100 if tp > 0 else 0:.1f}%)")
            print()

    # MetaQA
    metaqa_path = results_dir / "metaqa_results.json"
    if metaqa_path.exists():
        with metaqa_path.open("r", encoding="utf-8") as f:
            metaqa_results = json.load(f)

            tp = sum(1 for r in metaqa_results if r["true_label"] and r["predicted"])
            fp = sum(1 for r in metaqa_results if not r["true_label"] and r["predicted"])
            fn = sum(1 for r in metaqa_results if r["true_label"] and not r["predicted"])

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print("MetaQA:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print()

    print("="*80)


if __name__ == "__main__":
    print_comparison()
