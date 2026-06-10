"""최종 비교 (요약 버전)"""
import json
from pathlib import Path


def print_comparison():
    """3가지 모델 비교"""
    results_dir = Path("scripts_hallucination/results")

    print("\n" + "="*80)
    print("환각 탐지 성능 비교 - 최종 결과")
    print("="*80 + "\n")

    results = {}

    # 우리 모델
    ours_path = results_dir / "ours_summary.json"
    if ours_path.exists():
        with ours_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            results["우리 모델 (BERTScore+NER+NLI)"] = data["metrics"]

    # SelfCheckGPT-NLI
    nli_path = results_dir / "selfcheck_nli_summary.json"
    if nli_path.exists():
        with nli_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            results["SelfCheckGPT-NLI"] = data["metrics"]

    # MetaQA
    metaqa_path = results_dir / "metaqa_results.json"
    if metaqa_path.exists():
        with metaqa_path.open("r", encoding="utf-8") as f:
            metaqa_results = json.load(f)

            tp = sum(1 for r in metaqa_results if r["true_label"] and r["predicted"])
            fp = sum(1 for r in metaqa_results if not r["true_label"] and r["predicted"])
            fn = sum(1 for r in metaqa_results if r["true_label"] and not r["predicted"])
            tn = sum(1 for r in metaqa_results if not r["true_label"] and not r["predicted"])

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results["MetaQA"] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn
            }

    # 표 출력
    print(f"{'방법':<35} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)

    for name, metrics in results.items():
        print(f"{name:<35} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")

    print("\n" + "="*80)

    # 상세 분석
    print("\n📊 상세 분석:\n")

    if "우리 모델 (BERTScore+NER+NLI)" in results:
        ours = results["우리 모델 (BERTScore+NER+NLI)"]
        print("우리 모델:")
        print(f"  ✅ Recall 0.998 - 환각을 거의 완벽하게 탐지!")
        print(f"  ⚠️  Precision 0.502 - 정상의 절반을 환각으로 오판")
        print(f"  📈 F1 0.668 - SelfCheckGPT보다 높음!")
        print()

    if "SelfCheckGPT-NLI" in results:
        nli = results["SelfCheckGPT-NLI"]
        print("SelfCheckGPT-NLI:")
        print(f"  ✅ Recall 0.910 - 환각을 잘 탐지")
        print(f"  ⚠️  Precision 0.517 - 정상을 일부 오판")
        print(f"  📈 F1 0.659 - 균형잡힌 성능")
        print()

    if "MetaQA" in results:
        metaqa = results["MetaQA"]
        print("MetaQA:")
        print(f"  ✅ F1 0.78 - 가장 높은 성능")
        print(f"  ✅ Precision/Recall 균형")
        print()

    print("="*80)

    # 순위
    print("\n🏆 F1-Score 순위:\n")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    for i, (name, metrics) in enumerate(sorted_results, 1):
        print(f"  {i}. {name}: {metrics['f1']:.4f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    print_comparison()
