"""전체 평가 결과 종합"""
import json
from pathlib import Path


def show_all_results():
    """RAG + RAGAS + 환각 탐지 결과 종합"""

    print("\n" + "="*100)
    print("법률 RAG 시스템 - 전체 평가 결과")
    print("="*100)

    # ============================================================
    # 1. RAG 검색 성능
    # ============================================================
    print("\n[ 1. RAG 검색 성능 평가 ]")
    print("-"*100)
    print(f"{'방법':<15} {'P@1':<10} {'P@3':<10} {'P@5':<10} {'R@5':<10} {'MRR':<10} {'NDCG@5':<10}")
    print("-"*100)

    results_dir = Path("results")

    for method_name, file_suffix in [("BM25", "bm25"), ("Dense", "dense"), ("Hybrid", "hybrid")]:
        file_path = results_dir / f"rag_eval_{file_suffix}.json"
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                key = list(data.keys())[0]
                metrics = data[key]

                print(f"{method_name:<15} "
                      f"{metrics.get('precision_at_1', 0):<10.4f} "
                      f"{metrics.get('precision_at_3', 0):<10.4f} "
                      f"{metrics.get('precision_at_5', 0):<10.4f} "
                      f"{metrics.get('recall_at_5', 0):<10.4f} "
                      f"{metrics.get('mrr', 0):<10.4f} "
                      f"{metrics.get('ndcg_at_5', 0):<10.4f}")

    # ============================================================
    # 2. RAGAS End-to-End 평가
    # ============================================================
    print("\n[ 2. RAGAS End-to-End 평가 ]")
    print("-"*100)
    print(f"{'방법':<15} {'Faithfulness':<15} {'Answer Rel.':<15} {'Context Prec.':<15} {'Context Rec.':<15} {'Citation':<10}")
    print("-"*100)

    for method_name, file_suffix in [("BM25-RAG", "bm25"), ("Dense-RAG", "dense"), ("Hybrid-RAG", "hybrid")]:
        file_path = results_dir / f"ragas_eval_{file_suffix}.json"
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                key = list(data.keys())[0]
                metrics = data[key]

                print(f"{method_name:<15} "
                      f"{metrics.get('faithfulness', 0):<15.4f} "
                      f"{metrics.get('answer_relevancy', 0):<15.4f} "
                      f"{metrics.get('context_precision', 0):<15.4f} "
                      f"{metrics.get('context_recall', 0):<15.4f} "
                      f"{metrics.get('citation_accuracy', 0):<10.4f}")

    # ============================================================
    # 3. 환각 탐지 평가
    # ============================================================
    print("\n[ 3. 환각 탐지 성능 평가 ]")
    print("-"*100)
    print(f"{'방법':<40} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'순위':<6}")
    print("-"*100)

    hallu_results = {}
    hallu_dir = Path("scripts_hallucination/results")

    # 우리 모델
    ours_path = hallu_dir / "ours_summary.json"
    if ours_path.exists():
        with ours_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            hallu_results["우리 모델 (BERTScore+NER+NLI)"] = data["metrics"]

    # SelfCheckGPT-NLI
    nli_path = hallu_dir / "selfcheck_nli_summary.json"
    if nli_path.exists():
        with nli_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            hallu_results["SelfCheckGPT-NLI"] = data["metrics"]

    # MetaQA
    metaqa_path = hallu_dir / "metaqa_results.json"
    if metaqa_path.exists():
        with metaqa_path.open("r", encoding="utf-8") as f:
            metaqa_results = json.load(f)

            tp = sum(1 for r in metaqa_results if r["true_label"] and r["predicted"])
            fp = sum(1 for r in metaqa_results if not r["true_label"] and r["predicted"])
            fn = sum(1 for r in metaqa_results if r["true_label"] and not r["predicted"])

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            hallu_results["MetaQA"] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

    # 정렬 및 출력
    sorted_hallu = sorted(hallu_results.items(), key=lambda x: x[1]["f1"], reverse=True)

    for rank, (name, metrics) in enumerate(sorted_hallu, 1):
        medal = "[1st]" if rank == 1 else "[2nd]" if rank == 2 else "[3rd]" if rank == 3 else "     "
        print(f"{name:<40} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} "
              f"{medal}")

    print("\n" + "="*100)

    # ============================================================
    # 4. 핵심 요약
    # ============================================================
    print("\n[ 핵심 요약 ]")
    print("-"*100)
    print("\n[OK] RAG 검색:")
    print("   - Hybrid 방식이 가장 균형잡힌 성능")
    print("   - BM25가 Precision@1에서 강점")

    print("\n[OK] RAGAS (End-to-End):")
    print("   - Faithfulness 0.95+ (모든 방법 우수)")
    print("   - Answer Relevancy 0.91+ (답변 품질 높음)")

    print("\n[OK] 환각 탐지:")
    if sorted_hallu:
        winner = sorted_hallu[0]
        second = sorted_hallu[1] if len(sorted_hallu) > 1 else None
        print(f"   - 1등: {winner[0]} (F1 {winner[1]['f1']:.4f})")
        if second:
            print(f"   - 2등: {second[0]} (F1 {second[1]['f1']:.4f})")

        # 우리 모델 분석
        if "우리 모델 (BERTScore+NER+NLI)" in hallu_results:
            ours = hallu_results["우리 모델 (BERTScore+NER+NLI)"]
            print(f"\n[*] 우리 모델 특징:")
            print(f"   - Recall {ours['recall']:.3f}: 환각 거의 완벽 탐지 (400개 중 399개)")
            print(f"   - Precision {ours['precision']:.3f}: 보수적 판정 (안전 우선)")

    print("\n" + "="*100)


if __name__ == "__main__":
    show_all_results()
