"""
Phase 3: 결과 분석 및 시각화

- Phase 1 + Phase 2 결과 통합
- 종합 결과 표 생성
- Excel 내보내기
"""

import json
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_results():
    """결과 파일 로드 (병렬 실행 파일 합치기)"""
    phase1_data = {}
    phase2_data = None

    # Phase 1: 병렬 실행된 파일들 합치기
    for method in ["bm25", "dense", "hybrid"]:
        method_file = Path(f"results/rag_eval_{method}.json")
        if method_file.exists():
            with method_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                phase1_data.update(data)
            print(f"[OK] Phase 1 ({method}) 로드: {method_file}")

    # 통합 파일도 확인
    phase1_path = Path("results/rag_evaluation_results.json")
    if phase1_path.exists() and not phase1_data:
        with phase1_path.open("r", encoding="utf-8") as f:
            phase1_data = json.load(f)
        print(f"[OK] Phase 1 로드: {phase1_path}")

    if not phase1_data:
        print(f"[WARNING] Phase 1 결과 없음")

    # Phase 2: 병렬 실행된 파일들 합치기
    phase2_data = {}
    for method in ["bm25", "dense", "hybrid"]:
        method_file = Path(f"results/ragas_eval_{method}.json")
        if method_file.exists():
            with method_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                phase2_data.update(data)
            print(f"[OK] Phase 2 ({method}) 로드: {method_file}")

    # 통합 파일도 확인
    phase2_path = Path("results/ragas_evaluation_results.json")
    if phase2_path.exists() and not phase2_data:
        with phase2_path.open("r", encoding="utf-8") as f:
            phase2_data = json.load(f)
        print(f"[OK] Phase 2 로드: {phase2_path}")

    if not phase2_data:
        print(f"[WARNING] Phase 2 결과 없음")

    return phase1_data, phase2_data


def print_combined_table(phase1_data, phase2_data):
    """종합 결과 표"""
    print("\n" + "="*100)
    print("종합 평가 결과")
    print("="*100)

    if phase1_data:
        print("\n[ Phase 1: Retrieval 성능 ]")
        try:
            from tabulate import tabulate

            table = []
            for name, metrics in phase1_data.items():
                table.append([
                    name,
                    f"{metrics.get('precision_at_1', 0):.4f}",
                    f"{metrics.get('precision_at_3', 0):.4f}",
                    f"{metrics.get('precision_at_5', 0):.4f}",
                    f"{metrics.get('recall_at_5', 0):.4f}",
                    f"{metrics.get('mrr', 0):.4f}",
                    f"{metrics.get('ndcg_at_5', 0):.4f}"
                ])

            headers = ["방법", "P@1", "P@3", "P@5", "R@5", "MRR", "NDCG@5"]
            print(tabulate(table, headers=headers, tablefmt="grid"))

        except ImportError:
            for name, metrics in phase1_data.items():
                print(f"\n{name}:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")

    if phase2_data:
        print("\n[ Phase 2: RAG 품질 (RAGAS) ]")
        try:
            from tabulate import tabulate

            table = []
            for name, metrics in phase2_data.items():
                table.append([
                    name,
                    f"{metrics.get('faithfulness', 0):.4f}",
                    f"{metrics.get('answer_relevancy', 0):.4f}",
                    f"{metrics.get('context_precision', 0):.4f}",
                    f"{metrics.get('context_recall', 0):.4f}",
                    f"{metrics.get('citation_accuracy', 0):.4f}"
                ])

            headers = ["방법", "Faithfulness", "Answer Rel.", "Context Prec.", "Context Rec.", "Citation Acc."]
            print(tabulate(table, headers=headers, tablefmt="grid"))

            print("\n* Faithfulness: 높을수록 환각 적음")

        except ImportError:
            for name, metrics in phase2_data.items():
                print(f"\n{name}:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")

    print("="*100)


def export_to_excel(phase1_data, phase2_data):
    """Excel 내보내기"""
    if not HAS_PANDAS:
        print("\n[WARNING] pandas 없음. Excel 내보내기 건너뜀")
        print("설치: pip install pandas openpyxl")
        return

    try:
        output_path = Path("results/rag_evaluation_summary.xlsx")

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Phase 1
            if phase1_data:
                df1 = pd.DataFrame.from_dict(phase1_data, orient='index')
                df1.to_excel(writer, sheet_name='Phase1_Retrieval')

            # Phase 2
            if phase2_data:
                df2 = pd.DataFrame.from_dict(phase2_data, orient='index')
                df2.to_excel(writer, sheet_name='Phase2_RAGAS')

        print(f"\n[Excel 저장] {output_path}")

    except Exception as e:
        print(f"\n[WARNING] Excel 저장 실패: {e}")


def main():
    """메인 실행"""
    print("="*70)
    print("Phase 3: 결과 분석")
    print("="*70 + "\n")

    # 결과 로드
    phase1_data, phase2_data = load_results()

    if not phase1_data and not phase2_data:
        print("\n[ERROR] 분석할 결과가 없습니다.")
        print("먼저 Phase 1, 2를 실행하세요.")
        return

    # 종합 표
    print_combined_table(phase1_data, phase2_data)

    # Excel 내보내기
    export_to_excel(phase1_data, phase2_data)

    print("\n[완료] 결과 분석\n")


if __name__ == "__main__":
    main()
