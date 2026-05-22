"""
논문 Table 생성 스크립트

outputs/ 디렉토리의 평가 결과 JSON을 읽어 논문용 표 5개를 출력.

Table 1: NER 전체 Span F1 비교 (우리 모델 vs 베이스라인 3개)
Table 2: NER 엔티티별 F1 (공동 엔티티 + 법률 전용 엔티티)
Table 3: RAG 검색 성능 (BM25 / Dense / Hybrid - Hit@1, NDCG@5)
Table 4: 환각 탐지 성능 비교 - 4개 모델
         Ours(NER) / GPT-4o-mini(Judge) / NLI(klue/roberta) / MiniCheck(Flan-T5)
Table 5: Ablation Study (RAG없음 → RAG → RAG+SIM → RAG+SIM+NER)

사용법:
  python scripts/generate_paper_tables.py
  python scripts/generate_paper_tables.py --latex   # LaTeX 형식 추가 출력
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]

NER_RESULTS_PATH    = REPO_ROOT / "outputs" / "ner_compare_results.json"
RAG_RESULTS_PATH    = REPO_ROOT / "outputs" / "rag_eval_results.json"
HALLU_RESULTS_PATH  = REPO_ROOT / "outputs" / "hallucination_eval_results.json"
ABLATION_RESULTS_PATH = REPO_ROOT / "outputs" / "ablation_results.json"

ENTITY_TYPES = ["LAW", "ORG", "DATE", "AMOUNT", "CRIME", "PENALTY"]
COMMON_ENTITIES  = ["ORG", "DATE", "AMOUNT"]
LEGAL_ENTITIES   = ["LAW", "CRIME", "PENALTY"]


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"[SKIP] 파일 없음: {path.name}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def bold(val: float, best: float, delta: float = 0.001) -> str:
    """콘솔 출력에서 best 값에 * 표시."""
    s = f"{val:.4f}"
    return f"*{s}*" if abs(val - best) < delta else s


def sep(width: int = 70) -> None:
    print("-" * width)


# ── Table 1: NER 전체 ─────────────────────────────────────────────────────────

def table1_ner_overall(data: dict, latex: bool) -> None:
    print("\n" + "=" * 70)
    print("Table 1: NER 전체 Span F1 비교")
    print("=" * 70)

    header = f"{'모델':<40} {'Precision':>10} {'Recall':>8} {'F1':>8} {'MacroF1':>9}"
    print(header)
    sep()

    best_f1     = max(r.get("f1", 0) for r in data.values())
    best_macro  = max(r.get("macro_f1", 0) for r in data.values())

    for name, r in data.items():
        p   = r.get("precision", 0)
        rec = r.get("recall", 0)
        f1  = r.get("f1", 0)
        mf1 = r.get("macro_f1", 0)
        print(
            f"{name:<40} {p:>10.4f} {rec:>8.4f} "
            f"{bold(f1, best_f1):>8} {bold(mf1, best_macro):>9}"
        )

    if latex:
        print("\n[LaTeX]")
        print(r"\begin{table}[h]")
        print(r"\caption{NER Span F1 Comparison}")
        print(r"\begin{tabular}{lrrrr}")
        print(r"\hline")
        print(r"Model & Precision & Recall & F1 & Macro-F1 \\")
        print(r"\hline")
        for name, r in data.items():
            short = name.replace("_", r"\_")
            print(
                f"{short} & {r.get('precision',0):.4f} & {r.get('recall',0):.4f} "
                f"& {r.get('f1',0):.4f} & {r.get('macro_f1',0):.4f} \\\\"
            )
        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\end{table}")


# ── Table 2: NER 엔티티별 F1 ──────────────────────────────────────────────────

def table2_ner_per_entity(data: dict, latex: bool) -> None:
    print("\n" + "=" * 70)
    print("Table 2: NER 엔티티별 F1")
    print("         [공동 엔티티] ORG / DATE / AMOUNT")
    print("         [법률 전용]  LAW / CRIME / PENALTY")
    print("=" * 70)

    col_w = 9
    header = f"{'모델':<40}" + "".join(f" {t:>{col_w}}" for t in ENTITY_TYPES)
    print(header)
    sep(40 + col_w * len(ENTITY_TYPES) + len(ENTITY_TYPES))

    for name, r in data.items():
        row = f"{name:<40}"
        for t in ENTITY_TYPES:
            val = r.get(f"f1_{t}", 0.0)
            row += f" {val:>{col_w}.4f}"
        print(row)

    if latex:
        print("\n[LaTeX]")
        print(r"\begin{table}[h]")
        print(r"\caption{Per-Entity F1 Score}")
        print(r"\begin{tabular}{l" + "r" * len(ENTITY_TYPES) + "}")
        print(r"\hline")
        cols = " & ".join(ENTITY_TYPES)
        print(f"Model & {cols} \\\\")
        print(r"\hline")
        for name, r in data.items():
            short = name.replace("_", r"\_")
            vals = " & ".join(f"{r.get(f'f1_{t}',0):.4f}" for t in ENTITY_TYPES)
            print(f"{short} & {vals} \\\\")
        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\end{table}")


# ── Table 3: RAG 검색 성능 ────────────────────────────────────────────────────

def table3_rag(data: dict, latex: bool) -> None:
    print("\n" + "=" * 70)
    print("Table 3: RAG 검색 성능 비교")
    print("=" * 70)

    best_hit  = max(r.get("hit_at_1", 0) for r in data.values())
    best_ndcg = max(r.get("ndcg_at_5", 0) for r in data.values())

    header = f"{'방식':<15} {'Hit@1':>10} {'NDCG@5':>10} {'N':>6}"
    print(header)
    sep(45)
    for name, r in data.items():
        h = r.get("hit_at_1", 0)
        n = r.get("ndcg_at_5", 0)
        cnt = r.get("n", "-")
        print(f"{name:<15} {bold(h, best_hit):>10} {bold(n, best_ndcg):>10} {cnt:>6}")

    if latex:
        print("\n[LaTeX]")
        print(r"\begin{table}[h]")
        print(r"\caption{RAG Retrieval Performance}")
        print(r"\begin{tabular}{lrrl}")
        print(r"\hline")
        print(r"Method & Hit@1 & NDCG@5 & N \\")
        print(r"\hline")
        for name, r in data.items():
            print(
                f"{name} & {r.get('hit_at_1',0):.4f} & {r.get('ndcg_at_5',0):.4f} "
                f"& {r.get('n', '-')} \\\\"
            )
        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\end{table}")


# ── Table 4: 환각 탐지 비교 ───────────────────────────────────────────────────

def table4_hallucination(data: dict, latex: bool) -> None:
    print("\n" + "=" * 70)
    print("Table 4: 환각 탐지 성능 비교")
    print("=" * 70)

    numeric = {k: v for k, v in data.items() if "f1" in v}
    if not numeric:
        print("  (결과 없음)")
        return

    best_f1  = max(r["f1"] for r in numeric.values())
    best_rec = max(r["recall"] for r in numeric.values())

    header = f"{'모델':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}"
    print(header)
    sep(65)
    for name, r in data.items():
        if "note" in r:
            print(f"{name:<25} {'(생략)'}")
            continue
        p   = r.get("precision", 0)
        rec = r.get("recall", 0)
        f1  = r.get("f1", 0)
        acc = r.get("accuracy", 0)
        print(
            f"{name:<25} {p:>10.4f} {bold(rec, best_rec):>8} "
            f"{bold(f1, best_f1):>8} {acc:>10.4f}"
        )

    if latex:
        print("\n[LaTeX]")
        print(r"\begin{table}[h]")
        print(r"\caption{Hallucination Detection Performance}")
        print(r"\begin{tabular}{lrrrr}")
        print(r"\hline")
        print(r"Model & Precision & Recall & F1 & Accuracy \\")
        print(r"\hline")
        for name, r in data.items():
            if "note" in r:
                continue
            short = name.replace("_", r"\_")
            print(
                f"{short} & {r.get('precision',0):.4f} & {r.get('recall',0):.4f} "
                f"& {r.get('f1',0):.4f} & {r.get('accuracy',0):.4f} \\\\"
            )
        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\end{table}")


# ── Table 5: Ablation ─────────────────────────────────────────────────────────

def table5_ablation(data: dict, latex: bool) -> None:
    print("\n" + "=" * 70)
    print("Table 5: Ablation Study")
    print("         (환각 탐지 파이프라인 단계별 기여도)")
    print("=" * 70)

    label_map = {
        "A_no_rag":     "A: RAG 없음",
        "B_rag":        "B: RAG",
        "C_rag_sim":    "C: RAG + SIM",
        "D_rag_sim_ner": "D: RAG + SIM + NER",
    }

    best_f1 = max(r.get("f1", 0) for r in data.values())

    header = f"{'조건':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}"
    print(header)
    sep(62)
    for key, label in label_map.items():
        if key not in data:
            continue
        r = data[key]
        p   = r.get("precision", 0)
        rec = r.get("recall", 0)
        f1  = r.get("f1", 0)
        acc = r.get("accuracy", 0)
        print(
            f"{label:<22} {p:>10.4f} {rec:>8.4f} "
            f"{bold(f1, best_f1):>8} {acc:>10.4f}"
        )

    if latex:
        print("\n[LaTeX]")
        print(r"\begin{table}[h]")
        print(r"\caption{Ablation Study}")
        print(r"\begin{tabular}{lrrrr}")
        print(r"\hline")
        print(r"Condition & Precision & Recall & F1 & Accuracy \\")
        print(r"\hline")
        for key, label in label_map.items():
            if key not in data:
                continue
            r = data[key]
            print(
                f"{label} & {r.get('precision',0):.4f} & {r.get('recall',0):.4f} "
                f"& {r.get('f1',0):.4f} & {r.get('accuracy',0):.4f} \\\\"
            )
        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\end{table}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main(latex: bool = False) -> None:
    ner_data    = load_json(NER_RESULTS_PATH)
    rag_data    = load_json(RAG_RESULTS_PATH)
    hallu_data  = load_json(HALLU_RESULTS_PATH)
    ablation_data = load_json(ABLATION_RESULTS_PATH)

    if ner_data:
        table1_ner_overall(ner_data, latex)
        table2_ner_per_entity(ner_data, latex)
    else:
        print("\n[Table 1/2 생략] eval_ner_compare.py 먼저 실행하세요.")

    if rag_data:
        table3_rag(rag_data, latex)
    else:
        print("\n[Table 3 생략] evaluate_rag.py 먼저 실행하세요.")

    if hallu_data:
        table4_hallucination(hallu_data, latex)
    else:
        print("\n[Table 4 생략] evaluate_hallucination.py 먼저 실행하세요.")

    if ablation_data:
        table5_ablation(ablation_data, latex)
    else:
        print("\n[Table 5 생략] evaluate_hallucination.py (--ablation) 먼저 실행하세요.")

    print("\n" + "=" * 70)
    print("완료. * 표시는 각 표에서 최고 성능.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true", help="LaTeX 표 추가 출력")
    args = parser.parse_args()
    main(latex=args.latex)
