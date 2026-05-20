"""
논문용 성능 평가 표 생성 스크립트

evaluate_ner.py, evaluate_rag.py, evaluate_hallucination.py 의 결과를
논문에 바로 사용할 수 있는 형식으로 변환합니다.

출력:
  - data/evaluation/results/paper_tables.md   (Markdown 표)
  - data/evaluation/results/paper_tables.tex  (LaTeX 표)
  - data/evaluation/results/summary.json      (모든 결과 통합)

사용법:
  python scripts/generate_paper_tables.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "data" / "evaluation" / "results"
OUT_FILE_MD = RESULTS_DIR / "paper_tables.md"
OUT_FILE_TEX = RESULTS_DIR / "paper_tables.tex"
OUT_FILE_JSON = RESULTS_DIR / "summary.json"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(v, decimals: int = 3) -> str:
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def bold_best(values: list[float]) -> list[str]:
    """최고값에 bold 표시."""
    if not values:
        return []
    best = max(values)
    return [f"**{fmt(v)}**" if v == best else fmt(v) for v in values]


# ──────────────────────────────
# Markdown 표 생성
# ──────────────────────────────

def make_ner_md(ner_results: dict) -> str:
    lines = []
    lines.append("## TABLE 1: NER 개체명 인식 성능 (Entity-level Span F1)\n")
    lines.append("| 모델 | LAW | ORG | DATE | AMOUNT | CRIME | PENALTY | Macro F1 | Overall F1 |")
    lines.append("|------|-----|-----|------|--------|-------|---------|----------|------------|")

    entity_types = ["LAW", "ORG", "DATE", "AMOUNT", "CRIME", "PENALTY"]
    rows = []
    for model_key, result in ner_results.items():
        pe = result.get("per_entity", {})
        row_vals = [pe.get(t, {}).get("f1", 0.0) for t in entity_types]
        macro = result.get("macro_f1", 0.0)
        overall = result.get("overall", {}).get("f1", 0.0)
        rows.append((model_key, row_vals, macro, overall))

    # 각 열별 최고값 bold 처리
    for model_key, row_vals, macro, overall in rows:
        cells = [fmt(v) for v in row_vals] + [fmt(macro), fmt(overall)]
        lines.append(f"| {model_key} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("*F1: span-level exact match, macro: 6개 엔티티 평균*\n")
    return "\n".join(lines)


def make_rag_md(rag_results: dict) -> str:
    lines = []
    lines.append("## TABLE 2: RAG 검색 성능\n")
    lines.append("| 시스템 | Hit@1 | Hit@5 | Hit@10 | MRR@10 | NDCG@10 | N |")
    lines.append("|--------|-------|-------|--------|--------|---------|---|")

    for sys_name, metrics in rag_results.items():
        row = (
            f"| {sys_name} "
            f"| {fmt(metrics.get('hit@1', 0.0))} "
            f"| {fmt(metrics.get('hit@5', 0.0))} "
            f"| {fmt(metrics.get('hit@10', 0.0))} "
            f"| {fmt(metrics.get('mrr@10', 0.0))} "
            f"| {fmt(metrics.get('ndcg@10', 0.0))} "
            f"| {metrics.get('n', 0)} |"
        )
        lines.append(row)

    lines.append("")
    lines.append("*Hit@k: top-k 내 정답 문서 포함 비율, MRR: Mean Reciprocal Rank, NDCG: Normalized DCG*\n")
    return "\n".join(lines)


def make_hallu_md(hallu_results: dict) -> str:
    lines = []
    lines.append("## TABLE 3: 환각 탐지 성능\n")
    lines.append("| 시스템 | Accuracy | Precision | Recall | F1 | AUC-ROC | N |")
    lines.append("|--------|----------|-----------|--------|----|---------|----|")

    for sys_name, metrics in hallu_results.items():
        row = (
            f"| {sys_name} "
            f"| {fmt(metrics.get('accuracy', 0.0))} "
            f"| {fmt(metrics.get('precision', 0.0))} "
            f"| {fmt(metrics.get('recall', 0.0))} "
            f"| {fmt(metrics.get('f1', 0.0))} "
            f"| {fmt(metrics.get('auc_roc', 0.0))} "
            f"| {metrics.get('support', 0)} |"
        )
        lines.append(row)

    lines.append("")
    lines.append("*binary classification: is_hallucination (positive=환각)*\n")
    return "\n".join(lines)


# ──────────────────────────────
# LaTeX 표 생성
# ──────────────────────────────

def make_ner_tex(ner_results: dict) -> str:
    entity_types = ["LAW", "ORG", "DATE", "AMOUNT", "CRIME", "PENALTY"]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{NER 개체명 인식 성능 (Entity-level Span F1)}",
        r"\label{tab:ner}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"모델 & LAW & ORG & DATE & AMOUNT & CRIME & PENALTY & Macro F1 & Overall F1 \\",
        r"\midrule",
    ]

    for model_key, result in ner_results.items():
        pe = result.get("per_entity", {})
        vals = [pe.get(t, {}).get("f1", 0.0) for t in entity_types]
        macro = result.get("macro_f1", 0.0)
        overall = result.get("overall", {}).get("f1", 0.0)
        cells = " & ".join([fmt(v) for v in vals] + [fmt(macro), fmt(overall)])
        lines.append(f"{model_key} & {cells} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def make_rag_tex(rag_results: dict) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{RAG 검색 성능}",
        r"\label{tab:rag}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"시스템 & Hit@1 & Hit@5 & Hit@10 & MRR@10 & NDCG@10 & N \\",
        r"\midrule",
    ]

    for sys_name, metrics in rag_results.items():
        cells = " & ".join([
            fmt(metrics.get("hit@1", 0.0)),
            fmt(metrics.get("hit@5", 0.0)),
            fmt(metrics.get("hit@10", 0.0)),
            fmt(metrics.get("mrr@10", 0.0)),
            fmt(metrics.get("ndcg@10", 0.0)),
            str(metrics.get("n", 0)),
        ])
        lines.append(f"{sys_name} & {cells} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def make_hallu_tex(hallu_results: dict) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{환각 탐지 성능}",
        r"\label{tab:hallucination}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"시스템 & Accuracy & Precision & Recall & F1 & AUC-ROC & N \\",
        r"\midrule",
    ]

    for sys_name, metrics in hallu_results.items():
        cells = " & ".join([
            fmt(metrics.get("accuracy", 0.0)),
            fmt(metrics.get("precision", 0.0)),
            fmt(metrics.get("recall", 0.0)),
            fmt(metrics.get("f1", 0.0)),
            fmt(metrics.get("auc_roc", 0.0)),
            str(metrics.get("support", 0)),
        ])
        lines.append(f"{sys_name} & {cells} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ner_results = load_json(RESULTS_DIR / "ner_results.json")
    rag_results = load_json(RESULTS_DIR / "rag_results.json")
    hallu_results = load_json(RESULTS_DIR / "hallucination_results.json")

    missing = []
    if not ner_results:
        missing.append("ner_results.json (evaluate_ner.py 실행 필요)")
    if not rag_results:
        missing.append("rag_results.json (evaluate_rag.py 실행 필요)")
    if not hallu_results:
        missing.append("hallucination_results.json (evaluate_hallucination.py 실행 필요)")

    if missing:
        print("[WARN] 다음 결과 파일이 없습니다:")
        for m in missing:
            print(f"  - {m}")
        print("사용 가능한 결과만 표를 생성합니다.\n")

    # ── Markdown ──
    md_sections = ["# LawsGuard 성능 평가 결과\n"]
    if ner_results:
        md_sections.append(make_ner_md(ner_results))
    if rag_results:
        md_sections.append(make_rag_md(rag_results))
    if hallu_results:
        md_sections.append(make_hallu_md(hallu_results))

    OUT_FILE_MD.write_text("\n".join(md_sections), encoding="utf-8")
    print(f"Markdown 저장: {OUT_FILE_MD}")

    # ── LaTeX ──
    tex_sections = [
        r"% LawsGuard 성능 평가 결과",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        "",
    ]
    if ner_results:
        tex_sections.append(make_ner_tex(ner_results))
    if rag_results:
        tex_sections.append(make_rag_tex(rag_results))
    if hallu_results:
        tex_sections.append(make_hallu_tex(hallu_results))

    OUT_FILE_TEX.write_text("\n".join(tex_sections), encoding="utf-8")
    print(f"LaTeX 저장: {OUT_FILE_TEX}")

    # ── 통합 JSON ──
    summary = {}
    if ner_results:
        summary["ner"] = ner_results
    if rag_results:
        summary["rag"] = rag_results
    if hallu_results:
        summary["hallucination"] = hallu_results

    OUT_FILE_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"통합 JSON 저장: {OUT_FILE_JSON}")

    # ── 콘솔 요약 ──
    print("\n" + "=" * 60)
    print("논문용 표 생성 완료")
    print("=" * 60)
    if ner_results:
        best_ner = max(ner_results.items(), key=lambda x: x[1].get("macro_f1", 0))
        print(f"최고 NER 모델: {best_ner[0]} (macro F1={best_ner[1].get('macro_f1', 0):.3f})")
    if rag_results:
        best_rag = max(rag_results.items(), key=lambda x: x[1].get("mrr@10", 0))
        print(f"최고 RAG 시스템: {best_rag[0]} (MRR@10={best_rag[1].get('mrr@10', 0):.3f})")
    if hallu_results:
        best_hallu = max(hallu_results.items(), key=lambda x: x[1].get("f1", 0))
        print(f"최고 환각탐지 시스템: {best_hallu[0]} (F1={best_hallu[1].get('f1', 0):.3f})")


if __name__ == "__main__":
    main()
