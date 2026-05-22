"""
RAG 검색 성능 평가 스크립트 (Phase 3 - Step 2).

지원 입력 형식:
  A) rag_eval.jsonl          — positive_chunk_id 단일 정답 형식 (LLM 합성)
  B) golden_candidates.jsonl — golden_doc_ids 다중 정답+relevance 형식 (자동 키워드)
  C) rag_eval_golden.jsonl   — golden_doc_ids 다중 정답, 수동 3계층 검토 완료본 (권장)

계산 메트릭:
  Hit@1, Hit@3, Hit@5, MRR, NDCG@5 (graded: rel=2 핵심근거, rel=1 보조근거)

실행:
  python scripts/evaluate_rag_comprehensive.py
  python scripts/evaluate_rag_comprehensive.py --eval golden_reviewed  # 수동 검토 완료본 (권장)
  python scripts/evaluate_rag_comprehensive.py --eval golden           # 자동 키워드 후보
  python scripts/evaluate_rag_comprehensive.py --top-k 10             # 검색 범위 확장
  python scripts/evaluate_rag_comprehensive.py --skip-zero            # relevance=0 항목 제외
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.rag import retriever

# ── 파일 경로 ─────────────────────────────────────────────────────────────────
EVAL_FILES = {
    "bio":             REPO_ROOT / "data" / "evaluation" / "rag_eval.jsonl",
    "remapped":        REPO_ROOT / "data" / "evaluation" / "rag_eval_remapped.jsonl",
    "golden":          REPO_ROOT / "data" / "evaluation" / "golden_candidates.jsonl",
    "golden_reviewed": REPO_ROOT / "data" / "evaluation" / "rag_eval_golden.jsonl",
}
RESULTS_DIR = REPO_ROOT / "results"


# ── 메트릭 계산 ───────────────────────────────────────────────────────────────

def _hit_at_k(retrieved_ids: list[str], golden_ids: set[str], k: int) -> float:
    return float(any(cid in golden_ids for cid in retrieved_ids[:k]))


def _mrr(retrieved_ids: list[str], golden_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in golden_ids:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved_ids: list[str], golden_rel: dict[str, int], k: int) -> float:
    """NDCG@k. golden_rel = {chunk_id: relevance(0/1/2)}."""
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], 1):
        rel = golden_rel.get(cid, 0)
        dcg += (2**rel - 1) / math.log2(rank + 1)

    # Ideal DCG: 최대 relevance 순으로 정렬
    ideal_rels = sorted(golden_rel.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, 1))
    return dcg / idcg if idcg > 0 else 0.0


# ── 데이터 로더 ───────────────────────────────────────────────────────────────

_LABOR_KW  = ["임금", "근로", "해고", "노동", "최저임금", "퇴직", "알바", "계약서", "주휴"]
_SEXUAL_KW = ["성폭력", "성희롱", "강제추행", "불법촬영", "성범죄", "추행"]


def _infer_domain(query: str, positive_text: str) -> str:
    text = query + " " + positive_text
    if any(k in text for k in _LABOR_KW):
        return "노동"
    if any(k in text for k in _SEXUAL_KW):
        return "성폭력"
    return "기타"


def _load_bio(path: Path, skip_zero: bool, domain_filter: bool = False) -> list[dict[str, Any]]:
    """positive_chunk_id 형식 로드 → 내부 표준 형식으로 변환."""
    items = []
    skipped_domain = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            if not row.get("positive_chunk_id"):
                continue
            if skip_zero and row.get("positive_relevance", 1) == 0:
                continue
            rel    = row.get("positive_relevance", 1)
            query  = row["query"]
            domain = _infer_domain(query, row.get("positive_text", ""))
            if domain_filter and domain == "기타":
                skipped_domain += 1
                continue
            items.append({
                "id":             row.get("query_id", ""),
                "query":          query,
                "category":       row.get("category", "") or domain,
                "source":         row.get("source_file", ""),
                "golden_doc_ids": {row["positive_chunk_id"]: max(1, rel)},
                "format":         "bio",
            })
    if skipped_domain:
        print(f"  [domain_filter] 기타 도메인 {skipped_domain}개 제외")
    return items


def _load_golden(path: Path, skip_zero: bool) -> list[dict[str, Any]]:
    """golden_doc_ids 형식 로드."""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            golden = row.get("golden_doc_ids") or {}
            if not golden:
                continue
            if skip_zero:
                golden = {k: v for k, v in golden.items() if v > 0}
            if not golden:
                continue
            items.append({
                "id":       row.get("id", ""),
                "query":    row["query"],
                "category": row.get("category", ""),
                "source":   "golden_candidates",
                "golden_doc_ids": golden,
                "format": "golden",
            })
    return items


# ── chunk_id 정규화 ────────────────────────────────────────────────────────────

def _normalize_id(cid: str) -> str:
    """__part2 같은 suffix 제거해 base ID로 통일."""
    return cid.split("__part")[0] if "__part" in cid else cid


def _retrieved_ids(docs: list[dict[str, Any]]) -> list[str]:
    return [str(doc.get("chunk_id") or "") for doc in docs if doc.get("chunk_id")]


# ── 평가 실행 ─────────────────────────────────────────────────────────────────

def evaluate(items: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    results = []
    mismatch_count = 0

    for item in items:
        query    = item["query"]
        category = item["category"]
        golden   = item["golden_doc_ids"]  # {chunk_id: rel}

        docs = retriever.retrieve(query, top_k=top_k, legal_category=category)
        ret_ids = _retrieved_ids(docs)

        # 정규화 버전(part suffix 제거)으로도 매칭
        golden_norm  = {_normalize_id(k): v for k, v in golden.items()}
        ret_ids_norm = [_normalize_id(i) for i in ret_ids]

        # 정답 집합
        golden_set      = set(golden.keys())
        golden_set_norm = set(golden_norm.keys())

        # 검색된 chunk_id가 golden에 한 개도 없으면 mismatch 가능성
        exact_hit = any(i in golden_set for i in ret_ids)
        norm_hit  = any(i in golden_set_norm for i in ret_ids_norm)
        if not exact_hit and not norm_hit:
            mismatch_count += 1

        # 메트릭은 정규화 기준으로 계산 (더 관대, 실질적 의미)
        h1   = _hit_at_k(ret_ids_norm, golden_set_norm, 1)
        h3   = _hit_at_k(ret_ids_norm, golden_set_norm, 3)
        h5   = _hit_at_k(ret_ids_norm, golden_set_norm, 5)
        mrr  = _mrr(ret_ids_norm, golden_set_norm)
        ndcg = _ndcg_at_k(ret_ids_norm, golden_norm, 5)

        results.append({
            "id":       item["id"],
            "query":    query[:60],
            "category": category,
            "source":   item.get("source", ""),
            "format":   item.get("format", ""),
            "golden_n": len(golden),
            "hit@1":  h1,
            "hit@3":  h3,
            "hit@5":  h5,
            "mrr":    mrr,
            "ndcg@5": ndcg,
            "retrieved_ids": ret_ids[:5],
            "golden_ids": list(golden.keys())[:5],
        })

    n = len(results)
    if n == 0:
        return {"n": 0, "results": []}

    def avg(key: str) -> float:
        return round(sum(r[key] for r in results) / n, 4)

    summary = {
        "n":        n,
        "hit@1":    avg("hit@1"),
        "hit@3":    avg("hit@3"),
        "hit@5":    avg("hit@5"),
        "mrr":      avg("mrr"),
        "ndcg@5":   avg("ndcg@5"),
        "chunk_id_mismatch_count": mismatch_count,
        "chunk_id_mismatch_rate":  round(mismatch_count / n, 3),
    }
    return {"summary": summary, "results": results}


# ── 카테고리별 분석 ────────────────────────────────────────────────────────────

def breakdown_by_category(results: list[dict]) -> dict[str, dict]:
    cats: dict[str, list] = {}
    for r in results:
        cat = r.get("category") or "기타"
        cats.setdefault(cat, []).append(r)

    out = {}
    for cat, rows in cats.items():
        n = len(rows)
        out[cat] = {
            "n":      n,
            "hit@1":  round(sum(r["hit@1"]  for r in rows) / n, 4),
            "hit@5":  round(sum(r["hit@5"]  for r in rows) / n, 4),
            "mrr":    round(sum(r["mrr"]    for r in rows) / n, 4),
            "ndcg@5": round(sum(r["ndcg@5"] for r in rows) / n, 4),
        }
    return out


# ── 출력 ──────────────────────────────────────────────────────────────────────

def _print_summary(summary: dict, breakdown: dict, mismatch_warned: bool) -> None:
    n = summary["n"]
    print(f"\n{'='*60}")
    print(f"RAG 검색 성능 평가 결과  (n={n})")
    print(f"{'='*60}")
    print(f"  Hit@1  : {summary['hit@1']:.4f}  ({summary['hit@1']*100:.1f}%)")
    print(f"  Hit@3  : {summary['hit@3']:.4f}  ({summary['hit@3']*100:.1f}%)")
    print(f"  Hit@5  : {summary['hit@5']:.4f}  ({summary['hit@5']*100:.1f}%)")
    print(f"  MRR    : {summary['mrr']:.4f}")
    print(f"  NDCG@5 : {summary['ndcg@5']:.4f}")

    mismatch_rate = summary.get("chunk_id_mismatch_rate", 0)
    if mismatch_rate > 0.1:
        print(f"\n  [경고] chunk_id 불일치율 {mismatch_rate*100:.0f}% "
              f"({summary['chunk_id_mismatch_count']}/{n})")
        print(f"         평가셋 chunk_id가 retriever 인덱스와 다를 수 있습니다.")
        print(f"         (bio vs non-bio 데이터 불일치 가능성)")

    if breakdown:
        print(f"\n{'─'*60}")
        print(f"  카테고리별 분석")
        print(f"{'─'*60}")
        for cat, row in sorted(breakdown.items()):
            print(f"  {cat:<12}  n={row['n']:>3}  "
                  f"Hit@1={row['hit@1']:.3f}  Hit@5={row['hit@5']:.3f}  "
                  f"MRR={row['mrr']:.3f}  NDCG@5={row['ndcg@5']:.3f}")

    print(f"{'='*60}\n")


def _print_failures(results: list[dict], k: int = 10) -> None:
    failures = [r for r in results if r["hit@5"] == 0.0]
    if not failures:
        print("  Hit@5 실패 케이스 없음 (전수 검색 성공)")
        return
    print(f"\n  Hit@5 실패 케이스 ({len(failures)}개, 상위 {k}개 출력):")
    for r in failures[:k]:
        print(f"    [{r['id']}] {r['query']}")
        print(f"      golden  : {r['golden_ids'][:2]}")
        print(f"      retrieved: {r['retrieved_ids'][:2]}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", choices=["bio", "remapped", "golden", "golden_reviewed"],
                        default="remapped",
                        help="평가셋 선택: golden_reviewed=수동3계층검토(권장), remapped=재매핑, bio=원본, golden=자동키워드")
    parser.add_argument("--top-k", type=int, default=10,
                        help="검색 범위 (기본값: 10)")
    parser.add_argument("--skip-zero", action="store_true",
                        help="relevance=0 항목 제외")
    parser.add_argument("--domain-filter", action="store_true",
                        help="노동/성폭력 외 도메인 제외 (bio/remapped 전용)")
    parser.add_argument("--save", action="store_true",
                        help="results/ 폴더에 JSON 저장")
    args = parser.parse_args()

    eval_path = EVAL_FILES[args.eval]
    if not eval_path.exists():
        print(f"[오류] 평가셋 파일 없음: {eval_path}")
        sys.exit(1)

    print(f"평가셋 : {eval_path.name}  (--eval {args.eval})")
    print(f"top_k  : {args.top_k}")
    print(f"skip_zero: {args.skip_zero}  domain_filter: {args.domain_filter}")

    if args.eval in ("bio", "remapped"):
        items = _load_bio(eval_path, args.skip_zero, domain_filter=args.domain_filter)
    else:  # golden, golden_reviewed
        items = _load_golden(eval_path, args.skip_zero)

    print(f"로드된 케이스: {len(items)}개\n검색 중...")

    output  = evaluate(items, top_k=args.top_k)
    summary = output.get("summary", {})
    results = output.get("results", [])

    if not results:
        print("[오류] 평가 결과가 없습니다.")
        sys.exit(1)

    breakdown = breakdown_by_category(results)
    _print_summary(summary, breakdown, mismatch_warned=True)
    _print_failures(results)

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = RESULTS_DIR / f"rag_metrics_{args.eval}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "breakdown": breakdown, "results": results},
                      f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
