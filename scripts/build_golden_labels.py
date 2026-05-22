"""
1단계: evaluate_rag_quality.py의 32개 케이스로 golden_doc_ids 후보 자동 생성.

실행:
    python scripts/build_golden_labels.py

출력:
    data/evaluation/golden_candidates.jsonl  — 자동 추출된 후보 (사람 검토 전)
    data/evaluation/golden_candidates_review.txt — 사람이 읽기 쉬운 검토 보고서

후속 작업:
    golden_candidates.jsonl을 검토해 relevance 값을 조정한 뒤
    data/evaluation/rag_eval_golden.jsonl 로 저장하면 2단계 평가에 사용됩니다.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.rag import retriever
from scripts.evaluate_rag_quality import CASES

# ── 설정 ──────────────────────────────────────────────────────────────────────
CANDIDATE_TOP_K = 20          # 후보 수집 시 검색 범위
OUT_JSONL = REPO_ROOT / "data" / "evaluation" / "golden_candidates.jsonl"
OUT_REVIEW = REPO_ROOT / "data" / "evaluation" / "golden_candidates_review.txt"


def _relevance(doc: dict, must_have: list[str], top_any: list[str]) -> int:
    """
    등급 2: must_have 기준 충족 + top_any 1개 이상 포함
    등급 1: must_have 기준만 충족
    등급 0: 해당 없음
    """
    haystack = (doc.get("text") or "") + " " + json.dumps(doc.get("metadata") or {}, ensure_ascii=False)
    must_hits = sum(1 for term in must_have if term in haystack)
    top_hits   = sum(1 for term in top_any   if term in haystack)

    must_threshold = max(1, min(2, len(must_have)))
    if must_hits >= must_threshold and top_hits >= 1:
        return 2
    if must_hits >= must_threshold:
        return 1
    return 0


def _doc_label(doc: dict) -> str:
    meta = doc.get("metadata") or {}
    parts = [
        meta.get("law_name") or meta.get("source_file") or meta.get("manual_title") or "",
        meta.get("article_id") or meta.get("section_label") or "",
        meta.get("source_type") or "",
    ]
    return " | ".join(p for p in parts if p)


def _chunk_id(doc: dict) -> str:
    return str(doc.get("chunk_id") or "")


def build_golden_labels() -> None:
    print(f"총 {len(CASES)}개 케이스 처리 시작 (top_k={CANDIDATE_TOP_K})\n")

    results = []
    review_lines: list[str] = []

    for i, case in enumerate(CASES, 1):
        query    = case["query"]
        category = case.get("category", "")
        must     = case.get("must_have", [])
        top_any  = case.get("top_any", [])

        docs = retriever.retrieve(query, top_k=CANDIDATE_TOP_K, legal_category=category)

        golden: dict[str, int] = {}
        for doc in docs:
            score = _relevance(doc, must, top_any)
            if score > 0:
                cid = _chunk_id(doc)
                if cid:
                    # 동일 chunk_id가 여러 번 나오면 높은 등급 유지
                    golden[cid] = max(golden.get(cid, 0), score)

        entry = {
            "id":             f"case_{i:03d}",
            "query":          query,
            "category":       category,
            "expected_issues": case.get("expected_issues", []),
            "must_have":      must,
            "top_any":        top_any,
            "golden_doc_ids": golden,   # {chunk_id: relevance(1 or 2)}
            "source":         "auto_keyword",
            "reviewed":       False,    # 사람 검토 후 True로 변경
        }
        results.append(entry)

        # 검토 보고서 생성
        review_lines.append(f"{'='*70}")
        review_lines.append(f"[{i:02d}] {case['name']}")
        review_lines.append(f"  query   : {query}")
        review_lines.append(f"  category: {category}  |  issues: {case.get('expected_issues')}")
        review_lines.append(f"  must_have: {must}")
        review_lines.append(f"  top_any : {top_any}")
        review_lines.append(f"  golden 후보 ({len(golden)}개):")

        if golden:
            # 등급 2 → 1 순으로 정렬
            for cid, rel in sorted(golden.items(), key=lambda x: -x[1]):
                # 원본 doc에서 텍스트 미리보기 추출
                matched_doc = next((d for d in docs if _chunk_id(d) == cid), None)
                label   = _doc_label(matched_doc) if matched_doc else cid
                preview = re.sub(r"\s+", " ", (matched_doc.get("text") or ""))[:100] if matched_doc else ""
                review_lines.append(f"    [rel={rel}] {label}")
                review_lines.append(f"           chunk_id: {cid}")
                review_lines.append(f"           preview : {preview}...")
        else:
            review_lines.append("    → 키워드 기준 매칭 없음 (수동 레이블 필요)")

        review_lines.append("")
        status = "OK" if golden else "EMPTY"
        print(f"  [{i:02d}/{len(CASES)}] {case['name'][:40]:<40} → {len(golden)}개 후보  [{status}]")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with OUT_REVIEW.open("w", encoding="utf-8") as f:
        f.write("\n".join(review_lines))

    # ── 요약 ─────────────────────────────────────────────────────────────────
    total      = len(results)
    has_golden = sum(1 for r in results if r["golden_doc_ids"])
    empty      = total - has_golden
    total_docs = sum(len(r["golden_doc_ids"]) for r in results)
    grade2     = sum(
        1 for r in results
        for v in r["golden_doc_ids"].values() if v == 2
    )

    print(f"\n{'='*60}")
    print(f"완료: {total}개 케이스 처리")
    print(f"  golden 있음 : {has_golden}개")
    print(f"  golden 없음 : {empty}개  ← 수동 레이블 필요")
    print(f"  총 후보 문서: {total_docs}개  (등급2={grade2}, 등급1={total_docs - grade2})")
    print(f"\n출력 파일:")
    print(f"  {OUT_JSONL}")
    print(f"  {OUT_REVIEW}")
    print(f"\n다음 단계:")
    print(f"  1. {OUT_REVIEW.name} 을 열어 각 후보 chunk_id 검토")
    print(f"  2. reviewed=True, relevance 수정 후 rag_eval_golden.jsonl 로 저장")
    print(f"  3. golden 없는 {empty}개 케이스는 수동으로 chunk_id 직접 입력")


if __name__ == "__main__":
    build_golden_labels()
