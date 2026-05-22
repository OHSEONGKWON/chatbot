"""
2단계: golden_candidates.jsonl 대화형 3계층 검토 도구.

평가 기준:
  rel=2  핵심근거 — 이 문서를 직접 인용해 답변할 수 있다 (법령 조문, 핵심 판례)
  rel=1  보조근거 — 맥락 이해에 도움이 되지만 직접 인용은 어렵다
  rel=0  제외     — 키워드만 걸렸고 실제 답변에 불필요

최적화:
  - 동일 chunk_id가 여러 케이스에 반복 등장하면 한 번 평가 후 자동 재사용
  - 진행 상황 자동 저장 → 중단 후 동일 명령어로 재개

실행:
    python scripts/review_golden_labels.py

출력:
    data/evaluation/review_progress.json   — 중간 진행 상태 (자동 저장)
    data/evaluation/rag_eval_golden.jsonl  — 최종 골든셋 (모든 케이스 완료 시)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

IN_JSONL    = REPO_ROOT / "data" / "evaluation" / "golden_candidates.jsonl"
PROGRESS_F  = REPO_ROOT / "data" / "evaluation" / "review_progress.json"
OUT_JSONL   = REPO_ROOT / "data" / "evaluation" / "rag_eval_golden.jsonl"

REL_LABELS = {
    2: "핵심근거 — 직접 인용해 답변 가능 (법령 조문·핵심 판례)",
    1: "보조근거 — 맥락 보조, 직접 인용은 어렵다",
    0: "제외     — 키워드만 걸림, 답변에 불필요",
}

# ── 유틸 ──────────────────────────────────────────────────────────────────────

def load_cases() -> list[dict]:
    with IN_JSONL.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_progress() -> dict:
    if PROGRESS_F.exists():
        with PROGRESS_F.open(encoding="utf-8") as f:
            return json.load(f)
    return {"global_ratings": {}, "completed_cases": []}


def save_progress(progress: dict) -> None:
    with PROGRESS_F.open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def save_golden(cases: list[dict], reviewed_ids: dict[str, dict[str, int]]) -> None:
    """rel >= 1인 문서만 골든셋으로 저장."""
    results = []
    for case in cases:
        cid = case["id"]
        rated = reviewed_ids.get(cid, {})
        golden = {k: v for k, v in rated.items() if v >= 1}
        entry = {**case, "golden_doc_ids": golden, "reviewed": True}
        results.append(entry)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_docs = sum(len(r["golden_doc_ids"]) for r in results)
    rel2 = sum(v for r in results for v in r["golden_doc_ids"].values() if v == 2)
    rel1 = sum(v for r in results for v in r["golden_doc_ids"].values() if v == 1)
    print(f"\n골든셋 저장 완료: {OUT_JSONL}")
    print(f"  케이스 {len(results)}개 | 문서 {total_docs}개 (rel=2: {rel2}, rel=1: {rel1})")


def fmt_doc_type(chunk_id: str) -> str:
    if "_제" in chunk_id or "_sec_" in chunk_id:
        return "법령/매뉴얼"
    if "_이유_" in chunk_id or "_판결요지_" in chunk_id:
        return "판례"
    return "기타"


def fmt_preview(text: str, width: int = 120) -> str:
    import re
    return re.sub(r"\s+", " ", text).strip()[:width]


# ── 메인 검토 루프 ─────────────────────────────────────────────────────────────

def review_case(
    case: dict,
    case_idx: int,
    total: int,
    global_ratings: dict[str, int],
) -> dict[str, int]:
    """
    단일 케이스를 대화형으로 검토. 반환값: {chunk_id: rel} (신규 평가 포함 전체)
    """
    existing = case.get("golden_doc_ids", {})
    candidates = list(existing.keys())

    # 이전 케이스에서 이미 평가된 것 분리
    auto_carry  = {cid: global_ratings[cid] for cid in candidates if cid in global_ratings}
    new_cids    = [cid for cid in candidates if cid not in global_ratings]

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"케이스 {case_idx}/{total}: {case.get('category', '')} | {case.get('expected_issues', [])}")
    print(f"쿼리: {case['query']}")
    print(f"must_have: {case.get('must_have', [])}  top_any: {case.get('top_any', [])}")
    print(f"후보 {len(candidates)}개 — 신규 {len(new_cids)}개 / 자동 재사용 {len(auto_carry)}개")

    if auto_carry:
        print("\n[자동 재사용]")
        for cid, rel in auto_carry.items():
            print(f"  rel={rel}  {cid}")

    new_ratings: dict[str, int] = {}

    if not new_cids:
        print("  → 모든 후보가 이전 케이스에서 이미 평가됨.")
        confirm = input("  자동 재사용 결과로 확정하시겠습니까? (y/n/q) ").strip().lower()
        if confirm == "q":
            return None  # 종료 신호
        if confirm == "n":
            # 이 케이스의 자동 분류 전체를 수동 재평가
            new_cids = candidates
            for cid in candidates:
                global_ratings.pop(cid, None)
            auto_carry = {}
    else:
        print()

    for i, cid in enumerate(new_cids, 1):
        doc_type = fmt_doc_type(cid)
        print(f"  [{i}/{len(new_cids)}] {doc_type} | {cid}")

        # 현재 자동 점수가 있으면 참고로 표시
        cur_auto = existing.get(cid)
        if cur_auto is not None:
            print(f"  (자동 키워드 점수: rel={cur_auto})")

        print()
        for rel_val, label in sorted(REL_LABELS.items(), reverse=True):
            print(f"    {rel_val} = {label}")

        while True:
            raw = input("  평가 (0/1/2  s=건너뛰기  q=저장 후 종료): ").strip().lower()
            if raw == "q":
                # 지금까지 평가한 것 저장 후 중단 신호
                global_ratings.update(new_ratings)
                return None
            if raw == "s":
                break
            if raw in ("0", "1", "2"):
                rel = int(raw)
                new_ratings[cid] = rel
                global_ratings[cid] = rel
                break
            print("  잘못된 입력. 0/1/2/s/q 중 하나를 입력하세요.")
        print()

    # 케이스 최종 ratings = 자동 재사용 + 이번 신규 평가
    case_ratings = {**auto_carry, **new_ratings}
    return case_ratings


def main() -> None:
    cases = load_cases()
    progress = load_progress()

    global_ratings: dict[str, int] = progress.get("global_ratings", {})
    completed_ids:  set[str]       = set(progress.get("completed_cases", []))
    reviewed_map:   dict[str, dict[str, int]] = progress.get("reviewed_map", {})

    remaining = [c for c in cases if c["id"] not in completed_ids]
    total     = len(cases)

    if not remaining:
        print("모든 케이스 검토 완료. rag_eval_golden.jsonl 생성 중...")
        save_golden(cases, reviewed_map)
        return

    print(f"LawsGuard 골든셋 대화형 검토 ({len(completed_ids)}/{total} 완료, {len(remaining)}개 남음)")
    print("도중에 q를 입력하면 진행 상황이 저장됩니다.\n")

    for case in remaining:
        result = review_case(case, cases.index(case) + 1, total, global_ratings)

        if result is None:
            # 사용자가 q로 중단 — 저장 후 종료
            progress["global_ratings"] = global_ratings
            progress["reviewed_map"]   = reviewed_map
            save_progress(progress)
            print(f"\n진행 상황 저장 완료 ({len(completed_ids)}/{total} 케이스 완료).")
            print(f"재개하려면: python scripts/review_golden_labels.py")
            sys.exit(0)

        reviewed_map[case["id"]] = result
        completed_ids.add(case["id"])

        progress["global_ratings"]  = global_ratings
        progress["completed_cases"] = list(completed_ids)
        progress["reviewed_map"]    = reviewed_map
        save_progress(progress)

    print("\n모든 케이스 검토 완료!")
    save_golden(cases, reviewed_map)
    print("다음 단계: python scripts/evaluate_rag_quality.py 로 NDCG@5 측정")


if __name__ == "__main__":
    main()
