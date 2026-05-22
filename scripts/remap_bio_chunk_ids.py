"""
bio chunk_id → non-bio chunk_id 재매핑 스크립트.

rag_eval.jsonl의 positive_chunk_id가 bio 청크 기준이라 retriever 인덱스(non-bio)와
불일치하는 문제를 해결한다.

핵심 발견:
  bio chunk_id = non-bio chunk_id + "__part{n}" suffix
  → suffix 제거만으로 144/144 (100%) 매핑 성공
  → suffix 없는 케이스는 chunk_id 직접 조회로 보완

실행:
    python scripts/remap_bio_chunk_ids.py

출력:
    data/evaluation/rag_eval_remapped.jsonl  — 재매핑된 평가셋
    data/evaluation/remap_report.txt         — 매핑 결과 보고서
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import config

RAG_EVAL   = REPO_ROOT / "data" / "evaluation" / "rag_eval.jsonl"
OUT_JSONL  = REPO_ROOT / "data" / "evaluation" / "rag_eval_remapped.jsonl"
OUT_REPORT = REPO_ROOT / "data" / "evaluation" / "remap_report.txt"

_PART_RE = re.compile(r"__part\d+$")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_nonbio_index() -> tuple[set[str], dict[str, str]]:
    """
    반환:
        id_set     — 존재하는 non-bio chunk_id 전체 집합 (O(1) 조회용)
        text_index — normalized(text[:60]) → chunk_id (suffix 없는 케이스 fallback)
    """
    id_set:     set[str]       = set()
    text_index: dict[str, str] = {}
    collision:  set[str]       = set()
    total = 0

    for raw_path in config.rag.jsonl_paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"  [경고] 파일 없음: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk_id = row.get("chunk_id", "")
                text     = _normalize(row.get("text", ""))
                if not chunk_id:
                    continue
                total += 1
                id_set.add(chunk_id)

                key = text[:60]
                if key in text_index and text_index[key] != chunk_id:
                    collision.add(key)
                else:
                    text_index[key] = chunk_id

    for key in collision:
        text_index.pop(key, None)

    print(f"  non-bio 청크 로드: {total:,}개")
    print(f"  chunk_id 집합   : {len(id_set):,}개")
    print(f"  텍스트 인덱스   : {len(text_index):,}개 (충돌 제거: {len(collision)})")
    return id_set, text_index


def _map(bio_id: str, positive_text: str,
         id_set: set[str], text_index: dict[str, str]) -> tuple[str, str]:
    """
    반환: (non_bio_chunk_id, method)
    method: "part_stripped" | "direct_id" | "text_prefix" | "not_found"
    """
    # 1순위: __part suffix 제거
    if _PART_RE.search(bio_id):
        stripped = _PART_RE.sub("", bio_id)
        if stripped in id_set:
            return stripped, "part_stripped"

    # 2순위: suffix 없는 경우 bio_id 자체가 non-bio id인지 확인
    if bio_id in id_set:
        return bio_id, "direct_id"

    # 3순위: 텍스트 prefix 매칭 (fallback)
    key = _normalize(positive_text)[:60]
    if key in text_index:
        return text_index[key], "text_prefix"

    return "", "not_found"


def remap() -> None:
    print("non-bio 청크 인덱스 빌드 중...")
    id_set, text_index = build_nonbio_index()

    rows = []
    with RAG_EVAL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    print(f"\nrag_eval.jsonl 로드: {len(rows)}개 항목")
    print("매핑 중...\n")

    results = []
    stats   = {"part_stripped": 0, "direct_id": 0, "text_prefix": 0, "not_found": 0}
    report  = []

    for row in rows:
        bio_id        = row.get("positive_chunk_id", "")
        positive_text = row.get("positive_text", "")
        pos_rel       = row.get("positive_relevance", 1)

        nonbio_id, method = _map(bio_id, positive_text, id_set, text_index)
        stats[method] += 1

        new_row = row.copy()
        if nonbio_id:
            new_row["positive_chunk_id"]     = nonbio_id
            new_row["positive_chunk_id_bio"] = bio_id
            new_row["remap_method"]          = method
            results.append(new_row)

        report.append({
            "query":    row.get("query", "")[:60],
            "pos_rel":  pos_rel,
            "bio_id":   bio_id,
            "nonbio_id": nonbio_id,
            "method":   method,
            "preview":  positive_text[:80],
        })

    # ── 저장 ─────────────────────────────────────────────────────────────────
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mapped = len(results)
    total  = len(rows)
    report_lines = [
        "=" * 70,
        "bio → non-bio chunk_id 재매핑 보고서",
        "=" * 70,
        f"입력  : {total}개",
        f"성공  : {mapped}개  ({mapped/total*100:.1f}%)",
        f"  - part_stripped: {stats['part_stripped']}개  (__partN suffix 제거)",
        f"  - direct_id    : {stats['direct_id']}개  (bio_id == non-bio_id)",
        f"  - text_prefix  : {stats['text_prefix']}개  (텍스트 prefix 매칭)",
        f"실패  : {stats['not_found']}개  ({stats['not_found']/total*100:.1f}%)",
        "",
        "실패 케이스:",
        "-" * 70,
    ]
    for r in report:
        if r["method"] == "not_found":
            report_lines += [
                f"  {r['query']}",
                f"  bio_id : {r['bio_id']}",
                f"  preview: {r['preview']}",
                "",
            ]

    with OUT_REPORT.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # ── 요약 출력 ─────────────────────────────────────────────────────────────
    print(f"{'='*55}")
    print(f"재매핑 완료")
    print(f"  입력   : {total}개")
    print(f"  성공   : {mapped}개  ({mapped/total*100:.1f}%)")
    print(f"    part_stripped: {stats['part_stripped']}개")
    print(f"    direct_id    : {stats['direct_id']}개")
    print(f"    text_prefix  : {stats['text_prefix']}개")
    print(f"  실패   : {stats['not_found']}개  ({stats['not_found']/total*100:.1f}%)")
    print(f"\n출력 파일:")
    print(f"  {OUT_JSONL}")
    print(f"  {OUT_REPORT}")


if __name__ == "__main__":
    remap()
