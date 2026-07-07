"""3단계 L1: 결정적 라벨링 — 법령 DB 조회로 확정 가능한 클레임 판정 (LLM·API 미사용).

판정 규칙 (보수적: 확실한 것만 확정):
  - 클레임이 인용한 (법령, 조문)에서 법령이 DB에 있는데 그 조문이 없음
    → not_supported 확정 (nonexistent_article)
  - 그 외 전부 → pass (L2로)

사용: python pipeline\\label_l1.py
"""
import json
from pathlib import Path

from law_db import LawDB, extract_citations

ROOT = Path(__file__).resolve().parent.parent
C_PATH = ROOT / "data" / "collection" / "claims.jsonl"
OUT_PATH = ROOT / "data" / "collection" / "labels_l1.jsonl"


def main():
    db = LawDB()
    claims = [json.loads(l) for l in open(C_PATH, encoding="utf-8") if l.strip()]
    n_cited = n_refuted = n_unknown_law = 0

    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for c in claims:
            citations = extract_citations(c["claim_text"])
            statuses = []
            for law, art in citations:
                if not db.law_exists(law):
                    statuses.append({"law": law, "article": art, "status": "law_unknown"})
                    n_unknown_law += 1
                elif art and not db.article_exists(law, art):
                    statuses.append({"law": law, "article": art, "status": "nonexistent_article"})
                else:
                    statuses.append({"law": law, "article": art, "status": "article_exists"})
            if citations:
                n_cited += 1
            refuted = any(s["status"] == "nonexistent_article" for s in statuses)
            if refuted:
                n_refuted += 1
            fout.write(json.dumps({
                "c_id": c["c_id"],
                "l1_verdict": "not_supported" if refuted else "pass",
                "l1_code": "nonexistent_article" if refuted else "",
                "citations": statuses,
            }, ensure_ascii=False) + "\n")

    print(f"클레임 {len(claims)}개 | 인용 포함 {n_cited} | "
          f"L1 확정(조문 부재) {n_refuted} | 미등재 법령 인용 {n_unknown_law}")
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
