import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.rag import retriever


CASES = [
    {
        "name": "임금체불/근로계약서 없음",
        "query": "편의점 알바 주 20시간 월급 80만원을 못 받았고 근로계약서가 없습니다. 카톡 증거는 있습니다.",
        "category": "노동",
        "must_have": ["근로기준법", "임금", "체불"],
    },
    {
        "name": "동의 없는 신체접촉",
        "query": "동아리 술자리에서 선배가 동의 없이 몸을 만졌고 거절했는데 반복했습니다. 카톡과 목격자가 있습니다.",
        "category": "성폭력",
        "must_have": ["성폭력", "강제추행", "상담"],
    },
    {
        "name": "성희롱 신고",
        "query": "교수가 성희롱 발언을 반복했고 학교에 신고하고 싶습니다.",
        "category": "성폭력",
        "must_have": ["성희롱", "상담", "신고"],
    },
]


def main():
    failures = 0
    for case in CASES:
        docs = retriever.retrieve(case["query"], top_k=5, legal_category=case["category"])
        merged = "\n".join(
            f"{doc.get('text', '')} {doc.get('metadata', {})}" for doc in docs
        )
        hits = [term for term in case["must_have"] if term in merged]
        ok = len(hits) >= max(1, len(case["must_have"]) - 1)
        failures += 0 if ok else 1

        print(f"\n[{case['name']}] {'PASS' if ok else 'FAIL'}")
        print(f"hits: {hits}")
        for i, doc in enumerate(docs[:3], 1):
            metadata = doc.get("metadata") or {}
            title = metadata.get("law_name") or metadata.get("source_file") or "근거 문서"
            article = metadata.get("article_id") or metadata.get("section_label") or ""
            print(f"{i}. {round(float(doc.get('score', 0.0)), 2)} | {title} | {article}")

    if failures:
        raise SystemExit(f"{failures} RAG quality case(s) failed")


if __name__ == "__main__":
    main()
