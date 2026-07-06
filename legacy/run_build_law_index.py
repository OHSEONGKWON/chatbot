"""1단계: 법령 DB 인덱스 구축.

법령 청크(rag_law_chunks.jsonl) → (법령명, 조문번호) → 조문텍스트 인덱스.
GPU/API 불필요. 빠름.

실행:  venv\\Scripts\\python.exe scripts_hallucination\\run_build_law_index.py
출력:  scripts_hallucination/results/law_index.json
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alcv.law_index import build_index, LawIndex


def main():
    build_index()
    # 스모크 점검
    idx = LawIndex()
    checks = [
        ("근로기준법", "제43조", True),
        ("근로기준법", "제999조", False),
        ("형법", "제298조", True),
    ]
    print("\n[스모크 점검] (법령, 조문) → 존재여부")
    for law, art, expected in checks:
        got = idx.article_exists(law, art)
        mark = "OK" if got == expected else "X"
        print(f"  [{mark}] {law} {art}: {got} (기대 {expected})")


if __name__ == "__main__":
    main()
