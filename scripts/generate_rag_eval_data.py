"""
RAG 평가 데이터 생성 스크립트

data/external_processed/rag_chunks/rag_corpus.jsonl 에서
OpenAI API를 사용하여 Query-Document 쌍을 자동 생성합니다.

출력 형식:
  {
    "query_id": "...",
    "query": "...",
    "relevant_chunk_ids": ["chunk_id_1", ...],
    "source_chunk_id": "chunk_id",     # 쿼리 생성 기반 청크
    "difficulty": "easy|medium|hard"
  }

사용법:
  python scripts/generate_rag_eval_data.py
  python scripts/generate_rag_eval_data.py --n-queries 100
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]

# 입력 데이터 경로 (우선순위 순)
INPUT_PATHS = [
    REPO_ROOT / "data" / "external_processed" / "rag_chunks" / "rag_corpus.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_law_chunks.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_case_chunks.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_manual_chunks.jsonl",
]
OUT_FILE = REPO_ROOT / "data" / "evaluation" / "rag_eval.jsonl"

SYSTEM_PROMPT = """당신은 한국 법률 검색 시스템의 평가 데이터를 만드는 전문가입니다.
주어진 법률 문서 청크를 읽고, 실제 사용자가 물어볼 법한 자연스러운 질문을 생성하세요.

규칙:
1. 질문은 반드시 주어진 문서 내용에 대한 답을 찾을 수 있어야 합니다
2. 질문은 한국어로, 실제 법률 상담 질문처럼 작성하세요
3. 난이도별로 각 1개씩 총 3개 질문 생성:
   - easy: 문서에서 직접 찾을 수 있는 사실 질문
   - medium: 문서 내용을 이해해야 답할 수 있는 질문
   - hard: 법률 지식과 문서 내용을 조합해야 하는 질문

JSON 형식으로만 응답:
{
  "questions": [
    {"difficulty": "easy", "query": "..."},
    {"difficulty": "medium", "query": "..."},
    {"difficulty": "hard", "query": "..."}
  ]
}"""


def load_chunks(path: Path, max_chunks: int = 2000) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                # chunk_id 필드 정규화
                if "chunk_id" not in rec and "id" in rec:
                    rec["chunk_id"] = rec["id"]
                if "text" not in rec:
                    continue
                # 너무 짧은 청크 제외
                if len(rec["text"]) < 100:
                    continue
                chunks.append(rec)
                if len(chunks) >= max_chunks:
                    break
    return chunks


def call_openai_generate(client, chunk_text: str) -> list[dict]:
    import openai

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"다음 법률 문서에 대한 질문을 생성하세요:\n\n{chunk_text[:800]}"},
            ],
            temperature=0.7,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("questions", [])
    except Exception as e:
        print(f"[WARN] OpenAI 호출 실패: {e}", file=sys.stderr)
        return []


def find_relevant_chunks(query: str, all_chunks: list[dict], source_id: str, top_k: int = 3) -> list[str]:
    """키워드 기반으로 관련 청크 ID 찾기 (평가용 레이블)."""
    # 소스 청크는 항상 포함
    relevant = [source_id]

    # 법령명/사건번호 추출
    law_pattern = re.compile(r"[가-힣]+법(?:\s*제\d+조)?")
    case_pattern = re.compile(r"\d{4}[가나다라마바사아자차카타파하도][가-힣\d]+\d+")

    laws_in_query = set(law_pattern.findall(query))
    cases_in_query = set(case_pattern.findall(query))

    if not laws_in_query and not cases_in_query:
        return relevant

    for chunk in all_chunks:
        if chunk.get("chunk_id", chunk.get("id")) == source_id:
            continue
        text = chunk.get("text", "")
        score = 0
        for law in laws_in_query:
            if law in text:
                score += 2
        for case in cases_in_query:
            if case in text:
                score += 3
        if score > 0:
            relevant.append((score, chunk.get("chunk_id", chunk.get("id"))))

    # 점수 기준 정렬 후 top_k
    extra = sorted(
        [(s, cid) for s, cid in relevant[1:] if isinstance(s, int)],
        key=lambda x: -x[0],
    )[:top_k - 1]

    return [relevant[0]] + [cid for _, cid in extra]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 평가용 Query-Document 쌍 생성")
    parser.add_argument("--n-queries", type=int, default=100, help="생성할 총 쿼리 수 (기본 100)")
    parser.add_argument("--input", type=Path, default=None, help="입력 JSONL 파일 (기본: 자동 선택)")
    parser.add_argument("--output", type=Path, default=OUT_FILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    import openai

    client = openai.OpenAI()

    # 입력 파일 선택
    input_path = args.input
    if input_path is None:
        for p in INPUT_PATHS:
            if p.exists():
                input_path = p
                break
    if input_path is None or not input_path.exists():
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    print(f"입력 파일: {input_path}")
    all_chunks = load_chunks(input_path)
    print(f"로드된 청크: {len(all_chunks)}개")

    random.seed(args.seed)
    # 쿼리 당 3개 난이도 × n_queries / 3 ≈ 청크 수
    n_source_chunks = max(1, args.n_queries // 3)
    sample_chunks = random.sample(all_chunks, min(n_source_chunks, len(all_chunks)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    query_id = 0

    with args.output.open("w", encoding="utf-8") as fout:
        for i, chunk in enumerate(sample_chunks):
            if written >= args.n_queries:
                break

            source_id = chunk.get("chunk_id", chunk.get("id", f"chunk_{i}"))
            print(f"  [{written}/{args.n_queries}] {source_id[:40]}...")

            questions = call_openai_generate(client, chunk["text"])
            if not questions:
                continue

            for q in questions:
                if written >= args.n_queries:
                    break
                if not q.get("query"):
                    continue

                relevant_ids = find_relevant_chunks(q["query"], all_chunks, source_id)

                record = {
                    "query_id": f"rag_eval_{query_id:04d}",
                    "query": q["query"],
                    "relevant_chunk_ids": relevant_ids,
                    "source_chunk_id": source_id,
                    "difficulty": q.get("difficulty", "medium"),
                    "source_text_preview": chunk["text"][:200],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                query_id += 1

            if args.delay > 0:
                time.sleep(args.delay)

    print(f"\n완료: {written}개 쿼리 → {args.output}")
    _print_stats(args.output)


def _print_stats(path: Path) -> None:
    from collections import Counter

    difficulty_counts: Counter = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            difficulty_counts[rec.get("difficulty", "unknown")] += 1

    total = sum(difficulty_counts.values())
    print(f"\n=== 데이터 통계 ===")
    print(f"총 쿼리: {total}")
    for d, c in sorted(difficulty_counts.items()):
        print(f"  {d:8s}: {c:4d} ({c/total*100:.1f}%)")


if __name__ == "__main__":
    main()
