"""2단계-a: 질문에 대한 LLM 답변 생성 (방법론 v3 §3.1 Track A-(2)).

두 조건으로 자연 발생 환각을 유도한다:
  closed_book — 근거 없이 답변 (파라메트릭 지식의 환각)
  weak_rag    — BM25 top-2만 제공 (불완전 근거 오용 환각)

사용:
  python pipeline\\generate_answers.py --smoke      # 질문 10개 × 2조건
  python pipeline\\generate_answers.py             # 전체
"""
import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "real_data" / "New_Dataset"
Q_PATH = ROOT / "data" / "collection" / "questions.jsonl"
OUT_PATH = ROOT / "data" / "collection" / "answers.jsonl"

SYSTEM = ("당신은 한국 법률 상담 챗봇입니다. 질문에 대해 근거 법령과 조문 번호, "
          "관련 판례, 구체적인 기간·금액·기관을 들어 확신 있게 답하십시오. "
          "답을 모르더라도 아는 범위에서 최선을 다해 구체적으로 답하십시오.")

RAG_TEMPLATE = """[참고자료]
{context}

[질문]
{question}

참고자료를 활용하되, 부족한 부분은 아는 지식으로 보충하여 답하십시오."""


def tokenize(text: str) -> list[str]:
    """한글 문자 바이그램 + 영숫자 토큰 (간단 BM25용)."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text)
    hangul = re.sub(r"[^가-힣]", "", text)
    tokens += [hangul[i:i + 2] for i in range(len(hangul) - 1)]
    return tokens


def build_bm25():
    docs = []
    for fname in ["rag_law_chunks.jsonl", "rag_case_chunks.jsonl", "rag_manual_chunks.jsonl"]:
        with open(CORPUS_DIR / fname, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    docs.append(d["text"])
    print(f"BM25 인덱스 구축 중... ({len(docs):,} 청크)")
    return BM25Okapi([tokenize(t) for t in docs]), docs


def load_done() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    return {json.loads(l)["a_id"] for l in open(OUT_PATH, encoding="utf-8") if l.strip()}


async def answer_one(q: dict, condition: str, context: str | None, semaphore) -> dict | None:
    from llm_utils import chat_text
    if condition == "weak_rag":
        user = RAG_TEMPLATE.format(context=context, question=q["question"])
    else:
        user = q["question"]
    text = await chat_text(SYSTEM, user, semaphore=semaphore, temperature=0.7)
    if not text:
        return None
    return {
        "a_id": f"{q['q_id']}__{condition}",
        "q_id": q["q_id"],
        "condition": condition,
        "question": q["question"],
        "answer": text.strip(),
        "rag_context": context,
        "gen_model": "gpt-4o-mini",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


async def main(limit: int | None, concurrency: int):
    from llm_utils import cost_report
    questions = [json.loads(l) for l in open(Q_PATH, encoding="utf-8") if l.strip()]
    if limit:
        questions = questions[:limit]
    done = load_done()

    bm25, docs = build_bm25()
    jobs = []
    for q in questions:
        for cond in ["closed_book", "weak_rag"]:
            if f"{q['q_id']}__{cond}" in done:
                continue
            ctx = None
            if cond == "weak_rag":
                top = bm25.get_top_n(tokenize(q["question"]), docs, n=2)
                ctx = "\n\n---\n\n".join(t[:1500] for t in top)
            jobs.append((q, cond, ctx))
    print(f"기존 {len(done)}건, 신규 대상 {len(jobs)}건 (동시성 {concurrency})")

    semaphore = asyncio.Semaphore(concurrency)
    ok = 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "a", encoding="utf-8") as fout:
        tasks = [asyncio.create_task(answer_one(q, c, ctx, semaphore)) for q, c, ctx in jobs]
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            row = await task
            if row:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                ok += 1
            if i % 50 == 0:
                print(f"  진행 {i}/{len(tasks)} (성공 {ok})")
    print(f"완료: 신규 {ok}건 → {OUT_PATH}")
    print(cost_report())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="질문 10개만")
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()
    asyncio.run(main(10 if args.smoke else None, args.concurrency))
