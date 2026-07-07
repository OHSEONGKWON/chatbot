"""SelfCheckGPT 베이스라인용 샘플 답변 생성 (원 답변과 동일 조건, 온도 1.0, N개).

사용:
  python pipeline\\generate_samples.py --smoke     # 답변 10개만
  python pipeline\\generate_samples.py            # 전체 (782 × N)
"""
import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from generate_answers import SYSTEM, RAG_TEMPLATE

ROOT = Path(__file__).resolve().parent.parent
COLL = ROOT / "data" / "collection"
OUT_PATH = COLL / "selfcheck_samples.jsonl"

N_SAMPLES = 3


async def sample_one(ans, k, semaphore):
    from llm_utils import chat_text
    if ans["condition"] == "weak_rag":
        user = RAG_TEMPLATE.format(context=ans["rag_context"], question=ans["question"])
    else:
        user = ans["question"]
    text = await chat_text(SYSTEM, user, semaphore=semaphore, temperature=1.0)
    if not text:
        return None
    return {"s_id": f"{ans['a_id']}__s{k}", "a_id": ans["a_id"], "sample": text.strip(),
            "created_at": datetime.now(timezone.utc).isoformat()}


async def main(limit, concurrency):
    from llm_utils import cost_report
    answers = [json.loads(l) for l in open(COLL / "answers.jsonl", encoding="utf-8") if l.strip()]
    if limit:
        answers = answers[:limit]
    done = set()
    if OUT_PATH.exists():
        done = {json.loads(l)["s_id"] for l in open(OUT_PATH, encoding="utf-8") if l.strip()}
    jobs = [(a, k) for a in answers for k in range(N_SAMPLES)
            if f"{a['a_id']}__s{k}" not in done]
    print(f"샘플 생성 대상 {len(jobs)}건 (기존 {len(done)} 제외)")

    semaphore = asyncio.Semaphore(concurrency)
    ok = 0
    with open(OUT_PATH, "a", encoding="utf-8") as fout:
        tasks = [asyncio.create_task(sample_one(a, k, semaphore)) for a, k in jobs]
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            row = await task
            if row:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                ok += 1
            if i % 100 == 0:
                print(f"  진행 {i}/{len(tasks)} (성공 {ok})")
    print(f"완료: {ok}건 → {OUT_PATH}")
    print(cost_report())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()
    asyncio.run(main(10 if args.smoke else None, args.concurrency))
