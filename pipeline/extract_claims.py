"""2단계-b: 답변을 타입 태깅된 원자적 법률 클레임으로 분해 (방법론 v3 §3.1 Track A-(3)).

사용:
  python pipeline\\extract_claims.py --smoke      # 답변 10개만
  python pipeline\\extract_claims.py             # 전체
"""
import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A_PATH = ROOT / "data" / "collection" / "answers.jsonl"
OUT_PATH = ROOT / "data" / "collection" / "claims.jsonl"

CLAIM_TYPES = ["statute_citation", "case_citation", "numeric", "institution",
               "procedure", "interpretation"]

SYSTEM = "당신은 법률 텍스트를 사실 단위로 분해하는 전문가입니다. 반드시 JSON으로만 답합니다."

PROMPT = """아래 법률 상담 답변을 **원자적 클레임**(하나의 독립적으로 검증 가능한 사실 진술)으로 분해하세요.

[질문] {question}

[답변]
{answer}

규칙:
1. 각 클레임은 답변에 실제로 담긴 주장만 포함 (새 내용 추가 금지).
2. 대명사·생략은 질문 맥락으로 복원하여 클레임 단독으로 이해 가능하게 작성.
3. 각 클레임에 타입 1개 부여:
   - statute_citation: 특정 법령·조문 인용 ("~은 근로기준법 제60조에 규정")
   - case_citation: 특정 판례 인용
   - numeric: 기간·금액·비율·형량 등 수치 주장
   - institution: 기관명·연락처·관할 주장
   - procedure: 절차·방법·요건 주장
   - interpretation: 법리 해석·적용·결론 주장
4. 인사말, 일반적 권고("전문가와 상담하세요"), 검증 불가능한 의견은 제외.

JSON 형식: {{"claims": [{{"text": "...", "type": "..."}}, ...]}}"""


def load_done() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    return {json.loads(l)["a_id"] for l in open(OUT_PATH, encoding="utf-8") if l.strip()}


async def extract_one(ans: dict, semaphore) -> list[dict]:
    from llm_utils import chat_json
    result = await chat_json(SYSTEM, PROMPT.format(question=ans["question"],
                                                   answer=ans["answer"][:4000]),
                             semaphore=semaphore, temperature=0.0)
    if not result or not isinstance(result.get("claims"), list):
        return []
    rows = []
    for i, c in enumerate(result["claims"]):
        if not isinstance(c, dict) or not c.get("text"):
            continue
        ctype = c.get("type") if c.get("type") in CLAIM_TYPES else "interpretation"
        rows.append({
            "c_id": f"{ans['a_id']}__c{i:02d}",
            "a_id": ans["a_id"],
            "q_id": ans["q_id"],
            "condition": ans["condition"],
            "claim_text": c["text"].strip(),
            "claim_type": ctype,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    return rows


async def main(limit: int | None, concurrency: int):
    from llm_utils import cost_report
    answers = [json.loads(l) for l in open(A_PATH, encoding="utf-8") if l.strip()]
    if limit:
        answers = answers[:limit]
    done = load_done()
    todo = [a for a in answers if a["a_id"] not in done]
    print(f"기존 답변 {len(done)}건 분해됨, 신규 대상 {len(todo)}건 (동시성 {concurrency})")

    semaphore = asyncio.Semaphore(concurrency)
    n_claims = 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "a", encoding="utf-8") as fout:
        tasks = [asyncio.create_task(extract_one(a, semaphore)) for a in todo]
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            for row in await task:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_claims += 1
            fout.flush()
            if i % 50 == 0:
                print(f"  진행 {i}/{len(tasks)} (클레임 {n_claims})")
    print(f"완료: 클레임 {n_claims}개 → {OUT_PATH}")
    print(cost_report())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="답변 10개만")
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()
    asyncio.run(main(10 if args.smoke else None, args.concurrency))
