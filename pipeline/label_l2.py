"""3단계 L2: 근거 기반 silver 라벨링 (방법론 v3 §3.1 Track A-(4)).

라벨러는 탐지기와 달리 gold 소스 청크를 제공받는다(정보 비대칭 → 순환성 완화).
답변 단위로 클레임을 배칭하고, 근거 = gold 청크 + BM25 top-3 + 인용 조문 전문.
self-consistency 3표결(온도 0.7)로 SUPPORTED / NOT_SUPPORTED / NEI 판정.

사용:
  python pipeline\\label_l2.py --smoke      # 답변 10개만
  python pipeline\\label_l2.py             # 전체
"""
import argparse
import asyncio
import collections
import json
from datetime import datetime, timezone
from pathlib import Path

from generate_answers import build_bm25, tokenize
from law_db import LawDB, extract_citations

ROOT = Path(__file__).resolve().parent.parent
COLL = ROOT / "data" / "collection"
OUT_PATH = COLL / "labels_l2.jsonl"

N_VOTES = 3

SYSTEM = "당신은 법률 사실 검증 전문가입니다. 오직 제시된 근거만으로 판단하며, 반드시 JSON으로만 답합니다."

PROMPT = """아래 [근거]만을 기준으로 각 클레임을 판정하세요. 당신의 배경지식이 아니라 근거 텍스트가 유일한 판단 기준입니다.

[근거]
{evidence}

[클레임 목록]
{claims}

판정 기준:
- SUPPORTED: 근거가 클레임을 명시적으로 지지함
- NOT_SUPPORTED: 근거와 모순되거나, 근거상 명백히 틀림 (예: 근거의 수치·조문·기관과 다름)
- NEI: 이 근거만으로는 판단 불가 (근거에 관련 내용 없음)

JSON 형식: {{"labels": [{{"i": 0, "label": "SUPPORTED"}}, ...]}} (클레임 개수만큼, 순서대로)"""


def build_evidence(ans: dict, claims: list[dict], gold: dict, bm25, docs, db: LawDB) -> str:
    parts = [f"<근거 1: 질문의 원 출처 문서>\n{gold['gold_text'][:1000]}"]
    query = ans["question"] + " " + ans["answer"][:500]
    for j, t in enumerate(bm25.get_top_n(tokenize(query), docs, n=3), 2):
        parts.append(f"<근거 {j}: 검색 문서>\n{t[:800]}")
    # 클레임들이 인용한 조문 전문 (최대 6개)
    seen = set()
    for c in claims:
        for law, art in extract_citations(c["claim_text"]):
            text = db.article_text(law, art)
            if text and (law, art) not in seen and len(seen) < 6:
                seen.add((law, art))
                parts.append(f"<근거: {law} {art} 전문>\n{text[:700]}")
    return "\n\n".join(parts)


async def label_answer(ans, claims, gold, bm25, docs, db, semaphore) -> list[dict]:
    from llm_utils import chat_json
    claims_txt = "\n".join(f"{i}. [{c['claim_type']}] {c['claim_text']}"
                           for i, c in enumerate(claims))
    evidence = build_evidence(ans, claims, gold, bm25, docs, db)
    prompt = PROMPT.format(evidence=evidence, claims=claims_txt)

    votes = [collections.Counter() for _ in claims]
    results = await asyncio.gather(*[
        chat_json(SYSTEM, prompt, semaphore=semaphore, temperature=0.7)
        for _ in range(N_VOTES)])
    for r in results:
        if not r or not isinstance(r.get("labels"), list):
            continue
        for item in r["labels"]:
            i = item.get("i")
            lab = item.get("label")
            if isinstance(i, int) and 0 <= i < len(claims) \
                    and lab in ("SUPPORTED", "NOT_SUPPORTED", "NEI"):
                votes[i][lab] += 1

    rows = []
    for i, c in enumerate(claims):
        if not votes[i]:
            final = "NEI"
        else:
            top, top_n = votes[i].most_common(1)[0]
            final = top if top_n >= 2 else "NEI"  # 과반 없으면 NEI
        rows.append({"c_id": c["c_id"], "l2_label": final,
                     "votes": dict(votes[i]),
                     "created_at": datetime.now(timezone.utc).isoformat()})
    return rows


def load_done() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    return {json.loads(l)["c_id"] for l in open(OUT_PATH, encoding="utf-8") if l.strip()}


async def main(limit, concurrency):
    from llm_utils import cost_report
    questions = {q["q_id"]: q for q in (json.loads(l) for l in open(COLL / "questions.jsonl", encoding="utf-8") if l.strip())}
    answers = {a["a_id"]: a for a in (json.loads(l) for l in open(COLL / "answers.jsonl", encoding="utf-8") if l.strip())}
    all_claims = [json.loads(l) for l in open(COLL / "claims.jsonl", encoding="utf-8") if l.strip()]
    l1 = {json.loads(l)["c_id"]: json.loads(l)["l1_verdict"]
          for l in open(COLL / "labels_l1.jsonl", encoding="utf-8") if l.strip()}
    done = load_done()

    # L1 확정·기존 라벨 제외, 답변 단위 그룹핑
    by_answer = collections.defaultdict(list)
    for c in all_claims:
        if l1.get(c["c_id"]) == "not_supported" or c["c_id"] in done:
            continue
        by_answer[c["a_id"]].append(c)
    jobs = list(by_answer.items())
    if limit:
        jobs = jobs[:limit]
    print(f"대상 답변 {len(jobs)}건 / 클레임 {sum(len(v) for _, v in jobs)}개 "
          f"(L1 확정 {sum(1 for v in l1.values() if v=='not_supported')}건 제외, 3표결)")

    db = LawDB()
    bm25, docs = build_bm25()
    semaphore = asyncio.Semaphore(concurrency)
    n = 0
    with open(OUT_PATH, "a", encoding="utf-8") as fout:
        tasks = [asyncio.create_task(
            label_answer(answers[aid], cls, questions[cls[0]["q_id"]], bm25, docs, db, semaphore))
            for aid, cls in jobs]
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            for row in await task:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
            fout.flush()
            if i % 25 == 0:
                print(f"  진행 {i}/{len(tasks)} (클레임 {n})")
    print(f"완료: 클레임 {n}개 라벨 → {OUT_PATH}")
    print(cost_report())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="답변 10개만")
    ap.add_argument("--concurrency", type=int, default=15)
    args = ap.parse_args()
    asyncio.run(main(10 if args.smoke else None, args.concurrency))
