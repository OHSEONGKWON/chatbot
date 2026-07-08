"""베이스라인 3종 (API): vanilla judge / RAG judge / MetaQA — 전부 클레임 단위, gold 미접근.

  vanilla : 근거 없이 4o-mini 가 자체 지식으로 판정 (LLM-as-judge)
  ragjudge: BM25 top-3 근거 + 4o-mini 판정 (우리 방법에서 Stage1·인용조문 주입 제거)
  metaqa  : 클레임 → 메타질문 생성 → closed-book 자답 → 답과 클레임 대조 (legacy 이식)

사용:
  python pipeline\\baselines_run.py --method vanilla [--smoke]
  python pipeline\\baselines_run.py --method ragjudge [--smoke]
  python pipeline\\baselines_run.py --method metaqa [--smoke]
"""
import argparse
import asyncio
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLL = ROOT / "data" / "collection"

SYSTEM = "당신은 법률 사실 검증 전문가입니다. 반드시 JSON으로만 답합니다."

# 판정 루브릭은 우리 탐지기 Stage3 v2와 동일 (프롬프트 공정성: 방법 간 차이는 근거 접근만)
VANILLA_PROMPT = """당신의 한국 법률 지식에 근거해 아래 클레임을 판정하세요.

[클레임]
{claim}

판정 기준:
- SUPPORTED: 당신의 지식이 클레임을 명확히 지지함
- NOT_SUPPORTED: 당신의 지식과 클레임이 **같은 대상(같은 조문·제도·절차)**에 대해 **다르게** 규정함 (조문 번호·수치·주체·요건이 어긋남)
- NEI: 지식만으로는 판단 불가
주의: 단지 확신이 없다는 이유로 NOT_SUPPORTED를 고르지 말 것 — 그 경우는 NEI.
JSON 형식: {{"label": "..."}}"""

RAG_PROMPT = """아래 [근거]만을 기준으로 클레임을 판정하세요. 당신의 배경지식이 아니라 근거 텍스트가 유일한 판단 기준입니다.

[근거]
{evidence}

[클레임]
{claim}

판정 기준:
- SUPPORTED: 근거가 클레임을 명시적으로 지지함
- NOT_SUPPORTED: 근거가 클레임과 **같은 대상(같은 조문·제도·절차)**을 직접 다루면서 클레임과 **다르게** 규정함 (조문 번호·수치·주체·요건이 근거와 어긋남)
- NEI: 근거가 클레임의 대상을 직접 다루지 않음(주제만 관련), 또는 이 근거로는 판단 불가
주의: 근거에 해당 내용이 단지 '없다'는 이유로 NOT_SUPPORTED를 고르지 말 것 — 그 경우는 NEI.
JSON 형식: {{"label": "..."}}"""

META_Q_PROMPT = """아래 클레임을 검증하기 위한 단일 질문을 만드세요.
클레임: {claim}
JSON: {{"question": "..."}}"""

META_CMP_PROMPT = """[클레임] {claim}
[검증 답변] {meta_answer}

판정 기준:
- SUPPORTED: 검증 답변이 클레임을 명확히 지지함
- NOT_SUPPORTED: 검증 답변이 클레임과 **같은 대상**에 대해 **다르게** 서술함 (조문 번호·수치·주체·요건이 어긋남)
- NEI: 검증 답변만으로는 판단 불가
JSON 형식: {{"label": "..."}}"""

VALID = ("SUPPORTED", "NOT_SUPPORTED", "NEI")


async def run_vanilla(c, ctx, semaphore):
    from llm_utils import chat_json
    r = await chat_json(SYSTEM, VANILLA_PROMPT.format(claim=c["claim_text"]),
                        semaphore=semaphore, temperature=0.0)
    lab = (r or {}).get("label")
    return lab if lab in VALID else "NEI"


async def run_ragjudge(c, ctx, semaphore):
    from llm_utils import chat_json
    r = await chat_json(SYSTEM, RAG_PROMPT.format(evidence=ctx, claim=c["claim_text"]),
                        semaphore=semaphore, temperature=0.0)
    lab = (r or {}).get("label")
    return lab if lab in VALID else "NEI"


async def run_metaqa(c, ctx, semaphore):
    from llm_utils import chat_json, chat_text
    q = await chat_json(SYSTEM, META_Q_PROMPT.format(claim=c["claim_text"]),
                        semaphore=semaphore, temperature=0.0)
    if not q or not q.get("question"):
        return "NEI"
    ans = await chat_text("당신은 한국 법률 전문가입니다. 질문에 아는 대로 답하십시오.",
                          q["question"], semaphore=semaphore, temperature=0.0)
    if not ans:
        return "NEI"
    r = await chat_json(SYSTEM, META_CMP_PROMPT.format(claim=c["claim_text"], meta_answer=ans[:1500]),
                        semaphore=semaphore, temperature=0.0)
    lab = (r or {}).get("label")
    return lab if lab in VALID else "NEI"


RUNNERS = {"vanilla": run_vanilla, "ragjudge": run_ragjudge, "metaqa": run_metaqa}


async def main(method, limit, concurrency):
    from llm_utils import cost_report
    out_path = COLL / f"baseline_{method}.jsonl"
    claims = [json.loads(l) for l in open(COLL / "claims.jsonl", encoding="utf-8") if l.strip()]
    done = set()
    if out_path.exists():
        done = {json.loads(l)["c_id"] for l in open(out_path, encoding="utf-8") if l.strip()}
    claims = [c for c in claims if c["c_id"] not in done]
    if limit:
        claims = claims[:limit]

    # ragjudge 전용: detect_scores.jsonl 의 top_evidence 재사용 (동일 쿼리 BM25 top-3, 재계산 불필요)
    ctx_map = {}
    if method == "ragjudge":
        for l in open(COLL / "detect_scores.jsonl", encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                ctx_map[d["c_id"]] = "\n\n---\n\n".join(t[:700] for t in d["top_evidence"])

    print(f"[{method}] 대상 {len(claims)}건 (기존 {len(done)} 제외)")
    semaphore = asyncio.Semaphore(concurrency)
    runner = RUNNERS[method]
    with open(out_path, "a", encoding="utf-8") as fout:
        tasks = [asyncio.create_task(runner(c, ctx_map.get(c["c_id"]), semaphore)) for c in claims]
        for i, (c, task) in enumerate(zip(claims, tasks), 1):
            lab = await task
            fout.write(json.dumps({"c_id": c["c_id"], "verdict": lab}, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                fout.flush()
                print(f"  진행 {i}/{len(claims)}")
    print(f"완료 → {out_path}")
    print(cost_report())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(RUNNERS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()
    asyncio.run(main(args.method, 50 if args.smoke else None, args.concurrency))
