"""탐지기 Stage 3: NLI 불확실 구간만 LLM 근거-대조 판정 + 최종 verdict 산출 (방법론 v3 §4).

게이트 (detect_scores.jsonl 의 원점수 기준):
  stage1 == refuted            → NOT_SUPPORTED (확정)
  ent_max >= --ent-hi          → SUPPORTED (Stage 2 확정)
  ent_max <= --ent-lo          → NEI (Stage 2 확정: 지지 근거 없음)
  그 외                        → Stage 3 (LLM, 검색 근거만 제시)

사용:
  python pipeline\\detect_stage3.py --smoke
  python pipeline\\detect_stage3.py
"""
import argparse
import asyncio
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLL = ROOT / "data" / "collection"
IN_PATH = COLL / "detect_scores.jsonl"
OUT_PATH = COLL / "detect_final.jsonl"

SYSTEM = "당신은 법률 사실 검증 전문가입니다. 오직 제시된 근거만으로 판단하며, 반드시 JSON으로만 답합니다."

PROMPT = """아래 [근거]만을 기준으로 클레임을 판정하세요. 배경지식이 아니라 근거 텍스트가 유일한 판단 기준입니다.

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


async def judge(row, claim_text, semaphore):
    from llm_utils import chat_json
    evidence = "\n\n---\n\n".join(row["top_evidence"])
    r = await chat_json(SYSTEM, PROMPT.format(evidence=evidence, claim=claim_text),
                        semaphore=semaphore, temperature=0.0)
    lab = (r or {}).get("label")
    return lab if lab in ("SUPPORTED", "NOT_SUPPORTED", "NEI") else "NEI"


async def main(limit, ent_hi, ent_lo, concurrency):
    from llm_utils import cost_report
    claims = {c["c_id"]: c for c in (json.loads(l) for l in open(COLL / "claims.jsonl", encoding="utf-8") if l.strip())}
    scores = [json.loads(l) for l in open(IN_PATH, encoding="utf-8") if l.strip()]
    done = set()
    if OUT_PATH.exists():
        done = {json.loads(l)["c_id"] for l in open(OUT_PATH, encoding="utf-8") if l.strip()}

    finals, queue = [], []
    for r in scores:
        if r["c_id"] in done:
            continue
        if r["stage1"] == "refuted":
            finals.append({"c_id": r["c_id"], "verdict": "NOT_SUPPORTED", "stage": 1,
                           "ent_max": r["ent_max"]})
        elif r["ent_max"] >= ent_hi:
            finals.append({"c_id": r["c_id"], "verdict": "SUPPORTED", "stage": 2,
                           "ent_max": r["ent_max"]})
        elif r["ent_max"] <= ent_lo:
            finals.append({"c_id": r["c_id"], "verdict": "NEI", "stage": 2,
                           "ent_max": r["ent_max"]})
        else:
            queue.append(r)
    if limit:
        queue = queue[:limit]
    print(f"확정 {len(finals)} (S1 {sum(1 for f in finals if f['stage']==1)}) | Stage3 대상 {len(queue)}")

    semaphore = asyncio.Semaphore(concurrency)
    with open(OUT_PATH, "a", encoding="utf-8") as fout:
        for f in finals:
            fout.write(json.dumps(f, ensure_ascii=False) + "\n")
        tasks = [asyncio.create_task(judge(r, claims[r["c_id"]]["claim_text"], semaphore))
                 for r in queue]
        for i, (r, task) in enumerate(zip(queue, [*tasks]), 1):
            lab = await task
            fout.write(json.dumps({"c_id": r["c_id"], "verdict": lab, "stage": 3,
                                   "ent_max": r["ent_max"]}, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                fout.flush()
                print(f"  Stage3 진행 {i}/{len(queue)}")
    print(f"완료 → {OUT_PATH}")
    print(cost_report())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Stage3 50건만")
    ap.add_argument("--ent-hi", type=float, default=0.70)
    ap.add_argument("--ent-lo", type=float, default=0.10)
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()
    asyncio.run(main(50 if args.smoke else None, args.ent_hi, args.ent_lo, args.concurrency))
