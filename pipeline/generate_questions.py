"""1단계: 코퍼스 청크 기반 질문 생성 (방법론 v3 §3.1 Track A-(1)).

청크를 gold 근거로 기록하면서, 그 청크만으로 답할 수 있는 실무형 법률 질문을
GPT-4o-mini로 병렬 생성한다. 체크포인트(jsonl 증분 기록)·재개 지원.

사용:
  python pipeline\\generate_questions.py --smoke        # 15건 스모크
  python pipeline\\generate_questions.py --n 400        # 본 생성
  python pipeline\\generate_questions.py --dry-run      # API 없이 샘플링만 검증
"""
import argparse
import asyncio
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "real_data" / "New_Dataset"
OUT_PATH = ROOT / "data" / "collection" / "questions.jsonl"

CORPUS_FILES = {  # source_type -> 파일명
    "statute": "rag_law_chunks.jsonl",
    "case": "rag_case_chunks.jsonl",
    "manual": "rag_manual_chunks.jsonl",
}
MIX = {"statute": 0.4, "case": 0.3, "manual": 0.3}  # 질문 수 배분
MIN_CHARS = 250  # 헤더-only 청크 제거

# 자기완결적이지 않은 질문 걸러내기 (문서 참조 표현)
BAD_PATTERNS = re.compile(r"이 (문서|자료|판결|조문|글)|위 (문서|자료|판결|내용)|본 (문서|자료)|해당 문서")

SYSTEM = "당신은 한국 법률 상담 질문 데이터셋을 구축하는 전문가입니다. 반드시 JSON으로만 답합니다."

PROMPT = """아래 법률 자료를 읽고, **이 자료만으로 완전하고 정확하게 답할 수 있는** 법률 질문 1개를 만드세요.

[자료]
{text}

요구사항:
1. 일반인이 변호사나 상담 챗봇에게 실제로 물어볼 법한 자연스러운 한국어 질문.
2. 자료를 언급하지 말 것("이 문서에서", "위 조문에 따르면" 금지). 자료 없이 읽어도 이해되는 자기완결적 질문.
3. 질문의 정답 근거가 자료 안에 명시적으로 존재해야 함. answer_core는 자료에 쓰인 내용만으로 작성하고, 자료에 없는 지식을 보태지 말 것.
4. answer_core에는 자료에 근거한 핵심 정답을 1~2문장으로 요약.
5. 자료가 법률·제도·절차 정보(법령 내용, 판례 법리, 권리·의무, 신고·구제 절차)를 담고 있지 않으면(예: 심리상담 기법, 교육 프로그램 진행법, 빈칸 서식) 질문을 만들지 말고 {{"skip": true}}만 반환.

JSON 형식: {{"question": "...", "answer_core": "..."}} 또는 {{"skip": true}}"""


def load_chunks(source_type: str) -> list[dict]:
    chunks = []
    with open(CORPUS_DIR / CORPUS_FILES[source_type], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if len(d["text"]) < MIN_CHARS:
                continue
            if source_type == "statute" and not re.match(
                    r"제\d+조", d.get("metadata", {}).get("article_id") or ""):
                continue  # 부칙·개정문 등 조문 형식이 아닌 청크 제외
            if source_type == "case" and re.search(r"참조조문|참조판례|주문", d["chunk_id"]):
                continue  # 조문·판례 나열 섹션은 질문 소재로 부적합
            d["source_type"] = source_type
            chunks.append(d)
    return chunks


def sample_chunks(n: int, seed: int, used_ids: set[str]) -> list[dict]:
    rng = random.Random(seed)
    picked = []
    for st, ratio in MIX.items():
        pool = [c for c in load_chunks(st) if c["chunk_id"] not in used_ids]
        k = min(round(n * ratio), len(pool))
        picked += rng.sample(pool, k)
    rng.shuffle(picked)
    return picked


def load_done_ids() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    return {json.loads(l)["gold_chunk_id"] for l in open(OUT_PATH, encoding="utf-8") if l.strip()}


async def generate_one(chunk: dict, semaphore) -> dict | None:
    from llm_utils import chat_json
    result = await chat_json(SYSTEM, PROMPT.format(text=chunk["text"][:3000]),
                             semaphore=semaphore)
    if not result or result.get("skip") or not result.get("question") \
            or not result.get("answer_core"):
        return None
    if BAD_PATTERNS.search(result["question"]):
        return None  # 자기완결성 위반 → 드롭 (풀이 충분하므로 재생성 불필요)
    return {
        "q_id": f"q_{chunk['chunk_id']}",
        "question": result["question"].strip(),
        "answer_core": result["answer_core"].strip(),
        "gold_chunk_id": chunk["chunk_id"],
        "gold_source_type": chunk["source_type"],
        "gold_text": chunk["text"],
        "gold_metadata": chunk.get("metadata", {}),
        "gen_model": "gpt-4o-mini",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


async def main(n: int, seed: int, concurrency: int):
    from llm_utils import cost_report
    used = load_done_ids()
    todo = sample_chunks(n, seed, used)[: max(0, n - len(used))]
    print(f"기존 {len(used)}건, 신규 생성 대상 {len(todo)}건 (동시성 {concurrency})")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(concurrency)
    ok = 0
    with open(OUT_PATH, "a", encoding="utf-8") as fout:
        tasks = [asyncio.create_task(generate_one(c, semaphore)) for c in todo]
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            row = await task
            if row:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                ok += 1
            if i % 25 == 0:
                print(f"  진행 {i}/{len(tasks)} (성공 {ok})")
    print(f"완료: 신규 {ok}건 → {OUT_PATH}")
    print(cost_report())


def dry_run(n: int, seed: int):
    from collections import Counter
    picked = sample_chunks(n, seed, load_done_ids())
    print(f"샘플 {len(picked)}건, 유형 분포: {Counter(c['source_type'] for c in picked)}")
    for c in picked[:3]:
        print(f"\n--- {c['chunk_id']} ({c['source_type']}, {len(c['text'])}자)")
        print(c["text"][:200].replace("\n", " "))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--smoke", action="store_true", help="15건만 생성")
    ap.add_argument("--dry-run", action="store_true", help="API 호출 없이 샘플링만 확인")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()

    n = 15 if args.smoke else args.n
    if args.dry_run:
        dry_run(n, args.seed)
    else:
        asyncio.run(main(n, args.seed, args.concurrency))
