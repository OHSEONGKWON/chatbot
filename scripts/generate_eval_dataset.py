"""
평가 데이터셋 생성 스크립트

RAG 평가셋 200개 + 환각 테스트셋 800개를 GPT API로 생성.
asyncio.Semaphore(10)로 동시 10개 API 호출 병렬 처리.

출력:
  data/evaluation/rag_eval.jsonl         - RAG 평가셋
  data/evaluation/hallu_eval.jsonl       - 환각 탐지 테스트셋
  data/evaluation/generation_checkpoint.json - 체크포인트
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from openai import AsyncOpenAI

# ── 설정 ─────────────────────────────────────────────────────────────────────

RAG_EVAL_N        = 200   # RAG 평가 쿼리 수
HALLU_POSITIVE_N  = 400   # 환각 양성 (4타입 × 100)
HALLU_NEGATIVE_N  = 400   # 환각 음성 (정상 Q&A)
CHECKPOINT_EVERY  = 50    # 몇 개마다 체크포인트 저장
CONCURRENCY       = 10    # 동시 API 호출 수
RELEVANCE_REPEATS = 3     # 관련성 레이블 voting 횟수

BIO_SOURCES = [
    REPO_ROOT / "data" / "real_bio_data" / "New_Dataset" / "rag_law_chunks_bio.jsonl",
    REPO_ROOT / "data" / "real_bio_data" / "New_Dataset" / "rag_case_chunks_bio.jsonl",
    REPO_ROOT / "data" / "real_bio_data" / "New_Dataset" / "rag_manual_chunks_bio.jsonl",
]

OUT_DIR        = REPO_ROOT / "data" / "evaluation"
RAG_OUT        = OUT_DIR / "rag_eval.jsonl"
HALLU_OUT      = OUT_DIR / "hallu_eval.jsonl"
CHECKPOINT_OUT = OUT_DIR / "generation_checkpoint.json"

HALLU_TYPES = ["article_number_error", "forbidden_law_injection", "contact_mismatch", "semantic_error"]


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_chunks(seed: int = 42) -> list[dict]:
    """BIO JSONL에서 청크를 읽어 텍스트 복원."""
    chunks = []
    for path in BIO_SOURCES:
        if not path.exists():
            print(f"[SKIP] 파일 없음: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = "".join(rec["tokens"])
                if len(text) < 50:  # 너무 짧은 청크 제외
                    continue
                chunks.append({
                    "chunk_id": rec.get("chunk_id", ""),
                    "text": text,
                    "source_file": path.stem,
                })
    random.Random(seed).shuffle(chunks)
    print(f"총 {len(chunks)}개 청크 로드")
    return chunks


# ── 체크포인트 ────────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_OUT.exists():
        with CHECKPOINT_OUT.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"rag_done": [], "hallu_done": []}


def save_checkpoint(cp: dict) -> None:
    CHECKPOINT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_OUT.open("w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── RAG 평가셋 생성 ───────────────────────────────────────────────────────────

RAG_QUERY_SYSTEM = """당신은 한국 법률 전문가입니다.
주어진 법률 문서 청크를 읽고, 일반 시민이 법률 상담 시 할 법한 자연어 질문을 1개 생성하세요.

규칙:
- 질문에 법령 이름을 직접 언급하지 마세요 (검색 난이도 유지)
- 구체적인 상황을 포함한 질문이어야 합니다
- 질문만 출력하세요 (설명 없이)"""

RAG_RELEVANCE_SYSTEM = """당신은 법률 정보 검색 평가 전문가입니다.
주어진 [질문]과 [문서 청크]의 관련성을 판단하세요.

판단 기준:
- 2 (높음): 질문에 직접 답할 수 있는 핵심 내용 포함
- 1 (보통): 부분적으로 관련 있지만 완전한 답 불가
- 0 (없음): 질문과 무관

출력 형식 (JSON):
{
  "reasoning": "판단 근거를 2-3문장으로 설명",
  "score": 0 또는 1 또는 2
}"""

RAG_RELEVANCE_FEWSHOT = """예시 1:
질문: "회사에서 부당해고를 당했을 때 어떻게 대응할 수 있나요?"
청크: "[법령: 근로기준법] 제23조(해고 등의 제한) 사용자는 근로자에게 정당한 이유 없이 해고, 휴직, 정직, 전직, 감봉, 그 밖의 징벌을 하지 못한다."
→ {"reasoning": "질문은 부당해고 대응 방법에 관한 것이며, 이 청크는 해고 제한 조항으로 직접적인 법적 근거를 제공한다.", "score": 2}

예시 2:
질문: "음주운전 벌금이 얼마인가요?"
청크: "[법령: 근로기준법] 제34조(퇴직급여 제도) 사용자는 퇴직하는 근로자에게 급여를 지급하기 위하여 퇴직급여제도 중 하나 이상의 제도를 설정하여야 한다."
→ {"reasoning": "질문은 음주운전 벌금에 관한 것이고, 청크는 퇴직급여 관련 내용으로 전혀 관련이 없다.", "score": 0}

이제 판단하세요:"""


async def generate_rag_query(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    chunk: dict,
    model: str,
) -> dict | None:
    """청크로부터 RAG 평가 쿼리 1개 생성."""
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": RAG_QUERY_SYSTEM},
                    {"role": "user", "content": f"[문서 청크]\n{chunk['text'][:600]}"},
                ],
                temperature=0.8,
                max_tokens=150,
            )
            query = resp.choices[0].message.content.strip()
            return {
                "query_id": f"rag_{chunk['chunk_id']}",
                "query": query,
                "positive_chunk_id": chunk["chunk_id"],
                "positive_text": chunk["text"],
                "source_file": chunk["source_file"],
            }
        except Exception as e:
            print(f"[ERROR] 쿼리 생성 실패 ({chunk['chunk_id']}): {e}")
            return None


async def label_relevance(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    query: str,
    chunk_text: str,
    model: str,
) -> int:
    """쿼리-청크 관련성 점수 1회 판단 (0/1/2)."""
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": RAG_RELEVANCE_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"{RAG_RELEVANCE_FEWSHOT}\n\n"
                            f"질문: \"{query}\"\n"
                            f"청크: \"{chunk_text[:500]}\""
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            score = int(parsed.get("score", 0))
            return max(0, min(2, score))
        except Exception as e:
            print(f"[ERROR] 관련성 레이블 실패: {e}")
            return 0


async def label_relevance_voted(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    query: str,
    chunk_text: str,
    model: str,
    repeats: int = RELEVANCE_REPEATS,
) -> int:
    """3-repeat voting으로 관련성 점수 확정."""
    tasks = [label_relevance(client, sem, query, chunk_text, model) for _ in range(repeats)]
    scores = await asyncio.gather(*tasks)
    from collections import Counter
    return Counter(scores).most_common(1)[0][0]


async def build_rag_eval(
    client: AsyncOpenAI,
    chunks: list[dict],
    done_ids: set[str],
    model: str,
    n: int = RAG_EVAL_N,
) -> list[dict]:
    """RAG 평가셋 생성 (쿼리 생성 + 관련성 레이블링)."""
    sem = asyncio.Semaphore(CONCURRENCY)

    # 미완료 청크 선택
    candidates = [c for c in chunks if f"rag_{c['chunk_id']}" not in done_ids][:n]
    if not candidates:
        print("[RAG] 모두 완료됨 (체크포인트)")
        return []

    print(f"[RAG] 쿼리 생성 중... ({len(candidates)}개)")
    query_tasks = [generate_rag_query(client, sem, c, model) for c in candidates]

    results = []
    buffer = []
    cp = load_checkpoint()

    for i, coro in enumerate(asyncio.as_completed(query_tasks)):
        item = await coro
        if item is None:
            continue

        # 생성된 쿼리에 대해 positive 청크 관련성 레이블링 (검증용)
        pos_score = await label_relevance_voted(
            client, sem, item["query"], item["positive_text"], model
        )
        item["positive_relevance"] = pos_score

        results.append(item)
        buffer.append(item)
        cp["rag_done"].append(item["query_id"])

        if len(buffer) >= CHECKPOINT_EVERY:
            append_jsonl(RAG_OUT, buffer)
            save_checkpoint(cp)
            print(f"  [체크포인트] RAG {len(results)}/{len(candidates)}")
            buffer.clear()

    if buffer:
        append_jsonl(RAG_OUT, buffer)
        save_checkpoint(cp)

    print(f"[RAG] 완료: {len(results)}개")
    return results


# ── 환각 테스트셋 생성 ────────────────────────────────────────────────────────

HALLU_NORMAL_SYSTEM = """당신은 한국 법률 전문 상담사입니다.
아래 법률 청크를 근거로, 시민의 법률 질문과 정확한 답변을 생성하세요.

출력 형식 (JSON):
{
  "question": "시민이 할 법한 자연어 질문",
  "answer": "청크 내용에 근거한 정확한 법률 답변 (2-4문장)"
}"""

HALLU_INJECT_SYSTEM = """당신은 법률 환각 테스트 데이터 생성 전문가입니다.
아래 법률 청크를 근거로, 시민의 법률 질문과 [오류 유형]에 맞게 의도적으로 잘못된 답변을 생성하세요.

오류 유형 설명:
- article_number_error: 답변에서 실제 법조항 번호를 다른 번호로 교체 (예: 제23조 → 제32조)
- forbidden_law_injection: 실제로 존재하지 않는 판례 번호나 법령을 날조하여 답변에 삽입
- contact_mismatch: 답변에 등장하는 전화번호나 연락처를 다른 기관 번호로 교체 (예: 112 → 119)
- semantic_error: 법률 내용의 핵심 의미를 뒤집거나 조건을 잘못 서술 (예: "할 수 있다" → "할 수 없다")

출력 형식 (JSON):
{
  "question": "시민이 할 법한 자연어 질문",
  "answer": "오류가 포함된 잘못된 법률 답변 (2-4문장)",
  "injected_error": "어떤 오류를 어떻게 삽입했는지 설명"
}"""


async def generate_hallu_item(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    chunk: dict,
    hallu_type: str | None,
    model: str,
    idx: int,
) -> dict | None:
    """환각 테스트 항목 1개 생성. hallu_type=None이면 정상 답변."""
    async with sem:
        try:
            if hallu_type is None:
                system = HALLU_NORMAL_SYSTEM
                user_msg = f"[법률 청크]\n{chunk['text'][:600]}"
            else:
                system = HALLU_INJECT_SYSTEM
                user_msg = f"[오류 유형]: {hallu_type}\n[법률 청크]\n{chunk['text'][:600]}"

            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content.strip())

            item = {
                "hallu_id": f"hallu_{idx:04d}",
                "chunk_id": chunk["chunk_id"],
                "source_text": chunk["text"],
                "question": parsed.get("question", ""),
                "answer": parsed.get("answer", ""),
                "label": "hallucination" if hallu_type else "normal",
                "hallu_type": hallu_type or "none",
            }
            if hallu_type:
                item["injected_error"] = parsed.get("injected_error", "")
            return item
        except Exception as e:
            print(f"[ERROR] 환각 항목 생성 실패 (idx={idx}, type={hallu_type}): {e}")
            return None


async def build_hallu_eval(
    client: AsyncOpenAI,
    chunks: list[dict],
    done_ids: set[str],
    model: str,
) -> list[dict]:
    """환각 테스트셋 생성 (정상 400 + 오류 4타입×100)."""
    sem = asyncio.Semaphore(CONCURRENCY)

    # 작업 목록 구성
    tasks_spec: list[tuple[dict, str | None, int]] = []  # (chunk, hallu_type, idx)
    idx = 0

    # 음성 (정상): 400개
    neg_chunks = [c for c in chunks if f"hallu_neg_{c['chunk_id']}" not in done_ids]
    random.shuffle(neg_chunks)
    for c in neg_chunks[:HALLU_NEGATIVE_N]:
        tasks_spec.append((c, None, idx))
        idx += 1

    # 양성 (4타입 × 100): 400개
    pos_chunks = [c for c in chunks if c not in neg_chunks[:HALLU_NEGATIVE_N]]
    random.shuffle(pos_chunks)
    per_type = HALLU_POSITIVE_N // len(HALLU_TYPES)
    for hallu_type in HALLU_TYPES:
        key_prefix = f"hallu_{hallu_type}_"
        filtered = [c for c in pos_chunks if f"{key_prefix}{c['chunk_id']}" not in done_ids]
        for c in filtered[:per_type]:
            tasks_spec.append((c, hallu_type, idx))
            idx += 1

    if not tasks_spec:
        print("[HALLU] 모두 완료됨 (체크포인트)")
        return []

    print(f"[HALLU] 생성 중... ({len(tasks_spec)}개)")

    coros = [generate_hallu_item(client, sem, c, ht, model, i) for c, ht, i in tasks_spec]

    results = []
    buffer = []
    cp = load_checkpoint()

    for coro in asyncio.as_completed(coros):
        item = await coro
        if item is None:
            continue

        results.append(item)
        buffer.append(item)

        hallu_type = item["hallu_type"]
        chunk_id = item["chunk_id"]
        key = f"hallu_{hallu_type}_{chunk_id}" if hallu_type != "none" else f"hallu_neg_{chunk_id}"
        cp["hallu_done"].append(key)

        if len(buffer) >= CHECKPOINT_EVERY:
            append_jsonl(HALLU_OUT, buffer)
            save_checkpoint(cp)
            print(f"  [체크포인트] HALLU {len(results)}/{len(tasks_spec)}")
            buffer.clear()

    if buffer:
        append_jsonl(HALLU_OUT, buffer)
        save_checkpoint(cp)

    print(f"[HALLU] 완료: {len(results)}개")
    return results


# ── 통계 출력 ─────────────────────────────────────────────────────────────────

def print_stats(out_path: Path, label: str) -> None:
    if not out_path.exists():
        return
    items = []
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"\n[{label}] 총 {len(items)}개 저장: {out_path}")

    if label == "RAG":
        scores = [i.get("positive_relevance", -1) for i in items]
        from collections import Counter
        print(f"  관련성 분포: {dict(Counter(scores))}")
    elif label == "HALLU":
        from collections import Counter
        types = Counter(i.get("hallu_type", "?") for i in items)
        print(f"  타입 분포: {dict(types)}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

async def main_async(model: str, resume: bool) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY 환경변수 없음")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)

    if not resume and CHECKPOINT_OUT.exists():
        CHECKPOINT_OUT.unlink()
    if not resume:
        RAG_OUT.unlink(missing_ok=True)
        HALLU_OUT.unlink(missing_ok=True)

    cp = load_checkpoint()
    done_rag_ids = set(cp.get("rag_done", []))
    done_hallu_ids = set(cp.get("hallu_done", []))

    print(f"모델: {model}")
    print(f"재개 모드: {resume}")
    print(f"체크포인트 - RAG: {len(done_rag_ids)}개, HALLU: {len(done_hallu_ids)}개\n")

    chunks = load_chunks()

    # Step 1: RAG 평가셋
    print("=" * 50)
    print("Step 1: RAG 평가셋 생성")
    print("=" * 50)
    await build_rag_eval(client, chunks, done_rag_ids, model)

    # Step 2: 환각 테스트셋
    print("\n" + "=" * 50)
    print("Step 2: 환각 테스트셋 생성")
    print("=" * 50)
    await build_hallu_eval(client, chunks, done_hallu_ids, model)

    # 최종 통계
    print("\n" + "=" * 50)
    print_stats(RAG_OUT, "RAG")
    print_stats(HALLU_OUT, "HALLU")
    print("=" * 50)
    print("데이터셋 생성 완료")


def main() -> None:
    parser = argparse.ArgumentParser(description="평가 데이터셋 생성")
    parser.add_argument("--model", default="gpt-4o-mini", help="GPT 모델명")
    parser.add_argument("--resume", action="store_true", help="체크포인트에서 재개")
    args = parser.parse_args()

    asyncio.run(main_async(args.model, args.resume))


if __name__ == "__main__":
    main()
