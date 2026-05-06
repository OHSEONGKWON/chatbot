import argparse
import asyncio
import json
import re
from typing import Dict, List, Optional

from pipeline import pipeline
from modules.ner_checker import ner_checker


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def select_context_chunks(chunks: List[Dict], chunk_id: Optional[str], max_chunks: int) -> List[Dict]:
    if not chunks:
        return []

    if not chunk_id:
        return chunks[:max_chunks]

    selected = next((c for c in chunks if c.get("chunk_id") == chunk_id), None)
    if not selected:
        return chunks[:max_chunks]

    doc_id = selected.get("metadata", {}).get("doc_id")
    if not doc_id:
        return [selected]

    same_doc = [c for c in chunks if c.get("metadata", {}).get("doc_id") == doc_id]
    return same_doc[:max_chunks] if same_doc else [selected]


def build_default_answer(chunks: List[Dict]) -> str:
    if not chunks:
        return "관련 법령에 따르면 신청 기한은 60일 이내입니다."

    text = chunks[0].get("text", "")
    metadata = chunks[0].get("metadata", {})
    law_name = str(metadata.get("law_name") or "관련 법령").split("\n")[0].strip() or "관련 법령"
    article_id = str(metadata.get("article_id") or "").strip()

    m = re.search(r"(\d+)\s*일", text)
    if m:
        wrong_days = int(m.group(1)) + 30
        return f"{law_name} {article_id}에 따르면 신청은 {wrong_days}일 이내에 해야 합니다."

    return f"{law_name} {article_id}에 따르면 신청은 60일 이내에 해야 합니다."


def parse_args():
    parser = argparse.ArgumentParser(description="Run LawsGuard operational scenario")
    parser.add_argument("--mode", choices=["pipeline", "ner-only"], default="pipeline")
    parser.add_argument("--user-id", default="operational-user")
    parser.add_argument("--user-input", default="성추행 피해를 당했는데 어떻게 대응해야 하나요?")
    parser.add_argument("--rag-file", default="data/real_data/New_Dataset/rag_law_chunks.jsonl")
    parser.add_argument("--chunk-id", default="")
    parser.add_argument("--max-context-chunks", type=int, default=12)
    parser.add_argument("--answer", default="", help="LLM answer to verify in ner-only mode. If omitted, a synthetic answer is used.")
    return parser.parse_args()


async def run_pipeline_mode(user_id: str, user_input: str):
    result = await pipeline.process(user_id=user_id, user_input=user_input)

    print("\n=== Operational Scenario (pipeline) ===")
    print(f"User ID: {user_id}")
    print(f"User Input: {user_input}")
    print("\n=== Pipeline Result ===")
    print(f"step_reached: {result.step_reached}")
    print(f"needs_requery: {result.needs_requery}")
    print(f"session_ended: {result.session_ended}")
    print(f"consistency_score: {result.consistency_score}")
    print(f"was_ner_corrected: {result.was_ner_corrected}")
    print("\n=== Response ===")
    print(result.response_text)


async def run_ner_only_mode(args):
    all_chunks = load_jsonl(args.rag_file)
    rag_chunks = select_context_chunks(all_chunks, args.chunk_id or None, args.max_context_chunks)

    llm_answer = args.answer.strip() if args.answer else build_default_answer(rag_chunks)
    ner_result = await ner_checker.check_and_correct(answer=llm_answer, rag_docs=rag_chunks)

    print("\n=== Operational Scenario (ner-only) ===")
    print(f"RAG file: {args.rag_file}")
    print(f"Context chunks: {len(rag_chunks)}")
    print(f"Input answer: {llm_answer}")
    print("\n=== NER Check Result ===")
    print(json.dumps(
        {
            "was_corrected": ner_result.was_corrected,
            "corrected_answer": ner_result.corrected_answer,
            "mismatched_entities": ner_result.mismatched_entities,
        },
        ensure_ascii=False,
        indent=2,
    ))


async def main():
    args = parse_args()
    if args.mode == "pipeline":
        await run_pipeline_mode(user_id=args.user_id, user_input=args.user_input)
        return
    await run_ner_only_mode(args)


if __name__ == "__main__":
    asyncio.run(main())
