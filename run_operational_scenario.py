import argparse
import json
import re
from typing import Dict, List, Optional

from main import LawsGuardPipeline


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
    parser = argparse.ArgumentParser(description="Run LawsGuard operational scenario with real RAG chunks")
    parser.add_argument("--rag-file", default="data/real_data/New_Dataset/rag_law_chunks.jsonl")
    parser.add_argument("--chunk-id", default="")
    parser.add_argument("--max-context-chunks", type=int, default=12)
    parser.add_argument("--answer", default="", help="Actual LLM answer to verify. If omitted, a synthetic answer is used.")
    return parser.parse_args()


def main():
    args = parse_args()
    all_chunks = load_jsonl(args.rag_file)
    rag_chunks = select_context_chunks(all_chunks, args.chunk_id or None, args.max_context_chunks)

    llm_answer = args.answer.strip() if args.answer else build_default_answer(rag_chunks)

    pipeline = LawsGuardPipeline()
    result = pipeline.run(llm_answer, rag_chunks)

    print("\n=== Operational Scenario ===")
    print(f"RAG file: {args.rag_file}")
    print(f"Context chunks: {len(rag_chunks)}")
    print(f"Input answer: {llm_answer}")
    print("\n=== Pipeline Result ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
