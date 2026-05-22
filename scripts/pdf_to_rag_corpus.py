"""PDF -> RAG corpus + evaluation templates

Usage:
    python scripts/pdf_to_rag_corpus.py --input data/external_raw/legal_pdfs --out data/external_processed/rag_chunks/rag_corpus.jsonl --eval data/evaluation --workers 4

This script:
 - scans input folders for PDF files (under laws/ and cases/)
 - extracts text (pdfplumber or pymupdf if available)
 - splits law PDFs by article (`제\d+조`) when possible, otherwise by paragraph
 - splits case PDFs by paragraph/line blocks
 - writes chunk JSONL to output file and per-chunk files
 - generates three evaluation template JSONL files under eval_dir:
   - rag_eval_template.jsonl (queries with empty golden_doc_ids)
   - ner_eval_template.jsonl (texts for manual NER annotation)
   - hallucination_eval_template.jsonl (answers + rag_docs placeholders)

Notes:
 - The script produces templates for manual annotation; it does not auto-label gold answers.
 - Designed for iterating quickly with small sample sets before full-scale processing.
"""
from __future__ import annotations
import os
import re
import json
import argparse
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import fitz
except Exception:
    fitz = None


def extract_text_from_pdf(path: str) -> str:
    path = str(path)
    # debug: report available backends
    try:
        print(f"DBG: extract_text_from_pdf called for {path}")
        print(f"DBG: pdfplumber available={pdfplumber is not None}, fitz available={fitz is not None}")
    except Exception:
        # avoid failing on printing unicode paths
        print("DBG: extract_text_from_pdf called (path contains non-encodable chars)")

    if pdfplumber is not None:
        try:
            with pdfplumber.open(path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages)
        except Exception as e:
            import traceback
            print("DBG: pdfplumber failed:")
            traceback.print_exc()
    if fitz is not None:
        try:
            doc = fitz.open(path)
            pages = []
            for p in doc:
                pages.append(p.get_text())
            return "\n".join(pages)
        except Exception as e:
            import traceback
            print("DBG: fitz failed:")
            traceback.print_exc()
    raise RuntimeError("No PDF backend available (install pdfplumber or pymupdf) or file unreadable: %s" % path)


def split_by_article(text: str) -> List[str]:
    # Split by Korean article markers like '제20조' (keep marker with text)
    parts = re.split(r'(?=제\s*\d+\s*조)', text)
    parts = [p.strip() for p in parts if p and len(p.strip()) > 30]
    if len(parts) <= 1:
        # fallback: split by double newline paragraphs
        parts = [p.strip() for p in re.split(r'\n{2,}', text) if p and len(p.strip()) > 30]
    return parts


def split_case_text(text: str) -> List[str]:
    # heuristic: split by double newlines or by headings like '【' or '○'
    parts = [p.strip() for p in re.split(r'\n{2,}|\n\s*○|【', text) if p and len(p.strip()) > 30]
    return parts


def normalize_filename(name: str) -> str:
    name = re.sub(r'[^0-9a-zA-Z가-힣\-_\. ]+', '_', name)
    return name


def make_chunk_id(source_name: str, idx: int) -> str:
    base = normalize_filename(source_name)
    return f"{base}__chunk_{idx:04d}"


def write_jsonl(path: Path, records: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def collect_pdfs(root: Path) -> List[Path]:
    pdfs = []
    for p in root.rglob('*.pdf'):
        pdfs.append(p)
    return sorted(pdfs)


def _extract_and_chunk_pdf(pdf: Path) -> List[dict]:
    try:
        text = extract_text_from_pdf(str(pdf))
    except Exception as e:
        print(f"WARN: failed to extract {pdf}: {e}")
        return []

    # choose split method based on parent folder name
    parent = pdf.parent.name.lower()
    if 'law' in parent or 'laws' in parent or '법' in parent:
        parts = split_by_article(text)
    else:
        parts = split_case_text(text)

    records = []
    for i, part in enumerate(parts):
        cid = make_chunk_id(pdf.stem, i + 1)
        meta = {
            'id': cid,
            'text': part,
            'metadata': {
                'source_file': pdf.name,
                'source_path': str(pdf),
                'origin_dir': str(pdf.parent.name),
            }
        }
        # attempt to extract a law name or article id
        m = re.search(r'(제\s*\d+\s*조[^\n]*)', part)
        if m:
            meta['metadata']['article_id'] = m.group(1).strip()
        # Try to infer law name from beginning of doc if present
        head = part.strip()[:200]
        lawname_match = re.search(r'^(?:\s*|)([가-힣\s]{2,40}법)', head)
        if lawname_match:
            meta['metadata']['law_name'] = lawname_match.group(1).strip()

        records.append(meta)
    return records


def build_corpus(input_dir: Path, out_chunks_file: Path, workers: int = 1, executor_kind: str = 'thread') -> List[dict]:
    pdfs = collect_pdfs(input_dir)
    chunks = []
    if workers <= 1:
        for pdf in pdfs:
            chunks.extend(_extract_and_chunk_pdf(pdf))
    else:
        executor_cls = ThreadPoolExecutor if executor_kind == 'thread' else ProcessPoolExecutor
        with executor_cls(max_workers=workers) as executor:
            future_map = {executor.submit(_extract_and_chunk_pdf, pdf): pdf for pdf in pdfs}
            for future in as_completed(future_map):
                pdf = future_map[future]
                try:
                    chunks.extend(future.result())
                except Exception as e:
                    print(f"WARN: failed to process {pdf}: {e}")

    # stable output order for reproducibility
    chunks.sort(key=lambda item: (item['metadata'].get('source_file', ''), item['id']))

    # write combined JSONL
    write_jsonl(out_chunks_file, chunks)
    print(f"Wrote {len(chunks)} chunks to {out_chunks_file}")
    return chunks


def build_eval_templates(chunks: List[dict], eval_dir: Path, max_rag_queries: int = 50, max_ner: int = 300, max_hallu: int = 200):
    eval_dir.mkdir(parents=True, exist_ok=True)
    # RAG queries: use first sentence of chunk as a query candidate
    rag_queries = []
    for c in chunks[: max_rag_queries * 4]:
        text = c.get('text','')
        q = text.split('\n')[0].strip()
        if len(q) > 20:
            rag_queries.append({'id': f"q_{c['id']}", 'query': q, 'golden_doc_ids': [], 'notes': c['metadata'].get('source_file','')})
        if len(rag_queries) >= max_rag_queries:
            break

    write_jsonl(eval_dir / 'rag_eval_template.jsonl', rag_queries)

    # NER templates: collect sentences
    ner_recs = []
    for c in chunks:
        sents = re.split(r'[\.\?\!\n]+', c.get('text',''))
        for s in sents:
            s = s.strip()
            if 40 <= len(s) <= 600:
                ner_recs.append({'id': f'ner_{len(ner_recs):06d}', 'text': s, 'entities': [], 'source': c['metadata'].get('source_file','')})
            if len(ner_recs) >= max_ner:
                break
        if len(ner_recs) >= max_ner:
            break

    write_jsonl(eval_dir / 'ner_eval_template.jsonl', ner_recs)

    # Hallucination templates: take short answers and attach candidate rag_docs (ids)
    hallu_recs = []
    for c in chunks:
        text = c.get('text','').strip()
        if len(text) < 40 or len(text) > 800:
            continue
        # create an answer record referencing this chunk as a rag_doc
        hallu_recs.append({
            'id': f'h_{len(hallu_recs):06d}',
            'answer': text,
            'rag_docs': [ {'id': c['id'], 'text': c['text'], 'metadata': c.get('metadata',{})} ],
            'expected_mismatches': [],
            'no_mismatch': False,
            'source': c['metadata'].get('source_file','')
        })
        if len(hallu_recs) >= max_hallu:
            break

    write_jsonl(eval_dir / 'hallucination_eval_template.jsonl', hallu_recs)
    print(f"Wrote templates: rag={len(rag_queries)}, ner={len(ner_recs)}, hallucination={len(hallu_recs)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='data/external_raw/legal_pdfs', help='input folder with PDFs')
    parser.add_argument('--out', '-o', default='data/external_processed/rag_chunks/rag_corpus.jsonl', help='output combined chunks jsonl')
    parser.add_argument('--eval', '-e', default='data/evaluation', help='evaluation templates output dir')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1), help='number of worker threads/processes')
    parser.add_argument('--executor', choices=['thread', 'process'], default='thread', help='parallel executor type')
    parser.add_argument('--max_rag', type=int, default=50)
    parser.add_argument('--max_ner', type=int, default=300)
    parser.add_argument('--max_hallu', type=int, default=200)
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_file = Path(args.out)
    eval_dir = Path(args.eval)

    print(f"Scanning PDFs in {input_dir} ...")
    chunks = build_corpus(input_dir, out_file, workers=args.workers, executor_kind=args.executor)
    build_eval_templates(chunks, eval_dir, max_rag_queries=args.max_rag, max_ner=args.max_ner, max_hallu=args.max_hallu)


if __name__ == '__main__':
    main()
