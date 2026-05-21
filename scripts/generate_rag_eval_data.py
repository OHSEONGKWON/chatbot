"""
RAG 평가 데이터 생성 (template-based, API 없음)

rag_corpus.jsonl에서 청크 샘플링 후 템플릿 기반 질문 생성.

사용법:
  python scripts/generate_rag_eval_data.py
  python scripts/generate_rag_eval_data.py --n-queries 150
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATHS = [
    REPO_ROOT / "data" / "external_processed" / "rag_chunks" / "rag_corpus.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_law_chunks.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_case_chunks.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_manual_chunks.jsonl",
]
OUT_FILE = REPO_ROOT / "data" / "evaluation" / "rag_eval.jsonl"

# ── 엔티티 추출 정규식 ────────────────────────────────────────────────────────
LAW_ARTICLE_RE = re.compile(
    r"([가-힣]{2,12}법(?:률|전)?)\s*(제\d+조(?:의\d+)?)", re.UNICODE
)
LAW_NAME_RE = re.compile(r"[가-힣]{2,12}법(?:률|전)?", re.UNICODE)
ORG_RE = re.compile(
    r"(?:대법원|헌법재판소|고등법원|지방법원|가정법원|행정법원|특허법원|"
    r"대검찰청|검찰청|고용노동부|법무부|국토교통부|기획재정부|보건복지부|"
    r"행정안전부|경찰청|국세청|공정거래위원회|금융감독원)",
    re.UNICODE,
)
DATE_RE = re.compile(r"\d{4}년\s*\d{1,2}월|\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.", re.UNICODE)
AMOUNT_RE = re.compile(r"\d+\s*(?:억|만)\s*원|\d{1,3}(?:,\d{3})+\s*원", re.UNICODE)

# ── 질문 템플릿 (type → [(difficulty, template), ...]) ───────────────────────
TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "law_article": [
        ("easy",   "{law}의 {article}에서 규정하는 내용은 무엇인가요?"),
        ("medium", "{law}의 {article}에서 정한 요건을 충족하지 못하면 어떻게 되나요?"),
        ("hard",   "{law}의 {article}을 위반했을 때 적용되는 법적 제재는 무엇인가요?"),
    ],
    "law_only": [
        ("easy",   "{law}에서 규정하는 주요 내용은 무엇인가요?"),
        ("medium", "{law}에서 정하는 의무 사항과 권리는 무엇인가요?"),
        ("hard",   "{law}의 적용 범위와 예외 조항은 어떻게 되나요?"),
    ],
    "org": [
        ("easy",   "{org}은 어떤 법적 역할을 담당하고 있나요?"),
        ("medium", "{org}에서 처리하는 주요 법률 절차는 무엇인가요?"),
        ("hard",   "{org}의 결정이나 처분에 불복하는 법적 절차는 어떻게 되나요?"),
    ],
    "date": [
        ("easy",   "{date}에 적용되는 법률 규정은 무엇인가요?"),
        ("medium", "{date}를 기준으로 발생하는 법적 효력은 무엇인가요?"),
        ("hard",   "{date} 전후의 법률 적용 차이는 무엇인가요?"),
    ],
    "generic": [
        ("easy",   "이 법률 조항에서 규정하는 핵심 내용은 무엇인가요?"),
        ("medium", "이 규정의 적용 대상과 요건은 어떻게 되나요?"),
        ("hard",   "이 규정과 관련된 법적 분쟁이 발생할 경우 어떻게 처리되나요?"),
    ],
}


def extract_entities(text: str) -> dict:
    ent: dict = {}
    m = LAW_ARTICLE_RE.search(text)
    if m:
        ent["law"] = m.group(1)
        ent["article"] = m.group(2)
        return ent
    m = LAW_NAME_RE.search(text)
    if m:
        ent["law"] = m.group()
        return ent
    m = ORG_RE.search(text)
    if m:
        ent["org"] = m.group()
        return ent
    m = DATE_RE.search(text)
    if m:
        ent["date"] = m.group()
    m = AMOUNT_RE.search(text)
    if m:
        ent["amount"] = m.group()
    return ent


def generate_queries(text: str) -> list[tuple[str, str]]:
    ent = extract_entities(text)
    if "law" in ent and "article" in ent:
        key = "law_article"
    elif "law" in ent:
        key = "law_only"
    elif "org" in ent:
        key = "org"
    elif "date" in ent:
        key = "date"
    else:
        key = "generic"

    results: list[tuple[str, str]] = []
    for difficulty, tmpl in TEMPLATES[key]:
        try:
            query = tmpl.format(**ent)
        except KeyError:
            # 키가 없으면 generic 대체
            generic = TEMPLATES["generic"]
            query = generic[len(results) % len(generic)][1]
        results.append((difficulty, query))
    return results


def build_keyword_index(chunks: list[dict]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        cid = chunk["chunk_id"]
        text = chunk.get("text", "")
        for m in LAW_ARTICLE_RE.finditer(text):
            index[m.group(1) + "|" + m.group(2)].append(cid)
        for m in LAW_NAME_RE.finditer(text):
            index[m.group()].append(cid)
    return dict(index)


def find_relevant_chunks(query: str, source_id: str, index: dict, top_k: int = 3) -> list[str]:
    scores: dict[str, int] = defaultdict(int)
    for m in LAW_ARTICLE_RE.finditer(query):
        for cid in index.get(m.group(1) + "|" + m.group(2), []):
            if cid != source_id:
                scores[cid] += 3
    for m in LAW_NAME_RE.finditer(query):
        for cid in index.get(m.group(), []):
            if cid != source_id:
                scores[cid] += 1
    # 점수 있는 청크 전부 relevant로 포함 (최소 top_k개, 상한 없음)
    extras = [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]
    return [source_id] + extras


def load_chunks(max_n: int = 5000) -> list[dict]:
    chunks: list[dict] = []
    for path in INPUT_PATHS:
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = rec.get("text", "")
                    if len(text) < 100:
                        continue
                    chunks.append({
                        "chunk_id": rec.get("chunk_id") or rec.get("id", ""),
                        "text": text,
                        "metadata": rec.get("metadata", {}),
                    })
                    if len(chunks) >= max_n:
                        break
        except Exception as e:
            print(f"[WARN] 로드 실패: {path.name} - {e}", file=sys.stderr)
        gc.collect()
        if len(chunks) >= max_n:
            break
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=150)
    parser.add_argument("--output", type=Path, default=OUT_FILE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("청크 로드 중...")
    all_chunks = load_chunks()
    print(f"로드된 청크: {len(all_chunks)}개")
    if not all_chunks:
        print("[ERROR] 청크 없음", file=sys.stderr)
        sys.exit(1)

    print("키워드 인덱스 구축 중...", flush=True)
    index = build_keyword_index(all_chunks)
    print(f"인덱스 키워드: {len(index)}개")

    rng = random.Random(args.seed)
    queries_per_chunk = 3
    n_source = (args.n_queries + queries_per_chunk - 1) // queries_per_chunk
    source_chunks = rng.sample(all_chunks, min(n_source, len(all_chunks)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    print(f"쿼리 생성 중 (목표: {args.n_queries}개)...", flush=True)
    with args.output.open("w", encoding="utf-8") as fout:
        for i, chunk in enumerate(source_chunks):
            if written >= args.n_queries:
                break
            cid = chunk["chunk_id"]
            for difficulty, query in generate_queries(chunk["text"]):
                if written >= args.n_queries:
                    break
                relevant = find_relevant_chunks(query, cid, index)
                fout.write(json.dumps({
                    "query_id": f"rag_eval_{written:04d}",
                    "query": query,
                    "relevant_chunk_ids": relevant,
                    "source_chunk_id": cid,
                    "difficulty": difficulty,
                    "source_text_preview": chunk["text"][:200],
                }, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
            if i % 100 == 0:
                gc.collect()

    all_chunks = index = None
    gc.collect()

    print(f"\n완료: {written}개 쿼리 → {args.output}")
    _print_stats(args.output)


def _print_stats(path: Path) -> None:
    from collections import Counter
    diff_counts: Counter = Counter()
    total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            diff_counts[rec.get("difficulty", "unknown")] += 1
    print(f"\n=== RAG 통계 ===\n총 쿼리: {total}")
    for d, c in sorted(diff_counts.items()):
        print(f"  {d:8s}: {c:4d}")


if __name__ == "__main__":
    main()
