"""
RAG 성능 평가 스크립트

BM25, Dense, 하이브리드 검색 시스템과 다양한 임베딩 모델을 비교합니다.

비교 시스템:
  - BM25 단독
  - Dense (각 임베딩 모델별)
  - Hybrid (BM25 + Dense)

임베딩 모델:
  - intfloat/multilingual-e5-large (현재 사용 중)
  - jhgan/ko-sroberta-multitask
  - BM-K/KoSimCSE-roberta-multitask
  - upskyy/kure-roberta-base

평가 메트릭: Hit@1, Hit@5, MRR@10, NDCG@10

사용법:
  python scripts/evaluate_rag.py
  python scripts/evaluate_rag.py --eval-data data/evaluation/rag_eval.jsonl
  python scripts/evaluate_rag.py --systems bm25 e5_large  # 특정 시스템만
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EVAL_DATA_PATH = REPO_ROOT / "data" / "evaluation" / "rag_eval.jsonl"
OUT_DIR = REPO_ROOT / "data" / "evaluation" / "results"

CORPUS_PATHS = [
    REPO_ROOT / "data" / "external_processed" / "rag_chunks" / "rag_corpus.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_law_chunks.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_case_chunks.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_manual_chunks.jsonl",
]

# 임베딩 모델 레지스트리
EMBEDDING_MODELS = {
    "e5_large": {
        "name": "multilingual-E5-large",
        "model_id": "intfloat/multilingual-e5-large",
        "prefix": "query: ",
    },
    "ko_sroberta": {
        "name": "ko-SRoBERTa-multitask",
        "model_id": "jhgan/ko-sroberta-multitask",
        "prefix": "",
    },
    "kosimcse": {
        "name": "KoSimCSE-RoBERTa",
        "model_id": "BM-K/KoSimCSE-roberta-multitask",
        "prefix": "",
    },
    "kure_roberta": {
        "name": "KURE-RoBERTa",
        "model_id": "upskyy/kure-roberta-base",
        "prefix": "",
    },
}


def load_jsonl(path: Path, max_records: int = 0) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if "chunk_id" not in rec and "id" in rec:
                    rec["chunk_id"] = rec["id"]
                records.append(rec)
                if max_records > 0 and len(records) >= max_records:
                    break
    return records


def load_corpus(max_per_file: int = 5000) -> list[dict]:
    """여러 JSONL 파일에서 corpus 로드."""
    corpus = []
    for path in CORPUS_PATHS:
        if path.exists():
            recs = load_jsonl(path, max_records=max_per_file)
            corpus.extend(recs)
            print(f"  {path.name}: {len(recs)}개")
    return corpus


def build_bm25_index(corpus: list[dict]):
    """BM25 인덱스 구축."""
    from rank_bm25 import BM25Okapi

    def tokenize(text: str) -> list[str]:
        # 한국어 형태소 단위 분리 (간단 버전: 어절 + n-gram)
        words = text.split()
        tokens = []
        for w in words:
            tokens.append(w)
            # bi-gram
            if len(w) > 2:
                for i in range(len(w) - 1):
                    tokens.append(w[i : i + 2])
        return tokens

    tokenized_corpus = [tokenize(doc.get("text", "")) for doc in corpus]
    return BM25Okapi(tokenized_corpus), tokenize


def bm25_search(query: str, bm25, tokenize_fn, corpus: list[dict], top_k: int = 10) -> list[str]:
    q_tokens = tokenize_fn(query)
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [corpus[i].get("chunk_id", corpus[i].get("id", str(i))) for i in ranked]


def build_dense_index(corpus: list[dict], model_info: dict):
    """Dense 임베딩 인덱스 구축."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print(f"  임베딩 모델 로딩: {model_info['model_id']}")
    model = SentenceTransformer(model_info["model_id"])

    texts = [doc.get("text", "")[:512] for doc in corpus]
    prefix = model_info.get("prefix", "")
    if prefix:
        texts = [prefix + t for t in texts]

    print(f"  임베딩 계산 중 ({len(texts)}개)...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    return model, embeddings, prefix


def dense_search(
    query: str,
    model,
    doc_embeddings,
    corpus: list[dict],
    prefix: str,
    top_k: int = 10,
) -> list[str]:
    import numpy as np

    q_text = prefix + query if prefix else query
    q_emb = model.encode([q_text], normalize_embeddings=True)[0]
    scores = doc_embeddings @ q_emb
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [corpus[i].get("chunk_id", corpus[i].get("id", str(i))) for i in ranked]


def hybrid_search(
    query: str,
    bm25,
    tokenize_fn,
    model,
    doc_embeddings,
    corpus: list[dict],
    prefix: str,
    top_k: int = 10,
    alpha: float = 0.5,
) -> list[str]:
    """BM25 + Dense 하이브리드 (RRF 앙상블)."""
    import numpy as np

    bm25_results = bm25_search(query, bm25, tokenize_fn, corpus, top_k * 2)
    dense_results = dense_search(query, model, doc_embeddings, corpus, prefix, top_k * 2)

    # Reciprocal Rank Fusion
    rrf_scores: dict[str, float] = {}
    k = 60  # RRF parameter

    for rank, chunk_id in enumerate(bm25_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rank + k)

    for rank, chunk_id in enumerate(dense_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rank + k)

    ranked = sorted(rrf_scores.keys(), key=lambda cid: -rrf_scores[cid])[:top_k]
    return ranked


def compute_rag_metrics(
    queries: list[dict],
    retrieval_fn,
    top_k: int = 10,
) -> dict:
    """RAG 평가 메트릭 계산."""
    hit_1 = hit_5 = hit_10 = 0
    mrr_total = ndcg_total = 0.0
    n = len(queries)

    for q in queries:
        relevant_ids = set(q.get("relevant_chunk_ids", []))
        if not relevant_ids:
            continue

        retrieved = retrieval_fn(q["query"])

        # Hit@k
        if any(cid in relevant_ids for cid in retrieved[:1]):
            hit_1 += 1
        if any(cid in relevant_ids for cid in retrieved[:5]):
            hit_5 += 1
        if any(cid in relevant_ids for cid in retrieved[:10]):
            hit_10 += 1

        # MRR@10
        for rank, cid in enumerate(retrieved[:10], start=1):
            if cid in relevant_ids:
                mrr_total += 1 / rank
                break

        # NDCG@10
        dcg = 0.0
        for rank, cid in enumerate(retrieved[:10], start=1):
            if cid in relevant_ids:
                dcg += 1 / math.log2(rank + 1)
        # ideal DCG
        ideal_hits = min(len(relevant_ids), 10)
        idcg = sum(1 / math.log2(i + 2) for i in range(ideal_hits))
        if idcg > 0:
            ndcg_total += dcg / idcg

    return {
        "hit@1": round(hit_1 / n, 4),
        "hit@5": round(hit_5 / n, 4),
        "hit@10": round(hit_10 / n, 4),
        "mrr@10": round(mrr_total / n, 4),
        "ndcg@10": round(ndcg_total / n, 4),
        "n": n,
    }


def print_table(all_results: dict[str, dict]) -> None:
    print("\n" + "=" * 80)
    print("RAG 성능 평가 결과")
    print("=" * 80)
    header = f"{'시스템':<35} {'Hit@1':>6} {'Hit@5':>6} {'Hit@10':>7} {'MRR@10':>7} {'NDCG@10':>8} {'N':>5}"
    print(header)
    print("-" * 80)

    for sys_name, metrics in all_results.items():
        row = (
            f"{sys_name:<35}"
            f" {metrics.get('hit@1', 0.0):>6.3f}"
            f" {metrics.get('hit@5', 0.0):>6.3f}"
            f" {metrics.get('hit@10', 0.0):>7.3f}"
            f" {metrics.get('mrr@10', 0.0):>7.3f}"
            f" {metrics.get('ndcg@10', 0.0):>8.3f}"
            f" {metrics.get('n', 0):>5}"
        )
        print(row)

    print("=" * 80)


def save_csv(all_results: dict[str, dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["시스템", "Hit@1", "Hit@5", "Hit@10", "MRR@10", "NDCG@10", "N"])
        for sys_name, metrics in all_results.items():
            writer.writerow([
                sys_name,
                metrics.get("hit@1", 0.0),
                metrics.get("hit@5", 0.0),
                metrics.get("hit@10", 0.0),
                metrics.get("mrr@10", 0.0),
                metrics.get("ndcg@10", 0.0),
                metrics.get("n", 0),
            ])
    print(f"CSV 저장: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 검색 성능 비교 평가")
    parser.add_argument("--eval-data", type=Path, default=EVAL_DATA_PATH)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["all"],
        help="평가할 시스템 (bm25, e5_large, ko_sroberta, kosimcse, kure_roberta, hybrid_e5) 또는 all",
    )
    parser.add_argument("--max-corpus", type=int, default=5000, help="코퍼스 최대 크기")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if not args.eval_data.exists():
        print(f"[ERROR] 평가 데이터 없음: {args.eval_data}", file=sys.stderr)
        print("먼저 generate_rag_eval_data.py 를 실행하세요.", file=sys.stderr)
        sys.exit(1)

    queries = load_jsonl(args.eval_data)
    print(f"쿼리 수: {len(queries)}")

    print("\n코퍼스 로딩 중...")
    corpus = load_corpus(args.max_corpus)
    print(f"총 코퍼스: {len(corpus)}개")

    all_results = {}

    run_all = "all" in args.systems

    # --- BM25 ---
    if run_all or "bm25" in args.systems:
        print("\n[BM25] 인덱스 구축 중...")
        try:
            bm25, tokenize_fn = build_bm25_index(corpus)
            bm25_fn = lambda q: bm25_search(q, bm25, tokenize_fn, corpus, args.top_k)
            metrics = compute_rag_metrics(queries, bm25_fn, args.top_k)
            all_results["BM25"] = metrics
            print(f"  Hit@1={metrics['hit@1']:.3f} Hit@5={metrics['hit@5']:.3f} MRR@10={metrics['mrr@10']:.3f}")
        except ImportError:
            print("  [SKIP] rank_bm25 패키지 없음 (pip install rank-bm25)")
        except Exception as e:
            print(f"  [ERROR] {e}")

    # --- Dense 모델별 ---
    dense_keys = [k for k in EMBEDDING_MODELS if run_all or k in args.systems]
    for emb_key in dense_keys:
        emb_info = EMBEDDING_MODELS[emb_key]
        print(f"\n[Dense: {emb_info['name']}] 인덱스 구축 중...")
        try:
            model, embeddings, prefix = build_dense_index(corpus, emb_info)

            dense_fn = lambda q, m=model, e=embeddings, p=prefix: dense_search(
                q, m, e, corpus, p, args.top_k
            )
            metrics = compute_rag_metrics(queries, dense_fn, args.top_k)
            all_results[f"Dense ({emb_info['name']})"] = metrics
            print(f"  Hit@1={metrics['hit@1']:.3f} Hit@5={metrics['hit@5']:.3f} MRR@10={metrics['mrr@10']:.3f}")

            # Hybrid (BM25 + Dense)
            if "bm25" in all_results or run_all:
                try:
                    bm25_for_hybrid, tok_fn = build_bm25_index(corpus)
                    hybrid_fn = lambda q, m=model, e=embeddings, p=prefix: hybrid_search(
                        q, bm25_for_hybrid, tok_fn, m, e, corpus, p, args.top_k
                    )
                    hybrid_metrics = compute_rag_metrics(queries, hybrid_fn, args.top_k)
                    all_results[f"Hybrid ({emb_info['name']})"] = hybrid_metrics
                    print(
                        f"  [Hybrid] Hit@1={hybrid_metrics['hit@1']:.3f} "
                        f"Hit@5={hybrid_metrics['hit@5']:.3f} MRR@10={hybrid_metrics['mrr@10']:.3f}"
                    )
                except Exception as e:
                    print(f"  [WARN] Hybrid 실패: {e}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("[ERROR] 평가된 시스템이 없습니다.", file=sys.stderr)
        sys.exit(1)

    print_table(all_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "rag_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 저장: {json_path}")

    save_csv(all_results, args.output_dir / "rag_table.csv")


if __name__ == "__main__":
    main()
