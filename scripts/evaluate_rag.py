"""
RAG 검색 성능 평가: BM25 vs Dense(E5) vs Hybrid

평가 지표:
  - Hit@1  : Top-1 결과가 정답 청크인지 (0 or 1)
  - NDCG@5 : 정답 청크의 Top-5 내 순위 기반 할인 누적 이득

사용법:
  python scripts/evaluate_rag.py
  python scripts/evaluate_rag.py --top-k 5 --batch-size 32
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

RAG_EVAL_PATH = REPO_ROOT / "data" / "evaluation" / "rag_eval.jsonl"
CORPUS_PATHS = [
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_law_chunks.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_case_chunks.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_manual_chunks.jsonl",
]
OUT_PATH = REPO_ROOT / "outputs" / "rag_eval_results.json"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


# ── 코퍼스 로드 ───────────────────────────────────────────────────────────────

def load_corpus() -> tuple[list[str], list[str]]:
    """(chunk_ids, texts) 반환."""
    ids, texts = [], []
    for path in CORPUS_PATHS:
        if not path.exists():
            print(f"[SKIP] {path.name}")
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                ids.append(item["chunk_id"])
                texts.append(item["text"])
    print(f"코퍼스 로드: {len(ids)}개 청크")
    return ids, texts


def load_eval_queries() -> list[dict]:
    items = []
    with RAG_EVAL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"평가 쿼리: {len(items)}개")
    return items


# ── BM25 ──────────────────────────────────────────────────────────────────────

class BM25:
    def __init__(self, corpus_texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = [self._tokenize(t) for t in corpus_texts]
        self.n = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / self.n if self.n else 1

        df: dict[str, int] = defaultdict(int)
        for doc in self.corpus:
            for term in set(doc):
                df[term] += 1
        self.idf: dict[str, float] = {
            term: math.log((self.n - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        return re.findall(r"[가-힣a-zA-Z0-9]+", text.lower())

    def score(self, query: str, doc_idx: int) -> float:
        tokens = self._tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        tf: dict[str, int] = defaultdict(int)
        for t in doc:
            tf[t] += 1
        s = 0.0
        for t in tokens:
            if t not in self.idf:
                continue
            freq = tf[t]
            s += self.idf[t] * (freq * (self.k1 + 1)) / (
                freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
        return s

    def retrieve(self, query: str, top_k: int) -> list[int]:
        scores = [self.score(query, i) for i in range(self.n)]
        return sorted(range(self.n), key=lambda i: scores[i], reverse=True)[:top_k]


# ── Dense (E5) ────────────────────────────────────────────────────────────────

def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    prefix: str = "",
) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i : i + batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        embs = mean_pool(out.last_hidden_state, enc["attention_mask"])
        embs = torch.nn.functional.normalize(embs, dim=-1)
        all_embs.append(embs.cpu().numpy())
        if (i // batch_size) % 20 == 0:
            print(f"  임베딩 {i + len(batch)}/{len(texts)}")
    return np.vstack(all_embs)


# ── 지표 계산 ─────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_ids: list[str], relevant_id: str, k: int) -> float:
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid == relevant_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def hit_at_k(ranked_ids: list[str], relevant_id: str, k: int) -> float:
    return 1.0 if relevant_id in ranked_ids[:k] else 0.0


def compute_metrics(
    queries: list[dict],
    ranked_results: list[list[str]],
    top_k: int,
) -> dict:
    ndcg_scores, hit1_scores = [], []
    for q, ranked in zip(queries, ranked_results):
        pos_id = q["positive_chunk_id"]
        ndcg_scores.append(ndcg_at_k(ranked, pos_id, top_k))
        hit1_scores.append(hit_at_k(ranked, pos_id, 1))
    return {
        "ndcg_at_5": round(float(np.mean(ndcg_scores)), 4),
        "hit_at_1": round(float(np.mean(hit1_scores)), 4),
        "n": len(queries),
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main(top_k: int = 5, embed_batch: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    corpus_ids, corpus_texts = load_corpus()
    queries = load_eval_queries()

    results: dict[str, dict] = {}

    # ── BM25 ──────────────────────────────────────────────────────────────────
    print("\n[BM25] 인덱스 구축 중...")
    bm25 = BM25(corpus_texts)
    bm25_ranked = []
    for i, q in enumerate(queries):
        top_idxs = bm25.retrieve(q["query"], top_k)
        bm25_ranked.append([corpus_ids[idx] for idx in top_idxs])
        if i % 50 == 0:
            print(f"  BM25 {i}/{len(queries)}")
    results["BM25"] = compute_metrics(queries, bm25_ranked, top_k)
    print(f"  BM25 결과: {results['BM25']}")

    # ── Dense (E5) ────────────────────────────────────────────────────────────
    print(f"\n[Dense] E5 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(device)
    model.eval()

    print("  코퍼스 임베딩 중...")
    corpus_embs = encode_texts(
        corpus_texts, tokenizer, model, device, embed_batch, prefix="passage: "
    )

    print("  쿼리 임베딩 중...")
    query_texts = [q["query"] for q in queries]
    query_embs = encode_texts(
        query_texts, tokenizer, model, device, embed_batch, prefix="query: "
    )

    # 코사인 유사도 (이미 L2 정규화됨)
    sim_matrix = query_embs @ corpus_embs.T  # (n_queries, n_corpus)

    dense_ranked = []
    for i, sim in enumerate(sim_matrix):
        top_idxs = np.argsort(sim)[::-1][:top_k].tolist()
        dense_ranked.append([corpus_ids[idx] for idx in top_idxs])
    results["Dense(E5)"] = compute_metrics(queries, dense_ranked, top_k)
    print(f"  Dense 결과: {results['Dense(E5)']}")

    # ── Hybrid (BM25 0.4 + Dense 0.6) ────────────────────────────────────────
    print("\n[Hybrid] BM25 + Dense 결합 중...")
    bm25_alpha = 0.4
    dense_alpha = 0.6

    # BM25 스코어 행렬
    bm25_scores = np.array([
        [bm25.score(q["query"], j) for j in range(len(corpus_ids))]
        for q in queries
    ])
    # Min-max 정규화
    bm25_min = bm25_scores.min(axis=1, keepdims=True)
    bm25_max = bm25_scores.max(axis=1, keepdims=True)
    bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min + 1e-9)

    dense_min = sim_matrix.min(axis=1, keepdims=True)
    dense_max = sim_matrix.max(axis=1, keepdims=True)
    dense_norm = (sim_matrix - dense_min) / (dense_max - dense_min + 1e-9)

    hybrid_scores = bm25_alpha * bm25_norm + dense_alpha * dense_norm
    hybrid_ranked = []
    for i, scores in enumerate(hybrid_scores):
        top_idxs = np.argsort(scores)[::-1][:top_k].tolist()
        hybrid_ranked.append([corpus_ids[idx] for idx in top_idxs])
    results["Hybrid"] = compute_metrics(queries, hybrid_ranked, top_k)
    print(f"  Hybrid 결과: {results['Hybrid']}")

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("=== RAG 검색 성능 비교 ===")
    print("=" * 50)
    header = f"{'방식':<15} {'Hit@1':>8} {'NDCG@5':>8}"
    print(header)
    print("-" * 35)
    for name, r in results.items():
        print(f"{name:<15} {r['hit_at_1']:>8.4f} {r['ndcg_at_5']:>8.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {OUT_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    main(top_k=args.top_k, embed_batch=args.batch_size)
