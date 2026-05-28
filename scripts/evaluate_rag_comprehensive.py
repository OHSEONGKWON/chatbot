"""
RAG 검색 성능 평가 스크립트 (Phase 3 - Step 2).

지원 입력 형식:
  A) rag_eval.jsonl          — positive_chunk_id 단일 정답 형식 (LLM 합성)
  B) golden_candidates.jsonl — golden_doc_ids 다중 정답+relevance 형식 (자동 키워드)
  C) rag_eval_golden.jsonl   — golden_doc_ids 다중 정답, 수동 3계층 검토 완료본 (권장)

계산 메트릭:
  Hit@1, Hit@3, Hit@5, MRR, NDCG@5 (graded: rel=2 핵심근거, rel=1 보조근거)

실행:
  python scripts/evaluate_rag_comprehensive.py
  python scripts/evaluate_rag_comprehensive.py --eval golden_reviewed  # 수동 검토 완료본 (권장)
  python scripts/evaluate_rag_comprehensive.py --eval golden           # 자동 키워드 후보
  python scripts/evaluate_rag_comprehensive.py --top-k 10             # 검색 범위 확장
  python scripts/evaluate_rag_comprehensive.py --skip-zero            # relevance=0 항목 제외
  python scripts/evaluate_rag_comprehensive.py --compare-all          # BM25 vs Dense(E5) vs BM25+Reranker 동시 비교 (논문용)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.rag import retriever

OUTPUT_PATH = REPO_ROOT / "outputs" / "rag_eval_results.json"

# ── 파일 경로 ─────────────────────────────────────────────────────────────────
EVAL_FILES = {
    "bio":             REPO_ROOT / "data" / "evaluation" / "rag_eval.jsonl",
    "remapped":        REPO_ROOT / "data" / "evaluation" / "rag_eval_remapped.jsonl",
    "golden":          REPO_ROOT / "data" / "evaluation" / "golden_candidates.jsonl",
    "golden_reviewed": REPO_ROOT / "data" / "evaluation" / "rag_eval_golden.jsonl",
}
RESULTS_DIR = REPO_ROOT / "results"


# ── 메트릭 계산 ───────────────────────────────────────────────────────────────

def _hit_at_k(retrieved_ids: list[str], golden_ids: set[str], k: int) -> float:
    return float(any(cid in golden_ids for cid in retrieved_ids[:k]))


def _mrr(retrieved_ids: list[str], golden_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in golden_ids:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved_ids: list[str], golden_rel: dict[str, int], k: int) -> float:
    """NDCG@k. golden_rel = {chunk_id: relevance(0/1/2)}."""
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], 1):
        rel = golden_rel.get(cid, 0)
        dcg += (2**rel - 1) / math.log2(rank + 1)

    # Ideal DCG: 최대 relevance 순으로 정렬
    ideal_rels = sorted(golden_rel.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, 1))
    return dcg / idcg if idcg > 0 else 0.0


# ── 데이터 로더 ───────────────────────────────────────────────────────────────

_LABOR_KW  = ["임금", "근로", "해고", "노동", "최저임금", "퇴직", "알바", "계약서", "주휴"]
_SEXUAL_KW = ["성폭력", "성희롱", "강제추행", "불법촬영", "성범죄", "추행"]


def _infer_domain(query: str, positive_text: str) -> str:
    text = query + " " + positive_text
    if any(k in text for k in _LABOR_KW):
        return "노동"
    if any(k in text for k in _SEXUAL_KW):
        return "성폭력"
    return "기타"


def _load_bio(path: Path, skip_zero: bool, domain_filter: bool = False) -> list[dict[str, Any]]:
    """positive_chunk_id 형식 로드 → 내부 표준 형식으로 변환."""
    items = []
    skipped_domain = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            if not row.get("positive_chunk_id"):
                continue
            if skip_zero and row.get("positive_relevance", 1) == 0:
                continue
            rel    = row.get("positive_relevance", 1)
            query  = row["query"]
            domain = _infer_domain(query, row.get("positive_text", ""))
            if domain_filter and domain == "기타":
                skipped_domain += 1
                continue
            items.append({
                "id":             row.get("query_id", ""),
                "query":          query,
                "category":       row.get("category", "") or domain,
                "source":         row.get("source_file", ""),
                "golden_doc_ids": {row["positive_chunk_id"]: max(1, rel)},
                "format":         "bio",
            })
    if skipped_domain:
        print(f"  [domain_filter] 기타 도메인 {skipped_domain}개 제외")
    return items


def _load_golden(path: Path, skip_zero: bool) -> list[dict[str, Any]]:
    """golden_doc_ids 형식 로드."""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            golden = row.get("golden_doc_ids") or {}
            if not golden:
                continue
            if skip_zero:
                golden = {k: v for k, v in golden.items() if v > 0}
            if not golden:
                continue
            items.append({
                "id":       row.get("id", ""),
                "query":    row["query"],
                "category": row.get("category", ""),
                "source":   "golden_candidates",
                "golden_doc_ids": golden,
                "format": "golden",
            })
    return items


# ── chunk_id 정규화 ────────────────────────────────────────────────────────────

def _normalize_id(cid: str) -> str:
    """__part2 같은 suffix 제거해 base ID로 통일."""
    return cid.split("__part")[0] if "__part" in cid else cid


def _retrieved_ids(docs: list[dict[str, Any]]) -> list[str]:
    return [str(doc.get("chunk_id") or "") for doc in docs if doc.get("chunk_id")]


# ── 평가 실행 ─────────────────────────────────────────────────────────────────

def evaluate(
    items: list[dict[str, Any]],
    top_k: int,
    rerank: bool = False,
    rerank_model: str = "",
) -> dict[str, Any]:
    _reranker = None
    if rerank:
        if rerank_model == "llm":
            from src.modules.reranker import llm_reranker as _reranker
        else:
            from src.modules.reranker import CrossEncoderReranker
            _reranker = CrossEncoderReranker(rerank_model or "bongsoo/klue-cross-encoder-v1")

    results = []
    mismatch_count = 0

    for item in items:
        query    = item["query"]
        category = item["category"]
        golden   = item["golden_doc_ids"]  # {chunk_id: rel}

        fetch_k = top_k * 2 if rerank else top_k
        docs = retriever.retrieve(query, top_k=fetch_k, legal_category=category)
        if rerank and _reranker:
            docs = _reranker.rerank(query, docs, top_k=top_k)
        ret_ids = _retrieved_ids(docs)

        # 정규화 버전(part suffix 제거)으로도 매칭
        golden_norm  = {_normalize_id(k): v for k, v in golden.items()}
        ret_ids_norm = [_normalize_id(i) for i in ret_ids]

        # 정답 집합
        golden_set      = set(golden.keys())
        golden_set_norm = set(golden_norm.keys())

        # 검색된 chunk_id가 golden에 한 개도 없으면 mismatch 가능성
        exact_hit = any(i in golden_set for i in ret_ids)
        norm_hit  = any(i in golden_set_norm for i in ret_ids_norm)
        if not exact_hit and not norm_hit:
            mismatch_count += 1

        # 메트릭은 정규화 기준으로 계산 (더 관대, 실질적 의미)
        h1   = _hit_at_k(ret_ids_norm, golden_set_norm, 1)
        h3   = _hit_at_k(ret_ids_norm, golden_set_norm, 3)
        h5   = _hit_at_k(ret_ids_norm, golden_set_norm, 5)
        mrr  = _mrr(ret_ids_norm, golden_set_norm)
        ndcg = _ndcg_at_k(ret_ids_norm, golden_norm, 5)

        results.append({
            "id":       item["id"],
            "query":    query[:60],
            "category": category,
            "source":   item.get("source", ""),
            "format":   item.get("format", ""),
            "golden_n": len(golden),
            "hit@1":  h1,
            "hit@3":  h3,
            "hit@5":  h5,
            "mrr":    mrr,
            "ndcg@5": ndcg,
            "retrieved_ids": ret_ids[:5],
            "golden_ids": list(golden.keys())[:5],
        })

    n = len(results)
    if n == 0:
        return {"n": 0, "results": []}

    def avg(key: str) -> float:
        return round(sum(r[key] for r in results) / n, 4)

    summary = {
        "n":        n,
        "hit@1":    avg("hit@1"),
        "hit@3":    avg("hit@3"),
        "hit@5":    avg("hit@5"),
        "mrr":      avg("mrr"),
        "ndcg@5":   avg("ndcg@5"),
        "chunk_id_mismatch_count": mismatch_count,
        "chunk_id_mismatch_rate":  round(mismatch_count / n, 3),
    }
    return {"summary": summary, "results": results}


# ── 카테고리별 분석 ────────────────────────────────────────────────────────────

def breakdown_by_category(results: list[dict]) -> dict[str, dict]:
    cats: dict[str, list] = {}
    for r in results:
        cat = r.get("category") or "기타"
        cats.setdefault(cat, []).append(r)

    out = {}
    for cat, rows in cats.items():
        n = len(rows)
        out[cat] = {
            "n":      n,
            "hit@1":  round(sum(r["hit@1"]  for r in rows) / n, 4),
            "hit@5":  round(sum(r["hit@5"]  for r in rows) / n, 4),
            "mrr":    round(sum(r["mrr"]    for r in rows) / n, 4),
            "ndcg@5": round(sum(r["ndcg@5"] for r in rows) / n, 4),
        }
    return out


# ── 출력 ──────────────────────────────────────────────────────────────────────

def _print_summary(summary: dict, breakdown: dict, mismatch_warned: bool) -> None:
    n = summary["n"]
    print(f"\n{'='*60}")
    print(f"RAG 검색 성능 평가 결과  (n={n})")
    print(f"{'='*60}")
    print(f"  Hit@1  : {summary['hit@1']:.4f}  ({summary['hit@1']*100:.1f}%)")
    print(f"  Hit@3  : {summary['hit@3']:.4f}  ({summary['hit@3']*100:.1f}%)")
    print(f"  Hit@5  : {summary['hit@5']:.4f}  ({summary['hit@5']*100:.1f}%)")
    print(f"  MRR    : {summary['mrr']:.4f}")
    print(f"  NDCG@5 : {summary['ndcg@5']:.4f}")

    mismatch_rate = summary.get("chunk_id_mismatch_rate", 0)
    if mismatch_rate > 0.1:
        print(f"\n  [경고] chunk_id 불일치율 {mismatch_rate*100:.0f}% "
              f"({summary['chunk_id_mismatch_count']}/{n})")
        print(f"         평가셋 chunk_id가 retriever 인덱스와 다를 수 있습니다.")
        print(f"         (bio vs non-bio 데이터 불일치 가능성)")

    if breakdown:
        print(f"\n{'─'*60}")
        print(f"  카테고리별 분석")
        print(f"{'─'*60}")
        for cat, row in sorted(breakdown.items()):
            print(f"  {cat:<12}  n={row['n']:>3}  "
                  f"Hit@1={row['hit@1']:.3f}  Hit@5={row['hit@5']:.3f}  "
                  f"MRR={row['mrr']:.3f}  NDCG@5={row['ndcg@5']:.3f}")

    print(f"{'='*60}\n")


def _print_failures(results: list[dict], k: int = 10) -> None:
    failures = [r for r in results if r["hit@5"] == 0.0]
    if not failures:
        print("  Hit@5 실패 케이스 없음 (전수 검색 성공)")
        return
    print(f"\n  Hit@5 실패 케이스 ({len(failures)}개, 상위 {k}개 출력):")
    for r in failures[:k]:
        print(f"    [{r['id']}] {r['query']}")
        print(f"      golden  : {r['golden_ids'][:2]}")
        print(f"      retrieved: {r['retrieved_ids'][:2]}")


# ── Dense(E5) 인메모리 검색기 ─────────────────────────────────────────────────

class DenseRetriever:
    """JSONL 코퍼스를 dense 임베딩으로 인덱싱하고 코사인 유사도로 검색.

    model_type:
      "e5"   → intfloat/multilingual-e5-large  (query: / passage: prefix 사용)
      "bgem3" → BAAI/bge-m3                     (prefix 없음, 자체 정규화)
    """

    # E5는 query/passage prefix 필요, BGE-M3는 불필요
    _QUERY_PREFIX  = {"e5": "query: ",   "bgem3": ""}
    _PASSAGE_PREFIX = {"e5": "passage: ", "bgem3": ""}
    _MODEL_IDS = {
        "e5":    "intfloat/multilingual-e5-large",
        "bgem3": "BAAI/bge-m3",
    }

    def __init__(self, model_type: str = "e5"):
        import torch
        assert model_type in ("e5", "bgem3"), f"지원하지 않는 모델: {model_type}"
        self._model_type = model_type
        self._docs: list[dict[str, Any]] = []
        self._embeddings = None
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _cache_path(self, jsonl_paths: list[Path]) -> Path:
        """JSONL 파일 목록과 모델 타입으로 캐시 파일 경로 결정."""
        import hashlib
        key = self._model_type + "|" + "|".join(str(p) for p in sorted(jsonl_paths))
        h = hashlib.md5(key.encode()).hexdigest()[:10]
        return REPO_ROOT / "outputs" / f"dense_index_{self._model_type}_{h}.pt"

    def _cache_valid(self, cache_path: Path, jsonl_paths: list[Path]) -> bool:
        """캐시가 있고, JSONL 파일보다 최신이면 True."""
        if not cache_path.exists():
            return False
        cache_mtime = cache_path.stat().st_mtime
        return all(not p.exists() or p.stat().st_mtime <= cache_mtime for p in jsonl_paths)

    def build_index(self, jsonl_paths: list[Path]) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        label = self._MODEL_IDS[self._model_type]
        cache_path = self._cache_path(jsonl_paths)

        # ── 캐시 로드 시도 ────────────────────────────────────────────────────
        if self._cache_valid(cache_path, jsonl_paths):
            print(f"  [Dense/{label}] 캐시 로드: {cache_path.name}")
            saved = torch.load(cache_path, map_location="cpu", weights_only=False)
            self._docs = saved["docs"]
            self._embeddings = saved["embeddings"].to(self._device)
            print(f"  [Dense/{label}] 캐시에서 {len(self._docs):,}개 문서 / 임베딩 {self._embeddings.shape} ({self._device}) 복원")
            # 모델은 retrieve() 호출 시 지연 로드 (_load_model_if_needed)
            return

        # ── 문서 로딩 ─────────────────────────────────────────────────────────
        print(f"  [Dense/{label}] 문서 로딩 중...")
        for path in jsonl_paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("text") and row.get("chunk_id"):
                        self._docs.append(row)
        print(f"  [Dense/{label}] 총 {len(self._docs):,}개 문서 로드 완료")

        # ── 임베딩 생성 ───────────────────────────────────────────────────────
        print(f"  [Dense/{label}] 모델 로딩 및 임베딩 생성 중...")
        try:
            if self._device == "cuda":
                print(f"  [Dense/{label}] GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(f"  [Dense/{label}] CPU 모드 (CUDA 없음) — 느릴 수 있음")
            model_id = self._MODEL_IDS[self._model_type]
            try:
                self._model = SentenceTransformer(model_id, device=self._device, local_files_only=True)
            except Exception:
                print(f"  [Dense/{label}] 로컬 캐시 없음 → HuggingFace에서 다운로드 중...")
                self._model = SentenceTransformer(model_id, device=self._device)

            # GPU: fp16으로 변환해 메모리 절반 + 처리 속도 2배
            if self._device == "cuda":
                self._model = self._model.half()

            prefix = self._PASSAGE_PREFIX[self._model_type]
            # 텍스트 길이를 400자로 제한:
            # 한국어 1자 ≈ 1.5토큰 → 400자 ≈ 600토큰 → 토크나이저가 512토큰으로 최종 truncate
            # 모델이 실제로 처리할 수 있는 최대 한도(~340자)를 커버하면서,
            # 매우 긴 문서(최대 14,074자)로 인한 불필요한 패딩은 차단
            texts = [f"{prefix}{d['text'][:400]}" for d in self._docs]
            # GPU: batch_size=256, CPU: batch_size=32
            batch_size = 256 if self._device == "cuda" else 32
            print(f"  [Dense/{label}] 총 {len(texts):,}개 / batch_size={batch_size} / fp16={'on' if self._device == 'cuda' else 'off'}")
            self._embeddings = self._model.encode(
                texts, batch_size=batch_size, show_progress_bar=True,
                normalize_embeddings=True, convert_to_tensor=True,
            )
            print(f"  [Dense/{label}] 임베딩 완료: {self._embeddings.shape} ({self._device})")

            # ── 캐시 저장 ─────────────────────────────────────────────────────
            cache_path.parent.mkdir(exist_ok=True)
            torch.save({"docs": self._docs, "embeddings": self._embeddings.cpu()}, cache_path)
            print(f"  [Dense/{label}] 임베딩 캐시 저장: {cache_path.name}")
        except Exception as e:
            print(f"  [Dense/{label}] 모델 로드 실패: {e}")
            self._embeddings = None

    def _load_model_if_needed(self) -> None:
        """캐시 로드 후 쿼리 인코딩 시 모델이 없으면 지연 로드."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        model_id = self._MODEL_IDS[self._model_type]
        print(f"  [Dense/{model_id}] 쿼리 인코딩용 모델 지연 로드 중...")
        try:
            self._model = SentenceTransformer(model_id, device=self._device, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(model_id, device=self._device)
        print(f"  [Dense/{model_id}] 모델 로드 완료")

    @staticmethod
    def _doc_matches_category(doc: dict[str, Any], category: str) -> bool:
        """BM25 _metadata_boost와 동일한 기준으로 카테고리 매칭 여부 반환."""
        meta = doc.get("metadata") or {}
        title = " ".join(
            str(meta.get(key) or "")
            for key in ["law_name", "source_file", "article_title", "section_label"]
        )
        if category == "노동":
            return any(k in title for k in ["근로기준법", "노동", "고용"])
        if category == "성폭력":
            return any(k in title for k in ["성희롱", "성폭력", "강제추행", "강간"])
        return True

    @staticmethod
    def _metadata_score(doc: dict[str, Any], category: str) -> float:
        """BM25의 metadata_boost + source_boost를 Dense 점수 범위(0~1)에 맞게 정규화.

        BM25 전형 점수 범위: 8~15, Dense 코사인 유사도 범위: 0.7~0.95
        스케일 비율 ≈ 0.85 / 12 ≈ 0.07 적용:
          category_boost  +3.0 → +0.20
          statute_boost   +1.5 → +0.10
          manual_boost    +1.0 → +0.07
          case_boost      +0.5 → +0.03
        """
        meta = doc.get("metadata") or {}
        title = " ".join(
            str(meta.get(key) or "")
            for key in ["law_name", "source_file", "article_title", "section_label"]
        )

        # 카테고리 부스트 (BM25 +3.0 → +0.20)
        category_score = 0.0
        if category == "노동" and any(k in title for k in ["근로기준법", "노동", "고용"]):
            category_score = 0.20
        elif category == "성폭력" and any(k in title for k in ["성희롱", "성폭력", "강제추행", "강간"]):
            category_score = 0.20

        # 소스 타입 부스트 (statute +1.5 → +0.10, manual +1.0 → +0.07, case +0.5 → +0.03)
        source_type = str(meta.get("source_type") or "")
        source_score = {"statute": 0.10, "manual": 0.07, "case": 0.03}.get(source_type, 0.0)

        return category_score + source_score

    def retrieve(self, query: str, top_k: int = 5, category: str | None = None) -> list[dict[str, Any]]:
        if self._embeddings is None:
            return []
        self._load_model_if_needed()
        if self._model is None:
            return []
        import torch

        # BM25와 동일한 조건: 카테고리 메타데이터 기반 필터링
        if category:
            doc_indices = [
                i for i, d in enumerate(self._docs)
                if self._doc_matches_category(d, category)
            ]
            if not doc_indices:
                doc_indices = list(range(len(self._docs)))  # 매칭 없으면 전체 폴백
        else:
            doc_indices = list(range(len(self._docs)))

        prefix = self._QUERY_PREFIX[self._model_type]
        q_emb = self._model.encode(
            f"{prefix}{query}", normalize_embeddings=True, convert_to_tensor=True
        ).to(self._device)
        filtered_emb = self._embeddings[doc_indices].to(self._device)
        cosine_scores = torch.matmul(filtered_emb, q_emb).cpu().tolist()

        # BM25와 동일한 메타데이터 부스팅을 Dense 점수 범위로 정규화해 적용
        final_scores = [
            cos + self._metadata_score(self._docs[idx], category or "")
            for idx, cos in zip(doc_indices, cosine_scores)
        ]
        ranked = sorted(zip(doc_indices, final_scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            doc = self._docs[idx].copy()
            doc["score"] = score
            results.append(doc)
        return results


def _summary_to_paper_format(summary: dict, label: str) -> dict:
    """evaluate() 결과 summary → generate_paper_tables.py 형식으로 변환."""
    return {
        "hit_at_1": summary.get("hit@1", 0.0),
        "hit_at_3": summary.get("hit@3", 0.0),
        "hit_at_5": summary.get("hit@5", 0.0),
        "mrr":      summary.get("mrr", 0.0),
        "ndcg_at_5": summary.get("ndcg@5", 0.0),
        "n":        summary.get("n", 0),
    }


def _eval_dense(label: str, dense: DenseRetriever, items: list[dict[str, Any]], top_k: int) -> dict | None:
    """DenseRetriever 인스턴스로 items를 평가해 paper-format 결과를 반환.

    BM25와 동일한 조건:
      - _expand_query_tokens()로 카테고리·쟁점 키워드 확장
      - 카테고리 기반 문서 필터링
    """
    if dense._embeddings is None:
        print(f"  [SKIP] {label} 임베딩 없음 — 건너뜁니다.")
        return None
    results = []
    mismatch = 0
    for item in items:
        category = item.get("category", "")
        # BM25와 동일한 쿼리 확장 적용
        issues = retriever.infer_issues(item["query"], category=category)
        expanded_tokens = retriever._expand_query_tokens(item["query"], category, issues=issues)
        expanded_query = " ".join(expanded_tokens) if expanded_tokens else item["query"]

        docs = dense.retrieve(expanded_query, top_k=top_k, category=category)
        ret_ids = [_normalize_id(str(d.get("chunk_id", ""))) for d in docs]
        golden  = {_normalize_id(k): v for k, v in item["golden_doc_ids"].items()}
        golden_set = set(golden.keys())
        if not any(i in golden_set for i in ret_ids):
            mismatch += 1
        results.append({
            "hit@1":  _hit_at_k(ret_ids, golden_set, 1),
            "hit@3":  _hit_at_k(ret_ids, golden_set, 3),
            "hit@5":  _hit_at_k(ret_ids, golden_set, 5),
            "mrr":    _mrr(ret_ids, golden_set),
            "ndcg@5": _ndcg_at_k(ret_ids, golden, 5),
        })
    n = len(results)
    def avg(k): return round(sum(r[k] for r in results) / n, 4)
    summary = {k: avg(k) for k in ("hit@1", "hit@3", "hit@5", "mrr", "ndcg@5")}
    summary["n"] = n
    s = summary
    print(f"  Hit@1={s['hit@1']:.4f}  MRR={s['mrr']:.4f}  NDCG@5={s['ndcg@5']:.4f}  (n={n})")
    if mismatch:
        print(f"  [경고] chunk_id 불일치 {mismatch}/{n}개")
    return _summary_to_paper_format(summary, label)


def compare_all(items: list[dict[str, Any]], top_k: int) -> None:
    """BM25 / Dense(E5) / Dense(BGE-M3) 세 방법을 동일 데이터로 평가 후 저장."""
    from src.config import config

    all_results: dict[str, dict] = {}
    W = 65
    jsonl_paths = [Path(p) for p in config.rag.jsonl_paths]

    # ── 1. BM25 ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("[1/3] BM25 평가 중...")
    out_bm25 = evaluate(items, top_k=top_k, rerank=False)
    s = out_bm25["summary"]
    all_results["BM25"] = _summary_to_paper_format(s, "BM25")
    print(f"  Hit@1={s['hit@1']:.4f}  MRR={s['mrr']:.4f}  NDCG@5={s['ndcg@5']:.4f}  (n={s['n']})")

    # ── 2. Dense(E5) ─────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("[2/3] Dense(E5) 평가 중...")
    e5 = DenseRetriever(model_type="e5")
    e5.build_index(jsonl_paths)
    r = _eval_dense("Dense(E5)", e5, items, top_k)
    if r:
        all_results["Dense(E5)"] = r
    del e5  # VRAM 확보 후 다음 모델 로드

    # ── 3. Dense(BGE-M3) ─────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("[3/3] Dense(BGE-M3) 평가 중...")
    bgem3 = DenseRetriever(model_type="bgem3")
    bgem3.build_index(jsonl_paths)
    r = _eval_dense("Dense(BGE-M3)", bgem3, items, top_k)
    if r:
        all_results["Dense(BGE-M3)"] = r
    del bgem3

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print("RAG 검색 성능 비교 (동일 쿼리셋 · 동일 코퍼스)")
    print(f"{'='*W}")
    print(f"{'방법':<20} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'MRR':>7} {'NDCG@5':>8} {'N':>5}")
    print(f"{'─'*W}")
    for method, r in all_results.items():
        marker = " ◀ 채택" if method == "BM25" else ""
        print(
            f"{method:<20} {r['hit_at_1']:>7.4f} {r['hit_at_3']:>7.4f} {r['hit_at_5']:>7.4f}"
            f" {r['mrr']:>7.4f} {r['ndcg_at_5']:>8.4f} {r['n']:>5}{marker}"
        )
    print(f"{'='*W}")

    # ── outputs/rag_eval_results.json 저장 ───────────────────────────────────
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {OUTPUT_PATH}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", choices=["bio", "remapped", "golden", "golden_reviewed"],
                        default="remapped",
                        help="평가셋 선택: golden_reviewed=수동3계층검토(권장), remapped=재매핑, bio=원본, golden=자동키워드")
    parser.add_argument("--top-k", type=int, default=10,
                        help="검색 범위 (기본값: 10)")
    parser.add_argument("--skip-zero", action="store_true",
                        help="relevance=0 항목 제외")
    parser.add_argument("--domain-filter", action="store_true",
                        help="노동/성폭력 외 도메인 제외 (bio/remapped 전용)")
    parser.add_argument("--rerank", action="store_true",
                        help="Cross-Encoder Reranker 적용 (Before/After 비교 시 두 번 실행)")
    parser.add_argument("--compare", action="store_true",
                        help="Before/After 한 번에 비교 출력 (--rerank 포함)")
    parser.add_argument("--rerank-model", type=str, default="bongsoo/klue-cross-encoder-v1",
                        help="Cross-Encoder 모델명")
    parser.add_argument("--save", action="store_true",
                        help="results/ 폴더에 JSON 저장")
    parser.add_argument("--compare-all", action="store_true",
                        help="BM25 / Dense(E5) / BM25+Reranker 동시 비교 후 outputs/rag_eval_results.json 저장 (논문용)")
    args = parser.parse_args()

    # ── --compare-all: 세 방법 동시 비교 ─────────────────────────────────────
    if args.compare_all:
        eval_key = args.eval if args.eval != "remapped" else "golden_reviewed"
        eval_path = EVAL_FILES[eval_key]
        if not eval_path.exists():
            print(f"[오류] 평가셋 파일 없음: {eval_path}")
            sys.exit(1)
        items = _load_golden(eval_path, args.skip_zero)
        print(f"평가셋: {eval_path.name}  ({len(items)}개 쿼리)  top_k={args.top_k}")
        compare_all(items, top_k=args.top_k)
        return

    eval_path = EVAL_FILES[args.eval]
    if not eval_path.exists():
        print(f"[오류] 평가셋 파일 없음: {eval_path}")
        sys.exit(1)

    print(f"평가셋 : {eval_path.name}  (--eval {args.eval})")
    print(f"top_k  : {args.top_k}")
    print(f"skip_zero: {args.skip_zero}  domain_filter: {args.domain_filter}")

    if args.eval in ("bio", "remapped"):
        items = _load_bio(eval_path, args.skip_zero, domain_filter=args.domain_filter)
    else:  # golden, golden_reviewed
        items = _load_golden(eval_path, args.skip_zero)

    print(f"로드된 케이스: {len(items)}개\n검색 중...")

    do_compare = args.compare or (args.rerank and not args.compare)

    if do_compare:
        # Before
        print("[ Before: retriever only ]")
        out_before = evaluate(items, top_k=args.top_k, rerank=False)

        label = "LLM (Claude Haiku)" if args.rerank_model == "llm" else args.rerank_model
        print(f"\n[ After: + Reranker ({label}) ]")
        out_after = evaluate(items, top_k=args.top_k, rerank=True, rerank_model=args.rerank_model)

        s_before = out_before.get("summary", {})
        s_after  = out_after.get("summary", {})
        print(f"\n{'='*60}")
        print(f"Before vs After  (n={s_before.get('n', 0)})")
        print(f"{'='*60}")
        print(f"{'메트릭':<10}  {'Before':>8}  {'After':>8}  {'Delta':>8}")
        print(f"{'-'*42}")
        for key in ("hit@1", "hit@3", "hit@5", "mrr", "ndcg@5"):
            b = s_before.get(key, 0.0)
            a = s_after.get(key, 0.0)
            delta = a - b
            sign = "+" if delta >= 0 else ""
            print(f"{key:<10}  {b:>8.4f}  {a:>8.4f}  {sign}{delta:>7.4f}")
        print(f"{'='*60}\n")

        summary = s_after
        results = out_after.get("results", [])
        breakdown = breakdown_by_category(results)

        if args.save:
            RESULTS_DIR.mkdir(exist_ok=True)
            for tag, out in (("before", out_before), ("after", out_after)):
                out_path = RESULTS_DIR / f"rag_metrics_{args.eval}_{tag}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                print(f"저장: {out_path}")
        return

    output  = evaluate(items, top_k=args.top_k, rerank=args.rerank, rerank_model=args.rerank_model)
    summary = output.get("summary", {})
    results = output.get("results", [])

    if not results:
        print("[오류] 평가 결과가 없습니다.")
        sys.exit(1)

    breakdown = breakdown_by_category(results)
    _print_summary(summary, breakdown, mismatch_warned=True)
    _print_failures(results)

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        suffix = "_reranked" if args.rerank else ""
        out_path = RESULTS_DIR / f"rag_metrics_{args.eval}{suffix}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "breakdown": breakdown, "results": results},
                      f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
