"""
공통 Retriever 클래스
Phase 1, Phase 2에서 공통 사용
"""

from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError as e:
    print(f"[ERROR] 라이브러리 설치 필요: {e}")
    print("pip install sentence-transformers torch")
    import sys
    sys.exit(1)


class DenseRetriever:
    """Dense Retrieval (E5-base)"""

    def __init__(self):
        print("[Dense Retrieval 초기화]")
        model_name = "intfloat/multilingual-e5-base"
        print(f"  모델: {model_name}")

        try:
            self.model = SentenceTransformer(model_name)
            self.model.to(DEVICE)
            self.model.eval()

            # GPU 전용 최적화
            if DEVICE == "cuda":
                self.model = self.model.half()
                print("  - FP16 활성화")

            self.batch_size = 32 if DEVICE == "cuda" else 8
            print(f"  - 배치 크기: {self.batch_size}")
            print("[초기화 완료]\n")

        except Exception as e:
            print(f"[ERROR] 모델 로드 실패: {e}")
            raise

        self.doc_embeddings = None
        self.doc_list = []

    async def warmup(self):
        """문서 임베딩"""
        from src.modules.rag import retriever as bm25_retriever

        print("[문서 임베딩 시작]")

        try:
            docs = bm25_retriever._load_jsonl_cache()

            self.doc_list = []
            texts = []

            for doc in docs:
                texts.append(doc.text)
                self.doc_list.append(doc)

            total = len(texts)
            print(f"  총 문서: {total:,}개")

            # 안전하게 청크 분할
            chunk_size = 5000
            num_chunks = (total + chunk_size - 1) // chunk_size
            print(f"  청크 수: {num_chunks}개 (각 {chunk_size}개)")

            all_embeddings = []

            for i, start in enumerate(range(0, total, chunk_size), 1):
                end = min(start + chunk_size, total)
                chunk_texts = texts[start:end]

                print(f"  [{i}/{num_chunks}] 임베딩: {start+1:,}-{end:,} ({end*100//total}%)")

                with torch.no_grad():
                    chunk_emb = self.model.encode(
                        chunk_texts,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_tensor=True,
                        device=DEVICE,
                        normalize_embeddings=True
                    )

                all_embeddings.append(chunk_emb)

                # 메모리 정리
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            # 결합
            self.doc_embeddings = torch.cat(all_embeddings, dim=0)

            print(f"[완료] {total:,}개 문서 임베딩\n")

        except Exception as e:
            print(f"[ERROR] 임베딩 실패: {e}")
            raise

    async def retrieve(self, query: str, top_k: int = 5, **kwargs):
        """검색 수행"""
        if self.doc_embeddings is None:
            return []

        try:
            with torch.no_grad():
                query_emb = self.model.encode(
                    query,
                    convert_to_tensor=True,
                    device=DEVICE,
                    normalize_embeddings=True
                )

                scores = torch.cosine_similarity(
                    query_emb.unsqueeze(0),
                    self.doc_embeddings
                )

                top_k_actual = min(top_k, len(scores))
                top_scores, top_indices = torch.topk(scores, k=top_k_actual)

                top_indices = top_indices.cpu().numpy()
                top_scores = top_scores.cpu().numpy()

            results = []
            for idx, score in zip(top_indices, top_scores):
                doc = self.doc_list[idx]
                results.append({
                    "text": doc.text,
                    "chunk_id": doc.chunk_id,
                    "score": float(score)
                })

            return results

        except Exception as e:
            print(f"[WARNING] 검색 중 오류: {e}")
            return []


class HybridRetriever:
    """하이브리드 검색 (BM25 + Dense)"""

    def __init__(self, alpha: float, dense: DenseRetriever):
        from src.modules.rag import retriever as bm25_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha  # BM25 가중치
        self.dense = dense

    async def retrieve(self, query: str, top_k: int = 5, **kwargs):
        """하이브리드 검색"""
        try:
            # BM25 검색
            bm25_docs = await self.bm25_retriever.retrieve_async(
                query, top_k=top_k*2, **kwargs
            )

            # Dense 검색
            dense_docs = await self.dense.retrieve(query, top_k=top_k*2)

            if not dense_docs:
                return bm25_docs[:top_k]

            # 점수 결합
            scores = defaultdict(float)

            # BM25 점수 정규화
            bm25_scores = [d.get("score", 0) for d in bm25_docs]
            if bm25_scores:
                max_bm25 = max(bm25_scores)
                min_bm25 = min(bm25_scores)
                range_bm25 = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0

                for doc in bm25_docs:
                    cid = doc.get("chunk_id", "")
                    if cid:
                        norm = (doc.get("score", 0) - min_bm25) / range_bm25
                        scores[cid] += self.alpha * norm

            # Dense 점수 정규화
            dense_scores = [d.get("score", 0) for d in dense_docs]
            if dense_scores:
                max_dense = max(dense_scores)
                min_dense = min(dense_scores)
                range_dense = max_dense - min_dense if max_dense > min_dense else 1.0

                for doc in dense_docs:
                    cid = doc.get("chunk_id", "")
                    if cid:
                        norm = (doc.get("score", 0) - min_dense) / range_dense
                        scores[cid] += (1 - self.alpha) * norm

            # 정렬
            sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # 문서 매핑
            chunk_to_doc = {}
            for doc in bm25_docs + dense_docs:
                cid = doc.get("chunk_id", "")
                if cid and cid not in chunk_to_doc:
                    chunk_to_doc[cid] = doc

            # 결과 생성
            results = []
            for cid, score in sorted_chunks[:top_k]:
                if cid in chunk_to_doc:
                    doc = chunk_to_doc[cid].copy()
                    doc["score"] = score
                    results.append(doc)

            return results

        except Exception as e:
            print(f"[WARNING] 하이브리드 검색 중 오류: {e}")
            return []
