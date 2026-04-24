"""
RAG 검색 모듈
"""

import asyncio
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import config


class RAGRetriever:
	def __init__(self):
		cfg = config.rag
		self.top_k = cfg.top_k
		self._db = chromadb.PersistentClient(path=cfg.chroma_path, settings=Settings(anonymized_telemetry=False))
		self._collection = self._db.get_or_create_collection(name=cfg.collection_name, metadata={"hnsw:space": "cosine"})
		self._embedder = SentenceTransformer(cfg.embedding_model, device="cuda" if self._has_gpu() else "cpu")
		self._batch_size = cfg.embedding_batch_size

	@staticmethod
	def _has_gpu() -> bool:
		try:
			import torch
			return torch.cuda.is_available()
		except ImportError:
			return False

	def _embed(self, texts: list[str], is_query: bool = True) -> list[list[float]]:
		prefix = "query: " if is_query else "passage: "
		prefixed = [prefix + t for t in texts]
		return self._embedder.encode(prefixed, batch_size=self._batch_size, normalize_embeddings=True, show_progress_bar=False).tolist()

	def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
		k = top_k or self.top_k
		query_emb = self._embed([query], is_query=True)
		results = self._collection.query(query_embeddings=query_emb, n_results=k, include=["documents", "metadatas", "distances"])
		docs = []
		for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
			docs.append({"text": doc, "source": meta.get("source", ""), "score": 1.0 - dist, "metadata": meta})
		return docs

	async def retrieve_async(self, query: str, top_k: Optional[int] = None) -> list[dict]:
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(None, self.retrieve, query, top_k)

	async def batch_retrieve(self, queries: list[str]) -> list[list[dict]]:
		tasks = [self.retrieve_async(q) for q in queries]
		return await asyncio.gather(*tasks)

	def build_context(self, docs: list[dict]) -> str:
		parts = []
		for i, doc in enumerate(docs, 1):
			parts.append(f"[참고 {i}] 출처: {doc['source']}\n{doc['text']}")
		return "\n\n".join(parts)


retriever = RAGRetriever()
