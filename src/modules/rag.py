from __future__ import annotations

import asyncio
from collections import Counter
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import config


@dataclass
class RAGDocument:
    text: str
    metadata: dict[str, Any]
    chunk_id: str = ""
    score: float = 0.0
    retrieval_reason: str = ""


class LegalRetriever:
    def __init__(self):
        self._collection = None
        self._embedder = None
        self._jsonl_cache: list[RAGDocument] | None = None
        self._bm25_index: list[tuple[RAGDocument, list[str], Counter]] | None = None
        self._doc_freq: Counter | None = None
        self._avg_doc_len = 0.0

    async def retrieve_async(self, query: str, top_k: int | None = None, legal_category: str = "") -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.retrieve, query, top_k, legal_category)

    def retrieve(self, query: str, top_k: int | None = None, legal_category: str = "") -> list[dict[str, Any]]:
        top_k = top_k or config.rag.top_k
        docs = self._retrieve_from_chroma(query, top_k)
        if not docs:
            docs = self._retrieve_from_jsonl(query, top_k, legal_category=legal_category)
        return [doc.__dict__ for doc in self._dedupe_docs(docs)[:top_k]]

    def _retrieve_from_chroma(self, query: str, top_k: int) -> list[RAGDocument]:
        if not config.rag.enable_vector:
            return []
        try:
            import chromadb

            if self._collection is None:
                client = chromadb.PersistentClient(path=config.rag.chroma_path)
                self._collection = client.get_collection(config.rag.collection_name)

            embedding = self._embed_query(query)
            if embedding is None:
                return []

            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            docs = []
            ids = result.get("ids", [[]])[0]
            texts = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            for idx, text in enumerate(texts):
                distance = float(distances[idx]) if idx < len(distances) else 0.0
                docs.append(
                    RAGDocument(
                        text=text or "",
                        metadata=metadatas[idx] or {},
                        chunk_id=ids[idx] if idx < len(ids) else "",
                        score=1.0 / (1.0 + max(distance, 0.0)),
                    )
                )
            return docs
        except Exception:
            return []

    def _embed_query(self, query: str) -> list[float] | None:
        try:
            from sentence_transformers import SentenceTransformer

            if self._embedder is None:
                self._embedder = SentenceTransformer(config.rag.embedding_model, local_files_only=True)
            embedding = self._embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            return [float(x) for x in embedding]
        except Exception:
            return None

    def _load_jsonl_cache(self) -> list[RAGDocument]:
        if self._jsonl_cache is not None:
            return self._jsonl_cache

        docs: list[RAGDocument] = []
        for raw_path in config.rag.jsonl_paths:
            path = Path(raw_path)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    docs.append(
                        RAGDocument(
                            text=row.get("text", ""),
                            metadata=row.get("metadata", {}),
                            chunk_id=row.get("chunk_id", ""),
                        )
                    )
        self._jsonl_cache = docs
        return docs

    def _retrieve_from_jsonl(self, query: str, top_k: int, legal_category: str = "") -> list[RAGDocument]:
        category = legal_category or self._infer_category(query)
        query_tokens = self._expand_query_tokens(query, category)
        if not query_tokens:
            return []
        self._ensure_bm25_index()

        scored = []
        for doc, doc_tokens, token_counts in self._bm25_index or []:
            haystack = f"{doc.text} {json.dumps(doc.metadata, ensure_ascii=False)}"
            bm25 = self._bm25_score(query_tokens, doc_tokens, token_counts)
            if bm25 <= 0:
                continue
            score = bm25 + self._domain_boost(query, haystack, category) + self._metadata_boost(doc.metadata, category)
            score -= self._cross_domain_penalty(haystack, category)
            if score > 0:
                scored.append(
                    RAGDocument(
                        text=doc.text,
                        metadata=doc.metadata,
                        chunk_id=doc.chunk_id,
                        score=score,
                        retrieval_reason=f"bm25={bm25:.2f}, category={category or 'unknown'}",
                    )
                )
        scored.sort(key=lambda doc: doc.score, reverse=True)
        return self._balance_sources(scored, top_k, category)

    def _ensure_bm25_index(self):
        if self._bm25_index is not None:
            return

        index = []
        doc_freq = Counter()
        total_len = 0
        for doc in self._load_jsonl_cache():
            metadata_text = json.dumps(doc.metadata, ensure_ascii=False)
            tokens = self._tokenize_for_index(f"{doc.text} {metadata_text}")
            if not tokens:
                continue
            counts = Counter(tokens)
            doc_freq.update(counts.keys())
            total_len += len(tokens)
            index.append((doc, tokens, counts))

        self._bm25_index = index
        self._doc_freq = doc_freq
        self._avg_doc_len = total_len / len(index) if index else 0.0

    def _bm25_score(self, query_tokens: list[str], doc_tokens: list[str], token_counts: Counter) -> float:
        if not self._doc_freq or not self._bm25_index or not self._avg_doc_len:
            return 0.0

        k1 = 1.5
        b = 0.75
        total_docs = len(self._bm25_index)
        doc_len = len(doc_tokens)
        score = 0.0

        for token in query_tokens:
            freq = token_counts.get(token, 0)
            if not freq:
                continue
            df = self._doc_freq.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * doc_len / self._avg_doc_len)
            score += idf * numerator / denominator
        return score

    def _tokenize_for_index(self, text: str) -> list[str]:
        return re.findall(r"[0-9A-Za-z가-힣]{2,}", text.lower())

    def _expand_query_tokens(self, text: str, category: str = "") -> list[str]:
        tokens = self._tokenize_for_index(text)
        aliases = {
            "알바": "근로 임금 아르바이트 근로계약서 사업주",
            "알바비": "임금 급여 체불 근로기준법",
            "월급": "임금 급여 근로",
            "급여": "임금 체불 지급",
            "체불": "임금체불 지급 고용노동부",
            "못 받": "임금체불 임금 지급 근로기준법 제43조",
            "안 줬": "임금체불 임금 지급 근로기준법 제43조",
            "성추행": "강제추행 성폭력 신체접촉 동의",
            "성희롱": "성폭력 성희롱 학교 상담 신고",
            "동의": "성적인 행위 동의 피해자 진술",
            "거절": "동의 강제추행 피해자 진술",
            "해고": "근로 해고 부당해고",
        }
        for key, expansion in aliases.items():
            if key in text:
                tokens.extend(expansion.split())
        if category == "노동":
            tokens.extend("근로기준법 임금 근로계약서 고용노동부 사업주 지급".split())
        elif category == "성폭력":
            tokens.extend("성폭력 성희롱 강제추행 피해자 보호 상담 신고 동의".split())
        return tokens

    def _lexical_score(self, terms: set[str], text: str) -> float:
        lowered = text.lower()
        score = 0.0
        for term in terms:
            count = lowered.count(term)
            if count:
                score += 1.0 + math.log(count)
        return score

    def _domain_boost(self, query: str, text: str, category: str = "") -> float:
        boost = 0.0
        if category == "노동" or any(term in query for term in ["알바", "아르바이트", "임금", "월급", "체불", "근로", "급여", "못 받", "안 줬"]):
            if "근로계약서" in text:
                boost += 8.0
            if "임금체불" in text or "임금을 지급" in text or "임금은" in text:
                boost += 9.0
            if "근로기준법" in text:
                boost += 6.0
            if "제43조(임금 지급)" in text or "제43조" in text and "임금" in text and "지급" in text:
                boost += 12.0
            if "고용노동부" in text:
                boost += 3.0
        if category == "성폭력" or any(term in query for term in ["성폭력", "성추행", "성희롱", "강제추행", "불법촬영"]):
            if "피해자 보호" in text or "성폭력" in text:
                boost += 6.0
            if "신고" in text or "상담" in text:
                boost += 4.0
        if any(term in query for term in ["동의 없이", "동의없이", "몸을 만", "허리를 만", "거절했는데"]):
            if "성적인 행위" in text or "동의" in text:
                boost += 8.0
            if "강제추행" in text or "성추행" in text:
                boost += 6.0
            if "피해자 보호" in text or "상담 및 접수" in text:
                boost += 3.0
        return boost

    def _metadata_boost(self, metadata: dict[str, Any], category: str) -> float:
        source_type = str(metadata.get("source_type") or "")
        title = " ".join(str(metadata.get(key) or "") for key in ["law_name", "source_file", "article_title", "section_label"])

        boost = 0.0
        if source_type == "statute":
            boost += 1.5
        elif source_type == "manual":
            boost += 1.0
        elif source_type == "case":
            boost += 0.8

        if category == "노동":
            if "근로기준법" in title or "노동" in title or "고용" in title:
                boost += 5.0
            if "성범죄" in title and "고용" not in title and "노동" not in title:
                boost -= 3.0
        elif category == "성폭력":
            if "성희롱" in title or "성폭력" in title or "강제추행" in title:
                boost += 5.0
            if "강제추행" in title or "추행" in title:
                boost += 14.0
            if "신분위장수사" in title or "촬영물" in title:
                boost -= 10.0
            if "노동" in title and "성희롱" not in title and "성폭력" not in title:
                boost -= 3.0
        return boost

    def _cross_domain_penalty(self, text: str, category: str) -> float:
        if category == "노동":
            sexual_hits = sum(text.count(term) for term in ["성폭력", "성희롱", "강제추행", "성범죄"])
            labor_hits = sum(text.count(term) for term in ["근로", "임금", "노동", "고용", "계약"])
            if sexual_hits > labor_hits + 2:
                return 8.0
        elif category == "성폭력":
            labor_hits = sum(text.count(term) for term in ["근로", "임금", "퇴직금", "최저임금"])
            sexual_hits = sum(text.count(term) for term in ["성폭력", "성희롱", "강제추행", "동의", "피해자"])
            if labor_hits > sexual_hits + 2:
                return 8.0
            if "신분위장수사" in text or "촬영물 또는 복제물" in text:
                return 14.0
        return 0.0

    def _balance_sources(self, docs: list[RAGDocument], top_k: int, category: str) -> list[RAGDocument]:
        selected = []
        preferred_slots = ["statute", "manual", "statute"]
        if category == "성폭력":
            preferred_slots = ["manual", "statute", "manual"]

        remaining = list(docs)
        for wanted in preferred_slots:
            match = next((doc for doc in remaining if self._source_type(doc) == wanted), None)
            if match is None:
                continue
            selected.append(match)
            remaining.remove(match)

        for doc in remaining:
            selected.append(doc)
            if len(selected) >= top_k:
                break
        return selected

    def _source_type(self, doc: RAGDocument) -> str:
        return str((doc.metadata or {}).get("source_type") or "case")

    def _dedupe_docs(self, docs: list[RAGDocument]) -> list[RAGDocument]:
        deduped = []
        seen = set()
        for doc in docs:
            metadata = doc.metadata or {}
            key = (
                metadata.get("source_file") or metadata.get("law_name") or doc.chunk_id,
                metadata.get("article_id") or metadata.get("section_label") or metadata.get("article_title"),
                re.sub(r"\s+", " ", doc.text[:120]),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)
        return deduped

    def _infer_category(self, query: str) -> str:
        if any(term in query for term in ["성폭력", "성추행", "성희롱", "강제추행", "강간", "불법촬영", "동의 없이", "몸을 만", "거절"]):
            return "성폭력"
        if any(term in query for term in ["알바", "알바비", "아르바이트", "임금", "월급", "급여", "근로", "해고", "퇴직금"]):
            return "노동"
        return ""


retriever = LegalRetriever()
