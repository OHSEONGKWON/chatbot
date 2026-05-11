from __future__ import annotations

import asyncio
import json
import re
import statistics
from collections import OrderedDict
from difflib import SequenceMatcher
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from ..config import config
from .llm_client import llm_client


DEFAULT_SIMILAR_QUESTIONS = 3
MAX_SIMILAR_QUESTIONS = 10


SIMILAR_Q_SYSTEM = """당신은 법률 전문가입니다.
사용자 질문과 의미가 유사하지만 표현이 다른 질문을 정확히 {n}개 생성하세요.
반드시 JSON 배열로만 응답하세요."""

SIMILAR_Q_USER = """다음 질문과 의미가 유사한 법률 질문 {n}개를 생성하세요.
표현, 어휘, 문장 구조를 다양하게 바꾸되 핵심 법률 쟁점은 유지하세요.

[원본 질문]: {question}

응답 형식: ["질문1", "질문2", ..., "질문{n}"]"""

RAG_ANSWER_SYSTEM = """당신은 대한민국 법률 전문가입니다.
주어진 법률 참고 자료를 바탕으로 질문에 정확하고 간결하게 답변하세요.
참고 자료에 없는 조항번호·판례번호·형량·벌금액은 만들지 마세요.
사용자에게 유리한 결론도 사실관계가 부족하면 가능성으로 낮춰야 합니다."""

RAG_ANSWER_USER = """[법률 참고 자료]
{context}

[질문]
{question}

위 참고 자료를 바탕으로 답변하세요."""


class ConsistencyChecker:
    def __init__(self):
        cfg = config.hallucination
        requested = int(getattr(cfg, "num_similar_questions", DEFAULT_SIMILAR_QUESTIONS) or DEFAULT_SIMILAR_QUESTIONS)
        self.n_questions = max(1, min(requested, MAX_SIMILAR_QUESTIONS))
        self.threshold = cfg.consistency_threshold
        self._embedder = None
        self._embedder_name = config.rag.embedding_model
        self._similar_questions_cache: OrderedDict[str, list[str]] = OrderedDict()
        self._similar_questions_cache_size = 32

    @staticmethod
    def _get_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_embedder(self):
        if self._embedder is not None:
            return self._embedder
        if SentenceTransformer is None:
            return None
        try:
            self._embedder = SentenceTransformer(self._embedder_name, device=self._get_device())
        except Exception:
            try:
                self._embedder = SentenceTransformer(config.rag.embedding_model, device=self._get_device())
            except Exception:
                self._embedder = None
        return self._embedder

    async def generate_similar_questions(self, question: str) -> list[str]:
        cache_key = re.sub(r"\s+", " ", (question or "")).strip().lower()
        cached = self._similar_questions_cache.get(cache_key)
        if cached is not None:
            self._similar_questions_cache.move_to_end(cache_key)
            return list(cached)

        raw = await llm_client.complete(
            system_prompt=SIMILAR_Q_SYSTEM.format(n=self.n_questions),
            user_prompt=SIMILAR_Q_USER.format(n=self.n_questions, question=question),
            temperature=0.8,
            json_mode=True,
            model=llm_client.clarify_model,
        )
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                questions = [str(q).strip() for q in parsed if str(q).strip()]
            elif isinstance(parsed, dict):
                questions = [str(q).strip() for q in parsed.get("questions", []) if str(q).strip()]
            else:
                questions = []
            questions = questions[: self.n_questions]
            while len(questions) < self.n_questions:
                questions.append(question)
            self._similar_questions_cache[cache_key] = list(questions)
            self._similar_questions_cache.move_to_end(cache_key)
            while len(self._similar_questions_cache) > self._similar_questions_cache_size:
                self._similar_questions_cache.popitem(last=False)
            return questions
        except Exception:
            return [question] * self.n_questions

    async def warmup(self):
        await asyncio.to_thread(self._load_embedder)

    def _comparison_text(self, text: str) -> str:
        if not text:
            return ""

        lines: list[str] = []
        for raw_line in str(text).splitlines():
            line = re.sub(r"^\s*\d+\s*[\).:]\s*", "", raw_line).strip()
            line = re.sub(r"^\s*[-*•]\s*", "", line).strip()
            if not line:
                continue
            if line in {"상황 정리", "근거에서 확인한 내용", "지금 할 일", "주의사항"}:
                continue
            if line.startswith("이 답변은"):
                continue
            lines.append(line)

        compact = " ".join(lines)
        compact = re.sub(r"\s+", " ", compact).strip().lower()
        return compact

    def _format_docs(self, rag_docs: list[dict[str, Any]]) -> str:
        if not rag_docs:
            return "검색된 근거 문서가 없습니다."
        blocks = []
        for i, doc in enumerate(rag_docs[: config.rag.top_k], 1):
            metadata = doc.get("metadata") or {}
            title = metadata.get("law_name") or metadata.get("source_file") or metadata.get("manual_title") or "근거 문서"
            text = (doc.get("text") or "").strip()
            source_type = metadata.get("source_type") or "unknown"
            reason = doc.get("retrieval_reason") or ""
            blocks.append(f"[{i}] {title} ({source_type}) {reason}\n{text[:1200]}")
        return "\n\n".join(blocks)

    async def _generate_single_answer(self, question: str, rag_docs: list[dict[str, Any]], category: str = "") -> str:
        context = self._format_docs(rag_docs)
        answer = await llm_client.complete(
            system_prompt=RAG_ANSWER_SYSTEM,
            user_prompt=RAG_ANSWER_USER.format(context=context, question=question),
            temperature=0.1,
            model=llm_client.answer_model,
        )
        if answer:
            return answer
        return self._fallback_answer(question, rag_docs, category)

    async def generate_all_answers(
        self,
        original_question: str,
        similar_questions: list[str],
        rag_docs: list[dict[str, Any]],
        category: str = "",
    ) -> tuple[str, list[str]]:
        all_questions = [original_question] + similar_questions
        tasks = [self._generate_single_answer(q, rag_docs, category=category) for q in all_questions]
        results = await asyncio.gather(*tasks)
        return results[0], list(results[1:])

    def _fallback_answer(self, question: str, rag_docs: list[dict[str, Any]], category: str = "") -> str:
        if not rag_docs:
            return "현재 질문과 직접 연결되는 근거 문서를 찾지 못했습니다. 사실관계가 더 필요하므로 발생 시점, 장소/관계, 구체적 행위, 증거 유무를 알려주세요."
        summary = self._summarize_context(question, category)
        snippets = self._relevant_snippets(question, rag_docs, category, limit=3)
        basis = "\n".join(f"- {snippet}" for snippet in snippets) if snippets else "- 검색된 문서에서 직접 관련 근거를 충분히 추리지 못했습니다."
        return (
            "1) 상황 정리\n"
            f"{summary}\n\n"
            "2) 근거에서 확인한 내용\n"
            f"{basis}\n\n"
            "3) 지금 할 일\n"
            "- 증거와 사실관계를 먼저 정리하세요.\n"
            "- 긴급 위험이나 기한이 있으면 공공기관 또는 상담기관에 먼저 연결하세요.\n\n"
            "4) 주의사항\n"
            "이 답변은 입력한 사실관계와 검색 근거에 따른 일반 안내입니다."
        )

    def _summarize_context(self, question: str, category: str) -> str:
        if category == "노동":
            return "임금, 근로계약, 근무시간, 지급 여부를 우선 확인해야 합니다."
        if category == "성폭력":
            return "동의 여부, 구체적 행위, 장소, 관계, 사건 직후 증거가 중요합니다."
        return "사안의 법적 분류, 발생 시점, 상대방과의 관계, 증거 유무를 먼저 확인해야 합니다."

    def _relevant_snippets(self, question: str, rag_docs: list[dict[str, Any]], category: str, limit: int = 3) -> list[str]:
        query_terms = set(re.findall(r"[0-9A-Za-z가-힣]{2,}", question.lower()))
        if category == "노동":
            query_terms.update(["근로", "임금", "체불", "지급", "근로계약서", "고용노동부"])
        elif category == "성폭력":
            query_terms.update(["성폭력", "성희롱", "강제추행", "동의", "피해자", "상담", "신고"])
        snippets = []
        seen = set()
        for doc in rag_docs:
            metadata = doc.get("metadata") or {}
            title = self._clean_title(metadata.get("law_name") or metadata.get("source_file") or metadata.get("manual_title") or "근거 문서")
            text = re.sub(r"\s+", " ", (doc.get("text") or "")).strip()
            sentences = re.split(r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+", text)
            best = self._best_sentence(sentences, query_terms) or text[:240]
            best = best.strip()
            if len(best) > 260:
                best = best[:260].rstrip() + "..."
            key = (title, best)
            if not best or key in seen:
                continue
            seen.add(key)
            snippets.append(f"{title}: {best}")
            if len(snippets) >= limit:
                break
        return snippets

    def _clean_title(self, title: str) -> str:
        title = re.sub(r"\s+", " ", str(title or "")).strip()
        if len(title) > 80:
            title = title[:80].rstrip() + "..."
        return title

    def _best_sentence(self, sentences: list[str], query_terms: set[str]) -> str:
        best_sentence = ""
        best_score = 0
        for sentence in sentences:
            if len(sentence.strip()) < 25:
                continue
            lowered = sentence.lower()
            score = sum(1 for term in query_terms if term in lowered)
            if score > best_score:
                best_score = score
                best_sentence = sentence
        return best_sentence

    def _embed_answers(self, answers: list[str]) -> np.ndarray:
        embedder = self._load_embedder()
        if embedder is None:
            return np.zeros((len(answers), 1), dtype=float)
        prefixed = ["passage: " + self._comparison_text(a or "") for a in answers]
        embeddings = embedder.encode(
            prefixed,
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings)

    def _cosine_similarity_scores(self, original_answer: str, similar_answers: list[str]) -> list[float]:
        valid_answers = [a for a in similar_answers if isinstance(a, str) and a.strip()]
        if not original_answer.strip() or not valid_answers:
            return []
        embedder = self._load_embedder()
        if embedder is None:
            normalized_original = self._comparison_text(original_answer)
            return [float(SequenceMatcher(None, normalized_original, self._comparison_text(ans)).ratio()) for ans in valid_answers]
        all_answers = [original_answer] + valid_answers
        embeddings = self._embed_answers(all_answers)
        original_emb = embeddings[0]
        similar_embs = embeddings[1:]
        scores = similar_embs @ original_emb
        return [float(s) for s in scores]

    def score_answers(self, original_answer: str, similar_answers: list[str]) -> tuple[bool, float, list[float]]:
        scores = self._cosine_similarity_scores(original_answer, similar_answers)
        if not scores:
            return False, 0.0, []
        top_k = max(3, len(scores) // 2)
        robust_scores = sorted(scores, reverse=True)[:top_k]
        median_score = float(statistics.median(robust_scores))
        return median_score >= self.threshold, median_score, scores

    async def run(self, question: str, rag_docs: list[dict[str, Any]] | None = None, legal_category: str = "") -> tuple[bool, str, float, list[str]]:
        rag_docs = rag_docs or []
        similar_questions = await self.generate_similar_questions(question)
        original_answer, similar_answers = await self.generate_all_answers(
            original_question=question,
            similar_questions=similar_questions,
            rag_docs=rag_docs,
            category=legal_category,
        )
        is_reliable, score, _ = self.score_answers(original_answer, similar_answers)
        return is_reliable, original_answer, score, similar_answers


consistency_checker = ConsistencyChecker()
