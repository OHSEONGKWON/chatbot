"""
일관성 검사 모듈 (Step 1~4)
"""

import asyncio
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

from config import config
from modules.llm_client import llm_client
from modules.rag import retriever

SIMILAR_Q_SYSTEM = """당신은 법률 전문가입니다. 사용자 질문과 의미가 유사하지만
표현이 다른 질문을 정확히 {n}개 생성하세요. 반드시 JSON 배열로만 응답하세요."""

SIMILAR_Q_USER = """다음 질문과 의미가 유사한 법률 질문 {n}개를 생성하세요.
표현, 어휘, 문장 구조를 다양하게 바꾸되 핵심 법률 쟁점은 유지하세요.

[원본 질문]: {question}

응답 형식: [\"질문1\", \"질문2\", ..., \"질문{n}\"]"""

RAG_ANSWER_SYSTEM = """당신은 대한민국 법률 전문가입니다.
주어진 법률 참고 자료를 바탕으로 질문에 정확하고 간결하게 답변하세요.
참고 자료에 없는 내용은 추측하지 마세요."""

RAG_ANSWER_USER = """[법률 참고 자료]
{context}

[질문]
{question}

위 참고 자료를 바탕으로 답변하세요."""


class ConsistencyChecker:
	def __init__(self):
		cfg = config.hallucination
		self.n_questions = cfg.num_similar_questions
		self.threshold = cfg.consistency_threshold
		self.w_nli = cfg.similarity_weight_nli
		self.w_emb = cfg.similarity_weight_embed
		self._embedder = SentenceTransformer(config.rag.embedding_model, device=self._get_device())
		self._nli = hf_pipeline(
			task="text-classification",
			model=cfg.nli_model,
			tokenizer=cfg.nli_model,
			device=0 if self._get_device() == "cuda" else -1,
			top_k=None,
			truncation=True,
			max_length=512,
		)

	@staticmethod
	def _get_device() -> str:
		try:
			import torch
			return "cuda" if torch.cuda.is_available() else "cpu"
		except ImportError:
			return "cpu"

	async def generate_similar_questions(self, question: str) -> list[str]:
		raw = await llm_client.complete(
			system_prompt=SIMILAR_Q_SYSTEM.format(n=self.n_questions),
			user_prompt=SIMILAR_Q_USER.format(n=self.n_questions, question=question),
			temperature=0.8,
			json_mode=True,
		)
		try:
			questions = json.loads(raw)
			if isinstance(questions, list):
				questions = [str(q) for q in questions[: self.n_questions]]
				while len(questions) < self.n_questions:
					questions.append(question)
				return questions
		except (json.JSONDecodeError, TypeError):
			pass
		return [question] * self.n_questions

	async def _generate_single_answer(self, question: str) -> str:
		docs = await retriever.retrieve_async(question)
		context = retriever.build_context(docs)
		answer = await llm_client.complete(
			system_prompt=RAG_ANSWER_SYSTEM,
			user_prompt=RAG_ANSWER_USER.format(context=context, question=question),
			temperature=0.1,
		)
		return answer

	async def generate_all_answers(self, original_question: str, similar_questions: list[str]) -> tuple[str, list[str]]:
		all_questions = [original_question] + similar_questions
		tasks = [self._generate_single_answer(q) for q in all_questions]
		results = await asyncio.gather(*tasks)
		return results[0], list(results[1:])

	def _nli_score(self, premise: str, hypothesis: str) -> float:
		premise_trunc = premise[:400]
		hyp_trunc = hypothesis[:400]
		results = self._nli(f"{premise_trunc} [SEP] {hyp_trunc}")
		label_map = {r["label"].lower(): r["score"] for r in results}
		entail = label_map.get("entailment", label_map.get("함의", 0.0))
		neutral = label_map.get("neutral", label_map.get("중립", 0.0))
		return entail + 0.3 * neutral

	def _embedding_similarity(self, text1: str, text2: str) -> float:
		embs = self._embedder.encode(["query: " + text1, "query: " + text2], normalize_embeddings=True)
		return float(np.dot(embs[0], embs[1]))

	def compute_pairwise_score(self, original_answer: str, similar_answer: str) -> float:
		nli = self._nli_score(original_answer, similar_answer)
		emb = self._embedding_similarity(original_answer, similar_answer)
		return self.w_nli * nli + self.w_emb * emb

	def compute_consistency_score(self, original_answer: str, similar_answers: list[str]) -> tuple[float, list[float]]:
		scores = [self.compute_pairwise_score(original_answer, ans) for ans in similar_answers]
		return float(np.mean(scores)), scores

	def is_reliable(self, avg_score: float) -> bool:
		return avg_score >= self.threshold

	async def run(self, question: str) -> tuple[bool, str, float, list[float]]:
		similar_qs = await self.generate_similar_questions(question)
		original_answer, similar_answers = await self.generate_all_answers(question, similar_qs)
		avg_score, scores = self.compute_consistency_score(original_answer, similar_answers)
		return self.is_reliable(avg_score), original_answer, avg_score, scores


consistency_checker = ConsistencyChecker()
