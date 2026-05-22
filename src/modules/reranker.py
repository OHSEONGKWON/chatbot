from __future__ import annotations

import json
import re
from typing import Any

_DEFAULT_MODEL = "bongsoo/klue-cross-encoder-v1"
_FALLBACK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_LLM_SYSTEM = """\
당신은 한국 법률 RAG 시스템의 검색 결과 재순위 전문가입니다.
사용자 질문과 검색된 법률 문서 목록이 주어집니다.
각 문서가 질문에 대한 답변에 얼마나 직접적으로 유용한지 평가하고,
가장 관련성 높은 순서대로 문서 번호를 JSON 배열로 반환하세요.

반드시 아래 형식으로만 응답하세요 (다른 텍스트 없이):
{"ranked": [3, 1, 5, 2, 4, ...]}\
"""


class CrossEncoderReranker:
    """BM25 retriever 결과를 Cross-Encoder로 재순위화."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        try:
            self._model = CrossEncoder(self._model_name)
        except Exception:
            self._model = CrossEncoder(_FALLBACK_MODEL)

    def rerank(
        self,
        query: str,
        docs: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """docs를 cross-encoder score 기준으로 재정렬 후 top_k 반환."""
        if not docs:
            return docs
        self._load()

        pairs = [(query, (doc.get("text") or doc.get("content") or "")[:512]) for doc in docs]
        scores = self._model.predict(pairs)

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        result = []
        for score, doc in ranked:
            d = doc.copy()
            d["rerank_score"] = float(score)
            result.append(d)
        return result[:top_k] if top_k else result


class LLMReranker:
    """Claude API를 사용한 listwise 재순위화.

    쿼리 + top-K 문서를 한 번에 전달하고, Claude가 한국 법률 도메인
    이해를 바탕으로 관련도 순서로 재정렬한다. 쿼리당 1회 API 호출.
    """

    def __init__(self, model: str = "claude-haiku-4-5", max_docs: int = 20):
        self._model = model
        self._max_docs = max_docs
        self._client = None

    def _get_client(self):
        if self._client is None:
            from dotenv import load_dotenv
            from pathlib import Path
            import anthropic
            load_dotenv(Path(__file__).resolve().parents[2] / ".env")
            self._client = anthropic.Anthropic()
        return self._client

    def rerank(
        self,
        query: str,
        docs: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """docs를 Claude listwise reranking으로 재정렬 후 top_k 반환."""
        if not docs:
            return docs

        candidates = docs[: self._max_docs]
        doc_lines = []
        for i, doc in enumerate(candidates, 1):
            text = (doc.get("text") or doc.get("content") or "")[:400]
            doc_lines.append(f"[{i}] {text}")

        user_msg = (
            f"질문: {query}\n\n"
            f"검색된 문서 {len(candidates)}개:\n\n"
            + "\n\n".join(doc_lines)
        )

        try:
            client = self._get_client()
            resp = client.messages.create(
                model=self._model,
                max_tokens=200,
                system=[{
                    "type": "text",
                    "text": _LLM_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text.strip()
            # JSON 블록만 추출 (마크다운 코드블록 또는 중괄호 직접 탐색)
            match = re.search(r'\{[^{}]*"ranked"\s*:\s*\[[^\]]*\][^{}]*\}', raw, re.DOTALL)
            if match:
                raw = match.group(0)
            elif raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            order = json.loads(raw)["ranked"]
            # 1-based → 0-based, 범위 검증
            indices = [i - 1 for i in order if 1 <= i <= len(candidates)]
            # 누락된 인덱스는 뒤에 추가
            seen = set(indices)
            indices += [i for i in range(len(candidates)) if i not in seen]
        except Exception as e:
            print(f"    [LLMReranker error] {e} - 원래 순서 유지")
            indices = list(range(len(candidates)))

        reranked = [candidates[i] for i in indices]
        # max_docs를 초과한 원래 docs는 뒤에 붙임
        reranked += docs[self._max_docs:]

        return reranked[:top_k] if top_k else reranked


reranker = CrossEncoderReranker()
llm_reranker = LLMReranker()
