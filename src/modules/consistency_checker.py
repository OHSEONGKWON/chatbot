from __future__ import annotations

import asyncio
import json
import os
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
from .case_frame_contract import build_contract_context, build_safe_contract_answer


DEFAULT_SIMILAR_QUESTIONS = 10
MAX_SIMILAR_QUESTIONS = 10


SIMILAR_Q_SYSTEM = """당신은 법률 전문가입니다.
사용자 질문과 의미가 유사하지만 표현이 다른 질문을 정확히 {n}개 생성하세요.
반드시 JSON 배열로만 응답하세요."""

SIMILAR_Q_USER = """다음 질문과 의미가 유사한 법률 질문 {n}개를 생성하세요.
표현, 어휘, 문장 구조를 다양하게 바꾸되 핵심 법률 쟁점은 유지하세요.

[원본 질문]: {question}

응답 형식: ["질문1", "질문2", ..., "질문{n}"]"""

RAG_ANSWER_SYSTEM = """당신은 대한민국 법률 상담 챗봇입니다.
주어진 법률 참고 자료와 CaseFrame/IssuePlan/AnswerContract를 바탕으로 답변하세요.
참고 자료에 없는 조항번호·판례번호·형량·벌금액은 만들지 마세요.
사용자에게 유리한 결론도 사실관계가 부족하면 가능성으로 낮춰야 합니다.

반드시 다음 형식을 지키세요.
상황 정리
법적 판단
추가로 확인할 사항과 법적 의미
지금 할 일
도움을 받을 수 있는 곳

AnswerContract의 must_include는 가능한 한 반영하고, must_not_frame_as는 중심 쟁점으로 다루지 마세요.
특히 condition_or_exchange로 표시된 돈·시급·학점·근무조건은 이미 일한 임금 미지급으로 바꾸지 마세요."""

RAG_ANSWER_USER = """[법률 참고 자료]
{context}

[질문]
{question}

{contract_context}

위 참고 자료와 계약 조건을 바탕으로 답변하세요."""


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
        # 공식 품질지표(SQQS)와 디버그 출력용 최근 실행 리포트
        self.last_quality_report: dict[str, Any] = {}

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


    def _rule_based_similar_questions(self, question: str, n: int | None = None) -> list[str]:
        """LLM 유사 질문 생성 실패 시 원문 복붙 대신 사용하는 규칙 기반 fallback."""
        n = n or self.n_questions
        q = question or ""
        if any(x in q for x in ("허벅지", "가슴", "엉덩이", "만졌", "성추행", "강제추행")):
            candidates = [
                "선배가 술자리에서 제 신체를 만졌는데 강제추행이 될 수 있나요?",
                "원치 않는 허벅지 접촉을 당했는데 성추행으로 신고할 수 있나요?",
                "술자리에서 기습적으로 신체를 만진 경우 어떤 법적 문제가 있나요?",
                "학교 선배의 원치 않는 신체접촉에 어떻게 대응해야 하나요?",
                "증거가 부족해도 성추행 피해를 상담하거나 신고할 수 있나요?",
                "상대가 장난이었다고 해도 원치 않는 신체접촉이면 문제가 되나요?",
                "술에 취한 상태에서 신체접촉을 당했다면 준강제추행도 검토되나요?",
                "학교 안팎에서 선배에게 성추행을 당한 경우 학교 절차와 형사 절차를 어떻게 나눠야 하나요?",
                "원치 않는 신체접촉 직후 어떤 증거를 정리해야 하나요?",
                "허벅지 접촉이 강제추행으로 판단될 때 중요한 기준은 무엇인가요?",
            ]
        elif any(x in q for x in ("임금", "알바비", "주휴수당", "급여", "월급", "최저임금")):
            candidates = [
                "알바비 일부를 받지 못했는데 노동청에 진정할 수 있나요?",
                "주휴수당을 받지 못한 경우 어떤 자료를 모아야 하나요?",
                "근로계약서가 없어도 임금체불을 주장할 수 있나요?",
                "최저임금보다 적게 받은 경우 어떻게 대응해야 하나요?",
                "퇴사 후에도 밀린 임금을 청구할 수 있나요?",
            ]
        elif any(x in q for x in ("해고", "잘렸", "그만 나오", "권고사직")):
            candidates = [
                "갑자기 그만 나오라는 말을 들었는데 부당해고가 될 수 있나요?",
                "문자로 해고 통보를 받은 경우 어떤 절차를 확인해야 하나요?",
                "해고예고수당을 받을 수 있는지 판단하려면 무엇이 필요한가요?",
                "알바도 부당해고 구제신청을 할 수 있나요?",
                "해고 사유가 불명확할 때 어떻게 대응해야 하나요?",
            ]
        else:
            candidates = [
                f"{q} 이 경우 법적으로 어떤 쟁점이 있나요?",
                f"{q} 관련해서 어떤 증거가 필요할까요?",
                f"{q} 상황에서 어떻게 대응하는 것이 좋을까요?",
                f"{q} 사안에서 추가로 확인해야 할 사실은 무엇인가요?",
                f"{q} 문제를 상담이나 신고로 이어갈 수 있나요?",
            ]
        # 원문과 완전히 같은 문장은 제외하고 중복 제거
        result: list[str] = []
        norm_original = re.sub(r"\s+", " ", q).strip()
        for item in candidates:
            item = re.sub(r"\s+", " ", item).strip()
            if not item or item == norm_original:
                continue
            if item not in result:
                result.append(item)
            if len(result) >= n:
                break
        while len(result) < n:
            result.append(f"{norm_original} 관련 추가 법적 쟁점 {len(result)+1}은 무엇인가요?")
        return result[:n]

    def _dedupe_similar_questions(self, original_question: str, questions: list[str]) -> list[str]:
        cleaned: list[str] = []
        norm_original = re.sub(r"\s+", " ", (original_question or "")).strip()
        for q in questions or []:
            q = re.sub(r"\s+", " ", str(q or "")).strip()
            if not q:
                continue
            if q == norm_original:
                continue
            if any(self._jaccard_similarity(q, old) >= 0.90 for old in cleaned):
                continue
            cleaned.append(q)
            if len(cleaned) >= self.n_questions:
                break
        if len(cleaned) < max(1, min(3, self.n_questions)):
            fallback = self._rule_based_similar_questions(original_question, self.n_questions)
            for q in fallback:
                if not any(self._jaccard_similarity(q, old) >= 0.90 for old in cleaned):
                    cleaned.append(q)
                if len(cleaned) >= self.n_questions:
                    break
        return cleaned[: self.n_questions]

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
        questions: list[str] = []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                questions = [str(q).strip() for q in parsed if str(q).strip()]
            elif isinstance(parsed, dict):
                questions = [str(q).strip() for q in parsed.get("questions", []) if str(q).strip()]
        except Exception:
            questions = []

        questions = self._dedupe_similar_questions(question, questions)
        if not questions:
            questions = self._rule_based_similar_questions(question, self.n_questions)

        self._similar_questions_cache[cache_key] = list(questions)
        self._similar_questions_cache.move_to_end(cache_key)
        while len(self._similar_questions_cache) > self._similar_questions_cache_size:
            self._similar_questions_cache.popitem(last=False)
        return questions

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

    async def _generate_single_answer(self, question: str, rag_docs: list[dict[str, Any]], category: str = "", case_frame: Any | None = None, issue_plan: Any | None = None, answer_contract: Any | None = None) -> str:
        context = self._format_docs(rag_docs)
        contract_context = ""
        if case_frame is not None and issue_plan is not None and answer_contract is not None:
            contract_context = build_contract_context(case_frame, issue_plan, answer_contract)
        answer = await llm_client.complete(
            system_prompt=RAG_ANSWER_SYSTEM,
            user_prompt=RAG_ANSWER_USER.format(context=context, question=question, contract_context=contract_context),
            temperature=0.1,
            model=llm_client.answer_model,
        )
        if answer:
            return answer
        return build_safe_contract_answer(case_frame, issue_plan) if case_frame is not None and issue_plan is not None else self._fallback_answer(question, rag_docs, category)

    async def generate_all_answers(
        self,
        original_question: str,
        similar_questions: list[str],
        rag_docs: list[dict[str, Any]],
        category: str = "",
        case_frame: Any | None = None,
        issue_plan: Any | None = None,
        answer_contract: Any | None = None,
    ) -> tuple[str, list[str]]:
        all_questions = [original_question] + similar_questions
        tasks = [self._generate_single_answer(q, rag_docs, category=category, case_frame=case_frame, issue_plan=issue_plan, answer_contract=answer_contract) for q in all_questions]
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
        median_score = float(statistics.median(scores))
        return median_score >= self.threshold, median_score, scores


    def _tokenize_simple(self, text: str) -> set[str]:
        text = re.sub(r"\s+", " ", (text or "").strip().lower())
        text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
        return {t for t in text.split() if t}

    def _jaccard_similarity(self, a: str, b: str) -> float:
        ta, tb = self._tokenize_simple(a), self._tokenize_simple(b)
        if not ta and not tb:
            return 1.0
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(1, len(ta | tb))

    def _detect_duplicate_rate(self, questions: list[str], threshold: float = 0.90) -> float:
        if not questions:
            return 0.0
        duplicate_indices: set[int] = set()
        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                if self._jaccard_similarity(questions[i], questions[j]) >= threshold:
                    duplicate_indices.add(j)
        return len(duplicate_indices) / max(1, len(questions))

    def _core_slots_from_question(self, question: str) -> set[str]:
        """SQQS용 핵심 법률 슬롯 추정. 기존 라우터를 건드리지 않는 가벼운 휴리스틱."""
        q = question or ""
        slots: set[str] = set()

        if any(x in q for x in ("사장", "점장", "알바", "회사", "직장", "상사", "인턴")):
            slots.add("employment_relationship")
        if any(x in q for x in ("대학", "동기", "선배", "교수", "학교", "학생")):
            slots.add("school_relationship")

        if any(x in q for x in ("야근", "연장근무", "추가 근무")):
            slots.add("overtime_work")
        if any(x in q for x in ("돈 더", "추가수당", "수당", "시급", "월급", "급여", "알바비")):
            slots.add("wage_or_pay")
        if any(x in q for x in ("안 줬", "안줬", "못 받", "못받", "미지급", "체불", "지급하지")):
            slots.add("nonpayment")

        if any(x in q for x in ("성관계", "잠자리")):
            slots.add("sexual_intercourse")
        if any(x in q for x in ("술", "취해서", "만취", "기억")):
            slots.add("incapacity_or_memory_loss")

        if any(x in q for x in ("키스", "만졌", "가슴", "허벅지", "엉덩이", "신체접촉")):
            slots.add("sexual_physical_contact")
        if any(x in q for x in ("갑자기", "동의 없이", "싫다", "원치 않")):
            slots.add("nonconsent_or_surprise")

        if any(x in q for x in ("몰래", "동의 없이")) and any(x in q for x in ("사진", "영상", "촬영", "찍")):
            slots.add("secret_recording")
        if any(x in q for x in ("유포", "퍼뜨리", "올리겠", "공개하겠")):
            slots.add("distribution_or_threat")

        if any(x in q for x in ("차단", "거부")):
            slots.add("refusal")
        if any(x in q for x in ("계속 연락", "반복 연락", "자꾸 연락", "다른 계정", "부계")):
            slots.add("repeated_contact")

        if any(x in q for x in ("몸매", "외모", "성적 발언", "음담패설")):
            slots.add("sexual_comment")
        if any(x in q for x in ("계속", "반복", "자꾸")):
            slots.add("repetition")

        return slots

    def _calc_legal_core_preservation(self, original_question: str, similar_questions: list[str]) -> float:
        base = self._core_slots_from_question(original_question)
        if not base:
            return 1.0
        scores = []
        for question in similar_questions:
            slots = self._core_slots_from_question(question)
            scores.append(len(base & slots) / len(base))
        return float(sum(scores) / max(1, len(scores)))

    def calc_sqqs_simple(self, original_question: str, similar_questions: list[str]) -> dict[str, Any]:
        """SQQS = 0.5 * legal_core + 0.3 * semantic_similarity + 0.2 * diversity_non_duplication"""
        if not similar_questions:
            return {
                "score": 0.0,
                "legal_core_preservation": 0.0,
                "semantic_similarity": 0.0,
                "diversity_non_duplication": 0.0,
                "duplicate_rate": 1.0,
            }

        # 기존 embedder가 있으면 의미 유사도는 임베딩 기반, 실패하면 Jaccard 기반
        try:
            embedder = self._load_embedder()
            if embedder is not None:
                texts = ["query: " + original_question] + ["query: " + q for q in similar_questions]
                emb = np.asarray(embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False))
                original_emb = emb[0]
                similar_embs = emb[1:]
                semantic_similarity = float(np.mean(similar_embs @ original_emb))
            else:
                semantic_similarity = float(sum(self._jaccard_similarity(original_question, q) for q in similar_questions) / len(similar_questions))
        except Exception:
            semantic_similarity = float(sum(self._jaccard_similarity(original_question, q) for q in similar_questions) / len(similar_questions))

        legal_core = self._calc_legal_core_preservation(original_question, similar_questions)
        duplicate_rate = self._detect_duplicate_rate(similar_questions)
        diversity_non_duplication = 1.0 - duplicate_rate

        score = 0.45 * legal_core + 0.25 * semantic_similarity + 0.30 * diversity_non_duplication
        valid = duplicate_rate < 0.50 and diversity_non_duplication >= 0.50
        # 유사 질문이 복붙이면 SQQS를 강하게 캡핑한다.
        if duplicate_rate >= 0.50:
            score = min(score, 0.35)
        elif duplicate_rate >= 0.30:
            score = min(score, 0.55)
        return {
            "score": round(max(0.0, min(1.0, score)), 3),
            "legal_core_preservation": round(max(0.0, min(1.0, legal_core)), 3),
            "semantic_similarity": round(max(0.0, min(1.0, semantic_similarity)), 3),
            "diversity_non_duplication": round(max(0.0, min(1.0, diversity_non_duplication)), 3),
            "duplicate_rate": round(max(0.0, min(1.0, duplicate_rate)), 3),
            "valid": bool(valid),
        }

    def _print_quality_trace(self, question: str, original_answer: str, similar_questions: list[str], similar_answers: list[str], quality_report: dict[str, Any]) -> None:
        if os.getenv("LAWSGUARD_PRINT_QUALITY_TRACE", "1") != "1":
            return
        print("\n" + "=" * 72)
        print("LLM 생성 질문 및 각 질문에 대한 답변")
        print("=" * 72)
        print("\n[원 질문]")
        print(question)
        print("\n[원 질문 답변]")
        print(original_answer)
        for idx, (sq, ans) in enumerate(zip(similar_questions, similar_answers), 1):
            print("\n" + "-" * 72)
            print(f"[LLM 생성 질문 {idx}]")
            print(sq)
            print(f"\n[LLM 생성 질문 {idx}에 대한 답변]")
            print(ans)

        print("\n" + "=" * 72)
        print("공식 대표 지표")
        print("=" * 72)
        print(f"SQQS: {quality_report.get('SQQS')}")
        print(f"raw_consistency_score: {quality_report.get('raw_consistency_score')}")
        print(f"answer_reliability: {quality_report.get('answer_reliability')}")
    async def run(self, question: str, rag_docs: list[dict[str, Any]] | None = None, legal_category: str = "", case_frame: Any | None = None, issue_plan: Any | None = None, answer_contract: Any | None = None) -> tuple[bool, str, float, list[str]]:
        rag_docs = rag_docs or []
        similar_questions = await self.generate_similar_questions(question)
        original_answer, similar_answers = await self.generate_all_answers(
            original_question=question,
            similar_questions=similar_questions,
            rag_docs=rag_docs,
            category=legal_category,
            case_frame=case_frame,
            issue_plan=issue_plan,
            answer_contract=answer_contract,
        )

        is_reliable, score, raw_scores = self.score_answers(original_answer, similar_answers)

        sqqs_report = self.calc_sqqs_simple(question, similar_questions)
        answer_reliability = round(score * min(1.0, sqqs_report["score"] / 0.5), 3)

        self.last_quality_report = {
            "SQQS": sqqs_report["score"],
            "SQQS_detail": sqqs_report,
            "original_question": question,
            "original_answer": original_answer,
            "similar_questions": similar_questions,
            "similar_answers": similar_answers,
            "raw_answer_similarity_scores": [round(float(s), 3) for s in raw_scores],
            "raw_consistency_score": round(float(score), 3),
            "answer_reliability": answer_reliability,
            "legal_category": legal_category,
        }

        self._print_quality_trace(question, original_answer, similar_questions, similar_answers, self.last_quality_report)

        return is_reliable, original_answer, score, similar_answers


consistency_checker = ConsistencyChecker()
