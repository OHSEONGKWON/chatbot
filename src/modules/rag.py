from __future__ import annotations

import asyncio
from collections import Counter, OrderedDict
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


ISSUE_SPECS: dict[str, dict[str, Any]] = {
    "wage_unpaid": {
        "category": "노동",
        "triggers": ["임금체불", "체불", "알바비", "월급", "급여", "못 받", "안 줬", "미지급", "임금 못"],
        "expand": "임금체불 임금 지급 근로기준법 제43조 제36조 제37조 고용노동부 진정",
        "preferred": ["근로기준법 제43조", "제43조", "임금은", "임금 지급", "임금체불", "체불"],
        "secondary": ["제36조", "제37조", "고용노동부", "진정"],
    },
    "minimum_wage": {
        "category": "노동",
        "triggers": ["최저임금", "최저시급", "시급이 낮", "시급 적", "최저보다", "최저 임금"],
        "expand": "최저임금 최저임금법 제6조 최저시급 미달 임금",
        "preferred": ["최저임금법", "제6조", "최저임금", "최저시급"],
        "secondary": ["임금", "근로자"],
    },
    "dismissal": {
        "category": "노동",
        "triggers": ["해고", "잘렸", "그만 나오", "그만두라", "부당해고", "해고예고", "서면 통지"],
        "expand": "해고 부당해고 근로기준법 제23조 제26조 제27조 해고예고 서면통지 노동위원회",
        "preferred": ["근로기준법 제23조", "제23조", "부당해고", "정당한 이유"],
        "secondary": ["제26조", "해고예고", "제27조", "서면", "노동위원회"],
    },
    "missing_contract": {
        "category": "노동",
        "triggers": ["근로계약서", "계약서 안", "계약서 없", "계약서 미작성", "서면 계약", "근로조건", "근무시간", "구두로", "구두 조건"],
        "expand": "근로계약서 근로조건 서면 명시 근로기준법 제17조 계약서 미작성",
        "preferred": ["근로기준법 제17조", "제17조", "근로조건", "서면", "근로계약서"],
        "secondary": ["명시", "계약서"],
    },
    "workplace_harassment": {
        "category": "노동",
        "triggers": ["직장 내 괴롭힘", "직장내 괴롭힘", "괴롭힘", "폭언", "따돌림", "모욕", "갑질"],
        "expand": "직장 내 괴롭힘 근로기준법 제76조의2 제76조의3 예방 대응 매뉴얼",
        "preferred": ["근로기준법 제76조의2", "제76조의2", "직장 내 괴롭힘"],
        "secondary": ["제76조의3", "예방", "대응 매뉴얼", "조치"],
    },
    "sexual_harassment": {
        "category": "성폭력",
        "triggers": ["성희롱", "성적 농담", "외모 평가", "성적 발언", "음담패설"],
        "expand": "성희롱 신고 상담 양성평등기본법 제30조 공공부문 성희롱 성폭력 사건 처리 매뉴얼",
        "preferred": ["성희롱", "양성평등기본법 제30조", "제30조", "상담 및 신고 접수"],
        "secondary": ["공공부문 성희롱 성폭력 사건 처리 매뉴얼", "신고", "상담"],
    },
    "indecent_assault": {
        "category": "성폭력",
        "triggers": ["강제추행", "성추행", "동의 없이 몸", "몸을 만", "허리를 만", "가슴", "엉덩이", "입맞춤"],
        "expand": "강제추행 형법 제298조 동의 없는 신체접촉 성폭력 피해자 보호 상담 신고",
        "preferred": ["형법 제298조", "제298조", "강제추행"],
        "secondary": ["성폭력", "신체접촉", "상담", "신고"],
    },
    "illegal_filming": {
        "category": "성폭력",
        "triggers": ["불법촬영", "몰카", "몰래 촬영", "동의 없이 사진", "동의 없이 영상", "찍은 영상", "카메라", "촬영물", "유포", "단톡방"],
        "expand": "불법촬영 카메라 촬영 성폭력범죄의 처벌 등에 관한 특례법 제14조 촬영물 반포",
        "preferred": ["성폭력범죄의 처벌 등에 관한 특례법 제14조", "제14조", "카메라", "촬영"],
        "secondary": ["촬영물", "반포", "복제물", "불법촬영"],
    },
}


class LegalRetriever:
    def __init__(self):
        self._jsonl_cache: list[RAGDocument] | None = None
        self._bm25_index: list[tuple[RAGDocument, list[str], Counter]] | None = None
        self._doc_freq: Counter | None = None
        self._avg_doc_len = 0.0
        self._query_cache: OrderedDict[tuple, list[dict[str, Any]]] = OrderedDict()
        self._query_cache_size = 64

    async def retrieve_async(
        self,
        query: str,
        top_k: int | None = None,
        legal_category: str = "",
        case_frame: Any | None = None,
        issue_plan: Any | None = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.retrieve, query, top_k, legal_category, case_frame, issue_plan)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        legal_category: str = "",
        case_frame: Any | None = None,
        issue_plan: Any | None = None,
    ) -> list[dict[str, Any]]:
        top_k = top_k or config.rag.top_k
        cache_key = (
            re.sub(r"\s+", " ", (query or "")).strip().lower(),
            top_k,
            legal_category or "",
            str(getattr(issue_plan, "primary_issue", "")),
        )
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            self._query_cache.move_to_end(cache_key)
            return [doc.copy() for doc in cached]

        docs = self._retrieve_from_jsonl(query, top_k, legal_category=legal_category, issue_plan=issue_plan)
        result = []
        for doc in self._dedupe_docs(docs)[:top_k]:
            payload = doc.__dict__.copy()
            payload.setdefault("content", payload.get("text", ""))
            result.append(payload)
        self._query_cache[cache_key] = [doc.copy() for doc in result]
        self._query_cache.move_to_end(cache_key)
        while len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)
        return result

    async def warmup(self):
        await asyncio.to_thread(self._ensure_bm25_index)

    def evaluate_retrieval(
        self,
        docs: list[dict[str, Any]],
        query: str = "",
        legal_category: str = "",
        case_frame: Any | None = None,
        issue_plan: Any | None = None,
    ) -> dict[str, Any]:
        """RAG 검색 품질 게이트. 반환 score로 pipeline 라우팅을 결정한다."""
        n = len(docs or [])
        if n == 0:
            return {"score": 0.0, "passed": False, "action": "requery_or_safe_fallback", "doc_count": 0}
        top_scores = [float(doc.get("score") or 0.0) for doc in docs[:3]]
        max_score = max(top_scores)
        # BM25 스케일 완화: 전형적인 좋은 점수 ~15-20을 [0,1]로 포화 정규화
        normalized = min(1.0, max_score / (max_score + 10.0)) if max_score > 0 else 0.0
        count_bonus = min(0.15, n * 0.05)
        score = min(1.0, normalized + count_bonus)
        passed = score >= 0.72
        action = "answer" if score >= 0.85 else ("answer_with_caution" if passed else "requery_or_safe_fallback")
        return {"score": round(score, 3), "passed": passed, "action": action, "doc_count": n}

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

    def _retrieve_from_jsonl(
        self,
        query: str,
        top_k: int,
        legal_category: str = "",
        issue_plan: Any | None = None,
    ) -> list[RAGDocument]:
        category = legal_category or self._infer_category(query)
        issues = (
            [getattr(issue_plan, "primary_issue", "")]
            if getattr(issue_plan, "primary_issue", "")
            else self.infer_issues(query, category=category)
        )
        query_tokens = self._expand_query_tokens(query, category, issues=issues, issue_plan=issue_plan)
        if not query_tokens:
            return []
        self._ensure_bm25_index()

        scored: list[RAGDocument] = []
        for doc, doc_tokens, token_counts in self._bm25_index or []:
            if self._has_suspect_statute_metadata(doc.metadata):
                continue
            haystack = f"{doc.text} {json.dumps(doc.metadata, ensure_ascii=False)}"
            bm25 = self._bm25_score(query_tokens, doc_tokens, token_counts)
            if bm25 <= 0:
                continue
            issue_boost, issue_reasons = self._issue_boost(haystack, doc.metadata, issues)
            metadata_boost = self._metadata_boost(doc.metadata, category)
            score = bm25 + metadata_boost + issue_boost
            if score > 0:
                scored.append(
                    RAGDocument(
                        text=doc.text,
                        metadata=doc.metadata,
                        chunk_id=doc.chunk_id,
                        score=score,
                        retrieval_reason=f"bm25={bm25:.2f},cat={category or 'none'},issues={','.join(issues) or 'none'}{issue_reasons}",
                    )
                )
        scored.sort(key=lambda d: d.score, reverse=True)
        return self._balance_sources(scored, top_k, category, issues=issues, issue_plan=issue_plan)

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

    def _expand_query_tokens(
        self,
        text: str,
        category: str = "",
        issues: list[str] | None = None,
        issue_plan: Any | None = None,
    ) -> list[str]:
        tokens = self._tokenize_for_index(text)

        if category == "노동":
            tokens.extend("근로기준법 임금 근로계약서 고용노동부 사업주 지급".split())
        elif category == "성폭력":
            tokens.extend("성폭력 성희롱 강제추행 피해자 보호 상담 신고 동의".split())

        for issue in issues or []:
            if issue in ISSUE_SPECS:
                tokens.extend(str(ISSUE_SPECS[issue].get("expand", "")).split())

        if issue_plan is not None:
            tokens.extend(str(getattr(issue_plan, "primary_issue", "")).split())
            for law in getattr(issue_plan, "needed_laws", []) or []:
                tokens.extend(str(law).split())
            for focus in getattr(issue_plan, "answer_focus", []) or []:
                tokens.extend(str(focus).split())

        return tokens

    def _metadata_boost(self, metadata: dict[str, Any], category: str) -> float:
        source_type = str(metadata.get("source_type") or "")
        source_boost = {"statute": 1.5, "manual": 1.0, "case": 0.5}.get(source_type, 0.0)

        title = " ".join(
            str(metadata.get(key) or "")
            for key in ["law_name", "source_file", "article_title", "section_label"]
        )
        category_boost = 0.0
        if category == "노동" and any(k in title for k in ["근로기준법", "노동", "고용"]):
            category_boost = 3.0
        elif category == "성폭력" and any(k in title for k in ["성희롱", "성폭력", "강제추행", "강간"]):
            category_boost = 3.0

        return source_boost + category_boost

    def infer_issues(self, query: str, category: str = "") -> list[str]:
        normalized = re.sub(r"\s+", " ", query.lower())
        scored: list[tuple[int, str]] = []
        for issue, spec in ISSUE_SPECS.items():
            if category and spec.get("category") != category:
                continue
            score = sum(1 for trigger in spec.get("triggers", []) if trigger.lower() in normalized)
            if score:
                scored.append((score, issue))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [issue for _, issue in scored[:3]]

    def _issue_boost(self, text: str, metadata: dict[str, Any], issues: list[str]) -> tuple[float, str]:
        if not issues:
            return 0.0, ""
        haystack = f"{text} {json.dumps(metadata, ensure_ascii=False)}".lower()
        compact = re.sub(r"\s+", "", haystack)
        boost = 0.0
        reasons = []
        for issue in issues:
            spec = ISSUE_SPECS.get(issue, {})
            issue_boost = 0.0
            for term in spec.get("preferred", []):
                if term.lower() in haystack or re.sub(r"\s+", "", term.lower()) in compact:
                    issue_boost += 9.0
            for term in spec.get("secondary", []):
                if term.lower() in haystack or re.sub(r"\s+", "", term.lower()) in compact:
                    issue_boost += 3.0
            source_type = str(metadata.get("source_type") or "")
            if source_type == "case":
                issue_boost -= 2.0
            if issue_boost:
                boost += issue_boost
                reasons.append(f"{issue}:{issue_boost:.1f}")
        return boost, f", issue_boost={';'.join(reasons)}" if reasons else ""

    def _balance_sources(
        self,
        docs: list[RAGDocument],
        top_k: int,
        category: str,
        issues: list[str] | None = None,
        issue_plan: Any | None = None,
    ) -> list[RAGDocument]:
        preferred_slots = ["statute", "manual", "statute"]
        if category == "성폭력":
            preferred_slots = ["manual", "statute", "manual"]
        if issues and any(
            issue in {"wage_unpaid", "minimum_wage", "dismissal", "missing_contract", "indecent_assault", "illegal_filming"}
            for issue in issues
        ):
            preferred_slots = ["statute", "manual", "statute"]
        if issues and "workplace_harassment" in issues:
            preferred_slots = ["statute", "manual", "manual"]

        selected: list[RAGDocument] = []
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

    def _has_suspect_statute_metadata(self, metadata: dict[str, Any]) -> bool:
        law_name = re.sub(r"\s+", "", str((metadata or {}).get("law_name") or ""))
        article_id = str((metadata or {}).get("article_id") or "")
        if "성폭력범죄의처벌등에관한특례법" in law_name:
            match = re.search(r"제(\d+)조", article_id)
            if match and 297 <= int(match.group(1)) <= 305:
                return True
        return False

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
        if any(
            term in query
            for term in ["성폭력", "성추행", "성희롱", "강제추행", "강간", "불법촬영", "동의 없이", "몸을 만", "거절"]
        ):
            return "성폭력"
        if any(
            term in query
            for term in ["알바", "알바비", "아르바이트", "임금", "월급", "급여", "근로", "해고", "퇴직금"]
        ):
            return "노동"
        return ""


retriever = LegalRetriever()
