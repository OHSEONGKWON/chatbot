from __future__ import annotations

import asyncio
from collections import Counter
from collections import OrderedDict
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
        "penalty": ["성희롱", "성폭력", "강제추행", "불법촬영"],
    },
    "minimum_wage": {
        "category": "노동",
        "triggers": ["최저임금", "최저시급", "시급이 낮", "시급 적", "최저보다", "최저 임금"],
        "expand": "최저임금 최저임금법 제6조 최저시급 미달 임금",
        "preferred": ["최저임금법", "제6조", "최저임금", "최저시급"],
        "secondary": ["임금", "근로자"],
        "penalty": ["성희롱", "성폭력", "강제추행"],
    },
    "dismissal": {
        "category": "노동",
        "triggers": ["해고", "잘렸", "그만 나오", "그만두라", "부당해고", "해고예고", "서면 통지"],
        "expand": "해고 부당해고 근로기준법 제23조 제26조 제27조 해고예고 서면통지 노동위원회",
        "preferred": ["근로기준법 제23조", "제23조", "부당해고", "정당한 이유"],
        "secondary": ["제26조", "해고예고", "제27조", "서면", "노동위원회"],
        "penalty": ["임금체불", "강제추행", "성폭력"],
    },
    "missing_contract": {
        "category": "노동",
        "triggers": ["근로계약서", "계약서 안", "계약서 없", "계약서 미작성", "서면 계약", "근로조건", "근무시간", "구두로", "구두 조건"],
        "expand": "근로계약서 근로조건 서면 명시 근로기준법 제17조 계약서 미작성",
        "preferred": ["근로기준법 제17조", "제17조", "근로조건", "서면", "근로계약서"],
        "secondary": ["명시", "계약서"],
        "penalty": ["성희롱", "성폭력"],
    },
    "workplace_harassment": {
        "category": "노동",
        "triggers": ["직장 내 괴롭힘", "직장내 괴롭힘", "괴롭힘", "폭언", "따돌림", "모욕", "갑질"],
        "expand": "직장 내 괴롭힘 근로기준법 제76조의2 제76조의3 예방 대응 매뉴얼",
        "preferred": ["근로기준법 제76조의2", "제76조의2", "직장 내 괴롭힘"],
        "secondary": ["제76조의3", "예방", "대응 매뉴얼", "조치"],
        "penalty": ["강제추행", "불법촬영"],
    },
    "sexual_harassment": {
        "category": "성폭력",
        "triggers": ["성희롱", "성적 농담", "외모 평가", "성적 발언", "음담패설"],
        "expand": "성희롱 신고 상담 양성평등기본법 제30조 공공부문 성희롱 성폭력 사건 처리 매뉴얼",
        "preferred": ["성희롱", "양성평등기본법 제30조", "제30조", "상담 및 신고 접수"],
        "secondary": ["공공부문 성희롱 성폭력 사건 처리 매뉴얼", "신고", "상담"],
        "penalty": ["임금체불", "최저임금", "해고"],
    },
    "indecent_assault": {
        "category": "성폭력",
        "triggers": ["강제추행", "성추행", "동의 없이 몸", "몸을 만", "허리를 만", "가슴", "엉덩이", "입맞춤"],
        "expand": "강제추행 형법 제298조 동의 없는 신체접촉 성폭력 피해자 보호 상담 신고",
        "preferred": ["형법 제298조", "제298조", "강제추행"],
        "secondary": ["성폭력", "신체접촉", "상담", "신고"],
        "penalty": ["임금체불", "최저임금", "해고"],
    },
    "illegal_filming": {
        "category": "성폭력",
        "triggers": ["불법촬영", "몰카", "몰래 촬영", "동의 없이 사진", "동의 없이 영상", "찍은 영상", "카메라", "촬영물", "유포", "단톡방"],
        "expand": "불법촬영 카메라 촬영 성폭력범죄의 처벌 등에 관한 특례법 제14조 촬영물 반포",
        "preferred": ["성폭력범죄의 처벌 등에 관한 특례법 제14조", "제14조", "카메라", "촬영"],
        "secondary": ["촬영물", "반포", "복제물", "불법촬영"],
        "penalty": ["임금체불", "해고", "강제추행"],
    },
}


class LegalRetriever:
    def __init__(self):
        self._collection = None
        self._embedder = None
        self._jsonl_cache: list[RAGDocument] | None = None
        self._bm25_index: list[tuple[RAGDocument, list[str], Counter]] | None = None
        self._doc_freq: Counter | None = None
        self._avg_doc_len = 0.0
        self._query_cache: OrderedDict[tuple[str, int, str], list[dict[str, Any]]] = OrderedDict()
        self._query_cache_size = 64

    async def retrieve_async(self, query: str, top_k: int | None = None, legal_category: str = "", case_frame: Any | None = None, issue_plan: Any | None = None) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.retrieve, query, top_k, legal_category, case_frame, issue_plan)

    def retrieve(self, query: str, top_k: int | None = None, legal_category: str = "", case_frame: Any | None = None, issue_plan: Any | None = None) -> list[dict[str, Any]]:
        top_k = top_k or config.rag.top_k
        cache_key = (re.sub(r"\s+", " ", (query or "")).strip().lower(), top_k, legal_category or "", str(getattr(case_frame, "condition_or_exchange", "")), str(getattr(case_frame, "unpaid_claim", "")), str(getattr(issue_plan, "primary_issue", "")))
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            self._query_cache.move_to_end(cache_key)
            return [doc.copy() for doc in cached]

        docs = self._retrieve_from_chroma(query, top_k)
        if not docs:
            docs = self._retrieve_from_jsonl(query, top_k, legal_category=legal_category)
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
        await asyncio.to_thread(self._load_jsonl_cache)
        if config.rag.enable_vector:
            await asyncio.to_thread(self._embed_query, "법률 상담")

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
                result = self._collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
            else:
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
                metadata = metadatas[idx] or {}
                if self._has_suspect_statute_metadata(metadata):
                    continue
                distance = float(distances[idx]) if idx < len(distances) else 0.0
                docs.append(
                    RAGDocument(
                        text=text or "",
                        metadata=metadata,
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
                try:
                    self._embedder = SentenceTransformer(config.rag.embedding_model, local_files_only=True)
                except Exception:
                    self._embedder = SentenceTransformer(config.rag.embedding_model)
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

    def _retrieve_from_jsonl(self, query: str, top_k: int, legal_category: str = "", case_frame: Any | None = None, issue_plan: Any | None = None) -> list[RAGDocument]:
        category = legal_category or self._infer_category(query)
        issues = [getattr(issue_plan, "primary_issue", "")] if getattr(issue_plan, "primary_issue", "") else self.infer_issues(query, category=category)
        query_tokens = self._expand_query_tokens(query, category, issues=issues, case_frame=case_frame, issue_plan=issue_plan)
        if not query_tokens:
            return []
        self._ensure_bm25_index()

        scored = []
        for doc, doc_tokens, token_counts in self._bm25_index or []:
            if self._has_suspect_statute_metadata(doc.metadata):
                continue
            haystack = f"{doc.text} {json.dumps(doc.metadata, ensure_ascii=False)}"
            bm25 = self._bm25_score(query_tokens, doc_tokens, token_counts)
            if bm25 <= 0:
                continue
            issue_boost, issue_reasons = self._issue_boost(haystack, doc.metadata, issues)
            score = bm25 + self._domain_boost(query, haystack, category, case_frame=case_frame, issue_plan=issue_plan) + self._metadata_boost(doc.metadata, category) + issue_boost
            score -= self._cross_domain_penalty(haystack, category, case_frame=case_frame, issue_plan=issue_plan)
            if score > 0:
                scored.append(
                    RAGDocument(
                        text=doc.text,
                        metadata=doc.metadata,
                        chunk_id=doc.chunk_id,
                        score=score,
                        retrieval_reason=f"bm25={bm25:.2f}, category={category or 'unknown'}, issues={','.join(issues) or 'none'}{issue_reasons}",
                    )
                )
        scored.sort(key=lambda doc: doc.score, reverse=True)
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

    def _expand_query_tokens(self, text: str, category: str = "", issues: list[str] | None = None, case_frame: Any | None = None, issue_plan: Any | None = None) -> list[str]:
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
        suppress_wage = bool(getattr(case_frame, "condition_or_exchange", None)) and not bool(getattr(case_frame, "unpaid_claim", None))
        for key, expansion in aliases.items():
            if key in text:
                if suppress_wage and key in {"알바", "월급", "급여", "못 받", "안 줬", "체불", "알바비"}:
                    continue
                tokens.extend(expansion.split())
        if category == "노동":
            if suppress_wage:
                tokens.extend("고용 사업주 직장 내 성희롱 성적 요구 남녀고용평등법".split())
            else:
                tokens.extend("근로기준법 임금 근로계약서 고용노동부 사업주 지급".split())
        elif category == "성폭력":
            tokens.extend("성폭력 성희롱 강제추행 피해자 보호 상담 신고 동의".split())

        # IssuePlan이 있으면 검색어를 사실 역할과 필요한 법령 중심으로 보정한다.
        if issue_plan is not None:
            tokens.extend(str(getattr(issue_plan, "primary_issue", "")).split())
            for law in getattr(issue_plan, "needed_laws", []) or []:
                tokens.extend(str(law).split())
            for focus in getattr(issue_plan, "answer_focus", []) or []:
                tokens.extend(str(focus).split())

        for issue in issues or []:
            if issue in ISSUE_SPECS:
                tokens.extend(str(ISSUE_SPECS.get(issue, {}).get("expand", "")).split())
        if suppress_wage:
            blocked = {"임금체불", "미지급", "급여일", "주휴수당", "체불임금", "제43조"}
            tokens = [token for token in tokens if token not in blocked]
        return tokens

    def _lexical_score(self, terms: set[str], text: str) -> float:
        lowered = text.lower()
        score = 0.0
        for term in terms:
            count = lowered.count(term)
            if count:
                score += 1.0 + math.log(count)
        return score

    def _domain_boost(self, query: str, text: str, category: str = "", case_frame: Any | None = None, issue_plan: Any | None = None) -> float:
        boost = 0.0
        suppress_wage = bool(getattr(case_frame, "condition_or_exchange", None)) and not bool(getattr(case_frame, "unpaid_claim", None))
        if suppress_wage:
            if "직장 내 성희롱" in text or "성희롱" in text:
                boost += 12.0
            if "남녀고용평등" in text or "양성평등기본법" in text:
                boost += 10.0
            if "성적 요구" in text or "성적 언동" in text or "불리한 처우" in text:
                boost += 6.0
            if "임금체불" in text or "주휴수당" in text or "급여일" in text:
                boost -= 10.0
        elif category == "노동" or any(term in query for term in ["알바", "아르바이트", "임금", "월급", "체불", "근로", "급여", "못 받", "안 줬"]):
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
            for term in spec.get("penalty", []):
                if term.lower() in haystack or re.sub(r"\s+", "", term.lower()) in compact:
                    issue_boost -= 5.0
            source_type = str(metadata.get("source_type") or "")
            if source_type == "case":
                issue_boost -= 2.0
            if issue_boost:
                boost += issue_boost
                reasons.append(f"{issue}:{issue_boost:.1f}")
        return boost, f", issue_boost={';'.join(reasons)}" if reasons else ""

    def _cross_domain_penalty(self, text: str, category: str, case_frame: Any | None = None, issue_plan: Any | None = None) -> float:
        if bool(getattr(case_frame, "condition_or_exchange", None)) and not bool(getattr(case_frame, "unpaid_claim", None)):
            wage_hits = sum(text.count(term) for term in ["임금체불", "주휴수당", "급여일", "체불임금", "미지급"])
            harassment_hits = sum(text.count(term) for term in ["성희롱", "성적 요구", "성적 언동", "남녀고용평등", "불리한 처우"])
            if wage_hits > harassment_hits:
                return 12.0
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

    def _balance_sources(self, docs: list[RAGDocument], top_k: int, category: str, issues: list[str] | None = None, issue_plan: Any | None = None) -> list[RAGDocument]:
        selected = []
        preferred_slots = ["statute", "manual", "statute"]
        if category == "성폭력":
            preferred_slots = ["manual", "statute", "manual"]
        if issues and any(issue in {"wage_unpaid", "minimum_wage", "dismissal", "missing_contract", "indecent_assault", "illegal_filming"} for issue in issues):
            preferred_slots = ["statute", "manual", "statute"]
        if issues and "workplace_harassment" in issues:
            preferred_slots = ["statute", "manual", "manual"]

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
        if any(term in query for term in ["성폭력", "성추행", "성희롱", "강제추행", "강간", "불법촬영", "동의 없이", "몸을 만", "거절"]):
            return "성폭력"
        if any(term in query for term in ["알바", "알바비", "아르바이트", "임금", "월급", "급여", "근로", "해고", "퇴직금"]):
            return "노동"
        return ""


    def evaluate_retrieval(self, query: str, docs: list[dict[str, Any]], legal_category: str = "", case_frame: Any | None = None, issue_plan: Any | None = None) -> dict[str, Any]:
        """RRS: CaseFrame/IssuePlan 기준 RAG 근거 품질 평가."""
        docs = docs or []
        joined = "\n".join((doc.get("text") or "") + " " + json.dumps(doc.get("metadata") or {}, ensure_ascii=False) for doc in docs)
        top_scores = [float(doc.get("score") or 0.0) for doc in docs[:3]]
        if top_scores:
            # BM25와 vector score 스케일 차이를 완화하기 위한 saturating normalization
            top_doc_relevance = min(1.0, max(top_scores) / (max(top_scores) + 8.0)) if max(top_scores) > 1.0 else max(top_scores)
        else:
            top_doc_relevance = 0.0

        expected_laws = list(getattr(issue_plan, "needed_laws", []) or [])
        matched_laws = [law for law in expected_laws if law and any(part in joined for part in str(law).split())]
        legal_basis_coverage = len(matched_laws) / max(1, len(expected_laws)) if expected_laws else (top_doc_relevance * 0.5 if docs else 0.0)

        primary_issue = str(getattr(issue_plan, "primary_issue", "") or "")
        focus_terms = [primary_issue] + list(getattr(issue_plan, "answer_focus", []) or [])
        focus_tokens = []
        for term in focus_terms:
            focus_tokens.extend(re.findall(r"[0-9A-Za-z가-힣]{2,}", str(term)))
        focus_tokens = list(dict.fromkeys(focus_tokens))[:12]
        issue_hits = sum(1 for token in focus_tokens if token in joined)
        issue_match = issue_hits / max(1, min(len(focus_tokens), 6)) if focus_tokens else (0.5 if docs else 0.0)
        issue_match = min(1.0, issue_match)

        excluded = list(getattr(issue_plan, "excluded_issues", []) or []) + list(getattr(case_frame, "not_claimed", []) or [])
        excluded_hits = [term for term in excluded if term and term in joined]
        domain_consistency = max(0.0, 1.0 - 0.18 * len(excluded_hits))

        score = (
            0.30 * top_doc_relevance
            + 0.25 * issue_match
            + 0.25 * legal_basis_coverage
            + 0.20 * domain_consistency
        )
        # 필수 gate: 법률 근거가 매우 약하면 caution으로 낮춘다.
        action = "answer"
        if issue_match < 0.55 or domain_consistency < 0.65:
            score = min(score, 0.54)
            action = "requery_or_safe_fallback"
        elif legal_basis_coverage < 0.30:
            score = min(score, 0.71)
            action = "answer_with_caution"
        elif score < 0.72:
            action = "retry_retrieval_or_caution"
        elif score < 0.85:
            action = "answer_with_caution"

        return {
            "score": round(max(0.0, min(1.0, score)), 3),
            "passed": score >= 0.72,
            "action": action,
            "top_doc_relevance": round(top_doc_relevance, 3),
            "issue_match": round(issue_match, 3),
            "legal_basis_coverage": round(legal_basis_coverage, 3),
            "domain_consistency": round(domain_consistency, 3),
            "primary_issue": primary_issue,
            "expected_laws": expected_laws,
            "matched_laws": matched_laws,
            "excluded_issue_hits": excluded_hits,
            "doc_count": len(docs),
            "thresholds": {"excellent": 0.85, "passable": 0.72, "fail": 0.55},
        }


retriever = LegalRetriever()
