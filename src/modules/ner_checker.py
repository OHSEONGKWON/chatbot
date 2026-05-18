import re
import asyncio
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

try:
    import torch
except Exception:
    torch = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from .corrector import AnswerCorrector
except Exception:
    # fallback lightweight corrector for environments where module imports fail
    class AnswerCorrector:
        def fix_answer(self, llm_answer, hallucinations):
            return llm_answer

try:
    from ..config import config
except Exception:
    # minimal fallback config for embedding model name
    class _R:
        embedding_model = "intfloat/multilingual-e5-large"

    class _C:
        rag = _R()

    config = _C()


CASE_NUMBER_PATTERN = r"\b\d{4}[가-힣]{1,3}\d{1,6}\b"
# Civil cases: 가합(합의부), 가접(접수), 가단(단독), 가초(항소) 등
CASE_NUMBER_CIVIL_PATTERN = r"\b\d{4}(?:가합|가단|가초|가접|가상)\d{1,6}\b"
# Criminal cases: 고합(합의부), 고단(단독), 고초(항소), 고접(접수) 등
CASE_NUMBER_CRIMINAL_PATTERN = r"\b\d{4}(?:고합|고단|고초|고접|고상)\d{1,6}\b"
# Precedent/판례: 다, 나 등의 정정 판례 표기
CASE_NUMBER_PRECEDENT_PATTERN = r"\b(?:대판|대법원판례|판례)\s*\d{4}\s*다\d{4,5}\b|\b\d{4}\s*나\d{4,5}\b"

PHONE_PATTERN = r"\b(?:112|1366|1350|1331|132|1588-0075|\d{2,3}-\d{3,4}-\d{4})\b"

KNOWN_CONTACTS = {
    "경찰청": ["112"],
    "여성긴급전화": ["1366"],
    "고용노동부": ["1350"],
    "국가인권위원회": ["1331"],
    "대한법률구조공단": ["132"],
}

ISSUE_REGISTRY = {
    "wage_unpaid": {
        "allowed_laws": ["근로기준법", "임금채권보장법", "근로자퇴직급여 보장법", "고용보험법"],
        "forbidden_laws": ["형법", "형법 제298조", "민법", "성폭력범죄의 처벌 등에 관한 특례법"],
    },
    "dismissal": {
        "allowed_laws": ["근로기준법", "근로자참여 및 협력증진에 관한 법률"],
        "forbidden_laws": ["형법", "형법 제298조", "민법", "성폭력범죄의 처벌 등에 관한 특례법"],
    },
    "sexual_harassment": {
        "allowed_laws": ["양성평등기본법", "남녀고용평등법", "성폭력범죄의 처벌 등에 관한 특례법"],
        "forbidden_laws": ["근로기준법", "임금채권보장법", "근로자퇴직급여 보장법", "민법"],
    },
    "indecent_assault": {
        "allowed_laws": ["형법", "형법 제298조", "성폭력범죄의 처벌 등에 관한 특례법"],
        "forbidden_laws": ["근로기준법", "임금채권보장법", "근로자퇴직급여 보장법", "민법"],
    },
    "illegal_filming": {
        "allowed_laws": ["성폭력범죄의 처벌 등에 관한 특례법", "형법"],
        "forbidden_laws": ["근로기준법", "임금채권보장법", "근로자퇴직급여 보장법", "민법"],
    },
    "강간": {
        "allowed_laws": ["형법", "성폭력범죄의 처벌 등에 관한 특례법"],
        "forbidden_laws": ["근로기준법", "임금채권보장법", "근로자퇴직급여 보장법", "민법"],
    },
}

ISSUE_ALIASES = {
    "임금체불": "wage_unpaid",
    "부당해고": "dismissal",
    "교육기관 언어적 성희롱": "sexual_harassment",
    "언어적 성희롱": "sexual_harassment",
    "신체접촉형 강제추행": "indecent_assault",
    "강간": "강간",
    "불법촬영": "illegal_filming",
    "wage_unpaid": "wage_unpaid",
    "dismissal": "dismissal",
    "sexual_harassment": "sexual_harassment",
    "indecent_assault": "indecent_assault",
    "illegal_filming": "illegal_filming",
}


@dataclass
class NERCheckResult:
    original_answer: str
    corrected_answer: str
    found_entities: list
    mismatched_entities: list[dict]
    was_corrected: bool
    mid_confidence_warnings: list[dict] = None

    def __post_init__(self):
        if self.mid_confidence_warnings is None:
            self.mid_confidence_warnings = []

    @property
    def hallucinations(self):
        return self.mismatched_entities


class NERFactChecker:
    """Lightweight NER fact checker with law normalization, domain checks, and confidence scoring.

    Constructor avoids heavy model loading so the module can be imported in tests quickly.
    """
    def __init__(self, model_path=None):
        self.target_labels = ['LAW', 'PENALTY', 'AMOUNT', 'DATE', 'ORG', 'CRIME', 'CASE_NUMBER', 'PHONE']
        self.model_path = model_path or getattr(getattr(config, "ner", None), "model_path", None)
        self.use_model = bool(getattr(getattr(config, "ner", None), "use_model", True))
        self.min_model_confidence = float(getattr(getattr(config, "ner", None), "min_confidence", 0.70))
        self.max_length = int(getattr(getattr(config, "ner", None), "max_length", 510))

        # Domain keywords (simple examples)
        self.domain_keywords = {
            "sexual": ["성추행", "강제추행", "성희롱", "성폭력", "강간", "디지털성범죄", "성범죄", "성착취"],
            "labor": ["근로", "해고", "임금", "노동", "최저임금", "취업규칙", "근로기준"],
            "finance": ["사기", "대출", "채권", "금융", "이자", "변제", "채무"],
        }

        self.alias_map = {
            "대한민국": "한국",
            "우리나라": "한국",
            "대검": "대검찰청",
            "검찰": "검찰청",
            "대법": "대법원",
        }

        # Law domain mapping for co-occurrence checks
        self.law_domain_map = {
            r"근로기준법|근로기|근기법|근기": "labor",
            r"산업안전보건법|산안법|산업안전": "labor",
            r"민법|민사": "civil",
            r"형법|형사": "criminal",
            r"성폭력범죄의처벌등에관한특례법|성폭력|성범죄": "criminal_sexual",
        }

        self.entity_domain_compat = {
            "CRIME": ["criminal", "criminal_sexual", "ANY"],
            "PENALTY": ["criminal", "criminal_sexual", "labor", "civil", "ANY"],
            "ORG": ["ANY"],
            "DATE": ["ANY"],
            "AMOUNT": ["ANY"],
            "LAW": ["ANY"],
        }

        # Matching thresholds
        self.fuzzy_threshold = 0.90
        self.semantic_threshold = 0.82
        self.hallucination_report_threshold = 0.72
        self.enable_semantic_match = os.getenv("LAWSGUARD_ENABLE_SEMANTIC_NER", "0") == "1"
        self.semantic_allowed_labels = {"CRIME", "PENALTY"}

        # Lazy components
        self._semantic_embedder = None
        self._corrector = AnswerCorrector()
        self._ner_model = None
        self._ner_tokenizer = None

    async def check_and_correct(self, answer, rag_docs, route=None, route_issue=None):
        return await asyncio.to_thread(self.check_and_correct_sync, answer, rag_docs, route, route_issue)

    async def check(self, answer, rag_docs):
        return await self.check_and_correct(answer, rag_docs)

    async def warmup(self):
        await asyncio.to_thread(self._load_ner_model)
        await asyncio.to_thread(self._load_semantic_embedder)

    def check_and_correct_sync(self, answer, rag_docs, route=None, route_issue=None):
        found_entities = self.extract_entities(answer)
        all_hallucinations = self.find_hallucinations(answer, rag_docs, found_entities=found_entities, route=route, route_issue=route_issue)
        
        # 신뢰도에 따라 hallucinations와 mid_confidence_warnings로 분리
        mismatches = []
        mid_confidence_warnings = []
        
        for hallucination in all_hallucinations:
            confidence = hallucination.get("confidence", 0.0)
            if 0.80 <= confidence < 0.95:
                mid_confidence_warnings.append(hallucination)
            else:
                mismatches.append(hallucination)
        
        corrected = self._corrector.fix_answer(answer, mismatches)
        return NERCheckResult(
            original_answer=answer,
            corrected_answer=corrected,
            found_entities=found_entities,
            mismatched_entities=mismatches,
            was_corrected=corrected != answer,
            mid_confidence_warnings=mid_confidence_warnings,
        )

    def extract_entities(self, text):
        model_entities = self._extract_entities_with_model(text)
        rule_entities = self._extract_entities_with_rules(text)
        if model_entities:
            return self._merge_entity_lists(model_entities, rule_entities)
        return rule_entities

    def _extract_entities_with_model(self, text):
        model, tokenizer = self._load_ner_model()
        if model is None or tokenizer is None or torch is None:
            return []

        entities = []
        for offset, chunk in self._iter_text_chunks(text):
            try:
                encoded = tokenizer(
                    chunk,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                offsets = encoded.pop("offset_mapping")[0].tolist()
                device = next(model.parameters()).device
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.no_grad():
                    logits = model(**encoded).logits[0]
                probs = torch.softmax(logits, dim=-1)
                pred_ids = torch.argmax(probs, dim=-1).tolist()
                pred_scores = torch.max(probs, dim=-1).values.tolist()
                entities.extend(self._bio_predictions_to_entities(chunk, offset, offsets, pred_ids, pred_scores, model.config.id2label))
            except Exception:
                continue
        return self._dedupe_entities(entities)

    def _load_ner_model(self):
        if not self.use_model:
            return None, None
        if self._ner_model is not None and self._ner_tokenizer is not None:
            return self._ner_model, self._ner_tokenizer
        if not self.model_path or not Path(self.model_path).exists():
            return None, None
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, use_fast=True)
            model = AutoModelForTokenClassification.from_pretrained(self.model_path, local_files_only=True)
            device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
            model.to(device)
            model.eval()
            self._ner_model = model
            self._ner_tokenizer = tokenizer
        except Exception:
            self._ner_model = None
            self._ner_tokenizer = None
        return self._ner_model, self._ner_tokenizer

    def _iter_text_chunks(self, text):
        if len(text) <= 900:
            yield 0, text
            return
        start = 0
        while start < len(text):
            end = min(len(text), start + 900)
            if end < len(text):
                cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind(" ", start, end))
                if cut > start + 300:
                    end = cut + 1
            yield start, text[start:end]
            start = end

    def _bio_predictions_to_entities(self, chunk, chunk_offset, offsets, pred_ids, pred_scores, id2label):
        entities = []
        current = None

        for token_offset, pred_id, score in zip(offsets, pred_ids, pred_scores):
            start, end = token_offset
            if start == end:
                continue
            label = id2label.get(int(pred_id), "O")
            if label == "O":
                if current is not None:
                    entities.append(current)
                    current = None
                continue

            prefix, _, entity_label = label.partition("-")
            if entity_label not in self.target_labels:
                continue

            abs_start = chunk_offset + start
            abs_end = chunk_offset + end
            token_score = float(score)
            # The trained tokenizer/model can emit B-* for adjacent word pieces.
            # Trust the contiguous character span more than the BIO prefix here.
            should_start = current is None or current["entity_group"] != entity_label or abs_start > current["end"] + 1

            if should_start:
                if current is not None:
                    entities.append(current)
                current = {
                    "entity_group": entity_label,
                    "label": entity_label,
                    "word": chunk[start:end],
                    "start": abs_start,
                    "end": abs_end,
                    "score": token_score,
                    "_scores": [token_score],
                    "source": "model",
                }
            else:
                current["end"] = abs_end
                rel_start = current["start"] - chunk_offset
                rel_end = abs_end - chunk_offset
                current["word"] = chunk[rel_start:rel_end]
                current["_scores"].append(token_score)
                current["score"] = sum(current["_scores"]) / len(current["_scores"])

        if current is not None:
            entities.append(current)

        cleaned = []
        for entity in entities:
            entity.pop("_scores", None)
            entity["word"] = re.sub(r"\s+", " ", entity["word"]).strip()
            if entity["word"] and float(entity["score"]) >= self.min_model_confidence:
                cleaned.append(entity)
        return cleaned

    def _extract_entities_with_rules(self, text):
        entities = []
        seen = set()
        patterns = {
            "LAW": [
                r"[가-힣A-Za-z0-9·\s]{0,20}(?:법|시행령|시행규칙)\s*제?\s*\d+\s*조(?:의\s*\d+)?",
                r"(?:근로기준법|형법|민법|남녀고용평등법|성폭력범죄의 처벌 등에 관한 특례법|고용보험법|최저임금법)",
            ],
            "DATE": [
                r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일",
                r"\d{1,2}\s*월\s*\d{1,2}\s*일",
            ],
            "AMOUNT": [
                r"\d[\d,]*\s*(?:원|만원|천원)",
                r"시급\s*\d[\d,]*",
            ],
            "ORG": [
                r"(?:고용노동부|국가인권위원회|경찰청|경찰|검찰청|대법원|대한법률구조공단|법률구조공단|여성긴급전화|학교|대학교)",
            ],
            "CRIME": [
                r"(?:강제추행|성추행|성폭력|성희롱|강간|불법촬영|스토킹|폭행|협박)",
            ],
            "PENALTY": [
                r"\d+\s*년\s*(?:이하|이상)의?\s*징역",
                r"\d[\d,]*\s*만원\s*(?:이하|이상)의?\s*벌금",
            ],
            "CASE_NUMBER": [
                CASE_NUMBER_PATTERN,
            ],
            "PHONE": [
                PHONE_PATTERN,
            ],
        }
        for label, regexes in patterns.items():
            for pattern in regexes:
                for match in re.finditer(pattern, text):
                    word = re.sub(r"\s+", " ", match.group(0)).strip()
                    key = (label, match.start(), match.end(), word)
                    if not word or key in seen:
                        continue
                    seen.add(key)
                    entity_dict = {
                        "entity_group": label,
                        "label": label,
                        "word": word,
                        "start": match.start(),
                        "end": match.end(),
                        "score": 1.0,
                    }
                    # CASE_NUMBER의 경우 타입 메타데이터 추가
                    if label == "CASE_NUMBER":
                        entity_dict["case_type"] = self._categorize_case_number(word)
                    entities.append(entity_dict)
        entities.sort(key=lambda item: item["start"])
        return entities

    def _merge_entity_lists(self, primary, secondary):
        merged = list(primary)
        for entity in secondary:
            overlaps = [
                existing
                for existing in merged
                if existing.get("entity_group") == entity.get("entity_group")
                and not (entity.get("end", 0) <= existing.get("start", 0) or entity.get("start", 0) >= existing.get("end", 0))
            ]
            if overlaps:
                continue
            entity = dict(entity)
            entity.setdefault("source", "rule")
            merged.append(entity)
        return self._dedupe_entities(merged)

    def _dedupe_entities(self, entities):
        deduped = []
        seen = set()
        for entity in sorted(entities, key=lambda item: (item.get("start", 0), -(item.get("end", 0) - item.get("start", 0)))):
            word = re.sub(r"\s+", " ", str(entity.get("word", ""))).strip()
            if not word:
                continue
            key = (entity.get("entity_group"), entity.get("start"), entity.get("end"), word)
            if key in seen:
                continue
            seen.add(key)
            entity = dict(entity)
            entity["word"] = word
            deduped.append(entity)
        return deduped

    def find_hallucinations(self, answer, rag_docs, found_entities=None, route=None, route_issue=None):
        found_entities = found_entities if found_entities is not None else self.extract_entities(answer)
        candidates_by_label, context_law = self._collect_candidates(rag_docs)
        hallucinations = []

        # Route-aware checks can flag unsupported legal domains even if RAG is sparse.
        route_issue_key = self._resolve_route_issue(route, route_issue)
        rag_sparsity = len(candidates_by_label.get("LAW", [])) < 2  # RAG 자료 부족 여부

        for entity in found_entities:
            label = entity.get("entity_group") or entity.get("label")
            word = entity.get("word")
            if label not in {"LAW", "PENALTY", "AMOUNT", "DATE", "CRIME", "CASE_NUMBER", "PHONE", "ORG"} or not word:
                continue
            candidates = candidates_by_label.get(label, [])
            if not candidates:
                # LAW/ORG는 candidates가 없어도 route 기반 검사는 수행
                # CRIME/PENALTY/ORG는 RAG가 너무 부족할 때 정규 검사 스킵 (fallback은 뒤에서 처리)
                if label not in {"LAW", "ORG", "CRIME", "PENALTY"}:
                    continue
                # Route 검사를 거친 후 route에서도 지지되지 않으면 다음 엔티티로
                # This will be handled by _route_issue_hallucinations
                continue
            match = self._match_entity_against_candidates(label, word, candidates, context_law=context_law)
            if match.get("is_supported"):
                continue
            candidate = match.get("candidate")
            if not candidate:
                continue
            if float(match.get("score", 0.0)) < self.hallucination_report_threshold:
                continue
            support_score = max(0.0, min(1.0, float(match.get("confidence", 0.0))))
            replacement_confidence = self._replacement_confidence(label, word, candidate, match)
            risk_level = self._risk_level(label, support_score, match)
            if risk_level == "low" and replacement_confidence <= 0.0:
                continue
            hallucinations.append(
                {
                    "label": label,
                    "wrong_word": word,
                    "correct_word": candidate,
                    "start": entity.get("start"),
                    "end": entity.get("end"),
                    "confidence": replacement_confidence,
                    "support_score": support_score,
                    "match_score": float(match.get("score", 0.0)),
                    "risk_level": risk_level,
                    "reason_code": match.get("method", "unsupported_entity"),
                    "action": self._choose_action(label, risk_level, match, candidate),
                    "reason": "답변의 개체명이 검색 근거 문서에서 충분히 지지되지 않습니다.",
                }
            )

        hallucinations.extend(self._route_issue_hallucinations(answer, found_entities, candidates_by_label, context_law, route_issue_key))
        hallucinations.extend(self._known_contact_hallucinations(found_entities))
        # RAG 자료가 부족할 때 CRIME/PENALTY/ORG fallback 처리
        if rag_sparsity:
            hallucinations.extend(self._crime_penalty_fallback_hallucinations(found_entities, answer))
        return hallucinations

    def _collect_candidates(self, rag_docs):
        text_parts = []
        law_names = []
        for doc in rag_docs or []:
            if isinstance(doc, dict):
                text = doc.get("text") or doc.get("content", "")
                metadata = doc.get("metadata", {})
            else:
                text = getattr(doc, "text", "")
                metadata = getattr(doc, "metadata", {})
            text_parts.append(text or "")
            if isinstance(metadata, dict):
                for key in ("law_name", "source_file", "article_id", "article_title", "organization"):
                    value = metadata.get(key)
                    if value:
                        text_parts.append(str(value))
                if metadata.get("law_name"):
                    law_names.append(str(metadata["law_name"]))
                    article = metadata.get("article_id") or metadata.get("article_title")
                    if article:
                        text_parts.append(f"{metadata['law_name']} {article}")

        merged = "\n".join(text_parts)
        extracted = self.extract_entities(merged)
        candidates = {label: [] for label in self.target_labels}
        for entity in extracted:
            label = entity.get("entity_group")
            word = entity.get("word")
            if label in candidates and word:
                candidates[label].append(word)

        for law in law_names:
            candidates["LAW"].append(law)

        for label, values in candidates.items():
            deduped = []
            seen = set()
            for value in values:
                norm = self._normalize_entity_text(value, label)
                if norm and norm not in seen:
                    seen.add(norm)
                    deduped.append(value)
            candidates[label] = deduped

        context_law = law_names[0] if law_names else ""
        return candidates, context_law

    # --- Normalization ---
    def _normalize_law_text(self, text):
        if not text:
            return ""
        t = text.strip().lower()
        t = re.sub(r'\([^)]*\)', '', t)
        t = re.sub(r'\s+', ' ', t).strip()

        law_abbreviations = {
            # 근로 관련
            "근기": "근로기준법",
            "근기법": "근로기준법",
            "근로": "근로기준법",
            "최저임금": "최저임금법",
            "최저임금법": "최저임금법",
            # 남녀고용평등
            "남고평": "남녀고용평등과일가정양립지원에관한법률",
            "남녀고용": "남녀고용평등과일가정양립지원에관한법률",
            "고용평등법": "남녀고용평등과일가정양립지원에관한법률",
            "양성평등": "양성평등기본법",
            "양평기본법": "양성평등기본법",
            # 산업안전
            "산안": "산업안전보건법",
            "산안법": "산업안전보건법",
            # 산재보험
            "산재법": "산업재해보상보험법",
            "산재보험": "산업재해보상보험법",
            # 퇴직급여
            "퇴직금법": "근로자퇴직급여보장법",
            "퇴직급여": "근로자퇴직급여보장법",
            # 성폭력
            "성폭": "성폭력범죄의처벌등에관한특례법",
            "성폭법": "성폭력범죄의처벌등에관한특례법",
            "성폭력처벌법": "성폭력범죄의처벌등에관한특례법",
            "성폭력특례법": "성폭력범죄의처벌등에관한특례법",
            # 스토킹
            "스토킹법": "스토킹범죄의처벌등에관한법률",
            "스토킹범죄처벌": "스토킹범죄의처벌등에관한법률",
            # 기타
            "고용보험법": "고용보험법",
            "임금채권보장": "임금채권보장법",
            "형법": "형법",
            "민법": "민법",
        }
        compact = re.sub(r"[\s·ㆍ,.\-()「」『』<>\[\]]+", "", t)
        if compact in law_abbreviations:
            return law_abbreviations[compact]
        for alias, canonical in sorted(law_abbreviations.items(), key=lambda item: len(item[0]), reverse=True):
            if canonical in compact:
                continue
            compact = compact.replace(alias, canonical)
        return compact

    def _same_law_family(self, law1_text, law2_text):
        """약칭·약자 변형을 고려하여 두 법률명이 같은 법인지 판단합니다."""
        if not law1_text or not law2_text:
            return False
        norm1 = self._normalize_law_text(str(law1_text))
        norm2 = self._normalize_law_text(str(law2_text))
        if norm1 == norm2:
            return True
        # 둘 다 공백을 제거한 형태로 비교
        compact1 = re.sub(r"[\s·ㆍ,.\-()「」『』<>\[\]]+", "", norm1)
        compact2 = re.sub(r"[\s·ㆍ,.\-()「」『』<>\[\]]+", "", norm2)
        return compact1 == compact2

    def _law_article_key(self, text):
        compact = self._normalize_law_text(text)
        match = re.search(r"제\d+조(?:제\d+항)?(?:제\d+호)?", compact)
        return match.group(0) if match else ""

    def _law_name_key(self, text):
        compact = self._normalize_law_text(text)
        return re.sub(r"제\d+조(?:제\d+항)?(?:제\d+호)?", "", compact)

    def _categorize_case_number(self, case_number_text):
        """사건번호의 유형을 판단합니다: civil, criminal, precedent"""
        if not case_number_text:
            return "unknown"
        text = case_number_text.strip()
        # 판례 패턴 먼저 확인
        if re.search(CASE_NUMBER_PRECEDENT_PATTERN, text):
            return "precedent"
        # 형사 패턴
        if re.search(CASE_NUMBER_CRIMINAL_PATTERN, text):
            return "criminal"
        # 민사 패턴
        if re.search(CASE_NUMBER_CIVIL_PATTERN, text):
            return "civil"
        # 기본 패턴으로는 분류할 수 없으면 unknown 반환
        return "unknown"

    def _normalize_entity_text(self, text, label=None):
        if not isinstance(text, str):
            return ""
        t = text.strip().lower()
        t = self.alias_map.get(t, t)

        if label == "LAW":
            return self._normalize_law_text(t)

        if label == "CASE_NUMBER":
            return re.sub(r"\s+", "", t)

        if label == "PHONE":
            return re.sub(r"\s+", "", t)

        if label == "DATE":
            nums = re.findall(r"\d+", t)
            if nums:
                return "-".join(str(int(n)) for n in nums)
            return re.sub(r"\s+", "", t)

        if label == "AMOUNT":
            nums = re.findall(r"\d+", t)
            if nums:
                number = int(''.join(nums))
                if "만원" in t:
                    number *= 10000
                return str(number)
            return re.sub(r"\s+", "", t)

        t = re.sub(r"\([^)]*\)", "", t)
        t = re.sub(r"[^0-9a-z가-힣]", "", t)
        return t

    # --- Similarity utilities ---
    def _fuzzy_ratio(self, a, b):
        if not a or not b:
            return 0.0
        return float(SequenceMatcher(None, a, b).ratio())

    def _load_semantic_embedder(self):
        if self._semantic_embedder is None and SentenceTransformer is not None:
            device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
            try:
                self._semantic_embedder = SentenceTransformer(config.rag.embedding_model, device=device, local_files_only=True)
            except Exception:
                self._semantic_embedder = None
        return self._semantic_embedder

    def _semantic_similarity(self, left, right):
        if not self.enable_semantic_match:
            return 0.0
        embedder = self._load_semantic_embedder()
        if embedder is None or np is None:
            return 0.0
        try:
            embs = embedder.encode([left, right], normalize_embeddings=True, show_progress_bar=False)
            return float(np.dot(embs[0], embs[1]))
        except Exception:
            return 0.0

    # --- Domain checks ---
    def _get_law_domain(self, law_text):
        if not law_text:
            return "unknown"
        law_lower = law_text.strip().lower()
        for pattern, domain in self.law_domain_map.items():
            if re.search(pattern, law_lower):
                return domain
        return "unknown"

    def _validate_entity_combo(self, entity_label, law_text):
        if entity_label not in self.entity_domain_compat:
            return 1.0
        if not law_text:
            return 1.0
        allowed_domains = self.entity_domain_compat[entity_label]
        if "ANY" in allowed_domains:
            return 1.0
        law_domain = self._get_law_domain(law_text)
        if law_domain in allowed_domains:
            return 1.0
        if law_domain == "unknown":
            return 0.7
        return 0.3

    # --- Matching core (returns confidence and candidate) ---
    def _match_entity_against_candidates(self, label, word, candidates, context_law=None):
        norm_word = self._normalize_entity_text(word, label)
        if not norm_word:
            return {"is_supported": False, "candidate": None, "method": "empty", "score": 0.0, "confidence": 0.0, "combo_score": 1.0}

        # Strict numeric/date handling
        if label in {"DATE", "AMOUNT", "CASE_NUMBER", "PHONE"}:
            for cand in candidates:
                norm_cand = self._normalize_entity_text(cand, label)
                if norm_cand and norm_word == norm_cand:
                    return {"is_supported": True, "candidate": cand, "method": "exact", "score": 1.0, "confidence": 1.0, "combo_score": 1.0}
            return {"is_supported": False, "candidate": None, "method": "strict_numeric", "score": 0.0, "confidence": 0.0, "combo_score": 1.0}

        fuzzy_threshold = self.fuzzy_threshold
        semantic_threshold = self.semantic_threshold
        if label == "LAW":
            fuzzy_threshold = max(fuzzy_threshold, 0.95)
            semantic_threshold = 0.99
        elif label == "ORG":
            fuzzy_threshold = max(fuzzy_threshold, 0.94)
            semantic_threshold = max(semantic_threshold, 0.90)

        combo_score = self._validate_entity_combo(label, context_law) if context_law else 1.0

        best = {"is_supported": False, "candidate": None, "method": "none", "score": 0.0, "confidence": 0.0, "combo_score": combo_score}

        for cand in candidates:
            norm_cand = self._normalize_entity_text(cand, label)
            if not norm_cand:
                continue

            # exact
            if norm_word == norm_cand:
                exact_confidence = 1.0 * combo_score
                return {"is_supported": True, "candidate": cand, "method": "exact", "score": 1.0, "confidence": exact_confidence, "combo_score": combo_score}

            if label == "LAW":
                word_law = self._law_name_key(word)
                cand_law = self._law_name_key(cand)
                word_article = self._law_article_key(word)
                cand_article = self._law_article_key(cand)
                if word_law and cand_law and word_law == cand_law:
                    if word_article and cand_article and word_article != cand_article:
                        score = 0.93
                        confidence = score * combo_score
                        if confidence > best["confidence"]:
                            best = {
                                "is_supported": False,
                                "candidate": cand,
                                "method": "same_law_article_mismatch",
                                "score": score,
                                "confidence": confidence,
                                "combo_score": combo_score,
                            }
                        continue
                    if not word_article or not cand_article:
                        score = 0.88
                        confidence = score * combo_score
                        if confidence > best["confidence"]:
                            best = {
                                "is_supported": False,
                                "candidate": cand,
                                "method": "law_name_only",
                                "score": score,
                                "confidence": confidence,
                                "combo_score": combo_score,
                            }
                        continue

            # fuzzy
            fuzzy = self._fuzzy_ratio(norm_word, norm_cand)
            allow_semantic = self.enable_semantic_match and label in self.semantic_allowed_labels
            semantic = 0.0
            if allow_semantic and fuzzy < fuzzy_threshold:
                semantic = self._semantic_similarity(word, cand)

            match_score = max(fuzzy, semantic)
            weighted_confidence = match_score * combo_score

            if weighted_confidence > best["confidence"]:
                best = {"is_supported": False, "candidate": cand, "method": ("fuzzy" if fuzzy >= semantic else "semantic"), "score": match_score, "confidence": weighted_confidence, "combo_score": combo_score}

        if best["method"] == "fuzzy" and best["score"] >= fuzzy_threshold:
            if best["confidence"] >= fuzzy_threshold:
                best["is_supported"] = True
            return best
        if best["method"] == "semantic" and best["score"] >= semantic_threshold:
            if best["confidence"] >= semantic_threshold:
                best["is_supported"] = True
            return best
        return best

    def _replacement_confidence(self, label, word, candidate, match):
        """환각 수정의 신뢰도를 계산합니다. 라벨과 점수에 따라 차별화된 값을 반환합니다."""
        method = match.get("method")
        score = float(match.get("score", 0.0))
        confidence = float(match.get("confidence", 0.0))
        
        # LAW: 법령 이름이나 조문이 일치하는 경우
        if label == "LAW":
            if method == "exact":
                return 0.99
            if method == "same_law_article_mismatch":
                word_law = self._law_name_key(word)
                cand_law = self._law_name_key(candidate)
                if word_law and word_law == cand_law and self._law_article_key(word) and self._law_article_key(candidate):
                    return 0.98
            if method == "law_name_only" and score >= 0.88:
                return 0.92
            if method == "fuzzy" and score >= 0.95:
                return 0.88
            if confidence >= 0.90:  # combo_score 반영
                return 0.85
        
        # DATE/AMOUNT: 정확 매칭만 가능하므로 높은 신뢰도
        if label == "DATE":
            if method == "exact":
                return 0.99
        if label == "AMOUNT":
            if method == "exact":
                return 0.98
        
        # ORG: 조직명 유사도 높으면 교체 가능
        if label == "ORG":
            if method == "exact":
                return 0.99
            if method == "fuzzy" and score >= 0.95:
                return 0.93
            if method == "fuzzy" and score >= 0.90:
                return 0.85
            if confidence >= 0.85:
                return 0.80
        
        # CRIME/PENALTY: 시맨틱 또는 퍼지 매칭으로 높은 점수
        if label in {"CRIME", "PENALTY"}:
            if method == "exact":
                return 0.99
            if method == "semantic" and score >= 0.95:
                return 0.94
            if method == "fuzzy" and score >= 0.96:
                return 0.91
            if method == "fuzzy" and score >= 0.92:
                return 0.86
            if confidence >= 0.88:  # combo_score 반영된 신뢰도
                return 0.82
        
        # CASE_NUMBER/PHONE: 정확 매칭만 지원
        if label in {"CASE_NUMBER", "PHONE"}:
            if method == "exact":
                return 0.99
        
        # 폴백: 높은 지지도면 약간의 수정 신뢰도 부여
        if confidence >= 0.85:
            return 0.75
        
        return 0.0

    def _risk_level(self, label, support_score, match):
        if label == "LAW" and match.get("method") == "same_law_article_mismatch":
            return "high"
        if support_score < 0.35:
            return "high"
        if support_score < 0.75:
            return "medium"
        return "low"

    def _resolve_route_issue(self, route=None, route_issue=None):
        issue = route_issue
        if issue is None and route is not None:
            if isinstance(route, dict):
                issue = route.get("issue")
            else:
                issue = getattr(route, "issue", None)
        if not issue:
            return ""
        return ISSUE_ALIASES.get(str(issue).strip(), str(issue).strip())

    def _issue_registry_for(self, route_issue):
        if not route_issue:
            return None
        return ISSUE_REGISTRY.get(route_issue)

    def _text_contains_term(self, text, term):
        compact = re.sub(r"\s+", "", (text or "").lower())
        candidate = re.sub(r"\s+", "", (term or "").lower())
        return bool(candidate) and candidate in compact

    def _route_issue_hallucinations(self, answer, found_entities, candidates_by_label, context_law, route_issue_key):
        registry = self._issue_registry_for(route_issue_key)
        if not registry:
            return []

        hallucinations = []
        allowed_laws = registry.get("allowed_laws", [])
        forbidden_laws = registry.get("forbidden_laws", [])
        law_entities = [entity for entity in found_entities if (entity.get("entity_group") or entity.get("label")) == "LAW"]
        law_candidates = candidates_by_label.get("LAW", [])
        
        # RAG candidates가 부족할 때의 fallback 임계값
        candidate_scarcity_threshold = 2

        for entity in law_entities:
            word = entity.get("word") or ""
            if not word:
                continue
            # forbidden_laws: 약칭·약자도 고려한 검사
            if any(self._same_law_family(word, term) for term in forbidden_laws):
                hallucinations.append(
                    {
                        "label": "LAW",
                        "wrong_word": word,
                        "correct_word": allowed_laws[0] if len(allowed_laws) == 1 else "",
                        "start": entity.get("start"),
                        "end": entity.get("end"),
                        "confidence": 0.99,
                        "support_score": 0.0,
                        "match_score": 0.0,
                        "risk_level": "high",
                        "reason_code": "forbidden_law",
                        "action": "remove_sentence" if not allowed_laws else "soften",
                        "reason": f"{route_issue_key} 사안과 맞지 않는 법률이 답변에 포함되어 있습니다.",
                    }
                )
                continue

            # allowed_laws가 존재하고 법률이 allowed_laws 범위를 벗어나는 경우
            # RAG candidates가 부족할 때 더 강한 검사 수행
            is_in_allowed = any(self._same_law_family(word, term) for term in allowed_laws) if allowed_laws else True
            is_rag_sparse = len(law_candidates) < candidate_scarcity_threshold
            
            if allowed_laws and not is_in_allowed:
                hallucinations.append(
                    {
                        "label": "LAW",
                        "wrong_word": word,
                        "correct_word": allowed_laws[0],
                        "start": entity.get("start"),
                        "end": entity.get("end"),
                        "confidence": 0.98 if is_rag_sparse else 0.96,  # RAG가 부족하면 신뢰도 상향
                        "support_score": 0.0,
                        "match_score": 0.0,
                        "risk_level": "high",
                        "reason_code": "route_domain_mismatch",
                        "action": "remove_sentence",
                        "reason": f"{route_issue_key} 사안에서 허용된 법률 범위를 벗어나는 서술입니다.",
                    }
                )

        return self._dedupe_hallucinations(hallucinations)

    def _known_contact_hallucinations(self, found_entities):
        org_entities = [entity for entity in found_entities if (entity.get("entity_group") or entity.get("label")) == "ORG" and entity.get("word")]
        phone_entities = [entity for entity in found_entities if (entity.get("entity_group") or entity.get("label")) == "PHONE" and entity.get("word")]
        if not org_entities or not phone_entities:
            return []

        hallucinations = []
        for org in org_entities:
            org_word = str(org.get("word") or "")
            org_key = self._match_known_contact_org(org_word)
            if not org_key:
                continue
            expected_phones = KNOWN_CONTACTS.get(org_key, [])
            for phone in phone_entities:
                phone_word = str(phone.get("word") or "")
                phone_norm = self._normalize_entity_text(phone_word, "PHONE")
                if not phone_norm or phone_norm in expected_phones:
                    continue
                hallucinations.append(
                    {
                        "label": "PHONE",
                        "wrong_word": phone_word,
                        "correct_word": expected_phones[0] if len(expected_phones) == 1 else "",
                        "start": phone.get("start"),
                        "end": phone.get("end"),
                        "confidence": 0.97,
                        "support_score": 0.0,
                        "match_score": 0.0,
                        "risk_level": "high",
                        "reason_code": "known_contact_mismatch",
                        "action": "replace" if len(expected_phones) == 1 else "remove_sentence",
                        "reason": f"{org_key}에 연결된 전화번호가 답변에서 잘못 매핑되었습니다.",
                    }
                )
                break

        return self._dedupe_hallucinations(hallucinations)

    def _crime_penalty_fallback_hallucinations(self, found_entities, answer):
        """RAG 자료가 부족할 때 CRIME/PENALTY 엔티티 검증 (휴리스틱 기반)"""
        hallucinations = []
        
        # RAG가 없을 때만 fallback 처리
        crime_entities = [entity for entity in found_entities if (entity.get("entity_group") or entity.get("label")) == "CRIME" and entity.get("word")]
        penalty_entities = [entity for entity in found_entities if (entity.get("entity_group") or entity.get("label")) == "PENALTY" and entity.get("word")]
        
        # 범죄와 형벌의 연관성 검증 (같은 문장에 있으면 연관성 높음)
        for crime_entity in crime_entities:
            crime_word = crime_entity.get("word", "")
            crime_start = crime_entity.get("start", 0)
            crime_end = crime_entity.get("end", 0)
            
            # 같은 문장 내 penalty와 연관성 확인
            same_sentence_penalties = []
            for penalty_entity in penalty_entities:
                penalty_start = penalty_entity.get("start", 0)
                penalty_end = penalty_entity.get("end", 0)
                
                # 같은 문장인지 확인 (. ! ? \n으로 구분)
                crime_sentence_start = max(0, answer.rfind("\n", 0, crime_start))
                crime_sentence_end = answer.find("\n", crime_end)
                if crime_sentence_end == -1:
                    crime_sentence_end = answer.find(".", crime_end)
                if crime_sentence_end == -1:
                    crime_sentence_end = len(answer)
                
                if crime_sentence_start <= penalty_start < crime_sentence_end and \
                   crime_sentence_start <= penalty_end <= crime_sentence_end:
                    same_sentence_penalties.append(penalty_entity)
            
            # CRIME 엔티티가 있지만 같은 문장에 PENALTY가 없으면 경고 (낮은 신뢰도)
            if crime_entities and not same_sentence_penalties:
                hallucinations.append({
                    "label": "CRIME",
                    "wrong_word": crime_word,
                    "correct_word": "",
                    "start": crime_entity.get("start"),
                    "end": crime_entity.get("end"),
                    "confidence": 0.65,  # 낮은 신뢰도
                    "support_score": 0.0,
                    "match_score": 0.0,
                    "risk_level": "medium",
                    "reason_code": "sparse_rag_crime_unverified",
                    "action": "log_only",
                    "reason": "범죄 행위가 검색 근거에서 충분히 확인되지 않았습니다.",
                })
        
        return self._dedupe_hallucinations(hallucinations)

    def _match_known_contact_org(self, word):
        compact = self._normalize_entity_text(word, "ORG")
        for org in KNOWN_CONTACTS:
            if self._normalize_entity_text(org, "ORG") in compact or compact in self._normalize_entity_text(org, "ORG"):
                return org
        return ""

    def _choose_action(self, label, risk_level, match, candidate):
        method = match.get("method")
        if label == "LAW" and risk_level == "high" and method in {"same_law_article_mismatch", "law_name_only"}:
            return "soften"
        if label in {"LAW", "ORG", "PHONE", "CASE_NUMBER", "DATE", "AMOUNT"} and candidate:
            return "replace"
        if risk_level == "high":
            return "remove_sentence"
        return "log_only"

    def _dedupe_hallucinations(self, hallucinations):
        deduped = []
        seen = set()
        for item in hallucinations:
            key = (item.get("label"), item.get("start"), item.get("end"), item.get("wrong_word"), item.get("reason_code"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def debug_match_entity(self, label, word, candidates, context_law=None):
        return self._match_entity_against_candidates(label=label, word=word, candidates=candidates, context_law=context_law)


ner_checker = NERFactChecker()
