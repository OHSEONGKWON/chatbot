import logging
import re
import asyncio
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger(__name__)

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
    class AnswerCorrector:
        def fix_answer(self, llm_answer, hallucinations):
            return llm_answer

try:
    from ..config import config
except Exception:
    class _R:
        embedding_model = "intfloat/multilingual-e5-large"

    class _C:
        rag = _R()

    config = _C()


# ── 이슈별 허용/금지 법률 도메인 ────────────────────────────────────────────────

ISSUE_REGISTRY = {
    "wage_unpaid": {
        "allowed_laws": ["근로기준법", "임금채권보장법", "근로자퇴직급여 보장법", "고용보험법"],
        "forbidden_laws": ["형법", "형법 제298조", "민법", "성폭력범죄의 처벌 등에 관한 특례법"],
    },
    "dismissal": {
        "allowed_laws": ["근로기준법", "근로자참여 및 협력증진에 관한 법률"],
        "forbidden_laws": ["형법", "형법 제298조", "민법", "성폭력범죄의 처벌 등에 관한 특례법"],
    },
    "missing_contract": {
        "allowed_laws": ["근로기준법"],
        "forbidden_laws": ["형법 제298조", "성폭력범죄의 처벌 등에 관한 특례법"],
    },
    "minimum_wage": {
        "allowed_laws": ["최저임금법", "근로기준법"],
        "forbidden_laws": ["형법 제298조", "성폭력범죄의 처벌 등에 관한 특례법"],
    },
    "workplace_harassment": {
        "allowed_laws": ["근로기준법"],
        "forbidden_laws": ["형법 제298조", "성폭력범죄의 처벌 등에 관한 특례법"],
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
    "근로계약서미작성": "missing_contract",
    "교육기관 언어적 성희롱": "sexual_harassment",
    "언어적 성희롱": "sexual_harassment",
    "신체접촉형 강제추행": "indecent_assault",
    "강간": "강간",
    "불법촬영": "illegal_filming",
    "직장내괴롭힘": "workplace_harassment",
    "직장 내 괴롭힘": "workplace_harassment",
    "최저임금": "minimum_wage",
    "최저시급": "minimum_wage",
    "wage_unpaid": "wage_unpaid",
    "dismissal": "dismissal",
    "missing_contract": "missing_contract",
    "minimum_wage": "minimum_wage",
    "workplace_harassment": "workplace_harassment",
    "sexual_harassment": "sexual_harassment",
    "indecent_assault": "indecent_assault",
    "illegal_filming": "illegal_filming",
}

# ── 범죄 유형 → 근거 법조항 매핑 ────────────────────────────────────────────────
# 답변에서 범죄명과 인용 법조항 번호의 정합성을 검증하는 데 사용됩니다.

CRIME_LAW_ARTICLE_MAP: dict[str, list[tuple[str, str]]] = {
    "강제추행": [("형법", "제298조")],
    "성추행": [("형법", "제298조")],
    "강간": [("형법", "제297조")],
    "불법촬영": [("성폭력범죄의 처벌 등에 관한 특례법", "제14조")],
    "카메라촬영": [("성폭력범죄의 처벌 등에 관한 특례법", "제14조")],
    "성희롱": [("양성평등기본법", "제30조")],
    "임금체불": [("근로기준법", "제43조"), ("근로기준법", "제36조"), ("근로기준법", "제37조")],
    "부당해고": [("근로기준법", "제23조")],
    "직장내괴롭힘": [("근로기준법", "제76조의2"), ("근로기준법", "제76조의3")],
    "최저임금미달": [("최저임금법", "제6조")],
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
    """6개 법률 엔티티(LAW/PENALTY/AMOUNT/DATE/ORG/CRIME) 기반 환각 탐지기.

    모델 경로가 없거나 파일이 없으면 경고 로그를 남기고 규칙 기반으로 대체합니다.
    """

    def __init__(self, model_path=None, require_model: bool = False):
        self.target_labels = ["LAW", "PENALTY", "AMOUNT", "DATE", "ORG", "CRIME"]
        self.model_path = model_path or getattr(getattr(config, "ner", None), "model_path", None)
        self.use_model = bool(getattr(getattr(config, "ner", None), "use_model", True))
        self.min_model_confidence = float(getattr(getattr(config, "ner", None), "min_confidence", 0.70))
        self.max_length = int(getattr(getattr(config, "ner", None), "max_length", 510))

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

        self.fuzzy_threshold = 0.90
        self.semantic_threshold = 0.82
        self.hallucination_report_threshold = 0.72
        # 시맨틱 매칭 기본 활성화 (CRIME/PENALTY 유사 표현 탐지용)
        self.enable_semantic_match = os.getenv("LAWSGUARD_ENABLE_SEMANTIC_NER", "1") == "1"
        self.semantic_allowed_labels = {"CRIME", "PENALTY"}

        self._semantic_embedder = None
        self._corrector = AnswerCorrector()
        self._ner_model = None
        self._ner_tokenizer = None
        self._nli_model = None       # lazy-loaded: Huffon/klue-roberta-base-nli
        self._nli_tokenizer = None
        self._nli_contra_idx = None
        self._nli_device = "cpu"

        # 모델 경로 즉시 검증 (lazy 로딩 전 startup 단계에서 문제 노출)
        if self.use_model and self.model_path and not Path(self.model_path).exists():
            if require_model:
                raise FileNotFoundError(
                    f"NER 모델을 찾을 수 없습니다: '{self.model_path}'\n"
                    f"먼저 scripts/train_legal_ner.py로 모델을 훈련하거나, "
                    f"LAWSGUARD_USE_MODEL_NER=0으로 규칙 기반 모드로 실행하세요."
                )
            logger.warning(
                "NER 모델을 찾을 수 없습니다: '%s'. 규칙 기반 엔티티 추출만 사용됩니다.",
                self.model_path,
            )
            self.use_model = False

    async def check_and_correct(self, answer, rag_docs, route=None, route_issue=None):
        return await asyncio.to_thread(self.check_and_correct_sync, answer, rag_docs, route, route_issue)

    async def check(self, answer, rag_docs):
        return await self.check_and_correct(answer, rag_docs)

    async def warmup(self):
        await asyncio.to_thread(self._load_ner_model)
        await asyncio.to_thread(self._load_semantic_embedder)

    def check_and_correct_sync(self, answer, rag_docs, route=None, route_issue=None):
        found_entities = self.extract_entities(answer)
        all_hallucinations = self.find_hallucinations(
            answer, rag_docs, found_entities=found_entities, route=route, route_issue=route_issue
        )

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
                entities.extend(
                    self._bio_predictions_to_entities(chunk, offset, offsets, pred_ids, pred_scores, model.config.id2label)
                )
            except Exception:
                continue
        return self._dedupe_entities(entities)

    def _load_ner_model(self):
        if not self.use_model:
            return None, None
        if self._ner_model is not None and self._ner_tokenizer is not None:
            return self._ner_model, self._ner_tokenizer

        if not self.model_path or not Path(self.model_path).exists():
            self.use_model = False
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
        except Exception as exc:
            logger.warning("NER 모델 로드 실패: %s. 규칙 기반으로 대체합니다.", exc)
            self._ner_model = None
            self._ner_tokenizer = None
        return self._ner_model, self._ner_tokenizer

    def _iter_text_chunks(self, text: str):
        """슬라이딩 윈도우로 청크를 생성합니다. 경계 엔티티 누락을 방지합니다."""
        chunk_size = 800
        overlap = 120
        if len(text) <= chunk_size:
            yield 0, text
            return
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            if end < len(text):
                cut = max(
                    text.rfind("\n", start, end),
                    text.rfind(". ", start, end),
                    text.rfind(" ", start, end),
                )
                if cut > start + 300:
                    end = cut + 1
            yield start, text[start:end]
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)

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
            should_start = (
                current is None
                or current["entity_group"] != entity_label
                or abs_start > current["end"] + 1
            )

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
                r"(?:근로기준법|형법|민법|남녀고용평등법|성폭력범죄의 처벌 등에 관한 특례법|고용보험법|최저임금법|양성평등기본법)",
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
        }
        for label, regexes in patterns.items():
            for pattern in regexes:
                for match in re.finditer(pattern, text):
                    word = re.sub(r"\s+", " ", match.group(0)).strip()
                    key = (label, match.start(), match.end(), word)
                    if not word or key in seen:
                        continue
                    seen.add(key)
                    entities.append({
                        "entity_group": label,
                        "label": label,
                        "word": word,
                        "start": match.start(),
                        "end": match.end(),
                        "score": 1.0,
                    })
        entities.sort(key=lambda item: item["start"])
        return entities

    def _merge_entity_lists(self, primary, secondary):
        merged = list(primary)
        for entity in secondary:
            overlaps = [
                existing
                for existing in merged
                if existing.get("entity_group") == entity.get("entity_group")
                and not (
                    entity.get("end", 0) <= existing.get("start", 0)
                    or entity.get("start", 0) >= existing.get("end", 0)
                )
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
        for entity in sorted(
            entities, key=lambda item: (item.get("start", 0), -(item.get("end", 0) - item.get("start", 0)))
        ):
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

    # ── 환각 탐지: 4개 레이어 ─────────────────────────────────────────────────

    def find_hallucinations(self, answer, rag_docs, found_entities=None, route=None, route_issue=None):
        found_entities = found_entities if found_entities is not None else self.extract_entities(answer)
        candidates_by_label, context_law = self._collect_candidates(rag_docs)
        hallucinations = []

        route_issue_key = self._resolve_route_issue(route, route_issue)

        # Layer 1: 엔티티별 RAG 후보 매칭
        for entity in found_entities:
            label = entity.get("entity_group") or entity.get("label")
            word = entity.get("word")
            if label not in {"LAW", "PENALTY", "AMOUNT", "DATE", "CRIME", "ORG"} or not word:
                continue
            candidates = candidates_by_label.get(label, [])
            if not candidates:
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
            hallucinations.append({
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
            })

        # Layer 2: 이슈-도메인 법률 교차 검증
        hallucinations.extend(
            self._route_issue_hallucinations(answer, found_entities, candidates_by_label, context_law, route_issue_key)
        )

        # Layer 2.5: NLI 기반 클레임 일관성 검사 (논리 반전·의미 모순 탐지)
        hallucinations.extend(self._validate_claim_consistency(answer, found_entities, rag_docs or []))

        # Layer 2.6: 부정문 반전 패턴 검사 (의무 서술 ↔ 부정 서술 충돌)
        hallucinations.extend(self._validate_negation_consistency(answer, rag_docs or []))

        # Layer 3: 범죄명-법조항 번호 정합성 검증
        hallucinations.extend(self._validate_law_articles(answer, found_entities))

        # Layer 3.5: 숫자 범위 조건 비교 (이상/이하/미만/초과 방향 포함)
        hallucinations.extend(self._validate_numeric_ranges(found_entities, candidates_by_label))

        # Layer 4: RAG 미지지 엔티티 검증 (LAW/CRIME/ORG/PENALTY)
        hallucinations.extend(self._validate_ungrounded_entities(found_entities, candidates_by_label))

        return self._dedupe_hallucinations(hallucinations)

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

    # ── Layer 2.5: NLI 기반 클레임 일관성 검사 ──────────────────────────────────

    _NLI_MODEL_NAME = "Huffon/klue-roberta-base-nli"
    _NLI_CONTRA_THRESHOLD = 0.70   # contradiction 확률 이 이상이면 환각 판정

    def _load_claim_nli(self):
        """KLUE-RoBERTa NLI 모델 lazy 로드. 실패 시 (None, None, None) 반환."""
        if self._nli_model == "unavailable":
            return None, None, None
        if self._nli_model is not None:
            return self._nli_model, self._nli_tokenizer, self._nli_contra_idx
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            # 내부 NLI 서브모델은 항상 CPU (LAWSGUARD_NER_NLI_CPU=0 으로 해제 가능)
            # GPU에서 CUDA assertion을 유발해 외부 모델 로딩을 오염시키는 것을 방지
            force_cpu = os.getenv("LAWSGUARD_NER_NLI_CPU", "1") == "1"
            device = "cpu" if force_cpu else ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(self._NLI_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(self._NLI_MODEL_NAME)
            model.to(device).eval()
            id2label = model.config.id2label
            contra_idx = next(
                (k for k, v in id2label.items() if "contradict" in str(v).lower()), None
            )
            if contra_idx is None:
                entail_idx = next((k for k, v in id2label.items() if "entail" in str(v).lower()), 0)
                contra_idx = 2 if entail_idx == 0 else 0
            self._nli_model = model
            self._nli_tokenizer = tokenizer
            self._nli_contra_idx = contra_idx
            self._nli_device = device
            logger.info("NLI 모델 로드: %s (contradiction idx=%d)", self._NLI_MODEL_NAME, contra_idx)
            return model, tokenizer, contra_idx
        except Exception as exc:
            logger.warning("NLI 모델 로드 실패: %s — 임베딩 유사도로 대체합니다.", exc)
            self._nli_model = "unavailable"
            return None, None, None

    def _validate_claim_consistency(
        self, answer: str, found_entities: list, rag_docs: list
    ) -> list[dict]:
        """Layer 2.5: 핵심 엔티티를 포함한 문장이 RAG 근거와 모순되는지 NLI로 검사.

        NLI 모델이 없을 때는 임베딩 유사도(threshold 상향)로 fallback.
        """
        CLAIM_LABELS = {"LAW", "PENALTY", "AMOUNT"}
        target_entities = [
            e for e in found_entities
            if (e.get("entity_group") or e.get("label")) in CLAIM_LABELS
            and len(e.get("word") or "") >= 2
        ]
        if not target_entities or not rag_docs:
            return []

        rag_texts = []
        for doc in rag_docs:
            text = (doc.get("text") or doc.get("content", "")) if isinstance(doc, dict) else getattr(doc, "text", "")
            if text and text.strip():
                rag_texts.append(text.strip()[:500])
        if not rag_texts:
            return []

        raw_sents = re.split(r"(?<=[.?!。？！])\s+|\n+", answer)
        sentences = [s.strip() for s in raw_sents if len(s.strip()) >= 15]
        if not sentences:
            return []

        nli_model, nli_tok, contra_idx = self._load_claim_nli()
        if nli_model is not None:
            return self._nli_check_claims(sentences, target_entities, rag_texts, nli_model, nli_tok, contra_idx)
        return self._embedding_check_claims(sentences, target_entities, rag_texts)

    def _nli_check_claims(self, sentences, target_entities, rag_texts, model, tokenizer, contra_idx):
        """NLI contradiction 판정으로 클레임 환각 탐지."""
        checked: set[str] = set()
        hallucinations: list[dict] = []
        for entity in target_entities:
            word = entity.get("word", "")
            label = entity.get("entity_group") or entity.get("label")
            for sent in sentences:
                if word not in sent or sent in checked:
                    continue
                checked.add(sent)
                best_contra = 0.0
                best_rag = ""
                for rag_text in rag_texts:
                    try:
                        enc = tokenizer(
                            rag_text, sent,
                            return_tensors="pt", truncation=True, max_length=512, padding=True,
                        )
                        enc = {k: v.to(self._nli_device) for k, v in enc.items()}
                        with torch.no_grad():
                            probs = torch.softmax(model(**enc).logits, dim=-1)[0]
                        contra_prob = float(probs[contra_idx])
                        if contra_prob > best_contra:
                            best_contra = contra_prob
                            best_rag = rag_text
                    except Exception:
                        continue
                if best_contra >= self._NLI_CONTRA_THRESHOLD:
                    hallucinations.append({
                        "label": label,
                        "wrong_word": sent[:120],
                        "correct_word": None,
                        "start": None,
                        "end": None,
                        "confidence": round(best_contra, 3),
                        "reason_code": "nli_contradiction",
                        "action": "soften",
                        "reason": (
                            f"해당 문장이 RAG 근거 문서와 모순됩니다 "
                            f"(NLI contradiction={best_contra:.2f})"
                        ),
                        "nli_contradiction_score": round(best_contra, 3),
                    })
        return hallucinations

    def _embedding_check_claims(self, sentences, target_entities, rag_texts):
        """임베딩 유사도 기반 클레임 검사 (NLI 없을 때 fallback, threshold=0.55)."""
        embedder = self._load_semantic_embedder()
        if embedder is None or np is None:
            return []
        THRESHOLD = 0.55
        try:
            rag_embs = embedder.encode(rag_texts, normalize_embeddings=True, show_progress_bar=False)
        except Exception:
            return []
        checked: set[str] = set()
        hallucinations: list[dict] = []
        for entity in target_entities:
            word = entity.get("word", "")
            label = entity.get("entity_group") or entity.get("label")
            for sent in sentences:
                if word not in sent or sent in checked:
                    continue
                checked.add(sent)
                try:
                    sent_emb = embedder.encode([sent], normalize_embeddings=True, show_progress_bar=False)[0]
                    max_sim = float(np.max(np.dot(rag_embs, sent_emb)))
                except Exception:
                    continue
                if max_sim < THRESHOLD:
                    hallucinations.append({
                        "label": label,
                        "wrong_word": sent[:120],
                        "correct_word": None,
                        "start": None,
                        "end": None,
                        "confidence": 0.62,
                        "reason_code": "semantically_unsupported_claim",
                        "action": "log_only",
                        "reason": (
                            f"해당 문장이 RAG 근거 문서와 의미적으로 일치하지 않습니다 "
                            f"(유사도={max_sim:.2f} < {THRESHOLD})"
                        ),
                        "semantic_similarity": round(max_sim, 3),
                    })
        return hallucinations

    # ── 숫자 범위 조건 비교 ───────────────────────────────────────────────────────

    _RANGE_DIRECTIONS = {"이상": "gte", "초과": "gt", "이하": "lte", "미만": "lt", "이내": "lte"}

    def _parse_range_value(self, text: str) -> tuple:
        """형량/금액 텍스트에서 (숫자값, 방향코드)를 추출. 파싱 실패 시 (None, '')."""
        nums = re.findall(r"\d[\d,]*", text)
        if not nums:
            return None, ""
        value = float(nums[0].replace(",", ""))
        if "만원" in text or "만 원" in text:
            value *= 10_000
        elif "억" in text:
            value *= 100_000_000
        direction = next((code for word, code in self._RANGE_DIRECTIONS.items() if word in text), "")
        return value, direction

    def _validate_numeric_ranges(self, found_entities: list, candidates_by_label: dict) -> list[dict]:
        """Layer 3.5: PENALTY/AMOUNT의 수치 및 이상/이하 방향 불일치 탐지.

        Layer 1의 fuzzy 매칭은 '징역 3년 이하'와 '징역 5년 이하'를 유사하다고 볼 수 있어
        이 레이어에서 수치 수준까지 비교합니다.
        """
        hallucinations = []
        for entity in found_entities:
            label = entity.get("entity_group") or entity.get("label")
            if label not in {"PENALTY", "AMOUNT"}:
                continue
            word = entity.get("word", "")
            if not word:
                continue
            candidates = candidates_by_label.get(label, [])
            if not candidates:
                continue
            ans_val, ans_dir = self._parse_range_value(word)
            if ans_val is None:
                continue
            for cand in candidates:
                rag_val, rag_dir = self._parse_range_value(cand)
                if rag_val is None:
                    continue
                # 수치 차이 5% 초과
                if abs(ans_val - rag_val) / max(rag_val, 1) > 0.05:
                    hallucinations.append({
                        "label": label,
                        "wrong_word": word,
                        "correct_word": cand,
                        "start": entity.get("start"),
                        "end": entity.get("end"),
                        "confidence": 0.88,
                        "support_score": 0.0,
                        "match_score": 0.0,
                        "risk_level": "high",
                        "reason_code": "numeric_value_mismatch",
                        "action": "replace",
                        "reason": f"수치가 RAG 근거와 다릅니다: 답변 '{word}' ≠ 근거 '{cand}'",
                    })
                    break
                # 이상/이하 방향 불일치
                if ans_dir and rag_dir and ans_dir != rag_dir:
                    hallucinations.append({
                        "label": label,
                        "wrong_word": word,
                        "correct_word": cand,
                        "start": entity.get("start"),
                        "end": entity.get("end"),
                        "confidence": 0.92,
                        "support_score": 0.0,
                        "match_score": 0.0,
                        "risk_level": "high",
                        "reason_code": "range_direction_mismatch",
                        "action": "replace",
                        "reason": f"범위 방향이 반대입니다: 답변 '{word}' ↔ 근거 '{cand}'",
                    })
                    break
        return self._dedupe_hallucinations(hallucinations)

    # ── 법조항 정합성 검증 ───────────────────────────────────────────────────────

    def _validate_law_articles(self, answer: str, found_entities: list) -> list[dict]:
        """범죄명과 인용된 법조항 번호의 정합성을 검증합니다.

        같은 법률 계열이지만 조문 번호가 다른 경우에만 플래그합니다.
        복수 범죄 유형이 근접한 경우 오탐 방지를 위해 건너뜁니다.
        """
        crime_entities = [
            e for e in found_entities if (e.get("entity_group") or e.get("label")) == "CRIME"
        ]
        law_entities = [
            e for e in found_entities if (e.get("entity_group") or e.get("label")) == "LAW"
        ]
        if not crime_entities or not law_entities:
            return []

        hallucinations = []
        for crime_entity in crime_entities:
            crime_word = re.sub(r"\s+", "", crime_entity.get("word", ""))
            expected_pairs = CRIME_LAW_ARTICLE_MAP.get(crime_word, [])
            if not expected_pairs:
                continue

            crime_pos = crime_entity.get("start", 0)
            expected_law_families = [pair[0] for pair in expected_pairs]
            expected_articles = {pair[1] for pair in expected_pairs}

            for law_entity in law_entities:
                law_word = law_entity.get("word", "")
                law_pos = law_entity.get("start", 0)
                # 범죄명과 600자 이내 위치한 법조항만 검사
                if abs(law_pos - crime_pos) > 600:
                    continue

                # _same_law_family은 완전 일치를 요구하므로, 정규식으로 추출된 LAW 엔티티가
                # 앞 문맥을 포함할 수 있어 부분 포함 방식으로 검사합니다.
                law_compact = re.sub(r"\s+", "", law_word.lower())
                is_expected_family = any(
                    re.sub(r"\s+", "", exp_law.lower()) in law_compact
                    for exp_law in expected_law_families
                )
                if not is_expected_family:
                    continue

                cited_article = self._law_article_key(law_word)
                if not cited_article:
                    continue

                if cited_article not in expected_articles:
                    correct_law = f"{expected_pairs[0][0]} {expected_pairs[0][1]}"
                    hallucinations.append({
                        "label": "LAW",
                        "wrong_word": law_word,
                        "correct_word": correct_law,
                        "start": law_entity.get("start"),
                        "end": law_entity.get("end"),
                        "confidence": 0.95,
                        "support_score": 0.0,
                        "match_score": 0.0,
                        "risk_level": "high",
                        "reason_code": "wrong_article_for_crime",
                        "action": "replace",
                        "reason": f"'{crime_word}'에 대해 인용된 법조항 '{law_word}'이(가) 올바르지 않습니다.",
                    })

        return self._dedupe_hallucinations(hallucinations)

    def _validate_ungrounded_entities(self, found_entities: list, candidates_by_label: dict) -> list[dict]:
        """Layer 4: RAG에 근거 없는 엔티티 플래그 (LAW/CRIME/ORG/PENALTY).

        Layer 1은 RAG에 후보가 있을 때만 동작하므로, 후보 자체가 없는 경우를 여기서 잡습니다.
        """
        CONFIGS = {
            "LAW":     ("medium", 0.72, "law_ungrounded_by_rag",     "법률명이 검색 근거 문서에서 확인되지 않았습니다."),
            "CRIME":   ("high",   0.80, "crime_ungrounded_by_rag",   "범죄 유형이 검색 근거 문서에서 확인되지 않았습니다."),
            "ORG":     ("medium", 0.70, "org_ungrounded_by_rag",     "기관명이 검색 근거 문서에서 확인되지 않았습니다."),
            "PENALTY": ("medium", 0.70, "penalty_ungrounded_by_rag", "형량 정보가 검색 근거 문서에서 확인되지 않았습니다."),
        }
        hallucinations = []
        for entity in found_entities:
            label = entity.get("entity_group") or entity.get("label")
            if label not in CONFIGS:
                continue
            word = entity.get("word", "")
            if not word:
                continue
            if candidates_by_label.get(label):
                continue  # RAG에 후보 있음 → Layer 1에서 이미 처리
            risk_level, confidence, reason_code, reason = CONFIGS[label]
            hallucinations.append({
                "label": label,
                "wrong_word": word,
                "correct_word": "",
                "start": entity.get("start"),
                "end": entity.get("end"),
                "confidence": confidence,
                "support_score": 0.0,
                "match_score": 0.0,
                "risk_level": risk_level,
                "reason_code": reason_code,
                "action": "log_only",
                "reason": reason,
            })
        return self._dedupe_hallucinations(hallucinations)

    # ── 부정문 반전 패턴 검사 ─────────────────────────────────────────────────────

    _NEGATION_RE = re.compile(
        r"(?:하지\s*않|않아도\s*됩|할\s*필요\s*없|면제|불필요|아니어도|아니라도|하지\s*않아도)"
    )
    _OBLIGATION_RE = re.compile(
        r"(?:해야\s*합|해야\s*한다|필수|의무|반드시|필요합니다|해야만|해야\s*됩)"
    )
    _NEGATION_SIM_THRESHOLD = 0.72

    def _validate_negation_consistency(self, answer: str, rag_docs: list) -> list[dict]:
        """Layer 2.6: 답변의 부정 서술이 RAG 근거의 의무 서술과 충돌하는지 검사.

        예) 답변: "신고하지 않아도 됩니다" / RAG: "반드시 신고해야 합니다"
        임베딩으로 같은 주제임을 확인한 뒤 부정↔긍정 패턴 충돌을 플래그합니다.
        """
        embedder = self._load_semantic_embedder()
        if embedder is None or np is None or not rag_docs:
            return []

        raw_sents = re.split(r"(?<=[.?!。？！])\s+|\n+", answer)
        neg_sents = [s.strip() for s in raw_sents if len(s.strip()) >= 15 and self._NEGATION_RE.search(s)]
        if not neg_sents:
            return []

        rag_pos_sents = []
        for doc in rag_docs:
            text = (doc.get("text") or doc.get("content", "")) if isinstance(doc, dict) else getattr(doc, "text", "")
            for s in re.split(r"[.?!。？！\n]+", text or ""):
                s = s.strip()
                if len(s) >= 10 and self._OBLIGATION_RE.search(s):
                    rag_pos_sents.append(s)
        if not rag_pos_sents:
            return []

        try:
            neg_embs = embedder.encode(neg_sents, normalize_embeddings=True, show_progress_bar=False)
            pos_embs = embedder.encode(rag_pos_sents, normalize_embeddings=True, show_progress_bar=False)
        except Exception:
            return []

        hallucinations = []
        for neg_sent, neg_emb in zip(neg_sents, neg_embs):
            sims = np.dot(pos_embs, neg_emb)
            max_idx = int(np.argmax(sims))
            max_sim = float(sims[max_idx])
            if max_sim >= self._NEGATION_SIM_THRESHOLD:
                hallucinations.append({
                    "label": "LOGIC",
                    "wrong_word": neg_sent[:120],
                    "correct_word": rag_pos_sents[max_idx][:120],
                    "start": None,
                    "end": None,
                    "confidence": round(max_sim, 3),
                    "reason_code": "negation_reversal",
                    "action": "soften",
                    "reason": (
                        f"답변의 부정 서술이 RAG 근거의 의무 서술과 충돌 가능 "
                        f"(유사도={max_sim:.2f})"
                    ),
                    "similarity": round(max_sim, 3),
                })
        return hallucinations

    # ── 정규화 ────────────────────────────────────────────────────────────────

    def _normalize_law_text(self, text):
        if not text:
            return ""
        t = text.strip().lower()
        t = re.sub(r'\([^)]*\)', '', t)
        t = re.sub(r'\s+', ' ', t).strip()

        law_abbreviations = {
            "근기": "근로기준법",
            "근기법": "근로기준법",
            "근로": "근로기준법",
            "최저임금": "최저임금법",
            "최저임금법": "최저임금법",
            "남고평": "남녀고용평등과일가정양립지원에관한법률",
            "남녀고용": "남녀고용평등과일가정양립지원에관한법률",
            "고용평등법": "남녀고용평등과일가정양립지원에관한법률",
            "양성평등": "양성평등기본법",
            "양평기본법": "양성평등기본법",
            "산안": "산업안전보건법",
            "산안법": "산업안전보건법",
            "산재법": "산업재해보상보험법",
            "산재보험": "산업재해보상보험법",
            "퇴직금법": "근로자퇴직급여보장법",
            "퇴직급여": "근로자퇴직급여보장법",
            "성폭": "성폭력범죄의처벌등에관한특례법",
            "성폭법": "성폭력범죄의처벌등에관한특례법",
            "성폭력처벌법": "성폭력범죄의처벌등에관한특례법",
            "성폭력특례법": "성폭력범죄의처벌등에관한특례법",
            "스토킹법": "스토킹범죄의처벌등에관한법률",
            "스토킹범죄처벌": "스토킹범죄의처벌등에관한법률",
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
        if not law1_text or not law2_text:
            return False
        norm1 = self._normalize_law_text(str(law1_text))
        norm2 = self._normalize_law_text(str(law2_text))
        if norm1 == norm2:
            return True
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

    def _normalize_entity_text(self, text, label=None):
        if not isinstance(text, str):
            return ""
        t = text.strip().lower()
        t = self.alias_map.get(t, t)

        if label == "LAW":
            return self._normalize_law_text(t)

        if label == "DATE":
            nums = re.findall(r"\d+", t)
            if nums:
                return "-".join(str(int(n)) for n in nums)
            return re.sub(r"\s+", "", t)

        if label == "AMOUNT":
            nums = re.findall(r"\d+", t)
            if nums:
                number = int("".join(nums))
                if "만원" in t:
                    number *= 10000
                return str(number)
            return re.sub(r"\s+", "", t)

        t = re.sub(r"\([^)]*\)", "", t)
        t = re.sub(r"[^0-9a-z가-힣]", "", t)
        return t

    # ── 유사도 ────────────────────────────────────────────────────────────────

    def _fuzzy_ratio(self, a, b):
        if not a or not b:
            return 0.0
        return float(SequenceMatcher(None, a, b).ratio())

    def _load_semantic_embedder(self):
        if self._semantic_embedder is None:
            try:
                from .embedder import get_embedder
                self._semantic_embedder = get_embedder()
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

    # ── 도메인 검사 ───────────────────────────────────────────────────────────

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

    # ── 매칭 핵심 ─────────────────────────────────────────────────────────────

    def _match_entity_against_candidates(self, label, word, candidates, context_law=None):
        norm_word = self._normalize_entity_text(word, label)
        if not norm_word:
            return {"is_supported": False, "candidate": None, "method": "empty", "score": 0.0, "confidence": 0.0, "combo_score": 1.0}

        if label in {"DATE", "AMOUNT"}:
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
                            best = {"is_supported": False, "candidate": cand, "method": "same_law_article_mismatch", "score": score, "confidence": confidence, "combo_score": combo_score}
                        continue
                    if not word_article or not cand_article:
                        score = 0.88
                        confidence = score * combo_score
                        if confidence > best["confidence"]:
                            best = {"is_supported": False, "candidate": cand, "method": "law_name_only", "score": score, "confidence": confidence, "combo_score": combo_score}
                        continue

            fuzzy = self._fuzzy_ratio(norm_word, norm_cand)
            allow_semantic = self.enable_semantic_match and label in self.semantic_allowed_labels
            semantic = 0.0
            if allow_semantic and fuzzy < fuzzy_threshold:
                semantic = self._semantic_similarity(word, cand)

            match_score = max(fuzzy, semantic)
            weighted_confidence = match_score * combo_score

            if weighted_confidence > best["confidence"]:
                best = {
                    "is_supported": False,
                    "candidate": cand,
                    "method": ("fuzzy" if fuzzy >= semantic else "semantic"),
                    "score": match_score,
                    "confidence": weighted_confidence,
                    "combo_score": combo_score,
                }

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
        method = match.get("method")
        score = float(match.get("score", 0.0))
        confidence = float(match.get("confidence", 0.0))

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
            if confidence >= 0.90:
                return 0.85

        if label == "DATE":
            if method == "exact":
                return 0.99
        if label == "AMOUNT":
            if method == "exact":
                return 0.98

        if label == "ORG":
            if method == "exact":
                return 0.99
            if method == "fuzzy" and score >= 0.95:
                return 0.93
            if method == "fuzzy" and score >= 0.90:
                return 0.85
            if confidence >= 0.85:
                return 0.80

        if label in {"CRIME", "PENALTY"}:
            if method == "exact":
                return 0.99
            if method == "semantic" and score >= 0.95:
                return 0.94
            if method == "fuzzy" and score >= 0.96:
                return 0.91
            if method == "fuzzy" and score >= 0.92:
                return 0.86
            if confidence >= 0.88:
                return 0.82

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
        law_entities = [
            entity for entity in found_entities
            if (entity.get("entity_group") or entity.get("label")) == "LAW"
        ]
        law_candidates = candidates_by_label.get("LAW", [])
        candidate_scarcity_threshold = 2

        for entity in law_entities:
            word = entity.get("word") or ""
            if not word:
                continue
            if any(self._same_law_family(word, term) for term in forbidden_laws):
                hallucinations.append({
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
                })
                continue

            is_in_allowed = any(self._same_law_family(word, term) for term in allowed_laws) if allowed_laws else True
            is_rag_sparse = len(law_candidates) < candidate_scarcity_threshold

            if allowed_laws and not is_in_allowed:
                hallucinations.append({
                    "label": "LAW",
                    "wrong_word": word,
                    "correct_word": allowed_laws[0],
                    "start": entity.get("start"),
                    "end": entity.get("end"),
                    "confidence": 0.98 if is_rag_sparse else 0.96,
                    "support_score": 0.0,
                    "match_score": 0.0,
                    "risk_level": "high",
                    "reason_code": "route_domain_mismatch",
                    "action": "remove_sentence",
                    "reason": f"{route_issue_key} 사안에서 허용된 법률 범위를 벗어나는 서술입니다.",
                })

        return self._dedupe_hallucinations(hallucinations)

    def _choose_action(self, label, risk_level, match, candidate):
        method = match.get("method")
        if label == "LAW" and risk_level == "high" and method in {"same_law_article_mismatch", "law_name_only"}:
            return "soften"
        if label in {"LAW", "ORG", "DATE", "AMOUNT"} and candidate:
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
        return self._match_entity_against_candidates(
            label=label, word=word, candidates=candidates, context_law=context_law
        )


# LAWSGUARD_REQUIRE_NER_MODEL=1 이면 모델 없을 때 서비스 기동 실패 (프로덕션 권장)
# LAWSGUARD_REQUIRE_NER_MODEL=0 (기본값) 이면 경고 후 규칙 기반으로 폴백
ner_checker = NERFactChecker(require_model=os.getenv("LAWSGUARD_REQUIRE_NER_MODEL", "0") == "1")
