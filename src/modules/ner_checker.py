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


@dataclass
class NERCheckResult:
    original_answer: str
    corrected_answer: str
    found_entities: list
    mismatched_entities: list[dict]
    was_corrected: bool


class NERFactChecker:
    """Lightweight NER fact checker with law normalization, domain checks, and confidence scoring.

    Constructor avoids heavy model loading so the module can be imported in tests quickly.
    """
    def __init__(self, model_path=None):
        self.target_labels = ['LAW', 'PENALTY', 'AMOUNT', 'DATE', 'ORG', 'CRIME']
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
        self.enable_semantic_match = os.getenv("LAWSGUARD_ENABLE_SEMANTIC_NER", "0") == "1"
        self.semantic_allowed_labels = {"CRIME", "PENALTY"}

        # Lazy components
        self._semantic_embedder = None
        self._corrector = AnswerCorrector()
        self._ner_model = None
        self._ner_tokenizer = None

    async def check_and_correct(self, answer, rag_docs):
        return await asyncio.to_thread(self.check_and_correct_sync, answer, rag_docs)

    def check_and_correct_sync(self, answer, rag_docs):
        found_entities = self.extract_entities(answer)
        mismatches = self.find_hallucinations(answer, rag_docs, found_entities=found_entities)
        corrected = self._corrector.fix_answer(answer, mismatches)
        return NERCheckResult(
            original_answer=answer,
            corrected_answer=corrected,
            found_entities=found_entities,
            mismatched_entities=mismatches,
            was_corrected=corrected != answer,
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
                r"(?:고용노동부|경찰|검찰청|대법원|법률구조공단|여성긴급전화|학교|대학교)",
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
                    entities.append(
                        {
                            "entity_group": label,
                            "label": label,
                            "word": word,
                            "start": match.start(),
                            "end": match.end(),
                            "score": 1.0,
                        }
                    )
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

    def find_hallucinations(self, answer, rag_docs, found_entities=None):
        found_entities = found_entities if found_entities is not None else self.extract_entities(answer)
        candidates_by_label, context_law = self._collect_candidates(rag_docs)
        hallucinations = []

        for entity in found_entities:
            label = entity.get("entity_group") or entity.get("label")
            word = entity.get("word")
            if label not in {"LAW", "PENALTY", "AMOUNT", "DATE", "CRIME"} or not word:
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
            if float(match.get("score", 0.0)) < 0.75:
                continue
            confidence = 1.0 - float(match.get("confidence", 0.0))
            hallucinations.append(
                {
                    "label": label,
                    "wrong_word": word,
                    "correct_word": candidate,
                    "start": entity.get("start"),
                    "end": entity.get("end"),
                    "confidence": max(0.0, min(1.0, confidence)),
                    "reason_code": match.get("method", "unsupported_entity"),
                    "reason": "답변의 개체명이 검색 근거 문서에서 충분히 지지되지 않습니다.",
                }
            )
        return hallucinations

    def _collect_candidates(self, rag_docs):
        text_parts = []
        law_names = []
        for doc in rag_docs or []:
            text = doc.get("text", "") if isinstance(doc, dict) else getattr(doc, "text", "")
            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else getattr(doc, "metadata", {})
            text_parts.append(text or "")
            if isinstance(metadata, dict):
                for key in ("law_name", "source_file", "article_id", "article_title", "organization"):
                    value = metadata.get(key)
                    if value:
                        text_parts.append(str(value))
                if metadata.get("law_name"):
                    law_names.append(str(metadata["law_name"]))

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
        t = re.sub(r'제\s*\d+(?:조|항|절)', '', t)
        t = re.sub(r'\([^)]*\)', '', t)
        t = re.sub(r'\s+', ' ', t).strip()

        law_abbreviations = {
            "근기": "근로기준법",
            "근로": "근로기준법",
            "산안": "산업안전보건법",
            "성폭": "성폭력범죄의처벌등에관한특례법",
        }
        return law_abbreviations.get(t, t)

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

            # exact
            if norm_word == norm_cand:
                exact_confidence = 1.0 * combo_score
                return {"is_supported": True, "candidate": cand, "method": "exact", "score": 1.0, "confidence": exact_confidence, "combo_score": combo_score}

            # fuzzy
            fuzzy = self._fuzzy_ratio(norm_word, norm_cand)
            allow_semantic = self.enable_semantic_match and label in self.semantic_allowed_labels
            semantic = 0.0
            if allow_semantic and fuzzy < fuzzy_threshold:
                semantic = self._semantic_similarity(word, cand)

            weighted_score = 0.5 * (1.0 if fuzzy == 1.0 else 0.0) + 0.3 * fuzzy + 0.2 * semantic
            weighted_confidence = weighted_score * combo_score

            if weighted_confidence > best["confidence"]:
                best = {"is_supported": False, "candidate": cand, "method": ("fuzzy" if fuzzy >= semantic else "semantic"), "score": (fuzzy if fuzzy >= semantic else semantic), "confidence": weighted_confidence, "combo_score": combo_score}

        if best["method"] == "fuzzy" and best["score"] >= fuzzy_threshold:
            if best["confidence"] >= 0.95:
                best["is_supported"] = True
            return best
        if best["method"] == "semantic" and best["score"] >= semantic_threshold:
            if best["confidence"] >= 0.95:
                best["is_supported"] = True
            return best
        return best

    def debug_match_entity(self, label, word, candidates, context_law=None):
        return self._match_entity_against_candidates(label=label, word=word, candidates=candidates, context_law=context_law)


ner_checker = NERFactChecker()
