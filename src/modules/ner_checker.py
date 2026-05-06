import re
from dataclasses import dataclass
from difflib import SequenceMatcher

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
        self.enable_semantic_match = True
        self.semantic_allowed_labels = {"CRIME", "PENALTY"}

        # Lazy components
        self._semantic_embedder = None
        self._corrector = AnswerCorrector()
        self._ner_pipeline = None

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
                self._semantic_embedder = SentenceTransformer(config.rag.embedding_model, device=device)
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
