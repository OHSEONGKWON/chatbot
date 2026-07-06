"""유형별 결정적 검증 (근거 = RAG; 판단 = 규칙, LLM 미사용).

개체별 verdict:
  {type, word, verdict: ok|hallu|na, code, hard: bool}

  하드(고정밀, RAG 무관 또는 직접 모순):
    - nonexistent_article : 법령 DB에 그 조문이 없음 (article_number_error)
    - forbidden_law       : 사안에 금지된 법령 인용 (forbidden_law_injection)
    - numeric_mismatch    : 형량/금액이 RAG 근거의 값과 명백히 다름 (semantic_error)
  소프트(RAG 의존, 미존재일 수 있음):
    - org_ungrounded / date_ungrounded : 기관·날짜가 근거에 없음 (contact_mismatch 등)
"""
import re

from .config import ISSUE_KEYWORDS, NUMERIC_TOLERANCE
from .law_index import normalize_article

from src.modules.ner_checker import ISSUE_REGISTRY

_PENALTY_RE = re.compile(r"\d[\d,]*\s*년\s*(?:이하|이상)?\s*의?\s*징역|\d[\d,]*\s*만?\s*원\s*(?:이하|이상)?\s*의?\s*벌금")


def infer_issue(question: str) -> str:
    """질문에서 사안(issue) 추론. 못 찾으면 ''."""
    q = question or ""
    for issue, kws in ISSUE_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return issue
    return ""


def _grounded(word: str, rag_blob: str, ner) -> bool:
    """개체 표기가 RAG 근거에 (정규화 기준) 등장하는가."""
    norm = ner._normalize_entity_text(word, "ORG")
    if not norm:
        return True  # 빈 값은 검증 대상 아님
    blob = re.sub(r"\s+", "", rag_blob)
    return norm in blob


def verify_entities(question, entities, rag_texts, law_index, ner) -> list:
    issue = infer_issue(question)
    reg = ISSUE_REGISTRY.get(issue, {})
    forbidden = reg.get("forbidden_laws", [])
    rag_blob = "\n".join(rag_texts)

    # RAG 근거에서 비교용 형량/금액 값 추출
    ev_values = []
    for m in _PENALTY_RE.findall(rag_blob):
        v, _ = ner._parse_range_value(m)
        if v is not None:
            ev_values.append(v)

    verdicts = []
    for e in entities:
        label = e.get("entity_group") or e.get("label")
        word = e.get("word", "")
        v = {"type": label, "word": word, "verdict": "ok", "code": "", "hard": False}
        if not word:
            verdicts.append(v); continue

        if label == "LAW":
            base = ner._law_name_key(word)            # 정규화된 법령 base 이름
            art = normalize_article(word)
            # (a) 도메인 정합성: 금지 법령 인용
            if base and any(base == ner._law_name_key(f) for f in forbidden):
                v.update(verdict="hallu", code="forbidden_law", hard=True)
            # (b) 조문 존재: 법이 DB에 있고 조문이 DB에 없으면 반증
            elif art and law_index.law_exists(word) and not law_index.article_exists(word, art):
                v.update(verdict="hallu", code="nonexistent_article", hard=True)
            else:
                v["verdict"] = "ok" if law_index.law_exists(word) else "na"

        elif label in ("PENALTY", "AMOUNT"):
            val, _ = ner._parse_range_value(word)
            if val is None:
                v["verdict"] = "na"
            elif ev_values:
                # 근거에 비교 가능한 값이 있는데 어느 것과도 일치하지 않으면 수치 모순
                if not any(abs(val - c) / max(c, 1) <= NUMERIC_TOLERANCE for c in ev_values):
                    v.update(verdict="hallu", code="numeric_mismatch", hard=True)
            else:
                v.update(verdict="hallu", code="amount_ungrounded", hard=False)

        elif label == "ORG":
            if not _grounded(word, rag_blob, ner):
                v.update(verdict="hallu", code="org_ungrounded", hard=False)

        elif label == "DATE":
            if word.replace(" ", "") not in re.sub(r"\s+", "", rag_blob):
                v.update(verdict="hallu", code="date_ungrounded", hard=False)

        verdicts.append(v)
    return verdicts
