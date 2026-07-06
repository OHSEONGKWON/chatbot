"""법령 DB 인덱스: (법령명, 조문번호) → 조문 텍스트.

ALCV의 핵심 기호 검증 토대. 구조화된 인덱스로 다음을 결정적으로 판정한다.
  - 인용 조문이 실제로 존재하는가? (예: '근로기준법 제999조' → 부재 = 환각)
  - 인용 법령이 인덱스에 있는가?
규칙 기반·exact lookup 이므로 RAG 검색 품질에 종속되지 않는다(기존 NER 실패의 핵심 원인 해소).
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from .config import LAW_CHUNKS_PATH, LAW_INDEX_PATH

# 약어 → 정식 명칭 (claim 의 법령 표현을 인덱스 키에 맞추기 위함)
LAW_ALIASES = {
    "근기법": "근로기준법", "근기": "근로기준법",
    "최저임금": "최저임금법",
    "성폭력처벌법": "성폭력범죄의 처벌 등에 관한 특례법",
    "성폭력특례법": "성폭력범죄의 처벌 등에 관한 특례법",
    "남녀고용평등법": "남녀고용평등과 일·가정 양립 지원에 관한 법률",
    "퇴직급여법": "근로자퇴직급여 보장법",
}

_ARTICLE_RE = re.compile(r"제\s*\d+\s*조(?:\s*의\s*\d+)?")


def normalize_law_name(name: str) -> str:
    """법령명 정규화: 중복 newline 제거 + 공백 제거(매칭 키)."""
    if not name:
        return ""
    name = name.split("\n")[0].strip()
    return re.sub(r"\s+", "", name)


def normalize_article(text: str) -> str:
    """'제 43 조의 2' → '제43조의2' 형태로 정규화. 없으면 ''."""
    if not text:
        return ""
    m = _ARTICLE_RE.search(text)
    if not m:
        return ""
    return re.sub(r"\s+", "", m.group(0))


def build_index(law_chunks_path: Path = LAW_CHUNKS_PATH, out_path: Path = LAW_INDEX_PATH) -> dict:
    """법령 청크 jsonl → 인덱스 JSON 저장. 반환: 통계 dict."""
    law_chunks_path = Path(law_chunks_path)
    if not law_chunks_path.exists():
        raise FileNotFoundError(f"법령 청크 없음: {law_chunks_path}")

    # norm_name -> { article_id -> text },  그리고 표시용 raw name 보존
    articles = defaultdict(dict)
    raw_name = {}
    n = 0
    with law_chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            md = rec.get("metadata", {})
            ln = (md.get("law_name", "") or "").split("\n")[0].strip()
            art = normalize_article(md.get("article_id", "") or "")
            if not ln:
                continue
            key = normalize_law_name(ln)
            raw_name.setdefault(key, ln)
            if art and art not in articles[key]:
                articles[key][art] = (rec.get("text", "") or "")[:600]
            n += 1

    index = {"laws": {k: {"raw_name": raw_name[k], "articles": v} for k, v in articles.items()}}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    stats = {"chunks": n, "laws": len(articles),
             "articles_total": sum(len(v) for v in articles.values())}
    print(f"[법령 인덱스] 청크 {stats['chunks']} → 법령 {stats['laws']}개, 조문 {stats['articles_total']}개")
    print(f"[저장] {out_path}")
    return stats


class LawIndex:
    """저장된 법령 인덱스 조회기."""

    def __init__(self, index_path: Path = LAW_INDEX_PATH):
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(
                f"법령 인덱스 없음: {index_path}\n먼저 run_build_law_index.py 실행")
        data = json.loads(index_path.read_text(encoding="utf-8"))
        self.laws = data["laws"]
        # 정규화 키 목록 (부분 매칭용)
        self._keys = list(self.laws.keys())

    def _resolve_law_key(self, law_name: str):
        """claim 의 법령 표현을 인덱스 키로 해석. 실패 시 None."""
        if not law_name:
            return None
        canonical = LAW_ALIASES.get(law_name.strip(), law_name)
        key = normalize_law_name(canonical)
        if key in self.laws:
            return key
        # 부분 포함(예: '근로기준법 제43조' → '근로기준법')
        for k in self._keys:
            if k and (k in key or key in k):
                return k
        return None

    def law_exists(self, law_name: str) -> bool:
        return self._resolve_law_key(law_name) is not None

    def article_exists(self, law_name: str, article: str) -> bool:
        """해당 법령에 그 조문이 존재하는가. 법령을 못 찾으면 False."""
        key = self._resolve_law_key(law_name)
        if key is None:
            return False
        art = normalize_article(article)
        if not art:
            return False
        return art in self.laws[key]["articles"]

    def get_article_text(self, law_name: str, article: str) -> str:
        key = self._resolve_law_key(law_name)
        if key is None:
            return ""
        return self.laws[key]["articles"].get(normalize_article(article), "")
