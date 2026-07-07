"""법령 DB 조회 (legacy/alcv/law_index.py 에서 이식, 경로만 v3 구조로).

인덱스: legacy/law_index.json — {laws: {정규화법령명: {raw_name, articles: {제N조: 텍스트}}}}
"""
import json
import re
from pathlib import Path

INDEX_PATH = Path(__file__).resolve().parent.parent / "legacy" / "law_index.json"

LAW_ALIASES = {
    "근기법": "근로기준법", "근기": "근로기준법",
    "최저임금": "최저임금법",
    "성폭력처벌법": "성폭력범죄의 처벌 등에 관한 특례법",
    "성폭력특례법": "성폭력범죄의 처벌 등에 관한 특례법",
    "남녀고용평등법": "남녀고용평등과 일·가정 양립 지원에 관한 법률",
    "퇴직급여법": "근로자퇴직급여 보장법",
}

_ARTICLE_RE = re.compile(r"제\s*\d+\s*조(?:\s*의\s*\d+)?")

# 자유 텍스트에서 (법령명, 조문) 인용 추출
CITATION_RE = re.compile(
    r"[「『']?([가-힣][가-힣·\s]{1,40}?(?:법률|법|시행령|시행규칙))[」』']?\s*"
    r"(제\s*\d+\s*조(?:\s*의\s*\d+)?)")


def normalize_law_name(name: str) -> str:
    if not name:
        return ""
    name = name.split("\n")[0].strip()
    return re.sub(r"\s+", "", name)


def normalize_article(text: str) -> str:
    m = _ARTICLE_RE.search(text or "")
    return re.sub(r"\s+", "", m.group(0)) if m else ""


def extract_citations(text: str) -> list[tuple[str, str]]:
    """텍스트에서 (법령명, 조문) 쌍 추출. 예: '근로기준법 제60조' → ('근로기준법', '제60조')

    과잉 캡처(조사·수식어 포함)는 LawDB._resolve 의 부분 매칭이 흡수하므로 그대로 둔다.
    """
    return [(law.strip(), normalize_article(art))
            for law, art in CITATION_RE.findall(text or "")]


class LawDB:
    def __init__(self, index_path: Path = INDEX_PATH):
        data = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.laws = data["laws"]
        self._keys = list(self.laws.keys())

    def _resolve(self, law_name: str):
        if not law_name:
            return None
        key = normalize_law_name(LAW_ALIASES.get(law_name.strip(), law_name))
        if key in self.laws:
            return key
        for k in self._keys:
            if k and (k in key or key in k):
                return k
        return None

    def law_exists(self, law_name: str) -> bool:
        return self._resolve(law_name) is not None

    def article_exists(self, law_name: str, article: str) -> bool:
        key = self._resolve(law_name)
        return bool(key) and normalize_article(article) in self.laws[key]["articles"]

    def article_text(self, law_name: str, article: str) -> str | None:
        key = self._resolve(law_name)
        if not key:
            return None
        return self.laws[key]["articles"].get(normalize_article(article))
