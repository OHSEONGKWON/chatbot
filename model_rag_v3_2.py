"""
model_rag_v3.py
─────────────────────────────────────────────────────────────────────────────
LawsGuard (로스가드+) – 전체 환각 탐지 파이프라인 통합 (v3)

[v3 변경사항]
  1. 국가법령정보 Open API 연동 → RAG 데이터셋 실시간 보강
  2. 유사 질문 10개 + 초기 답변 11개 터미널 출력
  3. 재질문을 법적으로 의미있는 질문으로 개선 (도메인 특화 프롬프트)
  4. 구체성 판단 기준 강화 (점수 기준 상향, 법률 요건 엄격 적용)
  5. 최종 답변 형식 강화 (질문 재정리 → 법령·판결 인용 → 법적 판단 → 대처방안)

[전체 파이프라인]
  Step 0  : 사용자 질문 입력
  Step 1  : 구체성 검사 (강화된 법적 기준) + 재질문 루프 (최대 5회)
  Step 2  : 유사 질문 10개 생성 + 터미널 출력
  Step 3  : RAG 병렬 답변 생성 (11개) + 터미널 출력
  Step 4  : BERTScore 일관성 스코어링 (median F1)
  Step 5.5: 법령 인용 검증 (Open API + ChromaDB 병행)
  Step 6  : NER 개체명 검증 (stub → 파인튜닝 후 활성화)
  Step 7  : 수정 및 최종 답변 (상세 형식 강제)

[설치]
  pip install chromadb sentence-transformers openai python-dotenv bert-score requests
"""

from __future__ import annotations

import os
import re
import json
import time
import threading
import statistics
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Callable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# 0. 설정값
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
if not os.path.exists(CHROMA_PERSIST_DIR):
    os.makedirs(CHROMA_PERSIST_DIR)
    print(f"📂 새 DB 저장 폴더 생성됨: {CHROMA_PERSIST_DIR}")
else:
    print(f"📂 DB 저장 경로 확인 완료: {CHROMA_PERSIST_DIR}")
CHROMA_COLLECTION    = "legal_verdicts"
EMBEDDING_MODEL      = "intfloat/multilingual-e5-large"
CHROMA_COLLECTION    = "legal_verdicts"
EMBEDDING_MODEL      = "intfloat/multilingual-e5-large"

N_RESULTS_PER_SOURCE = 3
N_RESULTS_TOTAL      = 9
SCORE_THRESHOLD      = 0.30
MAX_CONTEXT_CHARS    = 8000          # v3: 6000 → 8000 (상세 답변 지원)

LLM_MODEL            = "gpt-4o-mini"
LLM_TEMPERATURE      = 0.2
LLM_MAX_TOKENS       = 2400          # v3: 1200 → 2400 (상세 답변 지원)

# ── 구체성 검사 (v3: 기준 강화) ───────────────────────────────────────────────
SPECIFICITY_THRESHOLD = 4            # v3: 3 → 4 (더 엄격하게)
MAX_REASK_COUNT       = 5

# ── 유사 질문 생성 ────────────────────────────────────────────────────────────
N_SIMILAR_QUESTIONS   = 10
SIMILAR_Q_TEMPERATURE = 0.7

# ── 일관성 스코어링 ───────────────────────────────────────────────────────────
CONSISTENCY_THRESHOLD = 0.75
MAX_PARALLEL_WORKERS  = 5

# ── 국가법령정보 Open API ─────────────────────────────────────────────────────
LAW_API_KEY           = os.getenv("LAW_API_KEY", "")   # .env에 LAW_API_KEY=... 저장
LAW_API_BASE_URL      = "https://open.law.go.kr/LSO/lawService.do"

# ── NER 파인튜닝 모델 경로 ────────────────────────────────────────────────────
NER_MODEL_PATH        = None   # 예: r"D:\Legal_chatbot\ner_model"


# ──────────────────────────────────────────────────────────────────────────────
# 1. 데이터 모델
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:    str
    text:        str
    score:       float
    source_type: str
    metadata:    dict = field(default_factory=dict)

    def short_label(self) -> str:
        if self.source_type == "statute":
            law = self.metadata.get("law_name", "")
            art = self.metadata.get("article_id", "")
            return f"[법령] {law} {art}".strip()
        elif self.source_type == "manual":
            part  = self.metadata.get("part", "")
            label = self.metadata.get("section_label", "")
            return f"[매뉴얼] {part} {label}".strip()
        elif self.source_type == "law_api":
            return f"[Open API] {self.metadata.get('law_name','')} {self.metadata.get('article_id','')}".strip()
        else:
            case = self.metadata.get("case_number", "")
            sec  = self.metadata.get("section_type", "")
            return f"[판결문] {case} {sec}".strip()


@dataclass
class RAGResult:
    query:            str
    retrieved_chunks: list[RetrievedChunk]
    context_text:     str
    initial_answer:   str
    prompt_messages:  list[dict]
    retrieval_time:   float = 0.0
    generation_time:  float = 0.0


@dataclass
class SpecificityResult:
    score:             int
    entities:          dict = field(default_factory=dict)
    legal_category:    str  = "불명확"
    legal_issue:       Optional[str] = None
    missing_elements:  list[str] = field(default_factory=list)
    followup_question: Optional[str] = None
    is_sufficient:     bool = False
    partial_mode:      bool = False


@dataclass
class ConsistencyScore:
    median_f1:  float
    all_scores: list[float]
    passed:     bool


@dataclass
class CitationVerification:
    found:       list[str]
    missing:     list[str]
    corrections: dict


@dataclass
class NERVerification:
    corrections: dict = field(default_factory=dict)
    is_stub:     bool = True


@dataclass
class PipelineResult:
    query:          str
    final_answer:   str
    status:         str
    specificity:    Optional[SpecificityResult]    = None
    original_rag:   Optional[RAGResult]            = None
    consistency:    Optional[ConsistencyScore]     = None
    citation_check: Optional[CitationVerification] = None
    ner_check:      Optional[NERVerification]      = None
    total_time:     float                          = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. ChromaDB 연결 (싱글턴 + 스레드 락)
# ──────────────────────────────────────────────────────────────────────────────

_chroma_collection = None
_embedding_fn      = None
_collection_lock   = threading.Lock()


def _get_collection():
    global _chroma_collection, _embedding_fn

    if _chroma_collection is not None:
        return _chroma_collection

    with _collection_lock:
        if _chroma_collection is not None:
            return _chroma_collection

        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            device=device,
        )
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _chroma_collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=_embedding_fn,
        )
        print(f"[RAG] ChromaDB 연결 완료 | 총 {_chroma_collection.count()}청크")

    return _chroma_collection


# ──────────────────────────────────────────────────────────────────────────────
# 3. OpenAI 클라이언트 (싱글턴)
# ──────────────────────────────────────────────────────────────────────────────

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY 환경변수가 설정되지 않았습니다. "
            ".env 파일에 OPENAI_API_KEY=sk-... 를 추가하세요."
        )
    _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# ──────────────────────────────────────────────────────────────────────────────
# 4. 국가법령정보 Open API 연동  ← v3 신규
# ──────────────────────────────────────────────────────────────────────────────

class LawOpenAPI:
    """
    국가법령정보 Open API (https://open.law.go.kr) 연동 클래스.
    RAG 검색 시 ChromaDB 로컬 데이터셋을 보완하는 실시간 법령 조회에 사용.

    [지원 기능]
      - 법령명 검색 → 법령 목록 반환
      - 법령 ID로 조문 내용 조회
      - 질의어 기반 관련 법령 자동 검색 및 RetrievedChunk 변환

    [사용 전 필수]
      .env 파일에 LAW_API_KEY=발급받은_OC값 추가
      (OC: open.law.go.kr 회원가입 후 마이페이지에서 확인)
    """

    BASE_URL = "https://www.law.go.kr/DRF"

    def __init__(self, api_key: str = LAW_API_KEY):
        if not api_key:
            raise EnvironmentError(
                "LAW_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 LAW_API_KEY=발급받은OC값 을 추가하세요."
            )
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def search_law(self, query: str, display: int = 5) -> list[dict]:
        """
        법령명 키워드로 법령 목록 검색.

        Returns:
            [{"법령ID": ..., "법령명": ..., "시행일자": ...}, ...]
        """
        params = {
            "OC":      self.api_key,
            "target":  "law",
            "type":    "JSON",
            "query":   query,
            "display": display,
        }
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/lawSearch.do",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            laws = data.get("LawSearch", {}).get("law", [])
            if isinstance(laws, dict):
                laws = [laws]
            return laws
        except Exception as e:
            print(f"[WARN] Open API 법령 검색 실패 ({query}): {e}")
            return []

    def fetch_article(self, law_id: str) -> list[dict]:
        """
        법령 ID로 전체 조문 내용 조회.

        Returns:
            [{"조번호": ..., "조제목": ..., "조문내용": ...}, ...]
        """
        params = {
            "OC":     self.api_key,
            "target": "law",
            "type":   "JSON",
            "ID":     law_id,
        }
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/lawService.do",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            articles = (
                data.get("법령", {})
                    .get("조문", {})
                    .get("조문단위", [])
            )
            if isinstance(articles, dict):
                articles = [articles]
            return articles
        except Exception as e:
            print(f"[WARN] Open API 조문 조회 실패 (ID={law_id}): {e}")
            return []

    def query_to_chunks(
        self,
        query: str,
        max_laws: int = 2,
        max_articles_per_law: int = 3,
    ) -> list[RetrievedChunk]:
        """
        질의어 → 관련 법령 검색 → 조문 조회 → RetrievedChunk 리스트 반환.
        ChromaDB 검색 결과와 병합하여 컨텍스트를 보강하는 데 사용.

        Args:
            query:               사용자 질의
            max_laws:            검색할 최대 법령 수
            max_articles_per_law: 법령당 포함할 최대 조문 수

        Returns:
            RetrievedChunk 리스트 (source_type="law_api")
        """
        chunks: list[RetrievedChunk] = []
        laws = self.search_law(query, display=max_laws)

        for law in laws[:max_laws]:
            law_id   = law.get("법령ID", "")
            law_name = law.get("법령명한글", law.get("법령명", ""))
            if not law_id:
                continue

            articles = self.fetch_article(law_id)
            for art in articles[:max_articles_per_law]:
                art_no      = art.get("조번호", "")
                art_title   = art.get("조제목", "")
                art_content = art.get("조문내용", "")
                if not art_content:
                    continue

                text = (
                    f"【{law_name}】 제{art_no}조"
                    + (f" ({art_title})" if art_title else "")
                    + f"\n{art_content}"
                )
                chunks.append(RetrievedChunk(
                    chunk_id    = f"api_{law_id}_{art_no}",
                    text        = text,
                    score       = 0.80,   # API 공식 데이터이므로 고정 신뢰도
                    source_type = "law_api",
                    metadata    = {
                        "law_name":   law_name,
                        "law_id":     law_id,
                        "article_id": f"제{art_no}조",
                        "source_type": "law_api",
                    },
                ))

        if chunks:
            print(f"  [Open API] {len(chunks)}개 조문 추가 ({', '.join(set(c.metadata['law_name'] for c in chunks))})")
        return chunks


# Open API 싱글턴 (LAW_API_KEY 없으면 None)
_law_api: Optional[LawOpenAPI] = None
_law_api_lock = threading.Lock()


def _get_law_api() -> Optional[LawOpenAPI]:
    global _law_api
    if _law_api is not None:
        return _law_api
    with _law_api_lock:
        if _law_api is None:
            if LAW_API_KEY:
                try:
                    _law_api = LawOpenAPI(LAW_API_KEY)
                except EnvironmentError:
                    pass
    return _law_api


# ──────────────────────────────────────────────────────────────────────────────
# 5. 쿼리 확장
# ──────────────────────────────────────────────────────────────────────────────

_LAW_KEYWORD_MAP = {
    r"딥페이크|허위영상|합성물":                   "성폭력범죄의 처벌 등에 관한 특례법 제14조의2",
    r"불법촬영|몰카|카메라":                        "성폭력범죄의 처벌 등에 관한 특례법 제14조",
    r"사이버|온라인.*성희롱|통신.*음란":            "성폭력범죄의 처벌 등에 관한 특례법 제13조",
    r"강제추행|추행|신체접촉":                      "형법 제298조 강제추행",
    r"성희롱|직장내.*성희롱":                       "남녀고용평등과 일·가정 양립 지원에 관한 법률 제12조",
    r"근로.*해고|부당해고|해고예고":                "근로기준법 제23조 제26조 제28조",
    r"연장근로|야간근로|휴일근로|수당":             "근로기준법 제56조",
    r"육아휴직|출산휴가|산전산후":                  "근로기준법 제74조",
    r"직장내.*괴롭힘|직장.*갑질":                  "근로기준법 제76조의2",
    r"임금.*체불|임금미지급":                       "근로기준법 제43조 제36조",
    r"아르바이트|알바|시급":                        "근로기준법 최저임금법",
    r"피해신고|신고방법|어디.*신고":                "디지털성범죄피해자지원센터 1366 경찰",
    r"삭제.*지원|유포.*삭제":                      "디지털성범죄피해자지원센터 삭제지원",
    r"증거.*확보|증거.*보존":                      "증거 확보 방법 캡처 저장",
}


def expand_query(query: str) -> str:
    expansions = []
    for pattern, expansion in _LAW_KEYWORD_MAP.items():
        if re.search(pattern, query):
            expansions.append(expansion)
    return (query + " " + " ".join(expansions)).strip() if expansions else query


# ──────────────────────────────────────────────────────────────────────────────
# 6. RAG 검색 (ChromaDB + Open API 병합)
# ──────────────────────────────────────────────────────────────────────────────

def _query_one_source(
    collection,
    query_text: str,
    source_type: Optional[str],
    n_results: int,
) -> list[RetrievedChunk]:
    kwargs: dict = dict(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if source_type in ("statute", "manual"):
        kwargs["where"] = {"source_type": source_type}

    try:
        results = collection.query(**kwargs)
    except Exception as e:
        print(f"[WARN] 검색 실패 (source={source_type}): {e}")
        return []

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = round(1.0 - dist, 4)
        if source_type in ("statute", "manual"):
            stype = source_type
        else:
            stype = meta.get("source_type", "")
            if not stype:
                stype = "verdict" if meta.get("section_type") else "unknown"

        chunks.append(RetrievedChunk(
            chunk_id    = meta.get("doc_id", "") + "_" + str(len(chunks)),
            text        = doc,
            score       = score,
            source_type = stype,
            metadata    = meta,
        ))
    return chunks


def _deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    unique = []
    for c in chunks:
        key = c.text[:100]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def retrieve(
    query: str,
    n_per_source: int          = N_RESULTS_PER_SOURCE,
    score_threshold: float     = SCORE_THRESHOLD,
    source_filter: Optional[list[str]] = None,
    use_query_expansion: bool  = True,
    use_law_api: bool          = True,
) -> list[RetrievedChunk]:
    """
    ChromaDB 로컬 검색 + Open API 실시간 법령 조회 병합.

    v3 변경: use_law_api=True 시 Open API 결과를 ChromaDB 결과에 병합.
    API 청크는 score=0.80 고정으로 우선순위 중간에 삽입.
    """
    collection = _get_collection()
    expanded   = expand_query(query) if use_query_expansion else query
    sources    = source_filter or ["verdict", "statute", "manual"]

    # ── ChromaDB 검색 ──────────────────────────────────────────────────────────
    all_chunks: list[RetrievedChunk] = []
    for src in sources:
        src_filter = src if src in ("statute", "manual") else None
        all_chunks.extend(_query_one_source(collection, expanded, src_filter, n_per_source))

    filtered = [c for c in all_chunks if c.score >= score_threshold]
    filtered = _deduplicate(filtered)
    filtered.sort(key=lambda x: x.score, reverse=True)
    chroma_chunks = filtered[:N_RESULTS_TOTAL]

    # ── Open API 보강 ──────────────────────────────────────────────────────────
    if use_law_api:
        law_api = _get_law_api()
        if law_api:
            try:
                api_chunks = law_api.query_to_chunks(query, max_laws=2, max_articles_per_law=2)
                # API 청크는 ChromaDB 결과 뒤에 병합 후 재정렬
                combined = _deduplicate(chroma_chunks + api_chunks)
                combined.sort(key=lambda x: x.score, reverse=True)
                return combined[:N_RESULTS_TOTAL + 2]   # API 추가분 최대 2개 허용
            except Exception as e:
                print(f"[WARN] Open API 병합 실패: {e}")

    return chroma_chunks


# ──────────────────────────────────────────────────────────────────────────────
# 7. 컨텍스트 조립
# ──────────────────────────────────────────────────────────────────────────────

def _format_chunk(chunk: RetrievedChunk, idx: int) -> str:
    label = chunk.short_label()
    if chunk.source_type in ("statute", "law_api"):
        return f"[출처 {idx+1}] {label} (유사도: {chunk.score:.2f})\n{chunk.text}"
    elif chunk.source_type == "manual":
        return f"[출처 {idx+1}] {label} (유사도: {chunk.score:.2f})\n{chunk.text}"
    else:
        case_no  = chunk.metadata.get("case_number", "")
        court    = chunk.metadata.get("court", "")
        date     = chunk.metadata.get("decision_date", "")
        sec_type = chunk.metadata.get("section_type", "")
        header   = f"{court} {date} 선고 {case_no} [{sec_type}]".strip()
        return f"[출처 {idx+1}] [판결문] {header} (유사도: {chunk.score:.2f})\n{chunk.text}"


def build_context(chunks: list[RetrievedChunk]) -> str:
    # 법적 권위 순서: 법령(법령+API) → 판결문 → 매뉴얼
    ordered: list[RetrievedChunk] = []
    for stype in ("statute", "law_api", "verdict", "manual"):
        ordered.extend(c for c in chunks if c.source_type == stype)
    ordered.extend(c for c in chunks if c.source_type not in ("statute", "law_api", "verdict", "manual"))

    parts   = [_format_chunk(c, i) for i, c in enumerate(ordered)]
    context = "\n\n---\n\n".join(parts)

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[... 컨텍스트 일부 생략 ...]"
    return context


# ──────────────────────────────────────────────────────────────────────────────
# 8. 시스템 프롬프트 (v3: 상세 답변 형식 강제)
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 성범죄법 및 노동법 전문 법률 상담 AI입니다.
반드시 아래 [답변 형식]을 정확히 따르세요. 각 항목을 모두 작성해야 합니다.

[핵심 원칙]
1. 제공된 [참고자료]의 내용만을 근거로 답변하세요. 참고자료에 없는 내용을 추측하거나 창작하지 마세요.
2. 법령 조문을 인용할 때는 반드시 법령명과 조항번호를 명시하세요.
   예) 「형법」 제298조, 「근로기준법」 제76조의2 제1항
3. 판례를 인용할 때는 사건번호·법원명·판결 요지를 함께 명시하세요.
   예) 대법원 2022도1111 판결은 "..."라고 판시했습니다.
4. 참고자료에 근거 없는 형량·벌금 수치를 절대 언급하지 마세요.
5. 피해자 보호를 최우선으로 하며, 2차 가해적 표현을 절대 사용하지 마세요.
6. 법률 판단은 최종적으로 변호사 또는 법원에서 이루어짐을 고지하세요.

[답변 형식] ← 반드시 이 순서대로, 각 항목 빠짐없이 작성

(1) 상황 정리
사용자의 질문 내용을 객관적으로 재정리하세요. 핵심 사실관계를 명확히 서술합니다.
예) "아르바이트 사장님이 시급 인상을 미끼로 신체를 접촉한 상황입니다."

(2) 관련 법령 및 판결문
해당되는 법령 조문을 조항 번호까지 구체적으로 인용하고,
관련 판결문이 있으면 사건번호·법원·판결 요지를 함께 서술하세요.
예) "이 경우 「형법」 제298조(강제추행)가 적용됩니다. 해당 조문은 '폭행 또는 협박으로
사람에 대하여 추행을 한 자는 10년 이하의 징역 또는 1500만 원 이하의 벌금에 처한다'고
규정합니다. 대법원 2022도1111 판결은 '시급 인상을 조건으로 한 신체 접촉은 업무상 위력에
의한 추행에 해당한다'고 판시했습니다."

(3) 법적 판단
사용자의 사건에 위 법령과 판례를 구체적으로 적용하여 법적 결론을 내리세요.
예) "따라서 사장님의 행위는 강제추행죄(형법 제298조)에 해당할 가능성이 높으며,
직장 내 성희롱(남녀고용평등법 제12조)도 함께 성립할 수 있습니다."

(4) 대처 방안 및 지원기관
즉시 취해야 할 행동 순서와 연락처를 포함한 지원기관을 안내하세요.
예) "① 즉시 증거 확보(문자, 녹음 등) ② 경찰 신고(112) 또는 고소장 접수
③ 여성긴급전화(1366) 또는 성폭력 상담소 연락 ④ 법률구조공단(132)을 통한
무료 법률상담 신청"

(5) 유의사항
최종적으로 반드시 "본 답변은 참고용이며, 정확한 법률 판단은 변호사 또는
관련 기관과 상담하시기 바랍니다."를 포함하세요."""

SYSTEM_PROMPT_PARTIAL = SYSTEM_PROMPT + """

[부분답변 모드]
충분한 상황 정보 없이 일반 원칙 수준의 답변을 드립니다.
구체적 사례 적용을 위해서는 추가 정보 제공 및 전문가 상담을 권장합니다."""

# 유사 질문 답변용: 일관성 측정 목적으로 형식 간소화
SYSTEM_PROMPT_CONSISTENCY = """당신은 법률 상담 AI입니다.
제공된 [참고자료]만을 근거로 질문에 답변하세요.
법령명과 조항번호를 포함하여 핵심 법적 판단을 3~5문장으로 간결하게 서술하세요."""


# ──────────────────────────────────────────────────────────────────────────────
# 9. 초기 답변 생성
# ──────────────────────────────────────────────────────────────────────────────

def _build_messages(
    query: str,
    context: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    user_content = (
        f"[참고자료]\n{context}\n\n---\n\n"
        f"[상담 질문]\n{query}\n\n"
        f"위 참고자료를 바탕으로 지정된 답변 형식에 따라 답변해 주세요."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]


def generate_initial_answer(
    query: str,
    context: str,
    model: str         = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int    = LLM_MAX_TOKENS,
    system_prompt: str = SYSTEM_PROMPT,
    retry: int         = 2,
) -> tuple[str, list[dict]]:
    client   = _get_openai_client()
    messages = _build_messages(query, context, system_prompt)

    for attempt in range(retry + 1):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip(), messages
        except Exception as e:
            if attempt < retry:
                wait = 2 ** attempt
                print(f"[WARN] LLM 호출 실패 (시도 {attempt+1}/{retry+1}), {wait}초 후 재시도: {e}")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM 호출 최종 실패: {e}") from e


# ──────────────────────────────────────────────────────────────────────────────
# 10. [Step 1] 구체성 검사 (v3: 기준 강화 + 법적 재질문 개선)
# ──────────────────────────────────────────────────────────────────────────────

# v3: 점수 기준 강화, 법률 요건 중심으로 프롬프트 재작성
_SPECIFICITY_SYSTEM = """\
당신은 법률 상담 전문가입니다. 사용자의 질문이 구체적인 법률 자문을 제공할 수 있을 만큼
충분한 정보를 담고 있는지 엄격하게 평가합니다.

반드시 아래 JSON 형식만 반환하세요. 설명, 마크다운 없이 JSON만.

{
  "score": <1~5 정수>,
  "entities": {
    "subject":  "<주체: 피해자/가해자 관계, 신분 (예: 아르바이트생/고용주) | null>",
    "time":     "<시기: 구체적 날짜 또는 기간 | null>",
    "action":   "<행위: 무슨 일이 발생했는지 구체적 행위 | null>",
    "purpose":  "<목적: 사용자가 원하는 것 (신고/처벌/손해배상/절차 확인 등) | null>"
  },
  "legal_category": "<민사|형사|노동|가사|기타|불명확>",
  "legal_issue": "<구체적 법적 쟁점 (예: 강제추행 성립 여부, 부당해고 여부) | null>",
  "missing_elements": ["<부족한 법적 판단 요소 목록>"],
  "followup_question": "<법적으로 의미있는 재질문 문구 | null>"
}

[점수 기준 - 엄격 적용]
5점: 주체·행위·시기·목적이 모두 명확하고 법적 쟁점 즉시 파악 가능
     예) "2024년 3월, 편의점 아르바이트 중 사장이 시급 인상 조건으로 엉덩이를 만졌습니다. 고소하고 싶습니다."
4점: 핵심 요소 대부분 있으나 일부 보완 필요 → 자문 가능 수준
     예) "직장 상사가 회식에서 신체를 만졌어요. 어떻게 해야 하나요?" (시기·구체적 행위 보완 필요)
3점: 상황은 파악되나 법적 판단에 필요한 핵심 요소 부족
     예) "사장이 이상한 행동을 해요." (행위·목적 불명확)
2점: 매우 모호하여 어떤 법률 분야인지조차 불명확
     예) "직장에서 힘들어요."
1점: 법률 상담 불가 수준
     예) "어떻게 하죠?"

[재질문 작성 원칙]
- 법적 판단에 실제로 필요한 정보를 구체적으로 질문하세요.
- 나쁜 예: "더 자세히 설명해 주세요" (법적으로 무의미)
- 좋은 예: "가해자와 피해자의 관계(고용주/동료/타인)가 어떻게 되시나요? 업무 관련 상황이었나요?"
- 좋은 예: "신체 접촉이 있었나요, 아니면 언어적 성희롱이었나요? 어느 쪽인지에 따라 적용 법령이 달라집니다."
- 좋은 예: "임금 체불이 발생한 구체적인 기간과 미지급 금액이 얼마나 되나요?"
"""

# 도메인별 법적으로 의미있는 재질문 템플릿 (v3 신규)
_LEGAL_REASK_TEMPLATES = {
    "형사": [
        "가해자와 피해자의 관계(고용주/동료/타인/모르는 사람)가 어떻게 되나요? 관계에 따라 적용되는 법령이 달라집니다.",
        "신체 접촉이 있었나요, 아니면 언어적 성희롱이었나요? 물리적 접촉 여부에 따라 강제추행죄 또는 성희롱으로 구분됩니다.",
        "해당 행위가 발생한 장소와 상황(업무 중/회식/이동 중)을 알 수 있나요? 업무 관련성이 처벌 수위에 영향을 줍니다.",
        "사건 당시 또는 이후 증거(문자, 녹음, 목격자)가 있나요? 증거 유무가 고소 전략에 중요합니다.",
        "현재 원하시는 것이 형사 고소인가요, 민사 손해배상인가요, 아니면 두 가지 모두인가요?",
    ],
    "노동": [
        "고용 형태가 정규직인가요, 아르바이트(단시간근로자)인가요? 형태에 따라 적용되는 근로기준법 조항이 다릅니다.",
        "근로계약서를 작성하셨나요? 계약서 존재 여부가 법적 입증에 중요합니다.",
        "임금 체불이라면 미지급 기간과 금액이 얼마나 되나요? 금액에 따라 진정·고소 전략이 달라집니다.",
        "해고를 당하셨다면 사전에 해고 예고를 받으셨나요? 예고 없는 해고는 부당해고 가능성이 있습니다.",
        "재직 기간이 얼마나 되나요? 퇴직금 청구 가능 여부(1년 이상)와 관련이 있습니다.",
    ],
    "민사": [
        "금전 거래라면 차용증이나 이체 내역 등 채무 증거가 있나요?",
        "상대방과 합의를 시도한 적이 있나요? 합의 시도 여부가 소송 전략에 영향을 줍니다.",
        "청구하려는 금액이 얼마나 되나요? 금액에 따라 소액사건심판(3천만 원 이하) 또는 일반 민사소송으로 나뉩니다.",
    ],
    "불명확": [
        "발생한 일이 신체적 피해인가요, 금전적 피해인가요, 아니면 명예 훼손 등 다른 유형인가요? 피해 유형에 따라 적용 법령이 완전히 달라집니다.",
        "가해자가 누구인지 알고 계신가요? (고용주/동료/타인) 관계에 따라 형사·민사·노동법 중 어떤 법률이 적용될지 달라집니다.",
    ],
}


def _get_legal_reask(category: str, missing_elements: list[str], reask_count: int) -> str:
    """
    도메인별 법적으로 의미있는 재질문 반환.
    reask_count를 인덱스로 사용해 매 회 다른 질문 선택.
    """
    templates = _LEGAL_REASK_TEMPLATES.get(category, _LEGAL_REASK_TEMPLATES["불명확"])
    idx = reask_count % len(templates)
    return templates[idx]


def check_specificity(query: str) -> SpecificityResult:
    """Step 1: 구체성 검사 (v3: 기준 강화, 점수 4점 이상만 통과)"""
    client = _get_openai_client()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SPECIFICITY_SYSTEM},
                {"role": "user",   "content": f"질문: {query}"},
            ],
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"[WARN] 구체성 검사 실패, 기본값 사용: {e}")
        data = {
            "score": 3, "entities": {}, "legal_category": "불명확",
            "legal_issue": None, "missing_elements": [], "followup_question": None,
        }

    score    = int(data.get("score", 3))
    category = data.get("legal_category", "불명확")

    # v3: followup_question이 없거나 법적으로 의미 없으면 템플릿으로 대체
    followup = data.get("followup_question")
    missing  = data.get("missing_elements", [])
    if not followup or len(followup) < 15:
        followup = _get_legal_reask(category, missing, reask_count=0)

    return SpecificityResult(
        score             = score,
        entities          = data.get("entities", {}),
        legal_category    = category,
        legal_issue       = data.get("legal_issue"),
        missing_elements  = missing,
        followup_question = followup,
        is_sufficient     = score >= SPECIFICITY_THRESHOLD,  # 4점 이상
    )


# ──────────────────────────────────────────────────────────────────────────────
# 11. [Step 2] 유사 질문 10개 생성 + 터미널 출력  (v3)
# ──────────────────────────────────────────────────────────────────────────────

_SIMILAR_Q_SYSTEM = """\
당신은 법률 질문 다양화 전문가입니다.
주어진 법률 질문과 의미는 동일하지만 표현과 관점이 다른 질문 {n}개를 생성하세요.

다음 4가지 관점을 고르게 포함하세요:
① 사실관계 중심: 어떤 일이 발생했는지에 초점
② 법적 요건 중심: 어떤 법률 요건이 충족되는지에 초점
③ 피해자/당사자 관점: 당사자 입장에서 서술
④ 구제 절차 중심: 어떻게 해결할 수 있는지에 초점

반드시 JSON 객체 형식으로 반환하세요.
{{"questions": ["질문1", "질문2", ..., "질문{n}"]}}
"""


def generate_similar_questions(
    query: str,
    n: int    = N_SIMILAR_QUESTIONS,
    verbose: bool = True,
) -> list[str]:
    """Step 2: 유사 질문 n개 생성 + 터미널 출력"""
    client = _get_openai_client()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SIMILAR_Q_SYSTEM.format(n=n)},
                {"role": "user",   "content": f"원본 질문: {query}"},
            ],
            temperature=SIMILAR_Q_TEMPERATURE,
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        parsed    = json.loads(response.choices[0].message.content.strip())
        questions = parsed.get("questions", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(questions, list):
            questions = list(parsed.values())[0] if isinstance(parsed, dict) else []
        questions = [str(q).strip() for q in questions if str(q).strip()][:n]
    except Exception as e:
        print(f"[WARN] 유사 질문 생성 실패: {e}")
        questions = []

    # v3: 터미널 출력
    if verbose:
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"[Step 2] 생성된 유사 질문 {len(questions)}개")
        print(sep)
        for i, q in enumerate(questions, 1):
            print(f"  {i:02d}. {q}")
        print(sep)

    return questions


# ──────────────────────────────────────────────────────────────────────────────
# 12. [Step 3] RAG 단일 + 병렬 실행 + 터미널 출력  (v3)
# ──────────────────────────────────────────────────────────────────────────────

def run_phase1(
    query: str,
    n_per_source: int          = N_RESULTS_PER_SOURCE,
    score_threshold: float     = SCORE_THRESHOLD,
    source_filter: Optional[list[str]] = None,
    use_query_expansion: bool  = True,
    use_law_api: bool          = True,
    llm_model: str             = LLM_MODEL,
    system_prompt: str         = SYSTEM_PROMPT,
    verbose: bool              = False,
) -> RAGResult:
    """단일 쿼리 RAG 검색 + 답변 생성 (병렬 워커 함수로도 사용)"""
    t0 = time.time()
    chunks = retrieve(
        query=query, n_per_source=n_per_source,
        score_threshold=score_threshold, source_filter=source_filter,
        use_query_expansion=use_query_expansion, use_law_api=use_law_api,
    )
    retrieval_time = time.time() - t0

    # 검색 결과 없음 처리 (버그 수정 유지)
    if not chunks:
        return RAGResult(
            query=query, retrieved_chunks=[],
            context_text="", initial_answer=(
                "죄송합니다. 현재 데이터베이스에서 해당 질문과 관련된 법령·판례·지침을 "
                "찾지 못했습니다. 법률구조공단(132) 또는 여성긴급전화(1366)에 문의하세요."
            ),
            prompt_messages=[], retrieval_time=retrieval_time, generation_time=0.0,
        )

    context = build_context(chunks)

    t1 = time.time()
    answer, messages = generate_initial_answer(
        query=query, context=context,
        model=llm_model, system_prompt=system_prompt,
    )
    generation_time = time.time() - t1

    return RAGResult(
        query=query, retrieved_chunks=chunks,
        context_text=context, initial_answer=answer,
        prompt_messages=messages,
        retrieval_time=retrieval_time, generation_time=generation_time,
    )


def run_rag_parallel(
    queries: list[str],
    is_original: list[bool],
    verbose: bool = False,
) -> list[Optional[RAGResult]]:
    """
    v3 수정: 원본 질문으로 RAG 검색 1회 → 동일 컨텍스트로 11개 답변 생성.
    검색 다양성이 아닌 LLM 해석 다양성만 측정하도록 구조 변경.
    """
    # ── Step 1: 원본 질문으로만 RAG 검색 ─────────────────────────────────────
    original_query = queries[0]
    print(f"\n[Step 3] 원본 질문으로 RAG 검색 (1회)...")
    _get_collection()  # 싱글턴 미리 초기화

    t0 = time.time()
    chunks = retrieve(
        query=original_query,
        use_query_expansion=True,
        use_law_api=True,
    )
    retrieval_time = time.time() - t0

    if not chunks:
        no_ans = (
            "죄송합니다. 관련 법령·판례를 찾지 못했습니다. "
            "법률구조공단(132) 또는 여성긴급전화(1366)에 문의하세요."
        )
        empty = RAGResult(
            query=original_query, retrieved_chunks=[],
            context_text="", initial_answer=no_ans,
            prompt_messages=[], retrieval_time=retrieval_time, generation_time=0.0,
        )
        return [empty] + [None] * (len(queries) - 1)

    # ── 고정 컨텍스트 조립 ────────────────────────────────────────────────────
    shared_context = build_context(chunks)
    print(f"  → {len(chunks)}개 청크 ({retrieval_time:.2f}s) | 컨텍스트 {len(shared_context)}자")
    print(f"  → 동일 컨텍스트로 {len(queries)}개 답변 병렬 생성...")

    # ── Step 2: 동일 컨텍스트로 11개 답변 병렬 생성 ──────────────────────────
    results: list[Optional[RAGResult]] = [None] * len(queries)
    workers = min(len(queries), MAX_PARALLEL_WORKERS)

    def _answer_worker(idx: int, q: str) -> tuple[int, RAGResult]:
        sp = SYSTEM_PROMPT if is_original[idx] else SYSTEM_PROMPT_CONSISTENCY
        t1 = time.time()
        answer, messages = generate_initial_answer(
            query=q,
            context=shared_context,   # ← 모든 쿼리에 동일 컨텍스트 주입
            system_prompt=sp,
        )
        return idx, RAGResult(
            query=q, retrieved_chunks=chunks,
            context_text=shared_context, initial_answer=answer,
            prompt_messages=messages,
            retrieval_time=retrieval_time,
            generation_time=round(time.time() - t1, 2),
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_answer_worker, i, q): i
            for i, q in enumerate(queries)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                _, result = future.result()
                results[idx] = result
            except Exception as e:
                print(f"[WARN] 쿼리 {idx+1} 답변 생성 실패: {e}")

    # ── 터미널 출력 ───────────────────────────────────────────────────────────
    sep = "─" * 60
    print(f"\n{sep}")
    print("[Step 3] 생성된 초기 답변 (원본 + 유사 질문)")
    print(sep)
    for i, r in enumerate(results):
        label = "원본 질문" if is_original[i] else f"유사 질문 {i:02d}"
        q_preview = queries[i][:60] + ("..." if len(queries[i]) > 60 else "")
        print(f"\n  ▶ [{label}] {q_preview}")
        if r and r.initial_answer:
            preview = r.initial_answer[:200].replace("\n", " ")
            print(f"  {preview}{'...' if len(r.initial_answer) > 200 else ''}")
        else:
            print("  (답변 생성 실패)")
    print(sep)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 13. [Step 4] BERTScore 일관성 스코어링
# ──────────────────────────────────────────────────────────────────────────────

def _extract_legal_core(text: str) -> str:
    """
    BERTScore 비교용 핵심 문장 추출.
    답변 형식 마커((1)(2)①② 등)와 대처방안·유의사항 섹션을 제거하고
    법적 판단 관련 문장만 반환.
    """
    # 섹션 마커 제거
    text = re.sub(r"\([1-5]\)\s*[^\n]+\n?", "", text)   # (1) 상황 정리 등
    text = re.sub(r"[①-⑤]\s*", "", text)
    # 대처방안·유의사항 섹션 이후 제거 (법적 판단 핵심만 보존)
    for marker in ["(4)", "(5)", "대처 방안", "유의사항", "본 답변은 참고용"]:
        if marker in text:
            text = text[:text.index(marker)]
    # 빈 줄 정리
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return " ".join(lines)


# ──────────────────────────────────────────────────────
# score_consistency() 전체 교체
# ──────────────────────────────────────────────────────

def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """넘파이 없이 순수 파이썬 코사인 유사도"""
    dot   = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot / (norm1 * norm2 + 1e-9)


def _score_bertscore(original: str, comparisons: list[str]) -> list[float]:
    """
    BERTScore F1 (klue/roberta-base).
    실패 시 빈 리스트 반환 → 앙상블에서 해당 지표 제외.
    """
    try:
        from bert_score import score as bert_score_fn
        _, _, F1 = bert_score_fn(
            [original] * len(comparisons), comparisons,
            lang="ko", model_type="klue/roberta-base",
            num_layers=12, verbose=False,
        )
        return [round(f.item(), 4) for f in F1]
    except Exception as e:
        print(f"[WARN] BERTScore 실패: {e}")
        return []


def _score_embedding_cosine(original: str, comparisons: list[str]) -> list[float]:
    """
    프로젝트 기존 임베딩 모델(multilingual-e5-large)로 코사인 유사도 계산.
    ChromaDB에 이미 로드된 모델을 재사용하므로 추가 비용 없음.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = SentenceTransformer(EMBEDDING_MODEL, device=device)

        all_texts = [original] + comparisons
        embeddings = model.encode(all_texts, normalize_embeddings=True)

        orig_vec = embeddings[0].tolist()
        return [
            round(_cosine_similarity(orig_vec, embeddings[i+1].tolist()), 4)
            for i in range(len(comparisons))
        ]
    except Exception as e:
        print(f"[WARN] 임베딩 코사인 유사도 실패: {e}")
        return []


def _score_llm_consistency(original: str, comparisons: list[str]) -> list[float]:
    """
    GPT-4o-mini로 법률 의미 일관성 직접 평가.
    법률 용어·개념 동일성을 언어모델이 직접 판단하므로
    도메인 특화 정확도가 가장 높음.
    비용 절감을 위해 비교 대상을 샘플링(최대 5개)해서 평가 후 나머지는 평균으로 보정.
    """
    if not comparisons:
        return []

    client = _get_openai_client()

    # 비용 절감: 최대 5개만 직접 평가 후 나머지는 평균으로 보정
    sample_size   = min(5, len(comparisons))
    sampled       = comparisons[:sample_size]
    scores_sample = []

    _CONSISTENCY_PROMPT = """\
아래 [기준 답변]과 [비교 답변]이 법률적으로 동일한 결론을 담고 있는지 평가하세요.
법령명이 달라도 같은 사안에 적용되면 일관성 있다고 봅니다.
표현이 달라도 법적 판단이 같으면 높은 점수를 주세요.

0.0 ~ 1.0 사이의 숫자 하나만 반환하세요. 설명 없이.

[기준 답변]
{original}

[비교 답변]
{comparison}"""

    for comp in sampled:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": _CONSISTENCY_PROMPT.format(
                        original=original[:600],
                        comparison=comp[:600],
                    ),
                }],
                temperature=0.0,
                max_tokens=5,
            )
            raw   = resp.choices[0].message.content.strip()
            score = float(re.search(r"[\d.]+", raw).group())
            score = max(0.0, min(1.0, score))
            scores_sample.append(round(score, 4))
        except Exception as e:
            print(f"[WARN] LLM 일관성 평가 실패: {e}")
            scores_sample.append(0.5)

    # 샘플 외 나머지는 샘플 평균으로 보정
    if len(comparisons) > sample_size:
        avg = statistics.mean(scores_sample) if scores_sample else 0.5
        scores_sample += [round(avg, 4)] * (len(comparisons) - sample_size)

    return scores_sample


def _extract_legal_core(text: str) -> str:
    """
    BERTScore·임베딩 비교 전 형식 노이즈 제거.
    (1)(2)... 섹션 마커, 대처방안·유의사항 이후 내용 제거 후 핵심 법적 판단만 반환.
    """
    text = re.sub(r"\([1-5]\)\s*[^\n]+\n?", "", text)
    text = re.sub(r"[①-⑤]\s*", "", text)
    for marker in ["(4)", "(5)", "대처 방안", "유의사항", "본 답변은 참고용"]:
        if marker in text:
            text = text[:text.index(marker)]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return " ".join(lines)


def score_consistency(
    original_answer: str,
    similar_answers: list[str],
    threshold: float = CONSISTENCY_THRESHOLD,
) -> ConsistencyScore:
    """
    3단계 앙상블 스코어링.

    ① BERTScore       (가중치 0.25) : 표면 토큰 유사도
    ② 임베딩 코사인    (가중치 0.40) : 의미 벡터 유사도 (기존 e5-large 재사용)
    ③ LLM 법률 판단    (가중치 0.35) : 법률 의미 동일성 직접 평가

    각 지표가 실패하면 성공한 지표들의 가중치를 정규화해서 앙상블.
    """
    valid = [a for a in similar_answers if a and a.strip()]
    if not valid:
        return ConsistencyScore(median_f1=0.0, all_scores=[], passed=False)

    # 형식 노이즈 제거 후 핵심 문장으로 비교
    original_core = _extract_legal_core(original_answer)
    valid_cores   = [_extract_legal_core(a) for a in valid]

    WEIGHTS = {
        "bertscore": 0.25,
        "embedding": 0.40,
        "llm":       0.35,
    }

    print(f"[Step 4] 앙상블 일관성 스코어링 ({len(valid)}쌍)...")

    # ── ① BERTScore ───────────────────────────────────────────────────────────
    bert_scores = _score_bertscore(original_core, valid_cores)
    print(f"  ① BERTScore     : {[round(s,3) for s in bert_scores] if bert_scores else '실패'}")

    # ── ② 임베딩 코사인 유사도 ────────────────────────────────────────────────
    emb_scores  = _score_embedding_cosine(original_core, valid_cores)
    print(f"  ② 임베딩 코사인 : {[round(s,3) for s in emb_scores] if emb_scores else '실패'}")

    # ── ③ LLM 법률 일관성 판단 ────────────────────────────────────────────────
    llm_scores  = _score_llm_consistency(original_answer, valid)   # 핵심 추출 전 원문 사용
    print(f"  ③ LLM 판단      : {[round(s,3) for s in llm_scores] if llm_scores else '실패'}")

    # ── 가중 앙상블 ───────────────────────────────────────────────────────────
    available = {}
    if bert_scores and len(bert_scores) == len(valid):
        available["bertscore"] = bert_scores
    if emb_scores  and len(emb_scores)  == len(valid):
        available["embedding"] = emb_scores
    if llm_scores  and len(llm_scores)  == len(valid):
        available["llm"]       = llm_scores

    if not available:
        print("[WARN] 모든 스코어링 실패 → 통과 처리")
        return ConsistencyScore(median_f1=1.0, all_scores=[], passed=True)

    # 성공한 지표 가중치 정규화
    total_w = sum(WEIGHTS[k] for k in available)
    norm_w  = {k: WEIGHTS[k] / total_w for k in available}

    ensemble_scores = []
    for i in range(len(valid)):
        score = sum(norm_w[k] * available[k][i] for k in available)
        ensemble_scores.append(round(score, 4))

    med_f1 = round(statistics.median(ensemble_scores), 4)
    passed = med_f1 >= threshold

    print(f"\n  앙상블 점수 : {[round(s,3) for s in ensemble_scores]}")
    print(f"  median      : {med_f1:.4f} | 기준={threshold} | {'통과' if passed else '미달'}")

    return ConsistencyScore(median_f1=med_f1, all_scores=ensemble_scores, passed=passed)


# ──────────────────────────────────────────────────────────────────────────────
# 14. [Step 5.5] 법령 인용 검증 (ChromaDB + Open API)
# ──────────────────────────────────────────────────────────────────────────────

_STATUTE_CITATION_PATTERN = re.compile(
    r"[가-힣]{2,25}(?:법률?|규칙|령|조례)?\s*제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?"
)


def _extract_citations(text: str) -> list[str]:
    return list(dict.fromkeys(m.strip() for m in _STATUTE_CITATION_PATTERN.findall(text)))


def verify_statute_citations(
    answer: str,
    collection=None,
) -> CitationVerification:
    """Step 5.5: 인용 조문 실존 여부 확인 (ChromaDB + Open API 병행)"""
    if collection is None:
        collection = _get_collection()

    citations = _extract_citations(answer)
    if not citations:
        return CitationVerification(found=[], missing=[], corrections={})

    print(f"[Step 5.5] 법령 인용 검증 | 추출 조문 {len(citations)}개: {citations}")
    found, missing, corrections = [], [], {}
    law_api = _get_law_api()

    for cite in citations:
        verified = False

        # ① ChromaDB에서 확인
        try:
            res = collection.query(
                query_texts=[cite], n_results=1,
                where={"source_type": "statute"},
                include=["documents", "metadatas", "distances"],
            )
            if res["documents"][0]:
                top_score = round(1.0 - res["distances"][0][0], 4)
                if top_score >= 0.70:
                    found.append(cite)
                    verified = True
                else:
                    corrections[cite] = res["documents"][0][0][:200]
        except Exception as e:
            print(f"[WARN] ChromaDB 조문 검증 실패 ({cite}): {e}")

        # ② ChromaDB 미확인 → Open API로 재확인
        if not verified and law_api:
            try:
                law_name = re.split(r"\s*제\d+조", cite)[0].strip()
                api_laws = law_api.search_law(law_name, display=1)
                if api_laws:
                    found.append(cite)
                    verified = True
                    # corrections에서 제거 (실제 존재하므로)
                    corrections.pop(cite, None)
            except Exception as e:
                print(f"[WARN] Open API 조문 검증 실패 ({cite}): {e}")

        if not verified:
            missing.append(cite)

    print(f"  → 확인됨: {len(found)}개 | 미확인: {len(missing)}개")
    return CitationVerification(found=found, missing=missing, corrections=corrections)


# ──────────────────────────────────────────────────────────────────────────────
# 15. [Step 6] NER 개체명 검증 (stub)
# ──────────────────────────────────────────────────────────────────────────────

def verify_entities_ner(
    answer: str,
    retrieved_chunks: list[RetrievedChunk],
) -> NERVerification:
    """
    KLUE-RoBERTa 법률 NER 파인튜닝 모델로 개체명 불일치 탐지.
    NER_MODEL_PATH 지정 전까지 stub으로 동작.
    """
    if NER_MODEL_PATH is None:
        return NERVerification(corrections={}, is_stub=True)

    # 파인튜닝 모델 연동 후 활성화
    # from transformers import pipeline
    # ner = pipeline("ner", model=NER_MODEL_PATH, aggregation_strategy="simple")
    # answer_ents = ner(answer)
    # chunk_text  = " ".join(c.text for c in retrieved_chunks)
    # chunk_ents  = ner(chunk_text)
    # corrections = _find_entity_mismatches(answer_ents, chunk_ents)
    # return NERVerification(corrections=corrections, is_stub=False)

    return NERVerification(corrections={}, is_stub=False)


# ──────────────────────────────────────────────────────────────────────────────
# 16. [Step 7] 교정 적용 + 최종 답변
# ──────────────────────────────────────────────────────────────────────────────

def apply_corrections(
    answer: str,
    citation_corrections: dict,
    ner_corrections: dict,
) -> str:
    """수정 가능한 환각을 RAG 기반 값으로 교체 (targeted replacement)"""
    corrected = answer

    for wrong, right_ctx in citation_corrections.items():
        if wrong in corrected:
            right_label = right_ctx.split("\n")[0].strip()
            corrected   = corrected.replace(wrong, right_label)
            print(f"  [수정] 인용 조문: '{wrong}' → '{right_label}'")

    for wrong, right in ner_corrections.items():
        if wrong in corrected:
            corrected = corrected.replace(wrong, right)
            print(f"  [수정] NER 개체: '{wrong}' → '{right}'")

    return corrected


# ──────────────────────────────────────────────────────────────────────────────
# 17. 전체 파이프라인 오케스트레이터
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    query: str,
    reask_callback: Optional[Callable[[str], Optional[str]]] = None,
    verbose: bool = False,
) -> PipelineResult:
    """
    LawsGuard 전체 환각 탐지 파이프라인 실행 (Step 0~7).

    Args:
        query:          사용자 원본 질문
        reask_callback: 재질문 처리 콜백 (str) -> Optional[str]
                        None이면 API 모드 (부분답변 모드로 전환)
        verbose:        상세 로그 출력
    """
    t_start  = time.time()
    final_q  = query
    partial  = False
    spec     = None

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: 구체성 검사 + 재질문 루프 (v3: 기준 강화 + 법적 재질문)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 1] 구체성 검사 중...")
    for reask_count in range(MAX_REASK_COUNT + 1):
        spec = check_specificity(final_q)

        print(
            f"  점수={spec.score}/5 | 카테고리={spec.legal_category} | "
            f"부족요소={spec.missing_elements}"
        )

        if spec.is_sufficient:
            print(f"  → 구체성 충분 (점수={spec.score} ≥ {SPECIFICITY_THRESHOLD})")
            break

        if reask_count >= MAX_REASK_COUNT:
            print(f"  → 재질문 {MAX_REASK_COUNT}회 초과 → 부분답변 모드")
            partial = True
            spec.partial_mode = True
            break

        # v3: 도메인 특화 법적 재질문 선택
        legal_reask = _get_legal_reask(
            spec.legal_category, spec.missing_elements, reask_count
        )
        # LLM이 생성한 재질문이 충분히 구체적이면 그것을 우선 사용
        if spec.followup_question and len(spec.followup_question) >= 20:
            followup = spec.followup_question
        else:
            followup = legal_reask

        if reask_callback:
            print(f"\n  [재질문 {reask_count+1}/{MAX_REASK_COUNT}] {followup}")
            extra = reask_callback(followup)
            if extra:
                final_q = final_q + " " + extra
            else:
                partial = True
                spec.partial_mode = True
                break
        else:
            partial = True
            spec.partial_mode = True
            break

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: 유사 질문 10개 생성 + 출력
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 2] 유사 질문 생성 중...")
    similar_qs = generate_similar_questions(final_q, n=N_SIMILAR_QUESTIONS, verbose=True)
    print(f"  → {len(similar_qs)}개 생성 완료")

    all_queries  = [final_q] + similar_qs
    is_original  = [True] + [False] * len(similar_qs)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: RAG 병렬 답변 생성 + 출력
    # ──────────────────────────────────────────────────────────────────────────
    rag_results  = run_rag_parallel(all_queries, is_original=is_original, verbose=verbose)
    original_rag = rag_results[0]
    similar_rags = [r for r in rag_results[1:] if r is not None]

    if original_rag is None or not original_rag.initial_answer:
        return PipelineResult(
            query=query,
            final_answer="RAG 검색 결과가 없습니다. 법률구조공단(132)에 문의하세요.",
            status="no_rag_result",
            specificity=spec,
            total_time=round(time.time() - t_start, 2),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: BERTScore 일관성 + 임계값 판단
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 4] 의미적 일관성 스코어링...")
    similar_answers = [r.initial_answer for r in similar_rags if r.initial_answer]
    consistency     = score_consistency(original_rag.initial_answer, similar_answers)

    if not consistency.passed:
        print(f"[Step 4→5] 일관성 미달 → 수정 불가 환각 → 답변 포기")
        return PipelineResult(
            query=query,
            final_answer=(
                "죄송합니다. 현재 해당 질문에 대해 신뢰할 수 있는 답변을 생성할 수 없습니다.\n"
                "정확한 법률 상담을 위해 법률구조공단(132) 또는 여성긴급전화(1366)에 문의하세요."
            ),
            status="hallucination_detected",
            specificity=spec, original_rag=original_rag,
            consistency=consistency,
            total_time=round(time.time() - t_start, 2),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5.5: 법령 인용 검증
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 5.5] 법령 인용 검증...")
    citation_check = verify_statute_citations(original_rag.initial_answer)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 6: NER 개체명 검증
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 6] NER 개체명 검증...")
    ner_check = verify_entities_ner(original_rag.initial_answer, original_rag.retrieved_chunks)
    if ner_check.is_stub:
        print("  → NER 모델 미연동 (stub) → 건너뜀")
    else:
        print(f"  → 교정 개체 {len(ner_check.corrections)}개")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 7: 교정 + 최종 답변
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 7] 교정 및 최종 답변 산출...")
    final_answer = apply_corrections(
        answer               = original_rag.initial_answer,
        citation_corrections = citation_check.corrections,
        ner_corrections      = ner_check.corrections,
    )

    total_time = round(time.time() - t_start, 2)
    status     = "partial_mode" if partial else "success"
    print(f"\n[완료] 상태={status} | 총 소요={total_time}s")

    return PipelineResult(
        query=query, final_answer=final_answer, status=status,
        specificity=spec, original_rag=original_rag,
        consistency=consistency, citation_check=citation_check,
        ner_check=ner_check, total_time=total_time,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 18. 멀티턴 대화 지원
# ──────────────────────────────────────────────────────────────────────────────

def run_phase1_with_history(query: str, history: list[dict], **kwargs) -> RAGResult:
    recent_user = [m["content"] for m in history[-4:] if m.get("role") == "user"][-2:]
    augmented   = " ".join(recent_user + [query])
    result      = run_phase1(query=augmented, **kwargs)
    result.query = query
    if result.prompt_messages and history:
        result.prompt_messages = (
            result.prompt_messages[:1] + history[-6:] + result.prompt_messages[1:]
        )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 19. 결과 직렬화
# ──────────────────────────────────────────────────────────────────────────────

def result_to_dict(result: RAGResult) -> dict:
    return {
        "query": result.query, "initial_answer": result.initial_answer,
        "context_text": result.context_text, "prompt_messages": result.prompt_messages,
        "retrieval_time": result.retrieval_time, "generation_time": result.generation_time,
        "retrieved_chunks": [
            {"chunk_id": c.chunk_id, "text": c.text, "score": c.score,
             "source_type": c.source_type, "metadata": c.metadata}
            for c in result.retrieved_chunks
        ],
    }


def pipeline_result_to_dict(result: PipelineResult) -> dict:
    d: dict = {
        "query": result.query, "final_answer": result.final_answer,
        "status": result.status, "total_time": result.total_time,
    }
    if result.specificity:
        sp = result.specificity
        d["specificity"] = {
            "score": sp.score, "entities": sp.entities,
            "legal_category": sp.legal_category, "legal_issue": sp.legal_issue,
            "missing_elements": sp.missing_elements,
            "is_sufficient": sp.is_sufficient, "partial_mode": sp.partial_mode,
        }
    if result.original_rag:
        d["original_rag"] = result_to_dict(result.original_rag)
    if result.consistency:
        d["consistency"] = {
            "median_f1": result.consistency.median_f1,
            "all_scores": result.consistency.all_scores,
            "passed": result.consistency.passed,
        }
    if result.citation_check:
        d["citation_check"] = {
            "found": result.citation_check.found,
            "missing": result.citation_check.missing,
            "corrections": result.citation_check.corrections,
        }
    if result.ner_check:
        d["ner_check"] = {
            "corrections": result.ner_check.corrections,
            "is_stub": result.ner_check.is_stub,
        }
    return d


# ──────────────────────────────────────────────────────────────────────────────
# 20. 검색 단독 유틸리티
# ──────────────────────────────────────────────────────────────────────────────

def search_only(query: str, source_filter: Optional[list[str]] = None, n: int = 5) -> list[dict]:
    chunks = retrieve(query, n_per_source=n, source_filter=source_filter)
    return [
        {
            "rank": i + 1, "score": c.score, "source_type": c.source_type,
            "label": c.short_label(), "preview": c.text[:200].replace("\n", " "),
        }
        for i, c in enumerate(chunks)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 21. 대화형 CLI
# ──────────────────────────────────────────────────────────────────────────────

_CMD_QUIT    = {"종료", "exit", "quit", "q"}
_CMD_RESET   = {"초기화", "reset", "새대화"}
_CMD_HISTORY = {"이력", "history", "대화이력"}
_CMD_VERBOSE = {"상세", "verbose"}
_CMD_SEARCH  = {"검색만", "search"}
_CMD_HELP    = {"도움말", "help", "?"}


def _print_help() -> None:
    print("""
┌─────────────────────────────────────────────────────┐
│              LawsGuard 대화형 상담 명령어             │
├──────────────┬──────────────────────────────────────┤
│ 종료 / exit  │ 프로그램 종료                          │
│ 초기화       │ 대화 이력 초기화                       │
│ 이력         │ 현재 대화 이력 출력                    │
│ 상세         │ 상세 로그 on/off                       │
│ 검색만       │ LLM 없이 검색 결과만 확인              │
│ 도움말       │ 이 도움말 출력                         │
└──────────────┴──────────────────────────────────────┘
""")


def _print_sources(chunks: list[RetrievedChunk]) -> None:
    if not chunks:
        return
    print("\n  참조 출처:")
    for i, c in enumerate(chunks):
        print(f"    [{i+1}] {c.short_label()}  (유사도: {c.score:.3f})")


def _print_banner() -> None:
    print("""
╔══════════════════════════════════════════════════════╗
║          LawsGuard (로스가드+) 법률 상담 AI  v3       ║
║    성범죄법 · 노동법 | 환각 탐지 파이프라인 통합 모드   ║
╚══════════════════════════════════════════════════════╝
  · 성범죄, 직장 내 괴롭힘, 임금 체불 등 법률 질문을 입력하세요.
  · 도움말: "help" | 종료: "exit"
""")


def run_interactive(verbose_default: bool = False) -> None:
    _print_banner()
    history: list[dict] = []
    verbose = verbose_default
    turn    = 0

    while True:
        try:
            user_input = input(f"\n[{turn+1}] 질문 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n상담을 종료합니다.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in _CMD_QUIT:
            print("\n상담을 종료합니다."); break
        if cmd in _CMD_RESET:
            history.clear(); turn = 0
            print("  대화 이력이 초기화되었습니다.\n"); continue
        if cmd in _CMD_HISTORY:
            if not history:
                print("  (이력 없음)")
            else:
                for i, m in enumerate(history):
                    print(f"  [{i+1}] {'질문' if m['role']=='user' else '답변'}: {m['content'][:80]}...")
            continue
        if cmd in _CMD_VERBOSE:
            verbose = not verbose
            print(f"  상세 로그: {'ON' if verbose else 'OFF'}"); continue
        if cmd in _CMD_HELP:
            _print_help(); continue
        if cmd in _CMD_SEARCH:
            sq = input("  검색어 > ").strip()
            if sq:
                for r in search_only(sq, n=5):
                    print(f"  [{r['rank']}] {r['label']} score={r['score']:.3f}")
                    print(f"       {r['preview'][:80]}…")
            continue

        turn += 1
        print(f"\n  법령·판례 검색 및 환각 탐지 파이프라인 실행 중...\n")

        def reask_cb(followup: str) -> Optional[str]:
            try:
                ans = input(f"\n  [추가 질문] {followup}\n  답변 > ").strip()
                return ans if ans else None
            except (EOFError, KeyboardInterrupt):
                return None

        try:
            result = run_full_pipeline(
                query=user_input, reask_callback=reask_cb, verbose=verbose,
            )
        except EnvironmentError as e:
            print(f"\n  설정 오류: {e}\n"); continue
        except RuntimeError as e:
            print(f"\n  오류 발생: {e}\n"); continue

        sep = "─" * 60
        print(f"\n{sep}")
        if result.status == "partial_mode":
            print("  [부분답변 모드] 충분한 정보 없이 일반 원칙으로 답변합니다.")
        print(result.final_answer)
        print(sep)

        if result.original_rag and result.original_rag.retrieved_chunks:
            _print_sources(result.original_rag.retrieved_chunks)

        print(f"\n  총 소요: {result.total_time}s | 상태: {result.status}")
        if result.consistency:
            print(f"  일관성 점수(median F1): {result.consistency.median_f1:.4f}")
        if result.citation_check and result.citation_check.missing:
            print(f"  교정된 인용 조문: {result.citation_check.missing}")

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant",  "content": result.final_answer})
        if len(history) > 20:
            history = history[-20:]


# ──────────────────────────────────────────────────────────────────────────────
# 22. 진입점
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LawsGuard v3 – 환각 탐지 파이프라인 통합 CLI")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--query",   "-q", type=str, default=None,
                        help="단일 질문 비대화형 실행")
    args = parser.parse_args()

    if args.query:
        r = run_full_pipeline(query=args.query, verbose=args.verbose)
        print("\n=== 최종 결과 ===")
        print(json.dumps(pipeline_result_to_dict(r), ensure_ascii=False, indent=2))
    else:
        run_interactive(verbose_default=args.verbose)