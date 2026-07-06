"""ALCV 설정값 한 곳 모음 (결정적 검증 버전: LLM/NLI 미사용)."""
from pathlib import Path

# 경로
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = SCRIPTS_DIR / "results"
LAW_INDEX_PATH = RESULTS_DIR / "law_index.json"

# 데이터
EVAL_PATH = REPO_ROOT / "data" / "evaluation" / "hallu_eval.jsonl"
LAW_CHUNKS_PATH = REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_law_chunks.jsonl"

# 검색 (근거 = RAG)
RETRIEVE_TOP_K = 8

# 수치 정합성 허용 오차(상대): 이보다 크게 다르면 numeric_mismatch
NUMERIC_TOLERANCE = 0.05

# 질문 -> 사안(issue) 추론 키워드 (도메인 정합성 검사용; ISSUE_REGISTRY 키로 매핑)
ISSUE_KEYWORDS = {
    "wage_unpaid": ["임금", "월급", "급여", "체불", "임금체불", "밀린 돈", "돈을 받지 못"],
    "dismissal": ["해고", "부당해고", "잘렸", "해고당"],
    "missing_contract": ["근로계약서", "계약서"],
    "minimum_wage": ["최저임금", "최저시급", "시급"],
    "workplace_harassment": ["직장 내 괴롭힘", "직장내괴롭힘", "괴롭힘"],
    "sexual_harassment": ["성희롱"],
    "indecent_assault": ["강제추행", "성추행", "추행"],
    "강간": ["강간"],
    "illegal_filming": ["불법촬영", "몰래카메라", "몰카", "촬영물", "성관계 동영상"],
}

# 위험 가중치 (집계 점수용; 법률에서 치명적인 유형에 높게)
RISK_WEIGHT = {
    "LAW": 1.0, "PENALTY": 0.9, "AMOUNT": 0.7, "ORG": 0.6, "DATE": 0.5, "CRIME": 0.6,
}

# 집계: 안전성 제약(재현율 하한)
RECALL_FLOOR = 0.90
