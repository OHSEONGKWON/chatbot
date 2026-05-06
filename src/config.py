"""LawsGuard 설정 파일."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


@dataclass
class RAGConfig:
	chroma_path: str = str(BASE_DIR / "data" / "RAG_data" / "chroma_db")
	collection_name: str = "legal_documents"
	embedding_model: str = "intfloat/multilingual-e5-large"
	enable_vector: bool = os.getenv("LAWSGUARD_ENABLE_VECTOR_RAG", "0") == "1"
	top_k: int = 5
	embedding_batch_size: int = 32
	jsonl_paths: tuple[str, ...] = (
		str(BASE_DIR / "data" / "real_data" / "New_Dataset" / "rag_law_chunks.jsonl"),
		str(BASE_DIR / "data" / "real_data" / "New_Dataset" / "rag_case_chunks.jsonl"),
		str(BASE_DIR / "data" / "real_data" / "New_Dataset" / "rag_manual_chunks.jsonl"),
	)


@dataclass
class NERConfig:
	model_path: str = os.getenv("LAWSGUARD_NER_MODEL", str(BASE_DIR / "outputs" / "legal-ner-lawsguard-v2-30k"))
	use_model: bool = os.getenv("LAWSGUARD_USE_MODEL_NER", "1") == "1"
	min_confidence: float = float(os.getenv("LAWSGUARD_NER_MIN_CONFIDENCE", "0.70"))
	max_length: int = int(os.getenv("LAWSGUARD_NER_MAX_LENGTH", "510"))


@dataclass
class ClarificationConfig:
	max_retries: int = 5
	min_score_threshold: float = 3.0
	fallback_message: str = "대답에 필요한 정보가 충분하지 않아 일반적인 기준으로 대답하겠습니다."


@dataclass
class HallucinationConfig:
	num_similar_questions: int = 10
	consistency_threshold: float = 0.60
	similarity_weight_nli: float = 0.6
	similarity_weight_embed: float = 0.4
	nli_model: str = "klue/roberta-large"
	contextual_model: str = "upskyy/kure-roberta-base"
	generation_model: str = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
	answer_give_up_message: str = "답변을 생성할 수 없습니다. 더 구체적인 질문을 입력해 주세요."


@dataclass
class LLMConfig:
	provider: str = os.getenv("LAWSGUARD_LLM_PROVIDER", "openai")
	model_name: str = os.getenv("LAWSGUARD_OPENAI_MODEL", "gpt-4o-mini")
	api_base: Optional[str] = os.getenv("OPENAI_BASE_URL") or None
	api_key: Optional[str] = os.getenv("OPENAI_API_KEY") or None
	temperature: float = float(os.getenv("LAWSGUARD_LLM_TEMPERATURE", "0.2"))
	max_tokens: int = int(os.getenv("LAWSGUARD_LLM_MAX_TOKENS", "2048"))
	request_timeout: int = int(os.getenv("LAWSGUARD_LLM_TIMEOUT", "60"))


@dataclass
class KakaoConfig:
	response_timeout_sec: float = 4.5
	use_callback: bool = True
	callback_message: str = "법률 자문을 준비 중이에요 ⚖️\n잠시만 기다려 주세요..."
	server_host: str = "0.0.0.0"
	server_port: int = 8000


@dataclass
class AppConfig:
	rag: RAGConfig = field(default_factory=RAGConfig)
	ner: NERConfig = field(default_factory=NERConfig)
	clarification: ClarificationConfig = field(default_factory=ClarificationConfig)
	hallucination: HallucinationConfig = field(default_factory=HallucinationConfig)
	llm: LLMConfig = field(default_factory=LLMConfig)
	kakao: KakaoConfig = field(default_factory=KakaoConfig)
	session_ttl_sec: int = 1800


config = AppConfig()
