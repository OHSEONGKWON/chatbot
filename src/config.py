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
	model_id: str = field(default_factory=lambda: os.getenv("NER_MODEL_ID", "legal-ner-v3"))
	use_model: bool = field(default_factory=lambda: os.getenv("LAWSGUARD_USE_MODEL_NER", "1") == "1")
	min_confidence: float = field(default_factory=lambda: float(os.getenv("LAWSGUARD_NER_MIN_CONFIDENCE", "0.70")))
	max_length: int = field(default_factory=lambda: int(os.getenv("LAWSGUARD_NER_MAX_LENGTH", "510")))
	model_path: str = field(default="", init=False)
	
	def __post_init__(self):
		# NER_MODEL_ID가 절대 경로면 그대로 사용, 아니면 outputs 폴더 내에서 찾기
		if os.path.isabs(self.model_id):
			self.model_path = self.model_id
		elif self.model_id.startswith(("outputs/", "outputs\\")):
			self.model_path = str(BASE_DIR / self.model_id)
		else:
			# 기본값: ID를 outputs 폴더 내 경로로 해석
			self.model_path = str(BASE_DIR / "outputs" / self.model_id)


@dataclass
class ClarificationConfig:
	max_retries: int = 5
	min_score_threshold: float = float(os.getenv("CLARIFICATION_MIN_SCORE", os.getenv("LAWSGUARD_CLARIFICATION_MIN_SCORE", "3.5")))
	fallback_message: str = "대답에 필요한 정보가 충분하지 않아 일반적인 기준으로 대답하겠습니다."


@dataclass
class HallucinationConfig:
	num_similar_questions: int = 10
	consistency_threshold: float = float(os.getenv("CONSISTENCY_THRESHOLD", os.getenv("LAWSGUARD_CONSISTENCY_THRESHOLD", "0.75")))
	similarity_weight_nli: float = 0.6
	similarity_weight_embed: float = 0.4
	nli_model: str = "klue/roberta-large"
	contextual_model: str = "upskyy/kure-roberta-base"
	generation_model: str = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
	answer_give_up_message: str = "답변을 생성할 수 없습니다. 더 구체적인 질문을 입력해 주세요."


@dataclass
class LLMConfig:
	provider: str = os.getenv("LAWSGUARD_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "openai"))
	model_name: str = os.getenv("LAWSGUARD_OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
	clarify_model_name: str = os.getenv("CLARIFY_MODEL_ID", model_name)
	answer_model_name: str = os.getenv("ANSWER_MODEL_ID", model_name)
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
	server_url: str = field(default_factory=lambda: os.getenv("SERVER_URL", "http://localhost:8000"))
	rest_api_key: str = field(default_factory=lambda: os.getenv("KAKAO_REST_API_KEY", ""))
	bot_id: str = field(default_factory=lambda: os.getenv("KAKAO_BOT_ID", "69c3fb758094aa665fd2c5f3"))


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
