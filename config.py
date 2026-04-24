"""
LawsGuard 설정 파일
모든 상수, 임계값, 모델 경로 등을 중앙 관리
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RAGConfig:
	chroma_path: str = r"D:\Law_Chatbot\chroma_db"
	collection_name: str = "legal_documents"
	embedding_model: str = "intfloat/multilingual-e5-large"
	top_k: int = 5
	embedding_batch_size: int = 32


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
	provider: str = "openai"
	model_name: str = "gpt-4o"
	api_base: Optional[str] = None
	api_key: Optional[str] = None
	temperature: float = 0.2
	max_tokens: int = 2048
	request_timeout: int = 60


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
	clarification: ClarificationConfig = field(default_factory=ClarificationConfig)
	hallucination: HallucinationConfig = field(default_factory=HallucinationConfig)
	llm: LLMConfig = field(default_factory=LLMConfig)
	kakao: KakaoConfig = field(default_factory=KakaoConfig)
	session_ttl_sec: int = 1800


config = AppConfig()
