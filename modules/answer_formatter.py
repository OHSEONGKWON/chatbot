"""
최종 답변 포맷 생성 모듈 (Step 7)
"""

from modules.llm_client import llm_client


FINAL_ANSWER_SYSTEM = """당신은 대한민국 법률 전문가입니다.
카카오톡으로 전달되는 법률 상담 답변을 작성합니다.
Markdown 문법(#, **, ``` 등)을 사용하지 마세요.
이모지와 줄바꿈으로 가독성을 높이세요.
반드시 아래 4단계 구조로 작성하세요."""

FINAL_ANSWER_USER = """[참고 법률 자료]
{context}

[사용자 질문]
{question}

[검증된 초안 답변]
{draft_answer}

[법률 카테고리]
{legal_category}

[일반 답변 여부]
{is_general}

다음 4단계 구조로 최종 답변을 작성하세요:

⚖️ 질문 정리
(사용자 질문을 한두 문장으로 명확하게 재정리)

📋 관련 법률 및 판례
(구체적인 법률 조항 번호와 판결문 번호를 인용하세요.
예: 형법 제297조, 2022도1234 판결 등
RAG 자료에 있는 내용만 인용하고, 없으면 일반 법리를 설명하세요.)

🔍 사건에 적용
(인용한 법률과 판례를 이 사건에 구체적으로 적용하여 
법률적 판단을 내려주세요. "~에 해당합니다", "~이 성립합니다" 등)

💡 대처 방안
(구체적인 행동 지침을 3~5가지로 제시하세요.
관련 기관명, 신고처, 지원센터를 명시하세요.)

---
⚠️ 이 답변은 AI 법률 정보 제공이며, 법적 효력이 있는 정확한 자문은 
변호사 상담을 받으시기 바랍니다."""


class AnswerFormatter:
	async def format(self, question: str, draft_answer: str, rag_docs: list[dict], legal_category: str, is_general: bool = False) -> str:
		context = "\n\n".join(f"[출처: {d['source']}]\n{d['text']}" for d in rag_docs)
		final = await llm_client.complete(
			system_prompt=FINAL_ANSWER_SYSTEM,
			user_prompt=FINAL_ANSWER_USER.format(
				context=context,
				question=question,
				draft_answer=draft_answer,
				legal_category=legal_category,
				is_general="예 (충분한 정보 없이 일반 기준으로 답변)" if is_general else "아니오",
			),
			temperature=0.2,
			max_tokens=1500,
		)
		return final


answer_formatter = AnswerFormatter()
