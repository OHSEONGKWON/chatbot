"""
LLM 호출 클라이언트 추상화 레이어.
OpenAI 호환 API (GPT-4o, vLLM + EXAONE 등) 를 동일 인터페이스로 호출.
"""

import asyncio
import os
from typing import Optional

from openai import AsyncOpenAI
from dotenv import load_dotenv

from config import config


# Ensure local .env values (e.g., OPENAI_API_KEY) are visible to this process.
load_dotenv()


class LLMClient:
	def __init__(self):
		cfg = config.llm
		api_key = cfg.api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
		base_url = cfg.api_base
		self.model = cfg.model_name
		self.temperature = cfg.temperature
		self.max_tokens = cfg.max_tokens
		self.timeout = cfg.request_timeout
		self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

	async def complete(self, system_prompt: str, user_prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None, json_mode: bool = False) -> str:
		temp_value = self.temperature if temperature is None else temperature
		token_limit = self.max_tokens if max_tokens is None else max_tokens
		kwargs = {}
		if json_mode:
			kwargs["response_format"] = {"type": "json_object"}

		resp = await self._client.chat.completions.create(
			model=self.model,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			temperature=temp_value,
			max_tokens=token_limit,
			timeout=self.timeout,
			**kwargs,
		)
		message = resp.choices[0].message
		content = message.content

		# Some providers can return None or structured content parts.
		if content is None:
			return ""

		if isinstance(content, str):
			return content.strip()

		parts: list[str] = []
		for item in content:
			if isinstance(item, dict):
				text = item.get("text")
				if isinstance(text, str):
					parts.append(text)

		return "\n".join(parts).strip()

	async def batch_complete(self, prompts: list[tuple[str, str]], temperature: Optional[float] = None) -> list[str]:
		tasks = [self.complete(sys_p, usr_p, temperature) for sys_p, usr_p in prompts]
		return await asyncio.gather(*tasks)


llm_client = LLMClient()
