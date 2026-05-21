from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ..config import config

logger = logging.getLogger("lawsguard.llm")


class LLMClient:
    def __init__(self):
        self._client = None

    @property
    def available(self) -> bool:
        return bool(config.llm.api_key)

    @property
    def clarify_model(self) -> str:
        return getattr(config.llm, "clarify_model_name", None) or config.llm.model_name

    @property
    def answer_model(self) -> str:
        return getattr(config.llm, "answer_model_name", None) or config.llm.model_name

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.available:
            return None
        try:
            from openai import AsyncOpenAI

            kwargs = {"api_key": config.llm.api_key, "timeout": config.llm.request_timeout}
            if config.llm.api_base:
                kwargs["base_url"] = config.llm.api_base
            self._client = AsyncOpenAI(**kwargs)
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            self._client = None
        return self._client

    async def complete(
        self,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        *,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        model: Optional[str] = None,
    ) -> str:
        client = self._get_client()
        if client is None:
            logger.error("LLM 클라이언트 없음 — API 키 또는 초기화 확인 필요")
            return ""

        if prompt is None:
            prompt = user_prompt or ""
        if system is None:
            system = system_prompt

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            request_kwargs = {
                "model": model or config.llm.model_name,
                "messages": messages,
                "temperature": config.llm.temperature if temperature is None else temperature,
                "max_tokens": max_tokens or config.llm.max_tokens,
            }
            if json_mode:
                request_kwargs["response_format"] = {"type": "json_object"}
            response = await client.chat.completions.create(**request_kwargs)
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"LLM 호출 실패: {e}")
            if json_mode:
                try:
                    request_kwargs = {
                        "model": model or config.llm.model_name,
                        "messages": messages,
                        "temperature": config.llm.temperature if temperature is None else temperature,
                        "max_tokens": max_tokens or config.llm.max_tokens,
                    }
                    response = await client.chat.completions.create(**request_kwargs)
                    return (response.choices[0].message.content or "").strip()
                except Exception as e2:
                    logger.error(f"LLM 재시도 실패: {e2}")
                    return ""
            return ""

    async def complete_many(self, prompts: list[str], system: Optional[str] = None) -> list[str]:
        return await asyncio.gather(*(self.complete(prompt, system=system) for prompt in prompts))


llm_client = LLMClient()
