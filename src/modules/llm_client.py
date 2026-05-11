from __future__ import annotations

import asyncio
from typing import Optional

from ..config import config


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
        except Exception:
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
        except Exception:
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
                except Exception:
                    return ""
            return ""

    async def complete_many(self, prompts: list[str], system: Optional[str] = None) -> list[str]:
        return await asyncio.gather(*(self.complete(prompt, system=system) for prompt in prompts))


llm_client = LLMClient()
