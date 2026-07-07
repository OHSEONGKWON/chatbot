"""공용 비동기 LLM 호출 유틸 (재시도·동시성 제한·사용량 집계)."""
import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError

load_dotenv()

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 누적 사용량 (비용 리포트용)
usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "failures": 0}

# gpt-4o-mini 단가 ($/1M tokens)
PRICE_IN, PRICE_OUT = 0.15, 0.60


async def chat_json(system: str, user: str, *, model: str = "gpt-4o-mini",
                    temperature: float = 0.7, semaphore: asyncio.Semaphore,
                    max_retries: int = 5) -> dict | None:
    """JSON 응답을 요구하는 단일 챗 호출. 실패 시 지수 백오프 재시도, 최종 실패는 None."""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                usage["prompt_tokens"] += resp.usage.prompt_tokens
                usage["completion_tokens"] += resp.usage.completion_tokens
                usage["calls"] += 1
                return json.loads(resp.choices[0].message.content)
            except (RateLimitError, APIConnectionError, APIError, json.JSONDecodeError) as e:
                if attempt == max_retries - 1:
                    usage["failures"] += 1
                    print(f"  [실패] {type(e).__name__}: {str(e)[:120]}")
                    return None
                await asyncio.sleep(2 ** attempt)


async def chat_text(system: str, user: str, *, model: str = "gpt-4o-mini",
                    temperature: float = 0.7, semaphore: asyncio.Semaphore,
                    max_retries: int = 5) -> str | None:
    """자유 텍스트 응답 챗 호출. 재시도 정책은 chat_json과 동일."""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                usage["prompt_tokens"] += resp.usage.prompt_tokens
                usage["completion_tokens"] += resp.usage.completion_tokens
                usage["calls"] += 1
                return resp.choices[0].message.content
            except (RateLimitError, APIConnectionError, APIError) as e:
                if attempt == max_retries - 1:
                    usage["failures"] += 1
                    print(f"  [실패] {type(e).__name__}: {str(e)[:120]}")
                    return None
                await asyncio.sleep(2 ** attempt)


def cost_report() -> str:
    cost = usage["prompt_tokens"] / 1e6 * PRICE_IN + usage["completion_tokens"] / 1e6 * PRICE_OUT
    return (f"호출 {usage['calls']}건(실패 {usage['failures']}), "
            f"토큰 in {usage['prompt_tokens']:,} / out {usage['completion_tokens']:,}, "
            f"비용 약 ${cost:.3f}")
