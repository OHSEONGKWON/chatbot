"""LawsGuard FastAPI 메인 애플리케이션"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import config
from .modules.kakao_response import (
    build_callback_response,
    build_simple_text,
    build_text_and_image_response,
    default_quick_replies,
)
from .modules.webtoon import OUTPUT_DIR, generate_webtoon
from .pipeline import pipeline
from .session_store import session_store


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("lawsguard")
CALLBACK_TOKEN_SAFETY_SEC = float(os.getenv("LAWSGUARD_CALLBACK_TOKEN_SAFETY_SEC", "50"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    warmup_task = asyncio.create_task(_warmup_components())
    logger.info("LawsGuard 서버 시작")
    await warmup_task
    yield
    cleanup_task.cancel()
    logger.info("LawsGuard 서버 종료")


async def _session_cleanup_loop():
    while True:
        await asyncio.sleep(1800)
        session_store.cleanup_expired()
        logger.info("만료 세션 정리 완료")


async def _warmup_components():
    try:
        await asyncio.gather(
            pipeline._retriever.warmup(),
            pipeline._consistency.warmup(),
            pipeline._ner.warmup(),
        )
        logger.info("핵심 모델 워밍업 완료")
    except Exception as e:
        logger.warning(f"모델 워밍업 실패: {e}")


app = FastAPI(title="LawsGuard API", description="RAG 기반 한국 법률 상담 챗봇 스킬 서버", version="1.0.0", lifespan=lifespan)
app.mount("/images", StaticFiles(directory=str(OUTPUT_DIR)), name="images")


async def send_callback(
    callback_url: str,
    response_text: str,
    needs_requery: bool = False,
    category: str = "",
    image_url: str | None = None,
) -> bool:
    quick_replies = default_quick_replies(needs_requery, category)
    if image_url:
        payload = build_text_and_image_response(response_text, image_url, quick_replies=quick_replies)
    else:
        payload = build_simple_text(response_text, quick_replies=quick_replies)
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(callback_url, json=payload)
            if resp.status_code >= 400:
                logger.error(f"콜백 전송 실패: {resp.status_code} | {resp.text[:300]}")
                return False
            else:
                logger.info(f"콜백 전송 완료: {resp.status_code}")
                return True
        except Exception as e:
            logger.error(f"콜백 전송 예외: {e}")
            return False


async def run_pipeline_and_callback(user_id: str, user_input: str, callback_url: str):
    start = time.monotonic()
    webtoon_task = asyncio.create_task(generate_webtoon(user_input))
    try:
        result = await pipeline.process(user_id=user_id, user_input=user_input)
        elapsed = time.monotonic() - start

        image_url = None
        if not result.needs_requery:
            remain_for_webtoon = max(0.0, CALLBACK_TOKEN_SAFETY_SEC - elapsed - 2.0)
            if remain_for_webtoon > 0:
                try:
                    filename = await asyncio.wait_for(webtoon_task, timeout=remain_for_webtoon)
                    if filename:
                        image_url = f"{config.kakao.server_url}/images/{filename}"
                except asyncio.TimeoutError:
                    logger.warning("콜백 안전시간 내 웹툰 생성 미완료 — 이번 턴은 텍스트만 전송")
            else:
                logger.warning("콜백 안전시간 부족 — 이번 턴은 텍스트만 전송")

        logger.info(
            f"파이프라인 완료 | user={user_id[:8]}... | step={result.step_reached} | "
            f"score={(f'{result.consistency_score:.3f}' if result.consistency_score is not None else 'N/A')} | "
            f"legal={(f'{result.legal_reasoning_score:.3f}' if result.legal_reasoning_score is not None else 'N/A')} | "
            f"time={elapsed:.2f}s"
        )
        callback_ok = await send_callback(
            callback_url,
            result.response_text,
            result.needs_requery,
            result.legal_category,
            image_url=image_url,
        )

        if not callback_ok:
            logger.warning("콜백 실패 — callback token 만료 가능성이 높습니다.")

    except Exception as e:
        logger.exception(f"파이프라인 오류: {e}")
        await send_callback(callback_url, "죄송합니다. 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
    finally:
        if not webtoon_task.done():
            webtoon_task.cancel()


@app.post("/webhook/kakao")
async def kakao_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content=build_simple_text("요청 형식이 올바르지 않습니다."), status_code=400)

    user_request = body.get("userRequest", {})
    user_id = user_request.get("user", {}).get("id", "anonymous")
    user_input = user_request.get("utterance", "").strip()
    callback_url = user_request.get("callbackUrl", "")

    if not user_input:
        return JSONResponse(content=build_simple_text("질문을 입력해 주세요.", quick_replies=default_quick_replies(True)))

    logger.info(f"수신 | user={user_id[:8]}... | input={user_input[:30]}...")

    if config.kakao.use_callback and callback_url:
        background_tasks.add_task(run_pipeline_and_callback, user_id=user_id, user_input=user_input, callback_url=callback_url)
        return JSONResponse(content=build_callback_response(config.kakao.callback_message))

    try:
        result = await asyncio.wait_for(pipeline.process(user_id=user_id, user_input=user_input), timeout=config.kakao.response_timeout_sec)
        logger.info("동기 응답 완료")
        return JSONResponse(content=build_simple_text(result.response_text, quick_replies=default_quick_replies(result.needs_requery, result.legal_category)))
    except asyncio.TimeoutError:
        logger.warning("응답 시간 초과")
        return JSONResponse(content=build_simple_text("처리 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."))
    except Exception as e:
        logger.exception(f"처리 오류: {e}")
        return JSONResponse(content=build_simple_text("오류가 발생했습니다. 잠시 후 다시 시도해 주세요."))


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "LawsGuard"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host=config.kakao.server_host, port=config.kakao.server_port, reload=False, workers=1, log_level="info")
