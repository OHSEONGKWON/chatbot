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
    try:
        while True:
            await asyncio.sleep(1800)
            session_store.cleanup_expired()
            logger.info("만료 세션 정리 완료")
    except asyncio.CancelledError:
        logger.info("세션 정리 태스크 종료")


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
    
    try:
        # 1. RAG 파이프라인부터 먼저 실행하여 '답변'을 완전히 얻어냅니다. (제한 시간 없음)
        result = await pipeline.process(user_id=user_id, user_input=user_input)
        
        image_url = None
        
        # 2. 재질의가 필요 없는 정상 답변인 경우, 얻어낸 '답변'을 기반으로 웹툰 생성을 시도합니다.
        if not result.needs_requery:
            logger.info("답변 기반 웹툰 생성을 시작합니다... (시간 제한 없음)")
            # 주의: user_input이 아니라 result.response_text(생성된 AI 답변)을 넘깁니다.
            filename = await generate_webtoon(result.response_text)
            
            if filename:
                image_url = f"{config.kakao.server_url}/images/{filename}"

        elapsed_total = time.monotonic() - start
        logger.info(
            f"파이프라인 완료 | user={user_id[:8]}... | step={result.step_reached} | "
            f"reliability={(f'{result.answer_reliability:.3f}' if result.answer_reliability is not None else 'N/A')} | "
            f"총 소요시간={elapsed_total:.2f}s"
        )
        
        # 3. 완성된 글과 그림을 카카오톡 콜백으로 전송합니다.
        callback_ok = await send_callback(
            callback_url,
            result.response_text,
            result.needs_requery,
            result.legal_category,
            image_url=image_url,
        )

        # 60초가 넘어가면 여기서 전송 실패 로그가 뜰 확률이 높습니다.
        if not callback_ok:
            logger.error(f"콜백 전송 실패! (소요시간: {elapsed_total:.1f}초) - 카카오의 60초 제한을 초과하여 토큰이 만료되었을 가능성이 높습니다.")

    except Exception as e:
        logger.exception(f"파이프라인 오류: {e}")
        await send_callback(callback_url, "죄송합니다. 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")


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