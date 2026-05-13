"""LawsGuard FastAPI 메인 애플리케이션"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import config
from .modules.kakao_response import build_callback_response, build_image_response, build_simple_text, build_text_and_image_response, default_quick_replies
from .modules.webtoon import OUTPUT_DIR, generate_webtoon
from .pipeline import pipeline
from .session_store import session_store


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("lawsguard")

# 마지막 완료된 질문 캐시 (user_id → 질문 내용)
_last_question: dict[str, str] = {}
# 사전 생성 중인 웹툰 태스크 (user_id → asyncio.Task)
_webtoon_tasks: dict[str, asyncio.Task] = {}

WEBTOON_TRIGGER = "만화로 보여줘"
# 콜백 URL 만료를 고려한 최대 대기 시간(초) — 카카오 콜백 유효시간 60초보다 여유 있게
_WEBTOON_WAIT_TIMEOUT = 55.0


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


def _webtoon_quick_reply() -> dict:
    return {"label": "만화로 보기", "action": "message", "messageText": WEBTOON_TRIGGER}


async def send_callback(callback_url: str, response_text: str, needs_requery: bool = False, category: str = "", add_webtoon_button: bool = False):
    quick_replies = default_quick_replies(needs_requery, category)
    if add_webtoon_button:
        quick_replies = [_webtoon_quick_reply()] + quick_replies
    payload = build_simple_text(response_text, quick_replies=quick_replies)
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(callback_url, json=payload)
            if resp.status_code >= 400:
                logger.error(f"콜백 전송 실패 응답: {resp.status_code} | {resp.text[:300]}")
            else:
                logger.info(f"콜백 전송 완료: {resp.status_code} → {callback_url[:50]}")
        except Exception as e:
            logger.error(f"콜백 전송 실패: {e}")


async def run_pipeline_and_callback(user_id: str, user_input: str, callback_url: str):
    start = time.monotonic()
    try:
        result = await pipeline.process(user_id=user_id, user_input=user_input)
        elapsed = time.monotonic() - start
        logger.info(
            f"파이프라인 완료 | user={user_id[:8]}... | step={result.step_reached} | "
            f"score={(f'{result.consistency_score:.3f}' if result.consistency_score is not None else 'N/A')} | "
            f"legal={(f'{result.legal_reasoning_score:.3f}' if result.legal_reasoning_score is not None else 'N/A')} | "
            f"time={elapsed:.2f}s"
        )
        if not result.needs_requery:
            _last_question[user_id] = user_input
            # 답변 전송과 동시에 웹툰 사전 생성 시작 — 버튼 탭 전에 준비
            old_task = _webtoon_tasks.pop(user_id, None)
            if old_task and not old_task.done():
                old_task.cancel()
            _webtoon_tasks[user_id] = asyncio.create_task(generate_webtoon(user_input))
            logger.info("웹툰 사전 생성 태스크 시작")
        await send_callback(
            callback_url,
            result.response_text,
            result.needs_requery,
            result.legal_category,
            add_webtoon_button=not result.needs_requery,
        )
    except Exception as e:
        logger.exception(f"파이프라인 오류: {e}")
        await send_callback(callback_url, "죄송합니다. 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")


async def run_webtoon_and_callback(user_id: str, question: str, callback_url: str):
    logger.info("웹툰 전송 시작")
    task = _webtoon_tasks.pop(user_id, None)
    if task:
        logger.info("사전 생성 태스크 대기 중...")
        try:
            # shield: 타임아웃 시에도 생성 태스크 자체는 취소하지 않음
            filename = await asyncio.wait_for(asyncio.shield(task), timeout=_WEBTOON_WAIT_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("사전 생성 대기 시간 초과 — 태스크 보존 후 재시도 안내")
            _webtoon_tasks[user_id] = task  # 완료되면 다음 탭에서 즉시 사용
            await send_callback(callback_url, "만화를 아직 그리고 있어요. 30초 후에 다시 '만화로 보기' 버튼을 눌러주세요!")
            return
        except Exception as e:
            logger.error(f"사전 생성 태스크 오류: {e}")
            filename = None
    else:
        filename = await generate_webtoon(question)

    if filename:
        image_url = f"{config.kakao.server_url}/images/{filename}"
        payload = build_image_response(image_url)
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.post(callback_url, json=payload)
                if resp.status_code >= 400:
                    logger.error(f"웹툰 콜백 전송 실패: {resp.status_code} | {resp.text[:200]}")
                else:
                    logger.info(f"웹툰 콜백 전송 완료: {resp.status_code}")
            except Exception as e:
                logger.error(f"웹툰 콜백 전송 실패: {e}")
    else:
        logger.warning("웹툰 생성 실패")
        await send_callback(callback_url, "만화 생성에 실패했습니다. 다시 시도해 주세요.")


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

    # 만화 요청 처리
    if user_input == WEBTOON_TRIGGER:
        question = _last_question.get(user_id)
        if not question:
            return JSONResponse(content=build_simple_text("먼저 법률 상담을 해주세요. 상담 후 만화로 보기 버튼이 표시됩니다."))
        if callback_url:
            background_tasks.add_task(run_webtoon_and_callback, user_id, question, callback_url)
        return JSONResponse(content=build_callback_response("만화를 그리고 있어요! 잠시만 기다려주세요 (약 60초)"))

    if config.kakao.use_callback and callback_url:
        background_tasks.add_task(run_pipeline_and_callback, user_id=user_id, user_input=user_input, callback_url=callback_url)
        return JSONResponse(content=build_callback_response(config.kakao.callback_message))

    try:
        result = await asyncio.wait_for(pipeline.process(user_id=user_id, user_input=user_input), timeout=config.kakao.response_timeout_sec)
        logger.info("동기 응답 완료")
        return JSONResponse(
            content=build_simple_text(
                result.response_text,
                quick_replies=default_quick_replies(result.needs_requery, result.legal_category),
            )
        )
    except asyncio.TimeoutError:
        logger.warning("응답 시간 초과 - 콜백 모드로 전환 필요")
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
