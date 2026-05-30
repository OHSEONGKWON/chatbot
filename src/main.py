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

# =========================================================================
# 웹툰 전용 임시 저장소 (세션이 지워져도 웹툰을 그릴 수 있도록 텍스트 보관)
# =========================================================================
webtoon_cache: dict[str, str] = {}

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

app = FastAPI(
    title="LawsGuard API", 
    description="RAG 기반 한국 법률 상담 챗봇 스킬 서버", 
    version="1.0.0", 
    lifespan=lifespan
)
app.mount("/images", StaticFiles(directory=str(OUTPUT_DIR)), name="images")

async def send_callback(
    callback_url: str,
    response_text: str,
    needs_requery: bool = False,
    category: str = "",
    image_url: str | None = None,  # 다시 단일 문자열(str)로 되돌림
    add_webtoon_btn: bool = False,
) -> bool:
    quick_replies = default_quick_replies(needs_requery, category)
    
    if add_webtoon_btn:
        quick_replies.append({
            "action": "message",
            "label": "웹툰으로 요약 보기",  # PC 호환을 위해 이모지(🎨) 제거 유지
            "messageText": "웹툰으로 요약 보기"
        })

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

# 1. 텍스트 답변 파이프라인
async def run_pipeline_and_callback(user_id: str, user_input: str, callback_url: str):
    start = time.monotonic()
    
    try:
        result = await pipeline.process(user_id=user_id, user_input=user_input)
        
        elapsed_total = time.monotonic() - start
        logger.info(f"텍스트 파이프라인 완료 | user={user_id[:8]}... | 총 소요시간={elapsed_total:.2f}s")
        
        can_make_webtoon = not result.needs_requery
        
        if can_make_webtoon:
            # 질문과 답변을 합쳐 웹툰 생성용 프롬프트로 캐시에 저장
            webtoon_cache[user_id] = f"상황: {user_input}\n법률해석: {result.response_text}"
        
        await send_callback(
            callback_url=callback_url,
            response_text=result.response_text,
            needs_requery=result.needs_requery,
            category=result.legal_category,
            add_webtoon_btn=can_make_webtoon
        )

    except Exception as e:
        logger.exception(f"파이프라인 오류: {e}")
        await send_callback(callback_url, "죄송합니다. 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")

# 2. 웹툰 단독 생성 파이프라인 (케로셀 제거)
async def run_webtoon_only_and_callback(user_id: str, callback_url: str):
    start = time.monotonic()
    
    try:
        logger.info("사용자 요청으로 웹툰 단독 생성을 시작합니다...")
        
        story_context = webtoon_cache.get(user_id) 
        
        if not story_context:
            await send_callback(callback_url, "이전 상담 내용이 만료되었거나 찾을 수 없습니다. 법률 질문을 먼저 다시 입력해 주세요.")
            return

        # 원본 2x2 웹툰 생성
        filename = await generate_webtoon(story_context)
        
        if filename:
            image_url = f"{config.kakao.server_url}/images/{filename}"
            
            await send_callback(
                callback_url=callback_url,
                response_text="요청하신 4컷 웹툰 요약이 완성되었습니다! 🎨",
                image_url=image_url
            )
        else:
            await send_callback(callback_url, "웹툰 생성에 실패했습니다. 다시 시도해 주세요.")
            
        logger.info(f"웹툰 생성 및 전송 완료 | 소요시간={time.monotonic() - start:.2f}s")

    except Exception as e:
        logger.exception(f"웹툰 생성 중 오류: {e}")
        await send_callback(callback_url, "죄송합니다. 웹툰 생성 중 오류가 발생했습니다.")

@app.post("/webhook/kakao")
async def kakao_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
        
        # 🟢 [추가된 디버깅 코드] 카카오에서 넘어온 JSON 데이터를 통째로 출력합니다.
        logger.info(f"카카오 원본 데이터: {body}")
        
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
        if user_input == "웹툰으로 요약 보기":
            background_tasks.add_task(run_webtoon_only_and_callback, user_id=user_id, callback_url=callback_url)
            return JSONResponse(content=build_callback_response("웹툰을 그리는 중입니다. 약 1~2분 정도 소요됩니다. 🎨"))
        else:
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