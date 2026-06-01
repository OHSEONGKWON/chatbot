"""LawsGuard FastAPI 메인 애플리케이션 (디버깅 강화 버전)"""

import asyncio
import logging
import os
import time
import traceback
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

logger = logging.getLogger("lawsguard")
logger.setLevel(logging.INFO)

CALLBACK_TOKEN_SAFETY_SEC = float(os.getenv("LAWSGUARD_CALLBACK_TOKEN_SAFETY_SEC", "50"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    warmup_task = asyncio.create_task(_warmup_components())
    print("[LAWSGUARD_INIT] 서버 라이프사이클 시작", flush=True)
    await warmup_task
    yield
    cleanup_task.cancel()
    print("[LAWSGUARD_INIT] 서버 라이프사이클 종료", flush=True)

async def _session_cleanup_loop():
    try:
        while True:
            await asyncio.sleep(1800)
            session_store.cleanup_expired()
            print("[LAWSGUARD_LOOP] 만료 세션 정리 완료", flush=True)
    except asyncio.CancelledError:
        pass

async def _warmup_components():
    try:
        await asyncio.gather(
            pipeline._retriever.warmup(),
            pipeline._consistency.warmup(),
            pipeline._ner.warmup(),
        )
        print("[LAWSGUARD_INIT] 핵심 AI 모델 워밍업 완료", flush=True)
    except Exception as e:
        print(f"[LAWSGUARD_INIT] ❌ 모델 워밍업 실패: {e}", flush=True)

app = FastAPI(title="LawsGuard API", version="1.0.0", lifespan=lifespan)
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
        print(f"[CALLBACK_SEND] 이미지 포함 응답 구성 -> URL: {image_url}", flush=True)
        payload = build_text_and_image_response(response_text, image_url, quick_replies=quick_replies)
    else:
        print("[CALLBACK_SEND] ⚠️ 이미지 없음 - 일반 텍스트 단독 응답 구성", flush=True)
        payload = build_simple_text(response_text, quick_replies=quick_replies)
        
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(callback_url, json=payload)
            print(f"[CALLBACK_SEND] 카카오 전송 완료 | 상태코드: {resp.status_code}", flush=True)
            return resp.status_code < 400
        except Exception as e:
            print(f"[CALLBACK_SEND] ❌ 카카오 전송 중 예외 발생: {e}", flush=True)
            return False

# 1. 텍스트 답변 + 웹툰 동시 생성 백그라운드 파이프라인
async def run_pipeline_and_callback(user_id: str, user_input: str, callback_url: str):
    start_time = time.monotonic()
    print(f"\n[TASK_START] 비동기 파이프라인 가동 (User: {user_id[:8]})", flush=True)
    
    try:
        # Step 1. RAG 텍스트 답변 생성
        print("[TASK_STEP1] RAG 파이프라인 텍스트 연산 시작...", flush=True)
        result = await pipeline.process(user_id=user_id, user_input=user_input)
        print(f"[TASK_STEP1] ✅ 텍스트 연산 완료 (소요시간: {time.monotonic() - start_time:.2f}s)", flush=True)
        
        can_make_webtoon = not result.needs_requery
        image_url = None
        
        # Step 2. 질문 재확인이 필요 없다면 이어서 바로 웹툰 생성
        if can_make_webtoon:
            print("[TASK_STEP2] 웹툰 생성 조건 충족. generate_webtoon 진입", flush=True)
            story_context = f"상황: {user_input}\n법률해석: {result.response_text}"
            
            # 외부 모듈 호출 및 결과 추적
            filename = await generate_webtoon(story_context)
            
            if filename:
                image_url = f"{config.kakao.server_url}/images/{filename}"
                print(f"[TASK_STEP2] ✅ 웹툰 생성 성공 -> 매핑 URL: {image_url}", flush=True)
            else:
                print("[TASK_STEP2] ❌ 웹툰 생성 실패 (generate_webtoon이 None 반환)", flush=True)
        else:
            print("[TASK_STEP2] ⚠️ 추가 질문 재확인이 필요하여 웹툰 생성을 스킵합니다.", flush=True)
                
        # Step 3. 텍스트와 이미지(있을 경우)를 한 번에 콜백으로 전송
        print("[TASK_STEP3] 최종 결과를 카카오 콜백 엔드포인트로 전송 시도", flush=True)
        await send_callback(
            callback_url=callback_url,
            response_text=result.response_text,
            needs_requery=result.needs_requery,
            category=result.legal_category,
            image_url=image_url
        )
        print(f"[TASK_END] 비동기 파이프라인 전체 완료 (총 소요시간: {time.monotonic() - start_time:.2f}s)", flush=True)

    except Exception as e:
        print(f"[TASK_ERROR] 🔥 백그라운드 파이프라인 치명적 오류 발생: {e}", flush=True)
        traceback.print_exc()
        await send_callback(callback_url, "죄송합니다. 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")

@app.post("/webhook/kakao")
async def kakao_webhook(request: Request, background_tasks: BackgroundTasks):
    print("\n" + "="*50, flush=True)
    print("[WEBHOOK_HIT] 카카오톡 웹훅 엔드포인트 요청 도달", flush=True)
    try:
        body = await request.json()
    except Exception:
        print("[WEBHOOK_HIT] ❌ 에러: JSON 바디 파싱 실패", flush=True)
        return JSONResponse(content=build_simple_text("요청 형식이 올바르지 않습니다."), status_code=400)

    user_request = body.get("userRequest", {})
    user_id = user_request.get("user", {}).get("id", "anonymous")
    user_input = user_request.get("utterance", "").strip()
    callback_url = user_request.get("callbackUrl", "")

    print(f"[WEBHOOK_INFO] 발화어: {user_input} | 콜백 URL 존재여부: {bool(callback_url)}", flush=True)

    if not user_input:
        return JSONResponse(content=build_simple_text("질문을 입력해 주세요.", quick_replies=default_quick_replies(True)))

    if config.kakao.use_callback and callback_url:
        print("[WEBHOOK_ROUTE] 비동기(Callback) 백그라운드 루틴 예약 완료", flush=True)
        background_tasks.add_task(
            run_pipeline_and_callback, 
            user_id=user_id, 
            user_input=user_input, 
            callback_url=callback_url
        )
        waiting_message = "법률 판례를 분석하고 4컷 웹툰 요약을 그리는 중입니다. 약 1~2분 정도 소요될 수 있습니다. 🎨"
        return JSONResponse(content=build_callback_response(waiting_message))

    print("[WEBHOOK_ROUTE] ⚠️ 경고: 콜백 미사용 또는 callbackUrl 누락으로 동기식 블록 진입", flush=True)
    try:
        result = await asyncio.wait_for(pipeline.process(user_id=user_id, user_input=user_input), timeout=config.kakao.response_timeout_sec)
        return JSONResponse(content=build_simple_text(result.response_text, quick_replies=default_quick_replies(result.needs_requery, result.legal_category)))
    except Exception as e:
        print(f"[WEBHOOK_SYNC_ERROR] 동기 처리 중 예외: {e}", flush=True)
        return JSONResponse(content=build_simple_text("오류가 발생했습니다. 잠시 후 다시 시도해 주세요."))