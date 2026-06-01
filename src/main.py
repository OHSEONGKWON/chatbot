"""LawsGuard FastAPI 메인 애플리케이션 (웹툰 선출력 수정 버전)"""

import asyncio
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager

print("[LAWSGUARD_DEBUG] >>> src/main.py 파일이 성공적으로 로드되었습니다! <<<", flush=True)

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
    print("[LAWSGUARD_INIT] 서버 라이프사이클(lifespan) 시작 완료", flush=True)
    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    warmup_task = asyncio.create_task(_warmup_components())
    await warmup_task
    yield
    cleanup_task.cancel()
    print("[LAWSGUARD_INIT] 서버 라이프사이클 종료", flush=True)

async def _session_cleanup_loop():
    try:
        while True:
            await asyncio.sleep(1800)
            session_store.cleanup_expired()
            print("[LAWSGUARD_LOOP] 만료 세션 정리 루프 작동 완료", flush=True)
    except asyncio.CancelledError:
        pass

async def _warmup_components():
    try:
        print("[LAWSGUARD_INIT] 핵심 AI 모델 워밍업 가동 시작...", flush=True)
        await asyncio.gather(
            pipeline._retriever.warmup(),
            pipeline._consistency.warmup(),
            pipeline._ner.warmup(),
        )
        print("[LAWSGUARD_INIT] ✅ 핵심 AI 모델 워밍업 전면 완료", flush=True)
    except Exception as e:
        print(f"[LAWSGUARD_INIT] ❌ 모델 워밍업 중 예외 발생: {e}", flush=True)
        traceback.print_exc()

app = FastAPI(title="LawsGuard API", version="1.0.0", lifespan=lifespan)
app.mount("/images", StaticFiles(directory=str(OUTPUT_DIR)), name="images")

async def send_callback(
    callback_url: str,
    response_text: str,
    needs_requery: bool = False,
    category: str = "",
    image_url: str | None = None,
) -> bool:
    print(f"[CALLBACK_SEND] 카카오 콜백 전송 진입. 이미지 URL 유무: {bool(image_url)}", flush=True)
    quick_replies = default_quick_replies(needs_requery, category)
    
    if image_url:
        print("[CALLBACK_SEND] 🚀 순서 조정: [1컷] 웹툰 이미지 선출력 -> [2컷] 법률 텍스트 후출력 구성", flush=True)
        # 카카오 표준 규격에 맞춰 outputs 배열 내부의 순서를 [이미지, 텍스트] 순서로 직접 정의합니다.
        payload = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleImage": {
                            "imageUrl": image_url,
                            "altText": "법률 요약 4컷 웹툰"
                        }
                    },
                    {
                        "simpleText": {
                            "text": response_text
                        }
                    }
                ],
                "quickReplies": quick_replies
            }
        }
    else:
        print("[CALLBACK_SEND] ⚠️ 이미지 없음 - 일반 텍스트 단독 페이로드 구성", flush=True)
        payload = build_simple_text(response_text, quick_replies=quick_replies)
        
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(callback_url, json=payload)
            print(f"[CALLBACK_SEND] 카카오 서버 전송 결과 응답코드: {resp.status_code}", flush=True)
            if resp.status_code >= 400:
                print(f"[CALLBACK_SEND] ❌ 카카오 응답 실패 본문: {resp.text[:300]}", flush=True)
            return resp.status_code < 400
        except Exception as e:
            print(f"[CALLBACK_SEND] ❌ 카카오 전송 중 통신 예외 발생: {e}", flush=True)
            traceback.print_exc()
            return False

# 1. 텍스트 답변 + 웹툰 동시 생성 백그라운드 파이프라인
async def run_pipeline_and_callback(user_id: str, user_input: str, callback_url: str):
    start_time = time.monotonic()
    print(f"\n[TASK_START] 비동기 백그라운드 태스크 가동 시작 (User ID: {user_id[:8]})", flush=True)
    
    try:
        # Step 1. RAG 텍스트 답변 생성
        print("[TASK_STEP1] RAG 파이프라인 텍스트 연산 요청...", flush=True)
        result = await pipeline.process(user_id=user_id, user_input=user_input)
        print(f"[TASK_STEP1] ✅ 텍스트 연산 완료 (구간 소요시간: {time.monotonic() - start_time:.2f}s)", flush=True)
        
        can_make_webtoon = not result.needs_requery
        image_url = None
        
        # Step 2. 질문 재확인이 필요 없다면 이어서 바로 웹툰 생성
        if can_make_webtoon:
            print("[TASK_STEP2] 웹툰 생성 조건 충족 (needs_requery가 False). generate_webtoon 진입합니다.", flush=True)
            story_context = f"상황: {user_input}\n법률해석: {result.response_text}"
            
            filename = await generate_webtoon(story_context)
            
            if filename:
                image_url = f"{config.kakao.server_url}/images/{filename}"
                print(f"[TASK_STEP2] ✅ 웹툰 생성 성공 -> 최종 이미지 URL 매핑 완료: {image_url}", flush=True)
            else:
                print("[TASK_STEP2] ❌ 웹툰 생성 실패 (generate_webtoon이 None을 반환함)", flush=True)
        else:
            print("[TASK_STEP2] ⚠️ 추가 질문 재확인이 필요하여 웹툰 이미지 생성을 스킵합니다.", flush=True)
                
        # Step 3. 텍스트와 이미지(있을 경우)를 한 번에 콜백으로 전송
        print("[TASK_STEP3] 최종 결과를 카카오 콜백 URL로 쏘기 시작합니다.", flush=True)
        await send_callback(
            callback_url=callback_url,
            response_text=result.response_text,
            needs_requery=result.needs_requery,
            category=result.legal_category,
            image_url=image_url
        )
        print(f"[TASK_END] 백그라운드 프로세스 최종 완료 (총 누적 소요시간: {time.monotonic() - start_time:.2f}s)", flush=True)

    except Exception as e:
        print(f"[TASK_ERROR] 🔥 백그라운드 파이프라인 내부에서 예기치 못한 치명적 예외 터짐: {e}", flush=True)
        traceback.print_exc()
        await send_callback(callback_url, "죄송합니다. 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")

@app.post("/webhook/kakao")
async def kakao_webhook(request: Request, background_tasks: BackgroundTasks):
    print("\n" + "="*60, flush=True)
    print("[WEBHOOK_HIT] 카카오톡 라우터 엔드포인트 요청 진입!", flush=True)
    try:
        body = await request.json()
        print(f"[WEBHOOK_BODY] 수신된 원본 데이터: {body}", flush=True)
    except Exception:
        print("[WEBHOOK_HIT] ❌ 에러: 카카오 요청 JSON 바디 파싱 실패", flush=True)
        return JSONResponse(content=build_simple_text("요청 형식이 올바르지 않습니다."), status_code=400)

    user_request = body.get("userRequest", {})
    user_id = user_request.get("user", {}).get("id", "anonymous")
    user_input = user_request.get("utterance", "").strip()
    callback_url = user_request.get("callbackUrl", "")

    print(f"[WEBHOOK_INFO] 유저 발화어: [{user_input}] | 콜백 URL 값 존재여부: {bool(callback_url)} (값: {callback_url})", flush=True)

    if not user_input:
        return JSONResponse(content=build_simple_text("질문을 입력해 주세요.", quick_replies=default_quick_replies(True)))

    # 분기 필터링 검증
    if config.kakao.use_callback and callback_url:
        print("[WEBHOOK_ROUTE] ✅ 조건 만족: 비동기(Callback) 백그라운드 태스크 예약 완료!", flush=True)
        background_tasks.add_task(
            run_pipeline_and_callback, 
            user_id=user_id, 
            user_input=user_input, 
            callback_url=callback_url
        )
        waiting_message = "법률 판례를 분석하고 4컷 웹툰 요약을 그리는 중입니다. 약 1~2분 정도 소요될 수 있습니다. 🎨"
        return JSONResponse(content=build_callback_response(waiting_message))

    print("[WEBHOOK_ROUTE] ⚠️ 경고: config 설정상 콜백 미사용 또는 callbackUrl 누락으로 즉시 동기식 블록 진입!", flush=True)
    try:
        result = await asyncio.wait_for(pipeline.process(user_id=user_id, user_input=user_input), timeout=config.kakao.response_timeout_sec)
        print("[WEBHOOK_SYNC] 동기 연산 응답 성공 및 즉시 반환 완료 (웹툰 미포함)", flush=True)
        return JSONResponse(content=build_simple_text(result.response_text, quick_replies=default_quick_replies(result.needs_requery, result.legal_category)))
    except Exception as e:
        print(f"[WEBHOOK_SYNC_ERROR] 동기 연산 처리 중 에러 또는 타임아웃 발생: {e}", flush=True)
        traceback.print_exc()
        return JSONResponse(content=build_simple_text("오류가 발생했습니다. 잠시 후 다시 시도해 주세요."))

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "LawsGuard"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host=config.kakao.server_host, port=config.kakao.server_port, reload=False, workers=1, log_level="info")