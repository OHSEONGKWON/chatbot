import os
import sys
import time
import asyncio
import base64
import uvicorn
import httpx
import io
import json
import re
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# ── 환경변수 로드 ──────────────────────────────────────────────
load_dotenv()
_LEGAL_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Legal_chatbot")
load_dotenv(dotenv_path=os.path.join(_LEGAL_BASE, ".env"))

# ── RAG 모듈 import ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(_LEGAL_BASE, "model"))
try:
    from model_rag_v3_2 import run_full_pipeline, check_specificity, _get_legal_reask
    RAG_AVAILABLE = True
    print("✅ RAG 모듈 로드 성공")
except Exception as _e:
    RAG_AVAILABLE = False
    print(f"⚠️  RAG 모듈 로드 실패 (Gemini 폴백 사용): {_e}")

# ── 상태 및 맥락 관리 ──────────────────────────────────────────
conversation_state: dict = {}
user_context: dict = {}
MAX_REASK = 3
STATE_EXPIRE_S = 600
BASE_URL = "https://superbeloved-thermochemically-lyndsey.ngrok-free.dev"

# ── HTTP 및 Gemini 클라이언트 ──────────────────────────────────
http_client = httpx.AsyncClient()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
TEXT_MODEL = "gemini-2.5-flash" # 최신 모델 반영
IMAGE_MODEL = "gemini-2.5-flash-image"

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists("static"):
        os.makedirs("static")
    yield
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── [핵심 추가] 4컷 레이아웃 합성 함수 ──────────────────────────
def create_4cut_layout(image_bytes_list, captions, output_path="static/legal_comic.png"):
    panel_size = 512
    text_height = 90  # 자막 영역 높이
    margin = 15       # 간격
    
    # 2x2 그리드 전체 크기 계산
    canvas_width = (panel_size * 2) + (margin * 3)
    canvas_height = ((panel_size + text_height) * 2) + (margin * 3)
    
    # 연회색 배경의 캔버스 생성
    canvas = Image.new("RGB", (canvas_width, canvas_height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    
    # 한글 폰트 설정 (폰트 파일이 static 폴더에 있어야 함)
    try:
        # 나눔고딕 등 한글 폰트 파일 경로 지정
        font = ImageFont.truetype("static/NanumGothic.ttf", 22)
    except:
        font = ImageFont.load_default()
        print("⚠️ 한글 폰트 로드 실패 - 기본 폰트를 사용합니다.")

    positions = [
        (margin, margin),
        (panel_size + (margin * 2), margin),
        (margin, panel_size + text_height + (margin * 2)),
        (panel_size + (margin * 2), panel_size + text_height + (margin * 2))
    ]

    for i in range(len(image_bytes_list)):
        if i >= 4: break
        # 바이트 데이터를 이미지로 변환 및 리사이즈
        img = Image.open(io.BytesIO(image_bytes_list[i])).resize((panel_size, panel_size))
        curr_x, curr_y = positions[i]
        
        # 이미지 붙이기
        canvas.paste(img, (curr_x, curr_y))
        
        # 하단 자막 영역 (흰색 박스)
        text_y_start = curr_y + panel_size
        draw.rectangle([curr_x, text_y_start, curr_x + panel_size, text_y_start + text_height], fill="white")
        
        # 텍스트 중앙 정렬
        text = captions[i] if i < len(captions) else ""
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        draw.text(
            (curr_x + (panel_size - text_w) / 2, text_y_start + (text_height - text_h) / 2),
            text, fill="black", font=font
        )

    canvas.save(output_path)
    return output_path

# ── [핵심 수정] 만화 데이터 생성 및 이미지 합성 로직 ──────────────
async def generate_legal_comic(user_utterance: str) -> str | None:
    try:
        print("🔍 [1/3] 법률 상황 분석 및 스토리보드 생성 중...")
        
        # 1. 4컷용 스토리보드(JSON) 생성
        analyze = await client.aio.models.generate_content(
            model=TEXT_MODEL,
            contents=f"""
당신은 법률 만화 작가입니다. 다음 상황을 4컷 만화로 구성하세요.
JSON 형식으로만 답하세요:
{{
  "panels": [
    {{"caption": "1컷 자막(한국어)", "image_prompt": "English prompt for panel 1"}},
    {{"caption": "2컷 자막(한국어)", "image_prompt": "English prompt for panel 2"}},
    {{"caption": "3컷 자막(한국어)", "image_prompt": "English prompt for panel 3"}},
    {{"caption": "4컷 자막(한국어)", "image_prompt": "English prompt for panel 4"}}
  ]
}}
- 각 자막은 20자 이내로 핵심만 요약하세요.
- image_prompt는 'Korean webtoon style, clean lines, no text'를 포함하여 영어로만 묘사하세요.
상황: {user_utterance}
"""
        )
        
        # JSON 파싱 (마크다운 태그 제거)
        json_str = re.sub(r'```json|```', '', analyze.text).strip()
        data = json.loads(json_str)

        print("🚀 [2/3] 4개 이미지 개별 생성 시작 (병렬)...")
        
        # 2. 4개 이미지를 병렬로 생성하여 속도 최적화
        async def fetch_image_part(prompt):
            res = await client.aio.models.generate_content(
                model=IMAGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["IMAGE"])
            )
            for part in res.parts:
                if part.inline_data:
                    data = part.inline_data.data
                    return base64.b64decode(data) if isinstance(data, str) else data
            return None

        tasks = [fetch_image_part(p['image_prompt']) for p in data['panels']]
        image_bytes_list = await asyncio.gather(*tasks)

        if any(b is None for b in image_bytes_list):
            return None

        print("🎨 [3/3] 레이아웃 합성 중...")
        captions = [p['caption'] for p in data['panels']]
        file_path = create_4cut_layout(image_bytes_list, captions)

        return f"{BASE_URL}/{file_path}"

    except Exception as e:
        print(f"🔥 만화 생성 에러: {e}")
        return None

# ── RAG 답변 로직 (기존 유지) ───────────────────────────────────
async def _get_legal_answer(utterance: str, user_id: str) -> str:
    if not RAG_AVAILABLE:
        response = await client.aio.models.generate_content(model=TEXT_MODEL, contents=utterance)
        return response.text.strip()

    state = conversation_state.get(user_id)
    if state and (time.time() - state.get("ts", 0)) > STATE_EXPIRE_S:
        conversation_state.pop(user_id, None)
        state = None

    combined = state["combined_query"] + " " + utterance if state else utterance
    reask_count = (state["reask_count"] + 1) if state else 0

    try:
        spec = await asyncio.to_thread(check_specificity, combined)
    except: spec = None

    if spec and not spec.is_sufficient and reask_count < MAX_REASK:
        followup = spec.followup_question or _get_legal_reask(spec.legal_category, spec.missing_elements, reask_count)
        conversation_state[user_id] = {"combined_query": combined, "reask_count": reask_count, "ts": time.time()}
        return followup

    conversation_state.pop(user_id, None)
    try:
        result = await asyncio.to_thread(run_full_pipeline, combined, None, False)
        if result and result.final_answer:
            user_context[user_id] = {"query": combined, "answer": result.final_answer}
            return result.final_answer
    except: pass

    response = await client.aio.models.generate_content(model=TEXT_MODEL, contents=combined)
    user_context[user_id] = {"query": combined, "answer": response.text.strip()}
    return response.text.strip()

# ── 콜백 처리 및 엔드포인트 (기존 유지) ──────────────────────────
async def process_and_callback(callback_url: str, utterance: str, user_id: str):
    try:
        if any(kw in utterance for kw in ["그림", "만화", "그려줘", "시각화"]):
            ctx = user_context.get(user_id)
            comic_input = f"질문: {ctx['query']}\n답변: {ctx['answer']}\n요청: {utterance}" if ctx else utterance
            image_url = await generate_legal_comic(comic_input)

            if image_url:
                payload = {"version": "2.0", "template": {"outputs": [{"simpleImage": {"imageUrl": image_url, "altText": "법률 상황 네컷만화"}}]}}
            else:
                payload = {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "🎨 이미지 생성에 실패했습니다."}}]}}
        else:
            answer = await _get_legal_answer(utterance, user_id)
            payload = {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": answer[:500]}}]}}

        await http_client.post(callback_url, json=payload, headers={"Content-Type": "application/json; charset=utf-8"}, timeout=30.0)
    except Exception as e:
        print(f"🔥 콜백 에러: {e}")

@app.post("/api/chat")
async def kakao_chat(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    user_req = data.get("userRequest", {})
    utterance = user_req.get("utterance", "")
    callback_url = user_req.get("callbackUrl")
    user_id = user_req.get("user", {}).get("id", "anonymous")

    if callback_url:
        background_tasks.add_task(process_and_callback, callback_url, utterance, user_id)
        wait_msg = "AI가 만화를 그리고 있습니다. 잠시만 기다려주세요!" if any(kw in utterance for kw in ["그림", "만화"]) else "⚖️ 답변을 분석 중입니다..."
        return {"version": "2.0", "useCallback": True, "template": {"outputs": [{"simpleText": {"text": wait_msg}}]}}
    return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "연결 완료"}}]}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)