"""
pip install google-genai fastapi uvicorn httpx python-dotenv
pip install chromadb sentence-transformers openai bert-score requests
"""

import os
import sys
import asyncio
import base64
import uvicorn
import httpx
from google import genai
from google.genai import types
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# ── 환경변수 로드 (chatbot .env → Legal_chatbot .env 순서로) ────
load_dotenv()

_LEGAL_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Legal_chatbot")
load_dotenv(dotenv_path=os.path.join(_LEGAL_BASE, ".env"))  # OPENAI_API_KEY 포함

# ── RAG 모듈 import ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(_LEGAL_BASE, "model"))
try:
    from model_rag_v3_2 import run_full_pipeline
    RAG_AVAILABLE = True
    print("✅ RAG 모듈 로드 성공")
except Exception as _e:
    RAG_AVAILABLE = False
    print(f"⚠️  RAG 모듈 로드 실패 (Gemini 폴백 사용): {_e}")

# ── HTTP 세션 ─────────────────────────────────────────────────
http_client = httpx.AsyncClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists("static"):
        os.makedirs("static")
        print("📁 static 폴더 생성 완료")
    yield
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Gemini 클라이언트 ─────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

TEXT_MODEL  = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash-image"

BASE_URL = "https://superbeloved-thermochemically-lyndsey.ngrok-free.dev"


# ── 로직 함수 ────────────────────────────────────────────────

async def get_comic_image_prompt(question: str) -> str | None:
    """법률 상황 분석 → 패널별 시각 묘사 포함한 이미지 프롬프트 생성"""
    try:
        print("🔍 [1/4] 법률 상황 분석 중...")

        # 1단계: 상황과 대처법을 패널별 시각 묘사까지 상세 분석 (한국어)
        analyze = await client.aio.models.generate_content(
            model=TEXT_MODEL,
            contents=f"""
당신은 법률 전문가이자 만화 스토리보드 작가입니다.
다음 법률 상황을 분석해서 4컷 만화 스토리보드를 만들어주세요.
반드시 아래 형식으로만 답해주세요:

[상황 요약] 15자 이내로 핵심 상황 요약
[상황 장면] 1컷에 그릴 구체적인 장면을 상세히 묘사. 등장인물의 표정, 행동, 배경, 소품을 구체적으로 설명. (예: 화난 표정의 세입자가 이삿짐 박스를 들고 아파트 현관 앞에 서있고, 집주인은 문을 닫으며 외면하는 장면. 배경은 아파트 복도.)

[1단계 요약] 15자 이내로 첫 번째 대처법 요약
[1단계 장면] 2컷에 그릴 구체적인 장면을 상세히 묘사. 등장인물의 표정, 행동, 배경, 소품을 구체적으로 설명.

[2단계 요약] 15자 이내로 두 번째 대처법 요약
[2단계 장면] 3컷에 그릴 구체적인 장면을 상세히 묘사. 등장인물의 표정, 행동, 배경, 소품을 구체적으로 설명.

[결과 요약] 15자 이내로 예상 결과 요약
[결과 장면] 4컷에 그릴 구체적인 장면을 상세히 묘사. 등장인물의 표정, 행동, 배경, 소품을 구체적으로 설명. (예: 세입자가 환한 표정으로 현금이 든 봉투를 받아들고, 집주인은 체념한 표정으로 건네주는 장면.)

법률 상황: {question}
"""
        )

        structured = analyze.text.strip()
        print(f"✅ 상황 분석 완료:\n{structured}")

        # 2단계: 분석 결과를 영어 이미지 프롬프트로 변환 (한국어 완전 제거)
        print("🎨 이미지 프롬프트 정밀 생성 중...")
        prompt_response = await client.aio.models.generate_content(
            model=TEXT_MODEL,
            contents=f"""
You are a professional comic strip prompt engineer.
Based on the following Korean legal situation analysis, write a highly detailed English image generation prompt for a 4-panel webtoon comic strip.

IMPORTANT: Translate ALL scene descriptions into English.
The final prompt must contain ONLY English words. Absolutely NO Korean characters allowed anywhere.

Analysis:
{structured}

Requirements:
- The image must be a strict 2x2 grid with 4 equal panels separated by thick black borders
- Korean webtoon art style: clean lines, expressive faces, black and white with light gray tones
- Each panel must visually match the described scene EXACTLY
- Characters must show clear emotions matching the situation
- Include specific props, settings, and actions described in each panel
- Short English captions or labels inside panels are allowed
- Panel 1 (top-left): Draw the situation scene
- Panel 2 (top-right): Draw the first action scene
- Panel 3 (bottom-left): Draw the second action scene
- Panel 4 (bottom-right): Draw the result scene

Write the prompt now in ENGLISH ONLY. Do not include any Korean characters:
"""
        )

        # ✅ structured 제거 - prompt_response만 사용해서 한국어 유입 차단
        final_prompt = f"""
CRITICAL LAYOUT RULE: This image MUST contain exactly 4 panels in a 2x2 grid layout.
- TOP-LEFT panel = Panel 1 (situation)
- TOP-RIGHT panel = Panel 2 (first action)
- BOTTOM-LEFT panel = Panel 3 (second action)
- BOTTOM-RIGHT panel = Panel 4 (result)
Each panel separated by thick black borders. All 4 panels must be equal size.
Short English captions or labels inside panels are allowed.
Korean webtoon style, expressive characters, clean lines.

{prompt_response.text.strip()}
"""

        print(f"✅ 최종 프롬프트 생성 완료: {final_prompt[:100]}...")
        return final_prompt

    except Exception as e:
        print(f"❌ 프롬프트 생성 에러: {e}")
        return None


async def generate_legal_comic(user_utterance: str) -> str | None:
    """이미지 생성 → static 저장 → URL 반환"""
    image_prompt = await get_comic_image_prompt(user_utterance)
    if not image_prompt:
        return None

    try:
        print("🚀 [2/4] 이미지 생성 중...")

        response = await client.aio.models.generate_content(
            model=IMAGE_MODEL,
            contents=image_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
        )

        print("📦 [3/4] 응답 수신 — 이미지 파트 탐색 중...")

        image_bytes = None
        for part in response.parts:
            if part.inline_data is not None:
                data = part.inline_data.data
                image_bytes = base64.b64decode(data) if isinstance(data, str) else data
                break

        if image_bytes is None:
            print("❌ 이미지 파트를 찾지 못했습니다.")
            print(f"   전체 parts: {response.parts}")
            return None

        file_path = "static/legal_comic.png"
        with open(file_path, "wb") as f:
            f.write(image_bytes)

        full_url = f"{BASE_URL}/{file_path}"
        print(f"✅ [4/4] 저장 완료: {full_url}")
        return full_url

    except Exception as e:
        print(f"🔥 이미지 생성 에러: {e}")
        return None


async def _get_legal_answer(utterance: str) -> str:
    """RAG 파이프라인으로 법률 답변 생성. 실패 시 Gemini 폴백."""
    if RAG_AVAILABLE:
        try:
            result = await asyncio.to_thread(run_full_pipeline, utterance, None, False)
            if result and result.final_answer:
                return result.final_answer
        except Exception as e:
            print(f"⚠️  RAG 실패 → Gemini 폴백: {e}")

    # Gemini 폴백
    response = await client.aio.models.generate_content(
        model=TEXT_MODEL,
        contents=utterance
    )
    return response.text.strip()


async def process_and_callback(callback_url: str, utterance: str):
    """백그라운드 처리 후 카카오톡 콜백 전송"""
    print(f"📩 콜백 시작 | utterance: {utterance[:30]}")

    try:
        if any(kw in utterance for kw in ["그림", "만화", "그려줘", "시각화"]):
            image_url = await generate_legal_comic(utterance)

            if image_url:
                payload = {
                    "version": "2.0",
                    "template": {
                        "outputs": [
                            {
                                "simpleImage": {
                                    "imageUrl": image_url,
                                    "altText": "법률 상황 네컷만화"
                                }
                            }
                        ]
                    }
                }
            else:
                payload = {
                    "version": "2.0",
                    "template": {
                        "outputs": [
                            {"simpleText": {"text": "🎨 이미지 생성에 실패했습니다. 다시 시도해주세요."}}
                        ]
                    }
                }

        else:
            answer = await _get_legal_answer(utterance)
            payload = {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {"simpleText": {"text": answer[:500]}}
                    ]
                }
            }

        print("📤 카카오톡 콜백 전송 중...")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        res = await http_client.post(callback_url, json=payload, headers=headers, timeout=30.0)

        print(f"🏁 콜백 결과: {res.status_code}")
        if res.status_code != 200:
            print(f"❗ 실패 응답: {res.text}")

    except Exception as e:
        print(f"🔥 콜백 처리 예외: {e}")


# ── 엔드포인트 ───────────────────────────────────────────────

@app.post("/api/chat")
async def kakao_chat(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    utterance    = data.get("userRequest", {}).get("utterance", "")
    callback_url = data.get("userRequest", {}).get("callbackUrl")

    if callback_url:
        background_tasks.add_task(process_and_callback, callback_url, utterance)

        is_comic = any(kw in utterance for kw in ["그림", "만화", "그려줘", "시각화"])
        wait_msg = (
            "AI가 상황을 분석하고 만화를 그리는 중입니다. 약 15초 정도 소요되니 잠시만 기다려주세요! 🎨"
            if is_comic else
            "⚖️ 법률 질문을 분석하고 있어요. 잠시만 기다려 주세요 😊"
        )

        return {
            "version": "2.0",
            "useCallback": True,
            "template": {
                "outputs": [
                    {"simpleText": {"text": wait_msg}}
                ]
            }
        }

    return {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": "상태 확인 완료"}}]}
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)