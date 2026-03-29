from fastapi import FastAPI, Request
from google.genai import Client  # 최신 SDK 임포트 방식
from dotenv import load_dotenv
import os
import asyncio
import aiohttp
import uvicorn

# .env 파일에서 GEMINI_API_KEY 로드
load_dotenv()

app = FastAPI()

# 1. 최신 Gemini 클라이언트 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = Client(api_key=GEMINI_API_KEY)

async def get_gemini_answer(question):
    """최신 SDK 방식으로 Gemini 2.5-flash 답변 생성 (RAG 제외)"""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"당신은 성범죄 법률 전문 AI입니다. 질문에 친절하게 답변해주세요: {question}"
        )
        return response.text
    except Exception as e:
        return f"Gemini 답변 생성 중 오류 발생: {str(e)}"

async def send_kakao_callback(callback_url, text):
    """카카오톡 서버로 최종 답변 전송 (콜백)"""
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": text}}]
        }
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(callback_url, json=payload) as resp:
                print(f"📡 콜백 전송 결과: {resp.status}")
        except Exception as e:
            print(f"❌ 콜백 전송 실패: {e}")

async def process_task(url, q):
    """비동기로 답변 생성 후 콜백 실행"""
    answer = await get_gemini_answer(q)
    await send_kakao_callback(url, answer)

@app.post("/api/chat")
async def kakao_chat(request: Request):
    data = await request.json()
    utterance = data.get("userRequest", {}).get("utterance", "")
    callback_url = data.get("userRequest", {}).get("callbackUrl", "")

    # 카카오 5초 타임아웃 방지
    if callback_url:
        # 백그라운드에서 Gemini 답변 생성 시작
        asyncio.create_task(process_task(callback_url, utterance))
        
        return {
            "version": "2.0",
            "useCallback": True,
            "data": {
                "text": "💬 답변을 생각하고 있습니다. 잠시만 기다려 주세요..."
            }
        }
    else:
        # 콜백 주소가 없는 경우 직접 응답
        answer = await get_gemini_answer(utterance)
        return {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": answer}}]}
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)