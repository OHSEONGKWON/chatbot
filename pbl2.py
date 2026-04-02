import os
import uvicorn
import httpx
import asyncio
import google.generativeai as genai  # 라이브러리 변경
from fastapi import FastAPI, Request, BackgroundTasks
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
MODEL_ID = "models/gemini-2.5-flash" 

async def get_gemini_answer(question):
    try:
        model = genai.GenerativeModel(MODEL_ID)
        prompt = (
            f"당신은 성범죄 법률 전문 AI입니다. 질문에 대해 전문적이고 상세하게 답변하세요.\n"
            f"마지막에는 반드시 '본 답변은 참고용이며 법적 효력이 없으므로 변호사와 상담하시기 바랍니다.'를 포함하세요.\n\n"
            f"질문: {question}"
        )
        
        # 동기 호출을 비동기로 실행
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Error]: {e}")
        return "법률 데이터를 분석하는 중 오류가 발생했습니다."

async def process_and_callback(callback_url: str, utterance: str):
    try:
        answer = await get_gemini_answer(utterance)
        callback_payload = {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": answer}}]}
        }
        async with httpx.AsyncClient() as http_client:
            await http_client.post(callback_url, json=callback_payload)
    except Exception as e:
        print(f"[Callback Error]: {e}")

@app.post("/api/chat")
async def kakao_chat(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    utterance = data.get("userRequest", {}).get("utterance", "")
    callback_url = data.get("userRequest", {}).get("callbackUrl")

    if callback_url:
        background_tasks.add_task(process_and_callback, callback_url, utterance)
        return {"version": "2.0", "useCallback": True, "template": {"outputs": []}}
    
    return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "상담 대기 중..."}}]}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)