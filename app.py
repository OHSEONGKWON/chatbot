from fastapi import FastAPI, Request
import uvicorn
import google.generativeai as genai
import asyncio
from fastapi.middleware.cors import CORSMiddleware

# API 키 설정
GEMINI_API_KEY = "AIzaSyDqYNVCmVxZMTgePa8x4ALu870rg3vtx3c"  # 여기에 실제 API 키를 입력하세요.
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('models/gemini-2.5-flash')

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat")
async def kakao_chat(request: Request):
    kakao_data = await request.json()
    user_message = kakao_data["userRequest"]["utterance"].strip()
    
    print(f"\n👤 사용자: {user_message}")

    try:
        response = await asyncio.wait_for(
            model.generate_content_async(user_message),
            timeout=4.5
        )
        llm_reply = response.text
        print(f"🤖 Gemini 성공: {llm_reply}")
        
    except asyncio.TimeoutError:
        llm_reply = "앗! 관련 법률을 꼼꼼히 뒤져보느라 5초가 넘어버렸어요."
        print("⏳ 타임아웃 방어 성공! (카카오 에러 회피)")
        
    except Exception as e:
        llm_reply = "서버 에러가 발생했습니다."
        print(f"에러 발생: {e}")

    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": llm_reply}}]
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)