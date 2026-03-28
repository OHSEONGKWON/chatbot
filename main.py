from fastapi import FastAPI, Request
import uvicorn
import google.generativeai as genai
import asyncio  # 🌟 시간 제어를 위한 비동기 모듈

GEMINI_API_KEY = "AIzaSyB14vLcdlNQt1zFapXWxqeyt01rlyn4wLs".strip()
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('models/gemini-2.0-flash')

app = FastAPI()

@app.post("/api/chat")
async def kakao_chat(request: Request):
    kakao_data = await request.json()
    user_message = kakao_data["userRequest"]["utterance"].strip()
    
    print(f"\n👤 사용자: {user_message}")

    try:
        # ⚡ 카카오가 문을 닫기 전(5초), 4.5초 만에 파이썬이 먼저 선수를 칩니다!
        # wait_for를 사용해 Gemini의 답변 시간을 강제로 제한합니다.
        response = await asyncio.wait_for(
            model.generate_content_async(user_message),
            timeout=4.5
        )
        llm_reply = response.text
        print(f"🤖 Gemini 성공: {llm_reply}")
        
    except asyncio.TimeoutError:
        # 4.5초가 넘어가면 봇이 뻗지 않고 자연스럽게 핑계를 댑니다.
        llm_reply = "앗! 관련 법률을 꼼꼼히 뒤져보느라 5초가 넘어버렸어요. "
        print("⏳ 타임아웃 방어 성공! (카카오 에러 회피)")
        
    except Exception as e:
        llm_reply = "서버 에러가 발생했습니다."
        print(f"에러 발생: {e}")

    # 카카오톡으로 즉시 쏴주기
    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": llm_reply}}]
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)