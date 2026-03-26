from fastapi import FastAPI, Request
import uvicorn
import google.generativeai as genai

# 1. Gemini API 설정 (현재 테스트에서는 멈춰두지만 뼈대는 유지합니다)
GEMINI_API_KEY = "AIzaSyDE-oZfNXe1hABFccenfbzq1mh_vMzmsIo"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI()

@app.post("/api/chat")
async def kakao_chat(request: Request):
    # 1. 카카오톡 메시지 수신
    kakao_data = await request.json()
    user_message = kakao_data["userRequest"]["utterance"]
    print(f"👤 사용자: {user_message}")

    # =========================================================
    # 2. 5초 타임아웃 원인 파악을 위한 0.1초 '칼답' 테스트 로직
    # =========================================================
    try:
        # 원래 Gemini에게 물어보던 코드를 잠깐 꺼둡니다 (# 주석 처리)
        # response = model.generate_content(user_message)
        # llm_reply = response.text
        
        # 대신, 우리가 직접 만든 문장을 바로 꽂아 넣습니다.
        llm_reply = "0.1초 만에 보내는 테스트 답변입니다! 서버 통신 완벽해요."
        print(f"🤖 테스트 응답: {llm_reply}")
        
    except Exception as e:
        llm_reply = "에러가 발생했습니다."
        print(f"에러 발생: {e}")

    # 3. 카카오톡 양식으로 포장해서 즉시 반환
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": llm_reply
                    }
                }
            ]
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)