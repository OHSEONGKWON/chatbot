from fastapi import FastAPI, Request
from google.genai import Client
from dotenv import load_dotenv
import os
import uvicorn

load_dotenv()
app = FastAPI()

# 1. API 키 설정 (ss.py에서 확인한 그 키!)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = Client(api_key=GEMINI_API_KEY)

async def get_gemini_fast_answer(question):
    """5초 타임아웃을 피하기 위해 최대한 짧고 빠르게 답변 생성"""
    try:
        # 답변 길이를 제한하여 생성 속도를 높임 (max_output_tokens)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"당신은 성범죄 법률 전문 AI입니다. 3문장 이내로 짧게 답변하세요: {question}",
            config={'max_output_tokens': 300} 
        )
        return response.text
    except Exception as e:
        return f"답변 생성 중 오류: {str(e)}"

@app.post("/api/chat")
async def kakao_chat(request: Request):
    data = await request.json()
    utterance = data.get("userRequest", {}).get("utterance", "")

    # Gemini에게 즉시 물어보고 답변 받기 (동기 방식처럼 작동)
    answer = await get_gemini_fast_answer(utterance)

    # 카카오 표준 응답 규격 (가장 단순한 형태)
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": answer
                    }
                }
            ]
        }
    }

if __name__ == "__main__":
    # 포트 8000번 사용
    uvicorn.run(app, host="0.0.0.0", port=8000)