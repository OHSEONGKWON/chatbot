import os
import uvicorn
from fastapi import FastAPI, Request
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 2. Gemini API 클라이언트 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_ID = "gemini-2.5-flash"

async def get_gemini_answer(question):
    """
    성범죄 법률 전문 AI로서 법적 근거를 포함한 답변을 생성합니다.
    2.5-flash 모델을 사용하여 카카오톡 5초 제한 내에 응답하도록 최적화되었습니다.
    """
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=(
                f"당신은 성범죄 법률 전문 AI입니다. 질문에 대해 전문적으로 답변하되 아래 규칙을 엄수하세요.\n"
                f"1. 답변은 반드시 3~4문장 내외로 짧고 명확하게 작성할 것.\n"
                f"2. 가독성을 위해 각 문장마다 줄바꿈을 할 것.\n"
                f"3. 법적 근거(형법 제00조 등)를 반드시 포함하되, 마지막은 '전문 변호사와 상담하세요'로 맺을 것.\n"
                f"4. 공백 포함 300자 이내로 핵심만 전달할 것.\n\n"
                f"질문: {question}"
            ),
            config=types.GenerateContentConfig(
                max_output_tokens=1000, 
                temperature=0.3,  # 법률 답변의 정확성을 위해 낮은 온도 설정
                top_p=0.8
            )
        )
        
        # 답변이 비어있거나 에러가 발생한 경우 처리
        if not response.text:
            return "죄송합니다. 현재 법률 데이터를 분석하는 데 어려움이 있습니다. 잠시 후 다시 질문해 주세요."
            
        return response.text.strip()

    except Exception as e:
        print(f"[API Error]: {e}")
        return "법률 자문 시스템 통신 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

@app.post("/api/chat")
async def kakao_chat(request: Request):
    """
    카카오톡 챗봇 스킬 요청을 처리하는 엔드포인트입니다.
    """
    try:
        data = await request.json()
        # 사용자의 발화 추출
        utterance = data.get("userRequest", {}).get("utterance", "")

        if not utterance:
            return {
                "version": "2.0",
                "template": {"outputs": [{"simpleText": {"text": "상담 내용을 입력해 주세요."}}]}
            }

        # Gemini 모델로부터 답변 생성 (비동기 호출)
        answer = await get_gemini_answer(utterance)

        # 카카오톡 표준 응답 규격 반환
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
    except Exception as e:
        print(f"[Server Error]: {e}")
        return {
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": "서버 연결이 원활하지 않습니다. 잠시 후 다시 시도해 주세요."}}]
            }
        }

if __name__ == "__main__":
    # 포트 8000번에서 서버 실행
    print(f"서버가 구동되었습니다. 모델: {MODEL_ID}")
    uvicorn.run(app, host="0.0.0.0", port=8000)