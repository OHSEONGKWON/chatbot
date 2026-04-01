import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
import httpx
import google.generativeai as genai

# [1순위] 앱 객체를 먼저 생성합니다. (이게 아래 @app보다 위에 있어야 함)
app = FastAPI()

# [2순위] Gemini 설정
GENI_API_KEY = "YOUR_GEMINI_API_KEY"  # 실제 API 키를 입력하세요
genai.configure(api_key=GENI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# [3순위] 도움을 줄 함수들 정의
async def get_gemini_answer(question):
    try:
        # 모델명은 실제 사용하시는 것에 맞춰 'gemini-1.5-flash' 등으로 수정 가능
        response = model.generate_content(question)
        return response.text.strip() if response.text else "답변을 생성할 수 없습니다."
    except Exception as e:
        print(f"Gemini API 오류: {e}")
        return "답변 생성 중 오류가 발생했습니다."

async def send_callback_answer(callback_url: str, utterance: str):
    # Gemini로부터 답변을 가져옴 (시간이 걸리는 작업)
    answer = await get_gemini_answer(utterance)
    
    # 카카오 콜백 규격에 맞춘 데이터
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": answer}}]
        }
    }
    
    # 카카오 서버로 답변 전송
    async with httpx.AsyncClient() as client:
        try:
            await client.post(callback_url, json=payload, timeout=10.0)
            print("콜백 전송 완료")
        except Exception as e:
            print(f"콜백 전송 실패: {e}")

# [4순위] API 경로 설정 (여기서 'app'을 사용하므로 위에서 정의되어야 함)
@app.post("/api/chat")
async def kakao_chat(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    
    # 카카오에서 보내주는 정보 추출
    user_request = data.get("userRequest", {})
    utterance = user_request.get("utterance", "")
    callback_url = user_request.get("callbackUrl") # 콜백 주소

    # 백그라운드 작업 등록 (Gemini 호출)
    if callback_url:
        background_tasks.add_task(send_callback_answer, callback_url, utterance)

    # 5초 이내에 즉시 응답 (이 메시지가 먼저 뜨고, 나중에 Gemini 답변이 옴)
    return {
        "version": "2.0",
        "useCallback": True,
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": "답변을 준비하고 있습니다. 잠시만 기다려 주세요! 🤖"
                    }
                }
            ]
        }
    }

# [5순위] 서버 실행 코드
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)