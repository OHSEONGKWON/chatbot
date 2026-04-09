import os
import uvicorn
import asyncio
import google.generativeai as genai  
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles # 사진 서빙을 위한 라이브러리 추가
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 💡 [핵심 추가] 내 컴퓨터의 'images' 폴더를 인터넷 주소('/images')로 연결합니다.
# 이제 ngrok주소/images/alba_unpaid.png 로 사진을 볼 수 있게 됩니다.
app.mount("/images", StaticFiles(directory="images"), name="images")

# 1. API 키 설정 (제미나이만 사용)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

MODEL_ID = "gemini-2.0-flash" 

# --- (STEP 1) 제미나이: 질문을 분석해서 사진 번호(카테고리)만 뽑기 ---
async def classify_question(question):
    try:
        model = genai.GenerativeModel(MODEL_ID)
        # 제미나이에게 긴 답변 대신, 오직 '숫자'만 대답하라고 강력하게 명령합니다.
        prompt = (
            f"당신은 성범죄 법률 상황 분류기입니다. "
            f"사용자의 질문을 읽고, 다음 카테고리 중 어디에 가장 잘 맞는지 분석해서 **오직 숫자(1, 2)만** 대답하세요.\n\n"
            f"1: 아르바이트생 월급/퇴직금 미지급, 임금 체불 관련 질문\n"
            f"2: 그 외 다른 모든 상황\n\n"
            f"질문: {question}"
        )
        
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text.strip() # 예: "1" 또는 "2" 반환
    except Exception as e:
        print(f"[Gemini Error]: {e}")
        return "2" # 에러 시 기본 카테고리

# --- 카카오톡 챗봇 메인 엔드포인트 ---
@app.post("/api/chat")
async def kakao_chat(request: Request):
    
    # 터미널 로그
    print("\n========== [카카오 요청 수신됨!] ==========")
    data = await request.json()
    utterance = data.get("userRequest", {}).get("utterance", "")
    
    print(f"사용자 질문: {utterance}")

    # 1. 제미나이에게 질문 분류 시키기 (1~2초 소요)
    category = await classify_question(utterance)
    print(f"제미나이 분류 결과: {category}번 카테고리")

    # [중요] 내 ngrok 주소를 여기에 적어주세요! (마지막에 '/' 붙이지 마세요)
    # 예: "https://abcdef.ngrok-free.dev"
    MY_NGROK_URL = "https://histopathological-grady-slabbery.ngrok-free.dev"

    # 2. 분류 결과에 따라 답변 구성
    if category == "1":
        # 알바비 미지급 상황 -> 미리 준비한 만화 URL을 보냅니다.
        comic_url = f"{MY_NGROK_URL}/images/alba.png"
        print(f"-> 만화 전송! 주소: {comic_url}")
        
        # 카카오톡 'simpleImage' (이미지 전송) 포맷
        response_data = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleImage": {
                            "imageUrl": comic_url,
                            "altText": "알바비 미지급 대처법 4컷 만화"
                        }
                    }
                ]
            }
        }
    else:
        # 그 외 상황 -> 기존 텍스트 답변
        response_data = {
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": "현재 알바비 미지급 상황에 대해서만 만화 답변이 가능합니다. 다른 질문은 준비 중입니다."}}]
            }
        }

    print("===========================================\n")
    return response_data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)