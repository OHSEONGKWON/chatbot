import os
import uvicorn
import httpx
import google.generativeai as genai
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()

# 1. HTTP 세션 관리
http_client = httpx.AsyncClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists("static"):
        os.makedirs("static")
        print("📁 static 폴더 확인 완료")
    yield
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# API 키 및 모델 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# 💡 준현님 리스트에서 확인된 모델 ID로 수정
TEXT_MODEL_ID = "models/gemini-2.5-flash" 
IMAGE_MODEL_ID = "models/gemini-2.5-flash-image" 

# --- [로직 함수] ---

async def get_comic_image_prompt(question):
    """Gemini를 사용해 네컷만화용 영어 프롬프트 생성"""
    try:
        print(f"🔍 [1/4] 프롬프트 생성 시작")
        text_model = genai.GenerativeModel(TEXT_MODEL_ID)
        instruction = (
            "Create a detailed 4-panel comic strip prompt in digital webtoon style. "
            "The comic should visually explain the following situation. "
            "Divide the image into 4 panels (2x2 grid). No text in panels."
        )
        response = await text_model.generate_content_async(f"{instruction}\nSituation: {question}")
        return response.text.strip()
    except Exception as e:
        print(f"❌ 프롬프트 에러: {e}")
        return None

async def generate_legal_comic(user_utterance):
    """나노바나나로 이미지 생성 및 로컬 저장"""
    refined_prompt = await get_comic_image_prompt(user_utterance)
    if not refined_prompt: return None

    try:
        print(f"🚀 [2/4] 나노바나나 이미지 생성 중...")
        image_model = genai.GenerativeModel(IMAGE_MODEL_ID)
        response = await image_model.generate_content_async(refined_prompt)
        
        print("📦 [3/4] 모델 응답 수신 완료")
        part = response.candidates[0].content.parts[0]
        
        if hasattr(part, 'inline_data'):
            image_data = part.inline_data.data
            file_path = "static/legal_comic.png"
            with open(file_path, "wb") as f:
                f.write(image_data)
            
            # 💡 학교 공인 IP 설정
            base_url = "http://203.255.63.29:8000" 
            full_url = f"{base_url}/{file_path}"
            
            print(f"✅ [4/4] 저장 완료: {full_url}")
            return full_url
        return None
    except Exception as e:
        print(f"🔥 이미지 생성 단계 에러: {e}")
        return None

async def process_and_callback(callback_url: str, utterance: str):
    """백그라운드 처리 및 카카오톡 콜백 전송"""
    print(f"📩 콜백 프로세스 시작")
    try:
        if any(kw in utterance for kw in ["그림", "만화", "그려줘", "시각화"]):
            image_url = await generate_legal_comic(utterance)
            
            if image_url:
                # 💡 [중요] 카카오 규격에 맞게 altText 필수 포함
                payload = {
                    "version": "2.0",
                    "template": {
                        "outputs": [
                            {
                                "simpleImage": {
                                    "imageUrl": f"{image_url}?v=1", # 캐싱 방지용 쿼리 스트링
                                    "altText": "요청하신 상황을 그린 네컷만화입니다."
                                }
                            }
                        ]
                    }
                }
            else:
                payload = {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "🎨 이미지 생성에 실패했습니다."}}]}}
        else:
            # 일반 텍스트 답변
            text_model = genai.GenerativeModel(TEXT_MODEL_ID)
            answer = await text_model.generate_content_async(utterance)
            payload = {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": answer.text.strip()[:500]}}]}}
        
        print(f"📤 카카오톡 콜백 전송 시도...")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        # 전송 시 timeout을 넉넉히 주어 안정성을 높입니다.
        res = await http_client.post(callback_url, json=payload, headers=headers, timeout=30.0)
        
        print(f"🏁 최종 결과: {res.status_code}")
        if res.status_code != 200:
            print(f"❗ 실패 사유: {res.text}")
            
    except Exception as e:
        print(f"🔥 최종 단계 예외: {str(e)}")

# --- [엔드포인트] ---

@app.post("/api/chat")
async def kakao_chat(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    utterance = data.get("userRequest", {}).get("utterance", "")
    callback_url = data.get("userRequest", {}).get("callbackUrl")

    if callback_url:
        background_tasks.add_task(process_and_callback, callback_url, utterance)
        return {
            "version": "2.0", 
            "useCallback": True, 
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "🎨 상황을 이해하기 쉽게 네컷만화로 그려드릴게요! 잠시만 기다려주세요. (약 15초)"
                        }
                    }
                ]
            }
        }
    return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "상태 확인 완료"}}]}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)