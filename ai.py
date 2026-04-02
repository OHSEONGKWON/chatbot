import os
import uvicorn
import httpx
import asyncio
import sqlite3
import urllib.parse
import random
from datetime import datetime
from fastapi import FastAPI, Request, BackgroundTasks
from google import genai
from google.genai import types
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
app = FastAPI()

# --- [설정 및 API 키] ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "models/gemini-2.5-flash"

# --- [1. 데이터베이스 초기화] ---
def init_db():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            question TEXT,
            answer TEXT,
            timestamp DATETIME
        )
    """)
    conn.commit()
    conn.close()

def save_chat_to_db(user_id, question, answer):
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO chat_logs (user_id, question, answer, timestamp) VALUES (?, ?, ?, ?)",
            (user_id, question, answer, now)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ DB 저장 에러: {e}")

# 서버 시작 시 DB 초기화
init_db()

# --- [2. Gemini 답변 생성 함수] ---
async def get_gemini_answer(question):
    try:
        prompt = (
            f"당신은 성범죄 법률 전문 AI입니다. 상황에 맞는 법조항이나 판례를 바탕으로 상세히 답변하세요.\n"
            f"마지막에는 반드시 '본 답변은 참고용이며 법적 효력이 없습니다.'를 포함하세요.\n\n"
            f"질문: {question}"
        )
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=1000, temperature=0.3)
        )
        return response.text.strip() if response.text else "답변을 생성할 수 없습니다."
    except Exception as e:
        print(f"❌ Gemini 에러: {e}")
        return "상담 서비스 점검 중입니다."

# --- [3. 무료 이미지 URL 생성 함수 (Pollinations.ai)] ---
async def get_image_url(utterance: str):
    try:
        # Gemini에게 그림을 그리기 위한 짧은 영어 묘사를 부탁함
        refine_prompt = (
            f"Create a very short English image prompt (under 15 words) "
            f"representing this situation: '{utterance}'. "
            f"Style: Soft webtoon art, professional, comforting, justice theme. No text."
        )
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL_ID,
            contents=refine_prompt
        )
        
        raw_prompt = response.text.strip() if response.text else "justice scales and hope"
        encoded_prompt = urllib.parse.quote(raw_prompt)
        
        # 랜덤 Seed를 넣어 매번 다른 그림이 나오게 함
        seed = random.randint(1, 1000)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=800&height=400&nologo=true&seed={seed}"
        print(f"🎨 생성된 이미지 주소: {image_url}")
        return image_url
    except Exception as e:
        print(f"❌ 이미지 생성 실패: {e}")
        return "https://t1.kakaocdn.net/kakaocorp/pw/notice/202311/notice_231122_1.png"

# --- [4. 비동기 콜백 처리 함수] ---
async def process_and_callback(callback_url: str, utterance: str, user_id: str):
    try:
        # 1. 답변 생성 및 이미지 URL 생성 병렬 처리 가능 (시간 단축)
        answer, image_url = await asyncio.gather(
            get_gemini_answer(utterance),
            get_image_url(utterance)
        )
        
        # 2. DB 저장
        save_chat_to_db(user_id, utterance, answer)

        # 3. 카카오톡 전송 (basicCard 형태)
        callback_payload = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "basicCard": {
                            "title": "⚖️ 성범죄 법률 AI 분석 결과",
                            "description": answer[:400] + "...", # 카드 설명 400자 제한
                            "thumbnail": {
                                "imageUrl": image_url
                            },
                            "buttons": [
                                {
                                    "action": "webLink",
                                    "label": "상세 내용 웹에서 보기",
                                    "webLinkUrl": "https://your-server.com/history" # 나중에 Vue.js 연동
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        async with httpx.AsyncClient() as http_client:
            await http_client.post(callback_url, json=callback_payload)
            print(f"✅ 콜백 전송 완료 (User: {user_id[:8]})")
            
    except Exception as e:
        print(f"❌ 콜백 처리 에러: {e}")

# --- [5. 메인 엔드포인트] ---
@app.post("/api/chat")
async def kakao_chat(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    user_id = data.get("userRequest", {}).get("user", {}).get("id", "guest")
    utterance = data.get("userRequest", {}).get("utterance", "")
    callback_url = data.get("userRequest", {}).get("callbackUrl")

    if callback_url:
        background_tasks.add_task(process_and_callback, callback_url, utterance, user_id)
        return {
            "version": "2.0",
            "useCallback": True,
            "template": {"outputs": []}
        }
    
    return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "상담을 시작합니다."}}]}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)