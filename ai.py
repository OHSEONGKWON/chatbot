import os
import uvicorn
import httpx
import asyncio
import urllib.parse
import random
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

# --- [1. Gemini 답변 생성 함수 (강제 자르기 제거 & 프롬프트 최적화)] ---
async def get_gemini_answer(question):
    try:
        # 프롬프트: 완벽한 문장으로 작성하되, 카톡 제한(1000자)을 넘지 않도록 900자 이내로 스스로 마무리하게 유도
        prompt = (
            f"당신은 성범죄 법률 전문 AI입니다. 관련 법조항과 판례를 바탕으로 상세하고 전문적인 답변을 제공하세요.\n"
            f"문장은 반드시 끝까지 완성되어야 하며, 중간에 끊기지 않게 자연스럽게 결론을 맺어주세요.\n"
            f"단, 카카오톡 메시지 전송 한계가 있으므로 전체 길이는 800~900자 사이로 조절해 주세요.\n"
            f"강조를 위한 별표(*)나 우물정(#) 같은 마크다운 특수기호는 절대 사용하지 마세요.\n"
            f"마지막에는 반드시 '\n\n본 답변은 참고용이며 법적 효력이 없습니다.'를 포함하세요.\n\n"
            f"질문: {question}"
        )
        # 문장을 끝까지 완성할 수 있도록 토큰은 아주 넉넉하게 1500으로 열어둡니다.
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=1500, temperature=0.3)
        )
        
        answer = response.text.strip() if response.text else "답변을 생성할 수 없습니다."
        
        # 카카오톡 파싱 오류를 일으킬 수 있는 마크다운 특수문자 강제 제거
        answer = answer.replace('*', '').replace('#', '')
        
        # 파이썬으로 무식하게 글자를 자르는 로직(answer[:limit])은 완전히 삭제했습니다!
        # AI가 만들어낸 완벽한 문장 그대로를 반환합니다.
                
        return answer
    except Exception as e:
        print(f"❌ Gemini 에러: {e}")
        return "상담 서비스 점검 중입니다."

# --- [2. 무료 이미지 URL 생성 함수 (Pollinations.ai)] ---
async def get_image_url(utterance: str):
    try:
        refine_prompt = (
            f"Create a short English image prompt (under 15 words) "
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
        
        seed = random.randint(1, 1000)
        return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=800&height=400&nologo=true&seed={seed}"
    except Exception as e:
        print(f"❌ 이미지 생성 실패: {e}")
        return None

# --- [3. 비동기 콜백 처리 함수 (단일 말풍선)] ---
async def process_and_callback(callback_url: str, utterance: str):
    try:
        # 이미지 키워드 체크
        image_keywords = ["그림", "그려줘", "묘사", "이미지", "상황", "어떤 모습", "어떤 상황"]
        should_generate_image = any(keyword in utterance for keyword in image_keywords)
        
        # 답변 및 이미지 병렬 생성
        answer, image_url = await asyncio.gather(
            get_gemini_answer(utterance),
            get_image_url(utterance) if should_generate_image else asyncio.sleep(0, result=None)
        )
        
        # 카카오톡 전송 응답 구성
        outputs = []
        
        if image_url:
            outputs.append({
                "basicCard": {
                    "title": "⚖️ 상황 묘사 및 법률 분석",
                    "description": "아래의 텍스트 답변을 확인해 주세요.",
                    "thumbnail": {"imageUrl": image_url}
                }
            })

        # 자르지 않은 완벽한 전체 문장을 말풍선에 담기
        outputs.append({
            "simpleText": {"text": answer}
        })

        callback_payload = {
            "version": "2.0",
            "template": {
                "outputs": outputs
            }
        }
        
        async with httpx.AsyncClient() as http_client:
            await http_client.post(callback_url, json=callback_payload)
            print(f"✅ 콜백 완료 (최종 텍스트 길이: {len(answer)}자)")
            
    except Exception as e:
        print(f"❌ 콜백 처리 에러: {e}")

# --- [4. 메인 엔드포인트] ---
@app.post("/api/chat")
async def kakao_chat(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    utterance = data.get("userRequest", {}).get("utterance", "")
    callback_url = data.get("userRequest", {}).get("callbackUrl")

    if callback_url:
        background_tasks.add_task(process_and_callback, callback_url, utterance)
        return {"version": "2.0", "useCallback": True, "template": {"outputs": []}}
    
    return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "상담을 시작합니다."}}]}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)