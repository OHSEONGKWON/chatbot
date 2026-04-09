import os
import uvicorn
import httpx
import google.generativeai as genai
from fastapi import FastAPI, Request, BackgroundTasks
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

# 1. HTTP 세션 재사용을 위한 전역 클라이언트 및 Lifespan 설정
http_client = httpx.AsyncClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await http_client.aclose() # 앱 종료 시 연결 닫기

app = FastAPI(lifespan=lifespan)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
MODEL_ID = "models/gemini-2.5-flash" 

async def get_gemini_answer(question):
    try:
        model = genai.GenerativeModel(MODEL_ID)
        
        # 💡 프롬프트 수정: 길이 제한 해제 및 완벽한 서술형 문장 강조
        prompt = (
            f"[System Instruction]\n"
            f"당신은 전문적이고 신뢰감 있는 법률 상담가입니다. 'AI로서~'와 같은 기계적인 서론을 절대 쓰지 말고 바로 본론으로 들어가세요.\n\n"
            f"[답변 작성 규칙]\n"
            f"1. 완벽한 문장 구사 (매우 중요): 답변 길이를 억지로 줄이거나 단답형(개조식)으로 요약하지 마세요. 시간이 걸리더라도 문맥이 자연스럽고 완성도 높은 완벽한 서술형 문장으로 상세히 작성해야 합니다.\n"
            f"2. 구조화 (소제목): 답변은 반드시 아래의 3가지 대괄호 소제목을 사용하여 구분하세요.\n"
            f"   - [ 판단 기준 ]: 해당 사안이 성립하기 위한 요건이나 판례상의 기준을 서술형으로 자세히 풀어쓰세요.\n"
            f"   - [ 주요 사례 ]: 이해를 돕기 위한 구체적인 예시를 포함하여 이야기하듯 설명하세요.\n"
            f"   - [ 관련 근거 ]: 답변의 바탕이 된 법령명만 제시, 벌금이나 형량은 생략하세요.\n"
            f"3. 가독성: 카카오톡 환경을 고려하여, 문단이 바뀔 때 적절히 줄바꿈을 사용하여 읽기 편하게 구성하세요.\n"
            f"4. 고정 면책 조항: 전체 답변의 맨 마지막 줄에 반드시 아래 문구를 토씨 하나 틀리지 않고 추가하세요.\n"
            f"   '※ AI 답변은 참고용이며 법적 효력이 없으므로, 정확한 판단은 법률 전문가와 상담하시기 바랍니다.'\n\n"
            f"질문: {question}"
        )
        
        # 2. 네이티브 비동기 메서드 사용 & 답변의 일관성을 높이기 위한 temperature 설정 추가
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2 # 답변이 너무 창의적으로 변하는 것을 방지 (0.0 ~ 0.3 추천)
            )
        )
        return response.text.strip()
    
    except Exception as e:
        print(f"[Gemini Error]: {e}")
        return "법률 데이터를 분석하는 중 오류가 발생했습니다."

async def process_and_callback(callback_url: str, utterance: str):
    try:
        answer = await get_gemini_answer(utterance)
        callback_payload = {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": answer}}]}
        }
        # 3. 전역 클라이언트를 사용하여 네트워크 연결 시간 단축
        await http_client.post(callback_url, json=callback_payload)
    except Exception as e:
        print(f"[Callback Error]: {e}")

@app.post("/api/chat")
async def kakao_chat(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    utterance = data.get("userRequest", {}).get("utterance", "")
    callback_url = data.get("userRequest", {}).get("callbackUrl")

    if callback_url:
        background_tasks.add_task(process_and_callback, callback_url, utterance)
        return {"version": "2.0", "useCallback": True, "template": {"outputs": []}}
    
    return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": "상담 대기 중..."}}]}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)