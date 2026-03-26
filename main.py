from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

# 앞서 카카오 스킬 URL에 적었던 경로와 동일하게 맞춰줍니다.
@app.post("/api/chat")
async def kakao_chat(request: Request):
    # 1. 카카오톡에서 보내는 복잡한 JSON 데이터 전체를 받습니다.
    kakao_data = await request.json()
    
    # 2. 그중에서 사용자가 실제로 입력한 텍스트(질문)만 쏙 뽑아냅니다.
    user_message = kakao_data["userRequest"]["utterance"]
    print(f"사용자가 보낸 메시지: {user_message}")

    # =========================================================
    # 3. LLM & RAG 로직 (나중에 이 부분에 코드를 채워 넣을 겁니다!)
    # - 외부 문서 검색
    # - LLM(Gemini, OpenAI 등)에게 프롬프트 전송 및 답변 받기
    # =========================================================
    
    # 지금은 LLM이 없으니, 메아리처럼 따라 하는 임시 답변을 만듭니다.
    llm_reply = f"'{user_message}'라고 하셨군요! 서버 연동 성공입니다."

    # 4. 카카오톡이 알아들을 수 있는 정해진 양식(SimpleText)으로 포장합니다.
    response_data = {
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
    
    # 5. 카카오 서버로 포장된 답변을 돌려보냅니다.
    return response_data

if __name__ == "__main__":
    # 포트 번호 8000번으로 서버를 실행합니다.
    uvicorn.run(app, host="0.0.0.0", port=8000)