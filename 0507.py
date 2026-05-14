import os
import io
import json
import base64
import uvicorn
import requests
import datetime
from PIL import Image
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# 정적 파일(이미지) 서빙 설정
os.makedirs("output", exist_ok=True)
try:
    app.mount("/images", StaticFiles(directory="output"), name="images")
except RuntimeError:
    pass

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY"))


def is_drawing_request(text: str) -> bool:
    """사용자 발화에 그림/만화 관련 키워드가 있는지 확인"""
    keywords = ["만화", "그림", "웹툰", "그려", "묘사", "컷"]
    return any(keyword in text for keyword in keywords)


def generate_storyboard(question: str) -> dict:
    try:
        system_prompt = """당신은 한국 법률 전문 웹툰 스토리보드 작가입니다.
다음 상담 내용을 바탕으로 4컷 만화를 구성하고 JSON으로 답하세요.

[구성 지침]
1. 1컷 (상황 발생): 사건이 일어난 구체적인 장소와 인물을 묘사하세요.
2. 인물 및 감정: 상담 내용의 맥락에 맞는 성별과 인상을 자유롭게 설정하고, 표정을 생생하게 묘사하세요.
   (주의: 1~4컷 내내 주인공의 외모, 헤어스타일, 옷차림을 완벽하게 동일하게 유지하도록 한국어로 아주 상세히 고정해두세요. 예: 파란색 정장을 입고 둥근 안경을 쓴 30대 한국 남성)
3. 시각적 동작: '설명한다' 대신 '경찰서 건물 앞에 서있다', '스마트폰으로 증거를 캡처한다' 등 시각적 행위 위주로 묘사하세요.
4. 텍스트 삽입: 각 컷 이미지 안에 다음 두 가지 한국어 텍스트를 자연스럽게 배치합니다.
   - top_text: 이미지 상단 소제목 (10자 이내, 예: "① 상황 발생")
   - bottom_text: 이미지 하단 조언/설명 (25자 이내, 핵심만 간결하게)

[중요 안전 지침]
- 민감한 사건(성희롱, 폭력, 학대 등)도 법률 교육 맥락에서 **간접적이고 비폭력적**으로만 묘사하세요.
- 실제 폭행, 신체 접촉, 성적 묘사는 절대 포함하지 마세요.
- 대신 피해자의 고통스러운 표정, 증거물, 법률 상담 장면 등으로 상황을 암시하세요.
- 예: "직장 상사와의 부당한 대우" → "회의실에서 상사에게 부정적 반응 받는 직원", "경찰서 방문하여 상담받는 장면"

[출력 형식 (JSON)]
{
  "panels": [
    {
      "top_text": "이미지 상단 소제목 (10자 이내)",
      "bottom_text": "이미지 하단 조언/설명 (25자 이내)",
      "image_prompt": "dall-e-3용 한국어 프롬프트 (주인공 외모 일관성 유지, 감정, 배경, 시각적 동작 상세 묘사. 폭력/성적 표현 없이)"
    }
  ]
}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"상담 내용: {question}"},
            ],
            max_tokens=1500,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as error:
        print(f"[GPT Error]: {error}")
        return {}


def sanitize_prompt(text: str) -> str:
    """민감한 표현을 법률 교육 맥락의 비폭력적 표현으로 변환"""
    replacements = {
        "성폭행": "부당한 대우",
        "폭행": "갈등",
        "강간": "성적 피해",
        "구타": "신체 피해",
        "학대": "부당한 대우",
        "성희롱": "직장 내 성희롱",
        "성적": "부적절한",
        "나체": "의상 입은",
        "누드": "의상 입은",
    }
    result = text
    for sensitive, safe in replacements.items():
        result = result.replace(sensitive, safe)
    return result


def build_image_prompt(panel: dict) -> str:
    top = panel.get("top_text", "")
    bottom = panel.get("bottom_text", "")
    image_prompt = sanitize_prompt(panel.get("image_prompt", ""))
    
    return (
        f"{image_prompt}. "
        f"한국 웹툰 스타일 일러스트. "
        f"법률 상담 및 권리 보호 교육 맥락의 그림. "
        f"중요: 이미지 안에 다음 한국어 텍스트를 반드시 포함하세요. "
        f"이미지 상단 중앙에는 어두운 반투명 배너 안에 '{top}' 이라는 텍스트를 선명한 웹툰 폰트로 적어주세요. "
        f"이미지 하단 중앙에는 깔끔한 흰색 말풍선이나 캡션 박스 안에 '{bottom}' 이라는 텍스트를 적어주세요. "
        f"모든 한국어 글자는 오타 없이 완벽하고 또렷하게 읽혀야 합니다. "
        f"1컷부터 4컷까지 주인공의 외모와 옷차림을 완벽하게 똑같이 유지하세요. "
        f"폭력, 성적 표현, 신체 접촉 묘사는 없어야 합니다."
    )


def url_to_pil(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def b64_to_pil(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def build_combined_4cut_prompt(panels: list) -> str:
    """4개 패널을 2x2 웹툰 그리드 스타일의 통합 프롬프트로 변환"""
    panel_descriptions = []
    for i, panel in enumerate(panels, 1):
        sanitized = sanitize_prompt(panel.get("image_prompt", ""))
        top = panel.get("top_text", "")
        bottom = panel.get("bottom_text", "")
        panel_descriptions.append(
            f"[패널 {i}] 상단텍스트:'{top}' 하단텍스트:'{bottom}' 장면:{sanitized}"
        )
    
    combined_desc = "\n".join(panel_descriptions)
    return (
        f"한국 웹툰 스타일 2x2 그리드 레이아웃 (4컷 만화). 법률 상담 및 권리 보호 교육 목적.\n"
        f"다음 4개의 독립적인 장면을 2x2 그리드(좌상단→우상단→좌하단→우하단)로 배치하세요:\n\n"
        f"{combined_desc}\n\n"
        f"각 패널마다:\n"
        f"- 상단 중앙에 어두운 배너 안에 지정된 상단 텍스트 작성\n"
        f"- 하단 중앙에 희색 박스/말풍선 안에 지정된 하단 텍스트 작성\n"
        f"- 1~4 패널의 주인공은 외모/옷차림이 완벽하게 동일\n"
        f"- 폭력, 성적 표현, 신체 접촉 묘사 없음\n"
        f"모든 한국어 텍스트는 오타 없이 선명하게 읽혀야 함."
    )


def generate_4cut_combined(panels: list) -> bytes:
    """4개 패널을 한 번의 API 호출로 2x2 그리드 이미지로 생성"""
    print("▶ 4컷 이미지 한 번에 생성 중...")
    try:
        prompt = build_combined_4cut_prompt(panels)
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        img = response.data[0]
        if img.url:
            pil_img = url_to_pil(img.url)
        elif img.b64_json:
            pil_img = b64_to_pil(img.b64_json)
        else:
            raise RuntimeError("이미지 응답에 url 또는 b64_json이 없습니다.")
        
        output = io.BytesIO()
        pil_img.save(output, format="JPEG", quality=90)
        output.seek(0)
        print("▶ 4컷 이미지 완성")
        return output.read()
    except Exception as error:
        print(f"[4Cut Generation Error]: {error}")
        raise


def upload_image_to_imgbb(image_bytes: bytes) -> str:
    """합쳐진 이미지를 imgbb에 업로드하고 URL 반환"""
    api_key = os.getenv("IMGBB_API_KEY")
    if not api_key:
        raise ValueError("IMGBB_API_KEY 환경변수가 설정되지 않았습니다.")

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    resp = requests.post(
        "https://api.imgbb.com/1/upload",
        data={"key": api_key, "image": encoded},
        timeout=30,
    )
    result = resp.json()
    if result.get("success"):
        url = result["data"]["url"]
        print(f"▶ imgbb 업로드 완료: {url}")
        return url
    else:
        raise Exception(f"imgbb 업로드 실패: {result}")


def send_error_callback(callback_url: str, message: str = "처리에 실패했습니다. 다시 시도해 주세요."):
    try:
        requests.post(
            callback_url,
            json={"version": "2.0", "template": {"outputs": [{"simpleText": {"text": message}}]}},
            timeout=10,
        )
    except Exception as e:
        print(f"[Error Callback 전송 실패]: {e}")


def process_text_callback(utterance: str, callback_url: str):
    """만화 요청이 아닐 때 일반 텍스트로 법률 상담 답변을 생성하여 콜백 전송"""
    print(f"\n▶ [백그라운드] '{utterance}' 일반 텍스트 답변 생성 시작...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 친절한 한국 법률 상담 챗봇입니다. 사용자의 질문에 대해 법률적 조언을 알기 쉽게 텍스트로 제공하세요."},
                {"role": "user", "content": utterance}
            ],
            max_tokens=800
        )
        answer_text = response.choices[0].message.content.strip()

        response_data = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": answer_text
                        }
                    }
                ]
            }
        }
        
        print("▶ [성공] 일반 텍스트 답변 전송 중...")
        requests.post(callback_url, json=response_data, timeout=30)
        
    except Exception as e:
        print(f"[ERROR] 일반 텍스트 답변 생성 실패: {e}")
        send_error_callback(callback_url, "답변을 생성하는 중 오류가 발생했습니다.")


def process_and_callback(utterance: str, callback_url: str):
    try:
        print(f"\n▶ [백그라운드] '{utterance}' 스토리보드 기획 시작...")

        # 1. GPT-4o 스토리보드 기획
        storyboard = generate_storyboard(utterance)
        panels = storyboard.get("panels", [])

        if not panels or len(panels) != 4:
            print("▶ 스토리보드 생성 실패")
            send_error_callback(callback_url, "만화 생성에 실패했습니다. 다시 시도해 주세요.")
            return

        print("▶ 4컷 스토리보드 완료! 이미지 생성 시작...")

        # 2. 모델로 4컷을 한 번에 2x2 그리드 이미지로 생성
        try:
            merged_bytes = generate_4cut_combined(panels)
        except Exception as error:
            print(f"▶ 4컷 이미지 생성 실패: {error}")
            send_error_callback(callback_url, "만화 생성에 실패했습니다. 다시 시도해 주세요.")
            return

        # 3. imgbb 서버에 이미지 업로드
        try:
            print("▶ imgbb 서버에 이미지 업로드 중...")
            image_url = upload_image_to_imgbb(merged_bytes)
        except Exception as e:
            print(f"▶ 이미지 업로드 실패: {e}")
            send_error_callback(callback_url, "이미지 호스팅 서버 업로드에 실패했습니다.")
            return

        # 4. 로컬 저장 (갤러리용)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"4cut_{timestamp}.jpeg"
        output_path = f"output/{filename}"
        with open(output_path, "wb") as f:
            f.write(merged_bytes)
        print(f"▶ 로컬 백업 저장 완료: {output_path}")

        # 5. 카카오톡에 이미지 전송
        response_data = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleImage": {
                            "imageUrl": image_url,
                            "altText": "4컷 법률 만화",
                        }
                    }
                ]
            },
        }

        print(f"▶ 카카오톡으로 전송되는 이미지 URL: {image_url}")
        print("▶ [성공] 4컷 만화 이미지 전송 중...")
        
        # 카카오톡 콜백 전송
        response = requests.post(callback_url, json=response_data, timeout=30)
        print(f"▶ ✓ 콜백 응답 상태: {response.status_code}")
        
        # ==========================================
        # 추가된 에러 확인 로직
        # ==========================================
        if response.status_code != 200:
            print("\n🚨 [카카오 400 에러 상세 분석] 🚨")
            print(f"▶ 카카오 거절 사유: {response.text}")
            print(f"▶ 보낸 데이터 확인: {json.dumps(response_data, ensure_ascii=False)}")
            print("=====================================\n")

    except Exception as error:
        print(f"\n[ERROR] 백그라운드 처리 중 예외 발생:")
        import traceback
        traceback.print_exc()
        send_error_callback(callback_url, "만화 생성에 실패했습니다. 다시 시도해 주세요.")


@app.post("/api/chat")
async def kakao_chat(request: Request, background_tasks: BackgroundTasks):
    print("\n========== [요청 수신] ==========")
    try:
        data = await request.json()
        utterance = data.get("userRequest", {}).get("utterance", "")
        callback_url = data.get("userRequest", {}).get("callbackUrl")

        print(f"사용자: {utterance}")

        if callback_url:
            if is_drawing_request(utterance):
                background_tasks.add_task(process_and_callback, utterance, callback_url)
            else:
                background_tasks.add_task(process_text_callback, utterance, callback_url)
            return {"version": "2.0", "useCallback": True}

        return {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": "콜백 URL 오류"}}]},
        }
    except Exception as e:
        print(f"[ERROR] 요청 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": "오류 발생"}}]}
        }

@app.get("/gallery", response_class=HTMLResponse)
async def view_gallery():
    """서버에 저장된 생성 이미지들을 모아보는 웹 페이지"""
    html_content = """
    <html>
    <head>
        <title>생성된 4컷 만화 갤러리</title>
        <style>
            body { font-family: 'Malgun Gothic', sans-serif; background-color: #f4f4f9; padding: 20px; }
            h1 { text-align: center; color: #333; }
            .gallery { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
            .card { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
            .card img { max-width: 300px; border-radius: 5px; cursor: pointer; transition: transform 0.2s; }
            .card img:hover { transform: scale(1.05); }
            .filename { margin-top: 10px; font-size: 14px; color: #555; }
        </style>
    </head>
    <body>
        <h1>생성된 4컷 만화 갤러리</h1>
        <div class="gallery">
    """
    
    if os.path.exists("output"):
        files = sorted(os.listdir("output"), reverse=True)
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                html_content += f"""
                <div class="card">
                    <a href="/images/{file}" target="_blank">
                        <img src="/images/{file}" alt="{file}">
                    </a>
                    <div class="filename">{file}</div>
                </div>
                """
                
    html_content += """
        </div>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    print("서버 시작: http://0.0.0.0:8000")
    print("▶ 갤러리 확인: http://localhost:8000/gallery")
    uvicorn.run(app, host="0.0.0.0", port=8000)