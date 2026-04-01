import os
from dotenv import load_dotenv
from google import genai

# 1. .env 파일의 환경 변수를 로드합니다.
load_dotenv()

# 2. .env 파일에 설정된 변수명(예: GEMINI_API_KEY)을 가져옵니다.
# 만약 .env에 API_KEY라고 적으셨다면 os.getenv('API_KEY')로 수정하세요.
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    print("오류: .env 파일에서 API 키를 찾을 수 없습니다. 변수명을 확인해주세요.")
else:
    # 3. 클라이언트 생성
    client = genai.Client(api_key=api_key)

    print(f"--- '{api_key[:5]}***' 키로 접근 가능한 모델 목록 ---")

    try:
        # 4. 모델 리스트 출력
        for model in client.models.list():
            # 실제 텍스트 생성 기능을 지원하는 모델만 필터링
            if 'generateContent' in model.supported_actions:
                print(f"모델명: {model.name}")
    except Exception as e:
        print(f"오류 발생: {e}")