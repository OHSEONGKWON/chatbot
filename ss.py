import os
from dotenv import load_dotenv

# .env 파일을 읽어옵니다
load_dotenv()

# 환경 변수에서 키를 가져옵니다
api_key = os.getenv("GEMINI_API_KEY")

print(f"현재 로드된 API 키: {api_key}")