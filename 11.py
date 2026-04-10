import os
import google.generativeai as genai
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# API 키 설정
genai.configure(api_key=GEMINI_API_KEY)

# 현재 내 API 키로 사용 가능한 모델 리스트 출력
print("--- 사용 가능한 모델 리스트 ---")
for m in genai.list_models():
    # 이미지 생성 기능이 포함된 모델 위주로 확인
    if 'generateContent' in m.supported_generation_methods or 'generateImage' in m.supported_generation_methods:
        print(f"Name: {m.name}")