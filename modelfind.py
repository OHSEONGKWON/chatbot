import os
import ssl
import certifi
from google import genai
from dotenv import load_dotenv

# 1. SSL 인증서 검증 강제 우회 (핵심!)
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

# .env에서 API 키 가져오기
api_key = os.getenv("GEMINI_API_KEY")

try:
    print("🚀 구글 서버에서 사용 가능한 모델 목록을 가져오는 중...")
    # 여기서 genai.Client를 사용합니다.
    client = genai.Client(api_key=api_key)
    
    # 모델 리스트 가져오기
    models = client.models.list()
    
    print("\n✅ [사용 가능한 모델 목록]")
    print("-" * 40)
    for model in models:
        # 모델의 이름(ID)과 표시 이름을 출력합니다.
        print(f"모델 ID: {model.name}")
        print(f"표시 이름: {model.display_name}")
        print("-" * 20)
    print("-" * 40)
    print("위 목록에서 'gemini-1.5-flash' 같은 ID를 복사해서 사용하세요.")

except Exception as e:
    print(f"\n❌ 에러 발생: {e}")