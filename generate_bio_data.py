import json
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# ⚙️ 설정부
# ==========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_FILE = "data/mock_data/law_hallu.jsonl"
OUTPUT_FILE = "data/mock_data/law_hallu_bio.jsonl"

# 동시에 처리할 스레드 개수 (사용자 티어에 따라 5~20 사이 조절 권장)
MAX_WORKERS = 10 

client = OpenAI(api_key=OPENAI_API_KEY)
def clean_legal_text(text):
    """
    법률 RAG 데이터의 노이즈(특수문자, 중복 공백, 불필요한 태그)를 제거합니다.
    """
    if not text:
        return ""

    # 1. 의미 없는 대괄호 및 매뉴얼 태그 제거
    # [매뉴얼: ], [ ], [] 등을 제거합니다.
    text = re.sub(r'\[\s*매뉴얼\s*:\s*\]', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    
    # 2. 깨진 특수문자 및 불필요한 기호 제거 (예: 󰀂)
    # 한글, 숫자, 영문, 기본 문장부호(. , ( ) [ ] - : /)만 남기고 나머지는 공백으로 치환
    text = re.sub(r'[^\w\s\d\.\,\(\)\[\]\-\:\/\%\+\>\<\=\&]', ' ', text)
    
    # 3. 중복된 제목이나 페이지 번호로 추정되는 패턴 정제 (선택 사항)
    # 동일한 단어가 3번 이상 반복되면 하나로 줄이는 등의 로직을 넣을 수 있지만, 
    # 법률 데이터는 중복 단어가 중요할 수 있으므로 여기서는 공백 위주로 정제합니다.

    # 4. 연속된 공백 및 줄바꿈을 하나의 공백으로 통합
    # \n, \r, \t 등을 모두 포함하여 정제합니다.
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_entities_with_gpt(text):
    """(기존과 동일) GPT를 사용하여 개체명 추출"""
    prompt = f"""
당신은 한국어 법률 개체명 인식(NER) 전문가입니다.
아래 원문에서 다음 6가지 범주를 **원문에 적힌 글자 그대로** 추출하세요.

1. LAW: 법령명, 조항 (예: 근로기준법, 제5조)
2. PENALTY: 형벌 (예: 징역 1년, 벌금)
3. AMOUNT: 금액 (예: 5천만 원)
4. DATE: 날짜, 기간 (예: 30일 이내, 2025.12.31)
5. ORG: 기관 (예: 중앙노동위원회)
6. CRIME: 죄명 (예: 부당해고, 특수공갈)

**주의사항**:
- 원문의 띄어쓰기까지 완벽하게 일치해야 합니다.
- 반드시 JSON 형식 {{"entities": [{{"word": "...", "label": "..."}}]}}으로만 응답하세요.

원문:
{text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[{"role": "system", "content": "You are a precise NER assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content).get("entities", [])
    except Exception as e:
        return []

def align_to_character_bio(text, entities):
    """(기존과 동일) 글자 단위 BIO 태깅"""
    tokens = list(text)
    ner_tags = ["O"] * len(tokens)
    entities = sorted(entities, key=lambda x: len(x['word']), reverse=True)
    
    for ent in entities:
        word, label = ent['word'], ent['label']
        if not word: continue
        for match in re.finditer(re.escape(word), text):
            start, end = match.start(), match.end()
            if all(tag == "O" for tag in ner_tags[start:end]):
                ner_tags[start] = f"B-{label}"
                for i in range(start + 1, end):
                    ner_tags[i] = "O" if tokens[i] == " " else f"I-{label}"
    return tokens, ner_tags

def process_single_line(line):
    record = json.loads(line)
    raw_text = record.get("text", "")
    
    cleaned_text = clean_legal_text(raw_text)
    
    if not cleaned_text: 
        return None
    
    entities = extract_entities_with_gpt(cleaned_text)
    tokens, tags = align_to_character_bio(cleaned_text, entities)
    
    return {
        "chunk_id": record.get("chunk_id"),
        "original_text": cleaned_text,  # 정제된 텍스트를 기준으로 저장
        "tokens": tokens,
        "ner_tags": tags
    }

def run_parallel_processing():
    print(f"🚀 [System] 병렬 처리 시작 (Workers: {MAX_WORKERS})...")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    # ThreadPoolExecutor를 사용해 병렬 처리 수행
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 진행 상황을 확인하기 위해 future 객체 활용
        future_to_line = {executor.submit(process_single_line, line): line for line in lines}
        
        for i, future in enumerate(as_completed(future_to_line)):
            result = future.result()
            if result:
                results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"✅ {i + 1}/{len(lines)} 건 처리 완료...")

    # 결과 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print(f"\n🎉 [System] 모든 작업 완료! 결과 저장됨: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_parallel_processing()