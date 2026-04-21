import json
import re
import os
import glob
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# ⚙️ 설정부
# ==========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_DIR = "data/real_data/"          # 팀원들이 준 파일들이 있는 폴더
OUTPUT_DIR = "data/real_bio_data/" # 결과가 저장될 폴더

# 병렬 처리 설정
MAX_FILE_WORKERS = 3   # 동시에 처리할 '파일' 개수 (CPU 코어 수에 맞게 조절)
MAX_LINE_WORKERS = 10  # 파일당 동시에 처리할 '라인(API 호출)' 개수

# API 안정성 설정
REQUEST_TIMEOUT_SEC = 45
REQUEST_MAX_RETRIES = 3
REQUEST_RETRY_BACKOFF_SEC = 2

# 학습 데이터 품질 설정
DROP_LINES_WITHOUT_ENTITY = True
MAX_TOKENS_PER_SAMPLE = 512

client = OpenAI(api_key=OPENAI_API_KEY)

# [정제/추출/태깅 함수들은 기존과 동일]
def clean_legal_text(text):
    if not text: return ""
    text = re.sub(r'\[\s*매뉴얼\s*:\s*\]|\[\s*\]', '', text)
    text = re.sub(r'[^\w\s\d\.\,\(\)\[\]\-\:\/\%\+\>\<\=\&]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_entities_with_gpt(text):
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
    for attempt in range(REQUEST_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": "You are a precise NER assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.1,
                timeout=REQUEST_TIMEOUT_SEC
            )
            return json.loads(response.choices[0].message.content).get("entities", [])
        except Exception:
            if attempt < REQUEST_MAX_RETRIES - 1:
                time.sleep(REQUEST_RETRY_BACKOFF_SEC * (attempt + 1))
            else:
                return []

def align_to_character_bio(text, entities):
    # [기존 BIO 태깅 로직 그대로]
    tokens = list(text)
    ner_tags = ["O"] * len(tokens)
    entities = sorted(entities, key=lambda x: len(x['word']), reverse=True)
    for ent in entities:
        word, label = ent['word'], ent['label']
        for match in re.finditer(re.escape(word), text):
            start, end = match.start(), match.end()
            if all(tag == "O" for tag in ner_tags[start:end]):
                ner_tags[start] = f"B-{label}"
                for i in range(start + 1, end):
                    ner_tags[i] = "O" if tokens[i] == " " else f"I-{label}"
    return tokens, ner_tags

def process_single_line(line):
    # [기존 라인 처리 로직 그대로]
    try:
        record = json.loads(line)
        raw_text = record.get("text", "")
        cleaned_text = clean_legal_text(raw_text)
        if not cleaned_text:
            return None
        entities = extract_entities_with_gpt(cleaned_text)
        tokens, tags = align_to_character_bio(cleaned_text, entities)
        return {"chunk_id": record.get("chunk_id"), "tokens": tokens, "ner_tags": tags}
    except Exception:
        return None

def split_long_sample(sample, max_tokens=MAX_TOKENS_PER_SAMPLE):
    tokens = sample["tokens"]
    tags = sample["ner_tags"]
    chunk_id = sample.get("chunk_id", "unknown_chunk")

    if len(tokens) <= max_tokens:
        return [sample]

    split_samples = []
    total = len(tokens)
    part = 1
    for start in range(0, total, max_tokens):
        end = min(start + max_tokens, total)
        sub_tokens = tokens[start:end]
        sub_tags = tags[start:end]

        if sub_tags and sub_tags[0].startswith("I-"):
            sub_tags[0] = "B-" + sub_tags[0][2:]

        split_samples.append({
            "chunk_id": f"{chunk_id}__part{part}",
            "tokens": sub_tokens,
            "ner_tags": sub_tags,
        })
        part += 1

    return split_samples

def process_file(file_path, position):
    file_name = os.path.basename(file_path)
    rel_path = os.path.relpath(file_path, INPUT_DIR)
    rel_dir = os.path.dirname(rel_path)
    file_stem, _ = os.path.splitext(file_name)
    output_dir = os.path.join(OUTPUT_DIR, rel_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{file_stem}_bio.jsonl")
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    success_count = 0
    failed_count = 0
    skipped_no_entity_count = 0
    skipped_duplicate_id_count = 0
    split_sample_count = 0
    seen_chunk_ids = set()
    # 파일 내부 진행 바 (position을 사용하여 바가 겹치지 않게 함)
    pbar = tqdm(total=len(lines), desc=f"📄 {file_name[:15]}", position=position + 1, leave=False)

    with open(output_path, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_LINE_WORKERS) as executor:
            future_to_line = {executor.submit(process_single_line, line): line for line in lines}
            for future in as_completed(future_to_line):
                try:
                    res = future.result()
                except Exception:
                    res = None

                if res:
                    if DROP_LINES_WITHOUT_ENTITY and all(tag == "O" for tag in res["ner_tags"]):
                        skipped_no_entity_count += 1
                        pbar.update(1)
                        pbar.set_postfix(ok=success_count, fail=failed_count, skip=skipped_no_entity_count)
                        continue

                    base_chunk_id = str(res.get("chunk_id", "unknown_chunk"))
                    if base_chunk_id in seen_chunk_ids:
                        skipped_duplicate_id_count += 1
                        pbar.update(1)
                        pbar.set_postfix(ok=success_count, fail=failed_count, skip=skipped_no_entity_count + skipped_duplicate_id_count)
                        continue

                    seen_chunk_ids.add(base_chunk_id)

                    split_samples = split_long_sample(res)
                    if len(split_samples) > 1:
                        split_sample_count += len(split_samples) - 1

                    for sample in split_samples:
                        out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        success_count += 1

                    if success_count % 100 == 0:
                        out_f.flush()
                else:
                    failed_count += 1

                pbar.update(1)
                pbar.set_postfix(
                    ok=success_count,
                    fail=failed_count,
                    skip=skipped_no_entity_count + skipped_duplicate_id_count,
                    split=split_sample_count,
                )
    
    pbar.close()

    return (
        f"✅ {file_name} 완료 "
        f"(ok={success_count}, fail={failed_count}, "
        f"skip_no_entity={skipped_no_entity_count}, skip_dup_id={skipped_duplicate_id_count}, split_added={split_sample_count})"
    )

def run_all_files():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    file_list = glob.glob(os.path.join(INPUT_DIR, "**", "*.jsonl"), recursive=True)
    
    print(f"🔥 총 {len(file_list)}개의 파일 처리를 시작합니다.")
    print("--------------------------------------------------")

    # 전체 파일 진행률 표시 (메인 바)
    with ProcessPoolExecutor(max_workers=MAX_FILE_WORKERS) as executor:
        # 파일별로 고유한 위치(position)를 할당하여 바가 쌓이게 함
        futures = [executor.submit(process_file, fp, i) for i, fp in enumerate(file_list)]
        
        # 전체 파일 완료 체크용 메인 tqdm
        for future in tqdm(as_completed(futures), total=len(file_list), desc="TOTAL PROGRESS", position=0):
            try:
                print(future.result())
            except Exception as e:
                print(f"❌ 파일 처리 실패: {e}")

    # 터미널 커서 위치 초기화 (바가 많을 경우 대비)
    print("\n" * (MAX_FILE_WORKERS + 1))
    print("🎉 모든 데이터 전처리가 완벽하게 끝났습니다!")

if __name__ == "__main__":
    run_all_files()