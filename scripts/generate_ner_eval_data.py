"""
NER 평가 데이터 생성 스크립트

data/external_raw/legal_pdfs/cases/ 의 PDF 파일에서 텍스트를 추출하고,
OpenAI API를 사용하여 BIO 태깅을 수행합니다.

출력 형식 (기존 case_hallu_bio.jsonl 동일):
  {"chunk_id": "...", "tokens": [...], "ner_tags": [...]}

사용법:
  python scripts/generate_ner_eval_data.py
  python scripts/generate_ner_eval_data.py --max-pdfs 20 --max-chunks 200
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time

# Windows stdout UTF-8 강제 설정
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

from dotenv import load_dotenv
from pdfminer.high_level import extract_text

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = REPO_ROOT / "data" / "external_raw" / "legal_pdfs" / "cases"
OUT_FILE = REPO_ROOT / "data" / "evaluation" / "ner_eval.jsonl"

# 청크 크기 (기존 데이터와 유사하게)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# 평가 대상 엔티티 레이블 (우리 모델 기준)
ENTITY_LABELS = ["LAW", "ORG", "DATE", "AMOUNT", "CRIME", "PENALTY"]

SYSTEM_PROMPT = """당신은 한국 법률 텍스트의 개체명 인식(NER) 전문가입니다.
주어진 텍스트에서 다음 엔티티를 찾아 정확한 위치(시작/끝 인덱스)와 함께 반환하세요.

엔티티 종류:
- LAW: 법령명, 조문 번호 (예: 근로기준법 제43조, 형법 제250조)
- ORG: 기관명, 법원명, 회사명 (예: 대법원, 서울고등법원, 고용노동부)
- DATE: 날짜, 기간 (예: 2024. 1. 15., 2023년 3월, 1년 이내)
- AMOUNT: 금액, 수량 (예: 1,000만 원, 100만원, 3개월)
- CRIME: 범죄명 (예: 강간, 사기, 횡령, 성희롱)
- PENALTY: 처벌 내용 (예: 징역 3년, 벌금 500만원, 집행유예 2년)

JSON 형식으로만 응답하세요:
{"entities": [{"label": "LAW", "start": 5, "end": 15, "text": "근로기준법 제43조"}, ...]}

- start, end는 텍스트 내 문자 인덱스 (end는 exclusive)
- 엔티티가 없으면 {"entities": []} 반환
- 반드시 JSON만 반환, 추가 설명 금지"""


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        text = extract_text(str(pdf_path))
        # 불필요한 공백/줄바꿈 정규화
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()
    except Exception as e:
        print(f"[WARN] PDF 추출 실패: {pdf_path.name} - {e}", file=sys.stderr)
        return ""


def split_into_chunks(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # 문장 경계에서 자르기 (마침표, 개행)
        if end < len(text):
            for boundary in [".\n", ". ", "\n", " "]:
                idx = text.rfind(boundary, start + size // 2, end)
                if idx != -1:
                    end = idx + len(boundary)
                    break
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 50]


def span_to_bio(text: str, entities: list[dict]) -> tuple[list[str], list[str]]:
    """텍스트를 문자 단위 토큰과 BIO 태그로 변환."""
    tokens = list(text)
    tags = ["O"] * len(tokens)

    # 겹치는 엔티티 처리 (긴 것 우선)
    sorted_ents = sorted(entities, key=lambda e: e["end"] - e["start"], reverse=True)
    occupied = [False] * len(tokens)

    for ent in sorted_ents:
        s, e, label = ent["start"], ent["end"], ent["label"]
        if label not in ENTITY_LABELS:
            continue
        if s < 0 or e > len(tokens):
            continue
        # 이미 점유된 범위와 겹치면 skip
        if any(occupied[s:e]):
            continue
        for i in range(s, e):
            occupied[i] = True
            if i == s:
                tags[i] = f"B-{label}"
            else:
                tags[i] = f"I-{label}"

    return tokens, tags


def call_openai_ner(client, text: str) -> list[dict]:
    """OpenAI API 호출로 엔티티 추출."""
    import openai

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("entities", [])
    except Exception as e:
        print(f"[WARN] OpenAI 호출 실패: {e}", file=sys.stderr)
        return []


def make_chunk_id(pdf_name: str, chunk_idx: int) -> str:
    h = hashlib.md5(pdf_name.encode()).hexdigest()[:16]
    return f"{h}_eval_chunk_{chunk_idx:04d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF에서 NER 평가 데이터 생성")
    parser.add_argument("--max-pdfs", type=int, default=30, help="처리할 최대 PDF 수 (기본 30)")
    parser.add_argument("--max-chunks", type=int, default=300, help="최대 청크 수 (기본 300)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--output", type=Path, default=OUT_FILE)
    parser.add_argument("--delay", type=float, default=0.5, help="API 호출 간격(초)")
    args = parser.parse_args()

    import openai

    client = openai.OpenAI()

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))[: args.max_pdfs]
    if not pdf_files:
        print(f"[ERROR] PDF 파일 없음: {PDF_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"처리 대상 PDF: {len(pdf_files)}개")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.output.open("w", encoding="utf-8") as fout:
        for pdf_path in pdf_files:
            if written >= args.max_chunks:
                break

            safe_name = pdf_path.name.encode("utf-8", errors="replace").decode("utf-8")
            print(f"  [{written}/{args.max_chunks}] {safe_name}")
            text = extract_pdf_text(pdf_path)
            if not text:
                continue

            chunks = split_into_chunks(text, args.chunk_size, CHUNK_OVERLAP)

            for idx, chunk in enumerate(chunks):
                if written >= args.max_chunks:
                    break

                entities = call_openai_ner(client, chunk)
                tokens, tags = span_to_bio(chunk, entities)

                record = {
                    "chunk_id": make_chunk_id(pdf_path.name, idx),
                    "tokens": tokens,
                    "ner_tags": tags,
                    "metadata": {
                        "source_file": pdf_path.name,
                        "chunk_index": idx,
                        "entity_count": sum(1 for t in tags if t.startswith("B-")),
                    },
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

                if args.delay > 0:
                    time.sleep(args.delay)

    print(f"\n완료: {written}개 청크 → {args.output}")

    # 엔티티 분포 통계 출력
    _print_stats(args.output)


def _print_stats(path: Path) -> None:
    from collections import Counter

    tag_counts: Counter = Counter()
    total_chunks = 0
    chunks_with_entity = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            total_chunks += 1
            has_entity = False
            for tag in rec["ner_tags"]:
                if tag.startswith("B-"):
                    tag_counts[tag[2:]] += 1
                    has_entity = True
            if has_entity:
                chunks_with_entity += 1

    print(f"\n=== 데이터 통계 ===")
    print(f"총 청크: {total_chunks}")
    print(f"엔티티 있는 청크: {chunks_with_entity} ({chunks_with_entity/total_chunks*100:.1f}%)")
    print("엔티티 분포:")
    for label, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:12s}: {count:5d}")


if __name__ == "__main__":
    main()
