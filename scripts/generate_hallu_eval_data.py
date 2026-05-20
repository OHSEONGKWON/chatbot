"""
환각 탐지 평가 데이터 생성 스크립트

기존 hallucination_data/와 mock_data/에서 positive(환각)/negative(정상) 샘플을 구성하고,
OpenAI API를 사용해 추가 환각 샘플을 생성합니다.

출력 형식:
  {
    "eval_id": "...",
    "answer": "...",         # 모델이 생성한 답변 (환각 포함 or 정상)
    "rag_docs": [...],       # 검색된 RAG 문서들
    "is_hallucination": true/false,
    "hallucination_type": "law_name|article_num|date|amount|org|none",
    "expected_mismatches": [{"label": ..., "wrong_word": ..., "correct_word": ...}]
  }

사용법:
  python scripts/generate_hallu_eval_data.py
  python scripts/generate_hallu_eval_data.py --n-positive 100 --n-negative 100
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = REPO_ROOT / "data" / "evaluation" / "hallu_eval.jsonl"

# 기존 데이터 경로
HALLU_DATA = {
    "case_bio": REPO_ROOT / "data" / "hallucination_data" / "case_hallu_bio.jsonl",
    "law_bio": REPO_ROOT / "data" / "hallucination_data" / "law_hallu_bio.jsonl",
    "manual_bio": REPO_ROOT / "data" / "hallucination_data" / "manual_hallu_bio.jsonl",
    "synthetic_bio": REPO_ROOT / "data" / "hallucination_data" / "synthetic_hallu_bio.jsonl",
    "case_hallu": REPO_ROOT / "data" / "mock_data" / "case_hallu.jsonl",
    "law_hallu": REPO_ROOT / "data" / "mock_data" / "law_hallu.jsonl",
    "manual_hallu": REPO_ROOT / "data" / "mock_data" / "manual_hallu.jsonl",
    "ner_factcheck": REPO_ROOT / "data" / "mock_data" / "ner_factcheck_eval.jsonl",
}
CLEAN_DATA = {
    "law": REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_law_chunks.jsonl",
    "case": REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_case_chunks.jsonl",
    "manual": REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_manual_chunks.jsonl",
}

SYSTEM_PROMPT_HALLU = """당신은 한국 법률 텍스트에서 환각(hallucination)을 탐지하는 평가 데이터를 생성하는 전문가입니다.
주어진 정상 법률 문서를 기반으로, 의도적으로 오류가 포함된 답변(환각)을 생성하세요.

오류 유형:
1. law_name: 법령 이름을 잘못된 것으로 교체 (예: 근로기준법 → 노동기준법)
2. article_num: 조문 번호 변경 (예: 제43조 → 제37조)
3. date: 날짜/기간 변경 (예: 30일 → 7일)
4. amount: 금액/수량 변경 (예: 500만원 → 300만원)
5. org: 기관명 변경 (예: 대법원 → 헌법재판소)

규칙:
- 원문에서 1-2개 항목만 변경하여 자연스러운 환각 답변 생성
- 나머지 내용은 원문과 동일하게 유지
- 변경 내용을 mismatches에 정확히 기록

JSON 형식으로만 응답:
{
  "hallucinated_answer": "...",
  "mismatches": [
    {"label": "LAW", "wrong_word": "노동기준법", "correct_word": "근로기준법"},
    ...
  ]
}"""


def load_jsonl(path: Path, max_records: int = 5000) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if len(records) >= max_records:
                    break
    return records


def bio_to_text(record: dict) -> str:
    """BIO 형식 레코드를 텍스트로 변환."""
    return "".join(record.get("tokens", []))


def extract_entities_from_bio(record: dict) -> list[dict]:
    """BIO 태그에서 엔티티 추출."""
    tokens = record.get("tokens", [])
    tags = record.get("ner_tags", [])
    entities = []
    i = 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            label = tags[i][2:]
            start = i
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{label}":
                j += 1
            span = "".join(tokens[start:j])
            entities.append({"label": label, "span": span, "start": start, "end": j})
            i = j
        else:
            i += 1
    return entities


def build_positive_from_ner_factcheck(records: list[dict]) -> list[dict]:
    """ner_factcheck_eval.jsonl에서 positive 샘플 추출."""
    positives = []
    for rec in records:
        mismatches = rec.get("expected_mismatches", [])
        if not mismatches:
            continue
        positives.append({
            "eval_id": f"factcheck_{rec.get('id', 'unknown')}",
            "answer": rec.get("answer", ""),
            "rag_docs": rec.get("rag_docs", []),
            "is_hallucination": True,
            "hallucination_type": mismatches[0].get("label", "unknown").lower() if mismatches else "unknown",
            "expected_mismatches": mismatches,
            "source": "ner_factcheck",
        })
    return positives


def build_negative_from_ner_factcheck(records: list[dict]) -> list[dict]:
    """ner_factcheck_eval.jsonl에서 negative 샘플 추출."""
    negatives = []
    for rec in records:
        if rec.get("expected_mismatches"):
            continue
        negatives.append({
            "eval_id": f"factcheck_{rec.get('id', 'unknown')}_neg",
            "answer": rec.get("answer", ""),
            "rag_docs": rec.get("rag_docs", []),
            "is_hallucination": False,
            "hallucination_type": "none",
            "expected_mismatches": [],
            "source": "ner_factcheck_neg",
        })
    return negatives


def build_negative_from_clean(clean_chunks: list[dict], n: int, rng: random.Random) -> list[dict]:
    """정상 청크에서 negative 샘플 생성."""
    sampled = rng.sample(clean_chunks, min(n, len(clean_chunks)))
    negatives = []
    for i, chunk in enumerate(sampled):
        text = chunk.get("text", "")
        if len(text) < 50:
            continue
        # 정상 답변 형태로 구성 (RAG 문서 = 자기 자신)
        negatives.append({
            "eval_id": f"clean_{i:04d}",
            "answer": text[:500],
            "rag_docs": [{"text": text, "metadata": chunk.get("metadata", {})}],
            "is_hallucination": False,
            "hallucination_type": "none",
            "expected_mismatches": [],
            "source": "clean_chunk",
        })
    return negatives


def call_openai_hallucinate(client, text: str) -> dict | None:
    """OpenAI API로 환각 데이터 생성."""
    import openai

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_HALLU},
                {"role": "user", "content": f"다음 법률 문서를 기반으로 환각 답변을 생성하세요:\n\n{text[:600]}"},
            ],
            temperature=0.8,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[WARN] OpenAI 호출 실패: {e}", file=sys.stderr)
        return None


def build_positive_from_openai(client, clean_chunks: list[dict], n: int, rng: random.Random, delay: float) -> list[dict]:
    """OpenAI로 환각 positive 샘플 생성."""
    sampled = rng.sample(clean_chunks, min(n, len(clean_chunks)))
    positives = []

    for i, chunk in enumerate(sampled):
        print(f"  [환각 생성] {i+1}/{len(sampled)}")
        text = chunk.get("text", "")
        if len(text) < 100:
            continue

        result = call_openai_hallucinate(client, text)
        if not result or not result.get("hallucinated_answer"):
            continue

        mismatches = result.get("mismatches", [])
        hallu_type = mismatches[0].get("label", "unknown").lower() if mismatches else "unknown"

        positives.append({
            "eval_id": f"openai_hallu_{i:04d}",
            "answer": result["hallucinated_answer"],
            "rag_docs": [{"text": text, "metadata": chunk.get("metadata", {})}],
            "is_hallucination": True,
            "hallucination_type": hallu_type,
            "expected_mismatches": mismatches,
            "source": "openai_generated",
        })

        if delay > 0:
            time.sleep(delay)

    return positives


def main() -> None:
    parser = argparse.ArgumentParser(description="환각 탐지 평가 데이터 생성")
    parser.add_argument("--n-positive", type=int, default=150, help="positive 샘플 수 (기본 150)")
    parser.add_argument("--n-negative", type=int, default=150, help="negative 샘플 수 (기본 150)")
    parser.add_argument("--output", type=Path, default=OUT_FILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument(
        "--no-openai",
        action="store_true",
        help="OpenAI API 없이 기존 데이터만 사용",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    import openai

    client = openai.OpenAI() if not args.no_openai else None

    print("=== 환각 탐지 평가 데이터 생성 ===")

    # --- Positive 샘플 구성 ---
    positives = []

    # 1) ner_factcheck_eval에서 기존 positive 추출
    factcheck_recs = load_jsonl(HALLU_DATA["ner_factcheck"])
    positives.extend(build_positive_from_ner_factcheck(factcheck_recs))
    print(f"기존 positive (factcheck): {len(positives)}개")

    # 2) manual_hallu.jsonl → positive (내용에 의도적 오류 포함된 청크)
    manual_hallu = load_jsonl(HALLU_DATA["manual_hallu"])
    for i, rec in enumerate(rng.sample(manual_hallu, min(50, len(manual_hallu)))):
        text = rec.get("text", "")
        if len(text) < 50:
            continue
        # manual_hallu는 오류 주입된 텍스트 — 자체가 positive
        positives.append({
            "eval_id": f"manual_hallu_{i:04d}",
            "answer": text[:500],
            "rag_docs": [{"text": text, "metadata": rec.get("metadata", {})}],
            "is_hallucination": True,
            "hallucination_type": "injected",
            "expected_mismatches": [],
            "source": "manual_hallu",
        })
    print(f"manual_hallu 추가 후 positive: {len(positives)}개")

    # 3) OpenAI로 추가 positive 생성
    remaining_positive = args.n_positive - len(positives)
    if remaining_positive > 0 and client is not None:
        clean_law = load_jsonl(CLEAN_DATA["law"], max_records=500)
        clean_case = load_jsonl(CLEAN_DATA["case"], max_records=500)
        clean_pool = clean_law + clean_case
        if clean_pool:
            print(f"OpenAI로 positive {remaining_positive}개 추가 생성 중...")
            new_positives = build_positive_from_openai(client, clean_pool, remaining_positive, rng, args.delay)
            positives.extend(new_positives)
    elif remaining_positive > 0:
        print(f"[INFO] --no-openai 옵션으로 OpenAI 생성 건너뜀")

    # positive 최대 n개로 제한
    positives = rng.sample(positives, min(args.n_positive, len(positives)))
    print(f"최종 positive: {len(positives)}개")

    # --- Negative 샘플 구성 ---
    negatives = []

    # 1) factcheck에서 negative 추출
    negatives.extend(build_negative_from_ner_factcheck(factcheck_recs))
    print(f"factcheck negative: {len(negatives)}개")

    # 2) 정상 청크에서 negative 생성
    all_clean = []
    for path in CLEAN_DATA.values():
        all_clean.extend(load_jsonl(path, max_records=1000))

    remaining_negative = args.n_negative - len(negatives)
    if all_clean and remaining_negative > 0:
        new_negatives = build_negative_from_clean(all_clean, remaining_negative, rng)
        negatives.extend(new_negatives)

    negatives = rng.sample(negatives, min(args.n_negative, len(negatives)))
    print(f"최종 negative: {len(negatives)}개")

    # --- 합치고 셔플 ---
    all_records = positives + negatives
    rng.shuffle(all_records)

    # eval_id 재부여
    for i, rec in enumerate(all_records):
        rec["eval_id"] = f"hallu_eval_{i:04d}"

    # --- 저장 ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fout:
        for rec in all_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n완료: {len(all_records)}개 → {args.output}")
    _print_stats(args.output)


def _print_stats(path: Path) -> None:
    from collections import Counter

    pos = neg = 0
    type_counts: Counter = Counter()
    source_counts: Counter = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["is_hallucination"]:
                pos += 1
            else:
                neg += 1
            type_counts[rec.get("hallucination_type", "none")] += 1
            source_counts[rec.get("source", "unknown")] += 1

    total = pos + neg
    print(f"\n=== 데이터 통계 ===")
    print(f"총: {total} (positive: {pos}, negative: {neg})")
    print(f"비율: positive {pos/total*100:.1f}% / negative {neg/total*100:.1f}%")
    print("\n환각 유형 분포:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:20s}: {c:4d}")
    print("\n데이터 출처:")
    for s, c in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {s:25s}: {c:4d}")


if __name__ == "__main__":
    main()
