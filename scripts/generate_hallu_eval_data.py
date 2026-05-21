"""
환각 탐지 평가 데이터 생성 (rule-based perturbation, API 없음)

rag_corpus.jsonl에서 청크 샘플링:
  - Positive (is_hallucination=True): 조문번호·금액·날짜·법령명 규칙 기반 변조
  - Negative (is_hallucination=False): 원본 그대로

사용법:
  python scripts/generate_hallu_eval_data.py
  python scripts/generate_hallu_eval_data.py --n-positive 150 --n-negative 150
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import random
import re
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATHS = [
    REPO_ROOT / "data" / "external_processed" / "rag_chunks" / "rag_corpus.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_law_chunks.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_case_chunks.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_manual_chunks.jsonl",
]
OUT_FILE = REPO_ROOT / "data" / "evaluation" / "hallu_eval.jsonl"

# ── 변조용 정규식 ─────────────────────────────────────────────────────────────
ARTICLE_RE = re.compile(r"제(\d+)조", re.UNICODE)
AMOUNT_MAN_RE = re.compile(r"(\d+)\s*만\s*원", re.UNICODE)
AMOUNT_EOK_RE = re.compile(r"(\d+)\s*억\s*원", re.UNICODE)
DATE_FULL_RE = re.compile(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.", re.UNICODE)
DATE_YEAR_RE = re.compile(r"(\d{4})년", re.UNICODE)

LAW_SUBSTITUTIONS: list[tuple[str, str]] = [
    ("근로기준법", "노동기준법"),
    ("형법", "특별형법"),
    ("민법", "상법"),
    ("상법", "민법"),
    ("형사소송법", "민사소송법"),
    ("민사소송법", "형사소송법"),
    ("행정소송법", "행정심판법"),
    ("국가공무원법", "지방공무원법"),
    ("지방공무원법", "국가공무원법"),
    ("개인정보보호법", "정보통신망법"),
    ("도로교통법", "교통사고처리특례법"),
    ("건축법", "국토계획법"),
    ("약사법", "의료기기법"),
    ("의료법", "보건의료기본법"),
    ("저작권법", "특허법"),
    ("특허법", "저작권법"),
]


# ── 변조 함수들 ───────────────────────────────────────────────────────────────

def _perturb_article(text: str, rng: random.Random) -> tuple[str, dict | None]:
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return text, None
    m = rng.choice(matches)
    orig = int(m.group(1))
    candidates = [n for n in range(max(1, orig - 10), orig + 11) if n != orig]
    if not candidates:
        return text, None
    new = rng.choice(candidates)
    orig_str, new_str = f"제{orig}조", f"제{new}조"
    return text.replace(orig_str, new_str, 1), {
        "label": "LAW", "wrong_word": new_str, "correct_word": orig_str,
    }


def _perturb_amount(text: str, rng: random.Random) -> tuple[str, dict | None]:
    for pat in (AMOUNT_MAN_RE, AMOUNT_EOK_RE):
        matches = list(pat.finditer(text))
        if not matches:
            continue
        m = rng.choice(matches)
        orig_num = int(m.group(1))
        factors = [0.5, 0.6, 0.7, 1.3, 1.5, 2.0]
        new_num = max(1, int(orig_num * rng.choice(factors)))
        while new_num == orig_num:
            new_num = orig_num + rng.randint(10, 100)
        unit = "만 원" if pat is AMOUNT_MAN_RE else "억 원"
        orig_str = m.group(0)
        new_str = f"{new_num}{unit}"
        return text.replace(orig_str, new_str, 1), {
            "label": "AMOUNT", "wrong_word": new_str, "correct_word": orig_str,
        }
    return text, None


def _perturb_date(text: str, rng: random.Random) -> tuple[str, dict | None]:
    m = DATE_FULL_RE.search(text)
    if m:
        year = int(m.group(1))
        orig_str = m.group(0)
        new_year = year + rng.choice([-2, -1, 1, 2])
        new_str = f"{new_year}. {m.group(2)}. {m.group(3)}."
        return text.replace(orig_str, new_str, 1), {
            "label": "DATE", "wrong_word": new_str, "correct_word": orig_str,
        }
    m = DATE_YEAR_RE.search(text)
    if m:
        year = int(m.group(1))
        orig_str = m.group(0)
        new_year = year + rng.choice([-2, -1, 1, 2])
        new_str = f"{new_year}년"
        return text.replace(orig_str, new_str, 1), {
            "label": "DATE", "wrong_word": new_str, "correct_word": orig_str,
        }
    return text, None


def _perturb_law_name(text: str, rng: random.Random) -> tuple[str, dict | None]:
    pairs = rng.sample(LAW_SUBSTITUTIONS, len(LAW_SUBSTITUTIONS))
    for orig_law, new_law in pairs:
        if orig_law in text:
            return text.replace(orig_law, new_law, 1), {
                "label": "LAW", "wrong_word": new_law, "correct_word": orig_law,
            }
    return text, None


_PERTURB_FUNCS = [_perturb_article, _perturb_amount, _perturb_date, _perturb_law_name]
_HALLU_TYPES = {
    "_perturb_article":  "article_num",
    "_perturb_amount":   "amount",
    "_perturb_date":     "date",
    "_perturb_law_name": "law_name",
}


def make_positive(chunk: dict, rng: random.Random) -> dict | None:
    text = chunk.get("text", "")[:500]
    for func in rng.sample(_PERTURB_FUNCS, len(_PERTURB_FUNCS)):
        new_text, mismatch = func(text, rng)
        if mismatch:
            return {
                "answer": new_text,
                "rag_docs": [{"text": text, "metadata": chunk.get("metadata", {})}],
                "is_hallucination": True,
                "hallucination_type": _HALLU_TYPES[func.__name__],
                "expected_mismatches": [mismatch],
            }
    return None


def load_chunks(max_n: int = 6000) -> list[dict]:
    chunks: list[dict] = []
    for path in CORPUS_PATHS:
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = rec.get("text", "")
                    if len(text) < 150:
                        continue
                    chunks.append({
                        "chunk_id": rec.get("chunk_id") or rec.get("id", ""),
                        "text": text,
                        "metadata": rec.get("metadata", {}),
                    })
                    if len(chunks) >= max_n:
                        break
        except Exception as e:
            print(f"[WARN] 로드 실패: {path.name} - {e}", file=sys.stderr)
        gc.collect()
        if len(chunks) >= max_n:
            break
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-positive", type=int, default=150)
    parser.add_argument("--n-negative", type=int, default=150)
    parser.add_argument("--output", type=Path, default=OUT_FILE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    need = args.n_positive + args.n_negative

    print("청크 로드 중...")
    all_chunks = load_chunks(max_n=need * 4)
    print(f"로드된 청크: {len(all_chunks)}개")
    if len(all_chunks) < need:
        print(f"[WARN] 청크 {len(all_chunks)}개로 목표 {need}개 미달, 가능한 만큼 생성합니다", file=sys.stderr)

    rng.shuffle(all_chunks)
    half = len(all_chunks) // 2
    pos_pool = all_chunks[:half]
    neg_pool = all_chunks[half:]

    # ── Positive 생성 ────────────────────────────────────────────────────────
    positives: list[dict] = []
    print(f"Positive 생성 중 (목표: {args.n_positive}개)...", flush=True)
    for chunk in pos_pool:
        if len(positives) >= args.n_positive:
            break
        rec = make_positive(chunk, rng)
        if rec:
            positives.append(rec)

    print(f"  → {len(positives)}개 생성")

    # ── Negative 생성 ────────────────────────────────────────────────────────
    negatives: list[dict] = []
    print(f"Negative 생성 중 (목표: {args.n_negative}개)...", flush=True)
    for chunk in neg_pool:
        if len(negatives) >= args.n_negative:
            break
        text = chunk.get("text", "")[:500]
        if len(text) < 100:
            continue
        negatives.append({
            "answer": text,
            "rag_docs": [{"text": text, "metadata": chunk.get("metadata", {})}],
            "is_hallucination": False,
            "hallucination_type": "none",
            "expected_mismatches": [],
        })

    print(f"  → {len(negatives)}개 생성")

    # ── 합치고 셔플 ──────────────────────────────────────────────────────────
    all_records = positives + negatives
    rng.shuffle(all_records)
    for i, rec in enumerate(all_records):
        rec["eval_id"] = f"hallu_eval_{i:04d}"

    all_chunks = positives = negatives = None
    gc.collect()

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
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["is_hallucination"]:
                pos += 1
            else:
                neg += 1
            type_counts[rec.get("hallucination_type", "none")] += 1
    total = pos + neg
    print(f"\n=== Hallu 통계 ===\n총 {total}개 (pos: {pos}, neg: {neg})")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:20s}: {c:4d}")


if __name__ == "__main__":
    main()
