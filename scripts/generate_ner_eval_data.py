"""
NER 평가 데이터 생성 (rule-based, API 없음, PDF 없음)

rag_corpus.jsonl 의 기존 청크에 정규식 기반 BIO 태깅 적용.
PDF 추출 없음 → 오류/멈춤 없음.

사용법:
  python scripts/generate_ner_eval_data.py
  python scripts/generate_ner_eval_data.py --max-chunks 500
"""
from __future__ import annotations

import argparse
import hashlib
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
INPUT_PATHS = [
    REPO_ROOT / "data" / "external_processed" / "rag_chunks" / "rag_corpus.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_law_chunks.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_case_chunks.jsonl",
    REPO_ROOT / "data" / "real_data" / "New_Dataset" / "rag_manual_chunks.jsonl",
]
OUT_FILE = REPO_ROOT / "data" / "evaluation" / "ner_eval.jsonl"

ENTITY_LABELS = ["LAW", "ORG", "DATE", "AMOUNT", "CRIME", "PENALTY"]

# ── NER 정규식 ────────────────────────────────────────────────────────────────
_LAW_NAMES = (
    r"(?:헌법|형법|민법|상법|노동법|세법|저작권법|특허법|"
    r"근로기준법|최저임금법|남녀고용평등법|고용보험법|산업재해보상보험법|"
    r"국민연금법|건강보험법|주택법|건축법|도로교통법|형사소송법|민사소송법|"
    r"행정소송법|행정절차법|공정거래법|독점규제법|개인정보보호법|"
    r"전자상거래법|소비자보호법|약사법|의료법|교육법|국가공무원법|"
    r"지방공무원법|경찰관직무집행법|검찰청법|법원조직법|"
    r"[가-힣]+법(?:률|전)?)"
)
LAW_RE = re.compile(
    rf"(?:{_LAW_NAMES})(?:\s*제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?)?"
    r"|제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?",
    re.UNICODE,
)
DATE_RE = re.compile(
    r"\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\."
    r"|\d{4}년\s*\d{1,2}월(?:\s*\d{1,2}일)?"
    r"|\d{4}년"
    r"|\d{1,2}\.\s*\d{1,2}\."
    r"|\d+\s*(?:년|개월|주|일)\s*(?:이내|이상|이하|동안|간)?",
    re.UNICODE,
)
AMOUNT_RE = re.compile(
    r"\d{1,3}(?:,\d{3})*\s*(?:억|만)?\s*원"
    r"|\d+억\s*(?:\d+만)?\s*원?"
    r"|\d+만\s*원?"
    r"|\d+\s*원"
    r"|\d+\s*개월"
    r"|\d+\s*년\s*(?:이하|이상|징역|금고)",
    re.UNICODE,
)
PENALTY_RE = re.compile(
    r"(?:징역|금고|구류)\s*\d+\s*(?:년|개월)"
    r"|벌금\s*\d{1,3}(?:,\d{3})*\s*원"
    r"|벌금\s*\d+\s*(?:만|억)?\s*원"
    r"|집행유예\s*\d+\s*(?:년|개월)"
    r"|사형|무기징역|무기금고"
    r"|과태료\s*\d+\s*(?:만|억)?\s*원"
    r"|몰수|추징",
    re.UNICODE,
)
ORG_RE = re.compile(
    r"(?:대법원|헌법재판소|고등법원|지방법원|가정법원|행정법원|특허법원|"
    r"서울중앙지방법원|수원지방법원|부산지방법원|대구지방법원|"
    r"광주지방법원|대전지방법원|인천지방법원|"
    r"대검찰청|검찰청|고등검찰청|지방검찰청|"
    r"고용노동부|법무부|국토교통부|기획재정부|보건복지부|"
    r"교육부|행정안전부|경찰청|국세청|관세청|"
    r"공정거래위원회|금융감독원|한국은행)"
    r"|[가-힣]{2,10}(?:주식회사|유한회사|합자회사|협동조합|재단법인|사단법인)"
    r"|(?:주식회사|유한회사)\s*[가-힣]{1,10}",
    re.UNICODE,
)
CRIME_RE = re.compile(
    r"(?:강제추행|명예훼손|무면허운전|음주운전|사문서위조|공문서위조|업무방해|"
    r"살인|강도|강간|절도|사기|횡령|배임|뇌물|위증|무고|모욕|협박|공갈|"
    r"폭행|상해|방화|방조|교사|공모|공범|성희롱|성폭력|스토킹|마약|도박)",
    re.UNICODE,
)


def rule_based_ner(text: str) -> list[dict]:
    entities: list[dict] = []
    seen: list[tuple[int, int]] = []

    def add(label: str, m: re.Match) -> None:
        s, e = m.start(), m.end()
        for ps, pe in seen:
            if s < pe and e > ps:
                return
        seen.append((s, e))
        entities.append({"label": label, "start": s, "end": e})

    for m in PENALTY_RE.finditer(text):
        add("PENALTY", m)
    for m in LAW_RE.finditer(text):
        add("LAW", m)
    for m in ORG_RE.finditer(text):
        add("ORG", m)
    for m in DATE_RE.finditer(text):
        add("DATE", m)
    for m in AMOUNT_RE.finditer(text):
        add("AMOUNT", m)
    for m in CRIME_RE.finditer(text):
        add("CRIME", m)
    return entities


def span_to_bio(text: str, entities: list[dict]) -> tuple[list[str], list[str]]:
    n = len(text)
    tags = ["O"] * n
    occupied = [False] * n
    for ent in sorted(entities, key=lambda e: e["end"] - e["start"], reverse=True):
        s, e, label = ent["start"], ent["end"], ent["label"]
        if s < 0 or e > n or any(occupied[s:e]):
            continue
        for i in range(s, e):
            occupied[i] = True
            tags[i] = f"B-{label}" if i == s else f"I-{label}"
    return list(text), tags


def make_chunk_id(source: str, idx: int) -> str:
    return f"{hashlib.md5(source.encode()).hexdigest()[:16]}_ner_eval_{idx:04d}"


def load_chunks(max_n: int = 10000) -> list[dict]:
    chunks: list[dict] = []
    for path in INPUT_PATHS:
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
                    # 한글이 포함된 청크만 사용 (인코딩 깨진 청크 제외)
                    korean_count = sum(1 for c in text if '가' <= c <= '힣')
                    if korean_count < 20:
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
        if len(chunks) >= max_n:
            break
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chunks", type=int, default=500)
    parser.add_argument("--output", type=Path, default=OUT_FILE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("청크 로드 중...", flush=True)
    all_chunks = load_chunks()
    print(f"사용 가능한 청크: {len(all_chunks)}개")

    if not all_chunks:
        print("[ERROR] 사용 가능한 청크 없음", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    selected = rng.sample(all_chunks, min(args.max_chunks, len(all_chunks)))
    print(f"선택된 청크: {len(selected)}개 → NER 태깅 중...")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.output.open("w", encoding="utf-8") as fout:
        for i, chunk in enumerate(selected):
            text = chunk["text"]
            tokens, tags = span_to_bio(text, rule_based_ner(text))
            fout.write(json.dumps({
                "chunk_id": chunk["chunk_id"] or make_chunk_id(str(i), i),
                "tokens": tokens,
                "ner_tags": tags,
                "metadata": {
                    **chunk.get("metadata", {}),
                    "chunk_index": i,
                    "entity_count": sum(1 for t in tags if t.startswith("B-")),
                },
            }, ensure_ascii=False) + "\n")
            written += 1
            if (i + 1) % 100 == 0:
                print(f"  [{written}/{len(selected)}] 완료", flush=True)

    print(f"\n완료: {written}청크 → {args.output}")
    _print_stats(args.output)


def _print_stats(path: Path) -> None:
    from collections import Counter
    tag_counts: Counter = Counter()
    total = chunks_with_entity = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            has = any(t.startswith("B-") for t in rec["ner_tags"])
            if has:
                chunks_with_entity += 1
            for t in rec["ner_tags"]:
                if t.startswith("B-"):
                    tag_counts[t[2:]] += 1
    print(f"\n=== NER 통계 ===\n총 청크: {total}")
    if total:
        print(f"엔티티 청크: {chunks_with_entity} ({chunks_with_entity/total*100:.1f}%)")
    for label, cnt in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:12s}: {cnt:5d}")


if __name__ == "__main__":
    main()
