import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

TARGET_LABELS = {"LAW", "PENALTY", "AMOUNT", "DATE", "ORG", "CRIME"}

LAW_POOL = [
    "근로기준법", "형법", "민법", "형사소송법", "민사소송법", "산업안전보건법",
    "성폭력처벌법", "정보통신망법", "근로기준법 시행령", "근로기준법 시행규칙",
]
ORG_POOL = [
    "중앙노동위원회", "지방노동위원회", "고용노동부", "대법원", "서울고등법원", "검찰청",
]
CRIME_POOL = [
    "강제추행", "성희롱", "사기", "횡령", "배임", "명예훼손", "업무방해", "폭행",
]
PENALTY_POOL = [
    "징역 6월", "징역 1년", "징역 2년", "징역 3년", "벌금 300만원", "벌금 500만원", "벌금 1000만원",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic hallucination BIO samples")
    parser.add_argument("--target-count", type=int, default=30000)
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=["data/hallucination_data", "data/real_bio_data"],
        help="Directories containing *_bio.jsonl files",
    )
    parser.add_argument(
        "--output-file",
        default="data/hallucination_data/synthetic_hallu_bio_30000.jsonl",
        help="Output synthetic JSONL path",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_samples(source_dirs, output_file):
    files = []
    out_resolved = Path(output_file).resolve()
    for src in source_dirs:
        base = Path(src)
        if not base.exists():
            continue
        files.extend(sorted(base.rglob("*_bio.jsonl")))

    samples = []
    for fp in files:
        if fp.resolve() == out_resolved:
            continue
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tokens = obj.get("tokens")
                tags = obj.get("ner_tags")
                if not isinstance(tokens, list) or not isinstance(tags, list):
                    continue
                if len(tokens) != len(tags) or len(tokens) == 0:
                    continue
                if not any(t.startswith("B-") or t.startswith("I-") for t in tags):
                    continue
                samples.append(obj)
    return samples


def extract_spans(tags):
    spans = []
    i = 0
    n = len(tags)
    while i < n:
        tag = tags[i]
        if tag.startswith("B-"):
            label = tag[2:]
            start = i
            i += 1
            while i < n and tags[i] == f"I-{label}":
                i += 1
            spans.append((start, i, label))
            continue
        if tag.startswith("I-"):
            label = tag[2:]
            start = i
            i += 1
            while i < n and tags[i] == f"I-{label}":
                i += 1
            spans.append((start, i, label))
            continue
        i += 1
    return spans


def make_entity_tags(label, text):
    out = []
    started = False
    for ch in text:
        if ch.isspace():
            out.append("O")
            continue
        if not started:
            out.append(f"B-{label}")
            started = True
        else:
            out.append(f"I-{label}")
    if not started:
        return ["O"] * len(text)
    return out


def mutate_law(orig, rng):
    m = re.search(r"제\s*(\d+)\s*조", orig)
    if m:
        n = int(m.group(1))
        new_n = max(1, n + rng.choice([-7, -5, -3, 3, 5, 7, 11]))
        return re.sub(r"제\s*\d+\s*조", f"제{new_n}조", orig, count=1)
    if "법" in orig or "규칙" in orig or "령" in orig:
        cand = rng.choice(LAW_POOL)
        if cand != orig:
            return cand
    return rng.choice(LAW_POOL)


def mutate_date(orig, rng):
    if re.search(r"\d{4}[\./-]\d{1,2}[\./-]\d{1,2}", orig):
        return re.sub(
            r"(\d{4})[\./-](\d{1,2})[\./-](\d{1,2})",
            lambda m: f"{int(m.group(1)) + rng.choice([-2,-1,1,2])}.{max(1,min(12,int(m.group(2))+rng.choice([-2,-1,1,2])))}.{max(1,min(28,int(m.group(3))+rng.choice([-5,-3,3,5])))}",
            orig,
            count=1,
        )
    if re.search(r"\d+\s*일", orig):
        return re.sub(r"(\d+)\s*일", lambda m: f"{max(1,int(m.group(1))+rng.choice([-20,-10,-5,5,10,20]))}일", orig, count=1)
    if re.search(r"\d+\s*개월", orig):
        return re.sub(r"(\d+)\s*개월", lambda m: f"{max(1,int(m.group(1))+rng.choice([-6,-3,3,6]))}개월", orig, count=1)
    if re.search(r"\d+\s*년", orig):
        return re.sub(r"(\d+)\s*년", lambda m: f"{max(1,int(m.group(1))+rng.choice([-2,-1,1,2]))}년", orig, count=1)
    return rng.choice(["7일", "14일", "30일", "60일", "90일", "1년"])


def mutate_amount(orig, rng):
    if re.search(r"\d", orig):
        nums = re.findall(r"\d+", orig)
        if nums:
            n = int(nums[0])
            k = rng.choice([2, 3, 5, 10])
            if rng.random() < 0.5:
                new_n = max(1, n * k)
            else:
                new_n = max(1, n // k)
            return re.sub(r"\d+", str(new_n), orig, count=1)
    return rng.choice(["300만원", "500만원", "1000만원", "5000만원", "1억원"])


def mutate_penalty(orig, rng):
    cand = rng.choice(PENALTY_POOL)
    return cand if cand != orig else rng.choice(PENALTY_POOL)


def mutate_org(orig, rng):
    cand = rng.choice(ORG_POOL)
    return cand if cand != orig else rng.choice(ORG_POOL)


def mutate_crime(orig, rng):
    cand = rng.choice(CRIME_POOL)
    return cand if cand != orig else rng.choice(CRIME_POOL)


def mutate_entity_text(label, orig, rng):
    if label == "LAW":
        return mutate_law(orig, rng)
    if label == "DATE":
        return mutate_date(orig, rng)
    if label == "AMOUNT":
        return mutate_amount(orig, rng)
    if label == "PENALTY":
        return mutate_penalty(orig, rng)
    if label == "ORG":
        return mutate_org(orig, rng)
    if label == "CRIME":
        return mutate_crime(orig, rng)
    return orig


def synthesize_one(sample, idx, rng):
    tokens = list(sample["tokens"])
    tags = list(sample["ner_tags"])

    spans = [s for s in extract_spans(tags) if s[2] in TARGET_LABELS]
    if not spans:
        return None

    num_changes = 1 if len(spans) == 1 else rng.choice([1, 2])
    chosen = rng.sample(spans, k=min(num_changes, len(spans)))
    chosen.sort(key=lambda x: x[0], reverse=True)

    changed_labels = []

    for start, end, label in chosen:
        orig_text = "".join(tokens[start:end])
        new_text = mutate_entity_text(label, orig_text, rng)
        if not new_text or new_text == orig_text:
            continue

        new_tokens = list(new_text)
        new_tags = make_entity_tags(label, new_text)

        tokens = tokens[:start] + new_tokens + tokens[end:]
        tags = tags[:start] + new_tags + tags[end:]
        changed_labels.append(label)

    if not changed_labels:
        return None

    base_id = str(sample.get("chunk_id", "unknown"))
    out = {
        "chunk_id": f"syn_hallu_{idx:05d}__{base_id}",
        "tokens": tokens,
        "ner_tags": tags,
    }
    return out, changed_labels


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_samples = load_samples(args.source_dirs, args.output_file)
    if not source_samples:
        raise RuntimeError("No valid source samples found for synthesis.")

    generated = 0
    attempts = 0
    max_attempts = args.target_count * 20
    label_stats = Counter()

    with output_path.open("w", encoding="utf-8") as out_f:
        while generated < args.target_count and attempts < max_attempts:
            attempts += 1
            sample = rng.choice(source_samples)
            result = synthesize_one(sample, generated + 1, rng)
            if result is None:
                continue
            syn, changed_labels = result
            out_f.write(json.dumps(syn, ensure_ascii=False) + "\n")
            generated += 1
            for lb in changed_labels:
                label_stats[lb] += 1

    if generated < args.target_count:
        raise RuntimeError(f"Only generated {generated}/{args.target_count} samples")

    print(f"Generated: {generated}")
    print(f"Output: {output_path}")
    print("Changed label counts:")
    for lb in sorted(TARGET_LABELS):
        print(f"  {lb}: {label_stats[lb]}")


if __name__ == "__main__":
    main()
