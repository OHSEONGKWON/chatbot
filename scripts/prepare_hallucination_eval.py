#!/usr/bin/env python3
"""
Prepare hallucination evaluation file by merging template entries with
streamed samples from hallucination_data JSONL files.

Usage examples:
  python scripts/prepare_hallucination_eval.py \
    --template data/evaluation/hallucination_eval_template.jsonl \
    --real data/hallucination_data/case_hallu_bio.jsonl,data/hallucination_data/law_hallu_bio.jsonl,data/hallucination_data/manual_hallu_bio.jsonl \
    --synthetic data/hallucination_data/synthetic_hallu_bio.jsonl \
    --output data/evaluation/hallucination_eval.jsonl --seed 42

This script streams source files (no full-file loads) and uses reservoir
sampling to select the requested number of samples per group.
"""
import argparse
import json
import random
from typing import List


def reservoir_sample(paths: List[str], k: int, seed: int = None) -> List[dict]:
    if k <= 0:
        return []
    rand = random.Random(seed)
    reservoir: List[dict] = []
    i = 0
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if i < k:
                    reservoir.append(item)
                else:
                    j = rand.randint(0, i)
                    if j < k:
                        reservoir[j] = item
                i += 1
    # If total seen < k, reservoir contains fewer items — that's fine.
    return reservoir


def load_templates(path: str) -> List[dict]:
    templates = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            templates.append(json.loads(line))
    return templates


def main():
    parser = argparse.ArgumentParser(description="Prepare hallucination_eval.jsonl")
    parser.add_argument("--template", required=True)
    parser.add_argument("--real", required=True,
                        help="Comma-separated real source jsonl files")
    parser.add_argument("--synthetic", required=True,
                        help="Comma-separated synthetic source jsonl files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--real-share", type=float, default=0.5,
                        help="Fraction of template entries to fill with real samples (0-1)")
    args = parser.parse_args()

    real_paths = [p.strip() for p in args.real.split(",") if p.strip()]
    synth_paths = [p.strip() for p in args.synthetic.split(",") if p.strip()]

    templates = load_templates(args.template)
    total = len(templates)
    if total == 0:
        print("템플릿 항목이 없습니다.")
        return

    # split according to requested real share
    share = max(0.0, min(1.0, args.real_share))
    n_real = int(round(total * share))
    n_synth = total - n_real

    print(f"템플릿 항목 수: {total} — real={n_real}, synthetic={n_synth}")

    real_samples = reservoir_sample(real_paths, n_real, seed=args.seed)
    synth_samples = reservoir_sample(synth_paths, n_synth, seed=args.seed + 1)

    # If sampled fewer than requested, notify and adjust
    if len(real_samples) < n_real:
        print(f"경고: real에서 {len(real_samples)}개만 샘플링됨 (요구 {n_real})")
    if len(synth_samples) < n_synth:
        print(f"경고: synthetic에서 {len(synth_samples)}개만 샘플링됨 (요구 {n_synth})")

    combined = real_samples + synth_samples
    rand = random.Random(args.seed)
    rand.shuffle(combined)

    # If we have fewer samples than templates, we'll repeat samples as needed.
    out_fh = open(args.output, "w", encoding="utf-8")
    for i, tmpl in enumerate(templates):
        if i < len(combined):
            sample = combined[i]
        else:
            sample = combined[i % len(combined)] if combined else {}

        # Attach sampled original item under 'hallu_sample' to preserve template
        merged = dict(tmpl)
        merged["hallu_sample"] = sample

        # Optionally update 'source' to be the sample's source file if present
        if isinstance(sample, dict):
            meta = sample.get("metadata") or sample.get("meta") or {}
            src = meta.get("source_file") or meta.get("source") or sample.get("source_file")
            if src:
                merged["source"] = src

        out_fh.write(json.dumps(merged, ensure_ascii=False) + "\n")
    out_fh.close()
    print(f"작성 완료: {args.output}")


if __name__ == "__main__":
    main()
