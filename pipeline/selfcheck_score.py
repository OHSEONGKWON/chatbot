"""SelfCheckGPT-NLI 베이스라인 채점 (로컬 NLI, API 미사용).

각 클레임을 같은 질문의 재생성 샘플 3개와 NLI 대조:
  score = 1 - mean(entailment)  (샘플들이 지지하지 않을수록 환각 점수↑)
contra_mean 도 저장. verdict 매핑은 평가 단계에서 임계값으로.

사용: python pipeline\\selfcheck_score.py [--limit 50]
"""
import argparse
import json
from pathlib import Path

from detect import load_nli

ROOT = Path(__file__).resolve().parent.parent
COLL = ROOT / "data" / "collection"
OUT_PATH = COLL / "baseline_selfcheck.jsonl"


def main(limit, batch, device):
    samples = {}
    for l in open(COLL / "selfcheck_samples.jsonl", encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            samples.setdefault(d["a_id"], []).append(d["sample"])
    claims = [json.loads(l) for l in open(COLL / "claims.jsonl", encoding="utf-8") if l.strip()]
    done = set()
    if OUT_PATH.exists():
        done = {json.loads(l)["c_id"] for l in open(OUT_PATH, encoding="utf-8") if l.strip()}
    claims = [c for c in claims if c["c_id"] not in done and c["a_id"] in samples]
    if limit:
        claims = claims[:limit]
    print(f"대상 {len(claims)}건")

    nli = load_nli(device)
    with open(OUT_PATH, "a", encoding="utf-8") as fout:
        for s in range(0, len(claims), 200):
            chunk = claims[s:s + 200]
            pairs, span = [], []
            for c in chunk:
                sams = samples[c["a_id"]][:3]
                start = len(pairs)
                pairs += [(sam[:2500], c["claim_text"]) for sam in sams]
                span.append((start, len(pairs)))
            scores = nli(pairs, batch)
            for c, (a, b) in zip(chunk, span):
                ents = [x.get("entailment", 0.0) for x in scores[a:b]]
                cons = [x.get("contradiction", 0.0) for x in scores[a:b]]
                fout.write(json.dumps({
                    "c_id": c["c_id"],
                    "hallu_score": 1 - sum(ents) / max(len(ents), 1),
                    "ent_mean": sum(ents) / max(len(ents), 1),
                    "contra_mean": sum(cons) / max(len(cons), 1),
                }, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"  진행 {min(s+200, len(claims))}/{len(claims)}")
    print(f"완료 → {OUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args.limit, args.batch, args.device)
