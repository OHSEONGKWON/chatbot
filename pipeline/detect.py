"""탐지기 Stage 1+2: DB 결정적 검증 + BM25 검색 + 로컬 NLI entailment (방법론 v3 §4).

탐지기는 gold를 모른다 — 질문·클레임만 입력으로 검색부터 수행한다(라벨러와의 정보 비대칭).
모든 원점수(ent/contra)를 저장하므로 임계값은 사후 조정 가능.

사용:
  python pipeline\\detect.py --limit 50      # 스모크
  python pipeline\\detect.py                # 전체 (GPU 권장, CPU도 가능)
"""
import argparse
import json
from pathlib import Path

from generate_answers import build_bm25, tokenize
from law_db import LawDB, extract_citations

ROOT = Path(__file__).resolve().parent.parent
COLL = ROOT / "data" / "collection"
OUT_PATH = COLL / "detect_scores.jsonl"

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
TOP_K = 5          # BM25 근거 수
EVIDENCE_CHARS = 700


def load_nli(device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device).eval()
    id2label = {i: l.lower() for i, l in model.config.id2label.items()}

    @torch.no_grad()
    def score(pairs: list[tuple[str, str]], batch_size=32) -> list[dict]:
        out = []
        for s in range(0, len(pairs), batch_size):
            batch = pairs[s:s + batch_size]
            enc = tok([p for p, _ in batch], [h for _, h in batch],
                      truncation=True, max_length=512, padding=True, return_tensors="pt").to(device)
            probs = model(**enc).logits.softmax(-1).cpu()
            for row in probs:
                out.append({id2label[i]: float(row[i]) for i in range(len(row))})
        return out
    return score


def main(limit, batch_size, device):
    db = LawDB()
    answers = {a["a_id"]: a for a in (json.loads(l) for l in open(COLL / "answers.jsonl", encoding="utf-8") if l.strip())}
    claims = [json.loads(l) for l in open(COLL / "claims.jsonl", encoding="utf-8") if l.strip()]
    done = set()
    if OUT_PATH.exists():
        done = {json.loads(l)["c_id"] for l in open(OUT_PATH, encoding="utf-8") if l.strip()}
    claims = [c for c in claims if c["c_id"] not in done]
    if limit:
        claims = claims[:limit]
    print(f"대상 클레임 {len(claims)}개 (기존 {len(done)} 제외)")

    bm25, docs = build_bm25()
    nli_score = load_nli(device)
    print(f"NLI 모델 로드 완료 ({device})")

    with open(OUT_PATH, "a", encoding="utf-8") as fout:
        for s in range(0, len(claims), 200):       # 200개 단위 체크포인트
            chunk = claims[s:s + 200]
            rows = []
            all_pairs, pair_span = [], []
            for c in chunk:
                question = answers[c["a_id"]]["question"]
                # Stage 1: 결정적 검증
                cites = extract_citations(c["claim_text"])
                stage1 = "pass"
                cited_texts = []
                for law, art in cites:
                    if db.law_exists(law) and art and not db.article_exists(law, art):
                        stage1 = "refuted"
                    t = db.article_text(law, art)
                    if t:
                        cited_texts.append(t[:EVIDENCE_CHARS])
                # 검색: BM25 top-k + 인용 조문 전문
                ev = [t[:EVIDENCE_CHARS] for t in
                      bm25.get_top_n(tokenize(question + " " + c["claim_text"]), docs, n=TOP_K)]
                ev += cited_texts[:2]
                rows.append({"c": c, "stage1": stage1, "ev": ev})
                span_start = len(all_pairs)
                all_pairs += [(e, c["claim_text"]) for e in ev]
                pair_span.append((span_start, len(all_pairs)))

            scores = nli_score(all_pairs, batch_size) if all_pairs else []
            for r, (a, b) in zip(rows, pair_span):
                sc = scores[a:b]
                ent = [x.get("entailment", 0.0) for x in sc]
                con = [x.get("contradiction", 0.0) for x in sc]
                best = max(range(len(ent)), key=ent.__getitem__) if ent else -1
                fout.write(json.dumps({
                    "c_id": r["c"]["c_id"],
                    "stage1": r["stage1"],
                    "ent_max": max(ent, default=0.0),
                    "contra_max": max(con, default=0.0),
                    "best_evidence": r["ev"][best][:EVIDENCE_CHARS] if best >= 0 else "",
                    "top_evidence": r["ev"][:3],
                }, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"  진행 {min(s+200, len(claims))}/{len(claims)}")
    print(f"완료 → {OUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None, help="cuda / cpu (기본: 자동)")
    args = ap.parse_args()
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args.limit, args.batch, args.device)
