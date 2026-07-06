"""2단계: ALCV 특징 수집 (결정적 검증, LLM 미사용).

각 케이스마다:
  답변 -> NER 개체 추출 -> 유형별 규칙 검증(근거=RAG) -> 답변 단위 특징
판정에 LLM/NLI 가 전혀 개입하지 않는다(추출은 NER, 판단은 규칙/DB/수치비교).

실행:  venv\Scripts\python.exe scripts_hallucination\run_collect_features.py
옵션:  --limit N  (앞 N건만; 스모크 점검용)
출력:  scripts_hallucination/results/alcv_features.json
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alcv.config import RESULTS_DIR, RETRIEVE_TOP_K
from alcv.dataset import load_dataset
from alcv.law_index import LawIndex
from alcv.extract import Extractor
from alcv.verify import verify_entities
from alcv.aggregate import case_features

from src.modules.rag import retriever


def _evidence_texts(rag_docs):
    out = []
    for d in rag_docs or []:
        t = (d.get("text") or d.get("content", "")) if isinstance(d, dict) else getattr(d, "text", "")
        if t and t.strip():
            out.append(t.strip()[:500])
    return out


async def main(limit=None):
    print("=" * 70)
    print("ALCV 특징 수집 (결정적 검증, LLM 미사용)")
    print("=" * 70)

    cases = load_dataset()
    if limit:
        cases = cases[:limit]
        print(f"[스모크] 앞 {len(cases)}건만 처리")

    extractor = Extractor()
    await extractor.warmup()
    law_index = LawIndex()
    await retriever.warmup()

    results = []
    for case in tqdm(cases, desc="ALCV"):
        try:
            entities = extractor.extract(case.answer)
            rag_docs = await retriever.retrieve_async(case.question, top_k=RETRIEVE_TOP_K)
            rag_texts = _evidence_texts(rag_docs)
            verdicts = verify_entities(case.question, entities, rag_texts, law_index, extractor.ner)
            feat = case_features(verdicts)
            results.append({
                "hallu_id": case.hallu_id,
                "true_label": bool(case.is_hallucinated),
                "hallu_type": case.hallu_type,
                "features": feat,
                "verdicts": verdicts,
            })
        except Exception as e:
            print(f"\n[ERROR] {case.hallu_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "hallu_id": case.hallu_id,
                "true_label": bool(case.is_hallucinated),
                "hallu_type": case.hallu_type,
                "features": case_features([]),
                "verdicts": [], "error": str(e),
            })

        if len(results) % 100 == 0:
            hard_h = sum(1 for r in results if r["features"]["n_hard_fail"] > 0 and r["true_label"])
            hard_n = sum(1 for r in results if r["features"]["n_hard_fail"] > 0 and not r["true_label"])
            print(f"\n[중간 {len(results)}] 하드게이트 발화: 환각 {hard_h} / 정상(오탐) {hard_n}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = "alcv_features_smoke.json" if limit else "alcv_features.json"
    out = RESULTS_DIR / name
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[저장] {out}  (총 {len(results)}건)")
    if not limit:
        print("다음:  venv\Scripts\python.exe scripts_hallucination\run_evaluate.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="앞 N건만 처리(스모크)")
    args = ap.parse_args()
    asyncio.run(main(limit=args.limit))
