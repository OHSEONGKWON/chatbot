"""
환각 탐지 성능 평가 + Ablation Study

평가 모델:
  - Ours             : NER 체커 (legal-ner-v3 기반)
  - XLM-RoBERTa-XNLI : joeddav/xlm-roberta-large-xnli (한국어 포함 15개 언어 XNLI, GPU 호환)
  - mDeBERTa-NLI     : MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (다국어 NLI)
  - DeBERTa-NLI      : cross-encoder/nli-deberta-v3-small (Cross-encoder NLI)
  - BERTScore        : xlm-roberta-large 기반 의미 유사도 (threshold=0.85)

Ablation 조건 (우리 시스템만):
  A: RAG 없음        (LLM 단독 답변)
  B: RAG             (RAG 검색 + LLM)
  C: RAG + SIM       (+ 일관성 검사, ConsistencyChecker)
  D: RAG + SIM + NER (전체 파이프라인)

사용법:
  python scripts/evaluate_hallucination.py
  python scripts/evaluate_hallucination.py --skip-ablation  # Ablation 생략 (빠른 실행)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from openai import AsyncOpenAI

HALLU_EVAL_PATH = REPO_ROOT / "data" / "evaluation" / "hallu_eval.jsonl"
OUT_PATH = REPO_ROOT / "outputs" / "hallucination_eval_results.json"
ABLATION_OUT_PATH = REPO_ROOT / "outputs" / "ablation_results.json"

NER_MODEL_PATH = str(REPO_ROOT / "outputs" / "legal-ner-v3")
CONCURRENCY = 10
BERTSCORE_THRESHOLD = 0.85


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_hallu_eval() -> list[dict]:
    items = []
    with HALLU_EVAL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"환각 평가셋: {len(items)}개 로드")
    from collections import Counter
    types = Counter(i["hallu_type"] for i in items)
    print(f"  분포: {dict(types)}")
    return items


# ── 지표 계산 ─────────────────────────────────────────────────────────────────

def classification_metrics(labels: list[int], preds: list[int]) -> dict:
    """Binary classification: 1=hallucination, 0=normal."""
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / len(labels) if labels else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ── 우리 NER 체커 (6개 레이어 전체 사용) ────────────────────────────────────────

def run_ner_checker(items: list[dict]) -> list[int]:
    """실제 NERFactChecker 6개 레이어 전체로 환각 탐지.

    Layer 1  : 엔티티 ↔ RAG 후보 fuzzy/semantic 매칭
    Layer 2  : 이슈-도메인 법률 교차 검증 (금지 법률)
    Layer 2.5: KLUE-RoBERTa NLI contradiction 검사
    Layer 2.6: 부정문 반전 패턴 (의무 서술 충돌)
    Layer 3  : 범죄명-법조항 번호 정합성
    Layer 3.5: 수치 범위 및 이상/이하 방향 비교
    Layer 4  : RAG 미지지 엔티티 플래그
    """
    from src.modules.ner_checker import NERFactChecker

    print(f"  NERFactChecker 초기화 (6-layer): {NER_MODEL_PATH}")
    checker = NERFactChecker(model_path=NER_MODEL_PATH)

    preds = []
    for i, item in enumerate(items):
        if i % 100 == 0:
            print(f"  NER {i}/{len(items)}")

        rag_docs = [{"text": item["source_text"], "metadata": {}}]
        hallucinations = checker.find_hallucinations(item["answer"], rag_docs)
        preds.append(1 if hallucinations else 0)

    return preds


# ── 공통 NLI 유틸 ─────────────────────────────────────────────────────────────

def _gpu_healthy() -> bool:
    """GPU가 실제로 사용 가능한 상태인지 간단한 텐서 연산으로 검증."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        t = torch.zeros(1, device="cuda")
        _ = (t + 1).sum().item()
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def _load_nli_model(model_name: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if _gpu_healthy() else "cpu")
    print(f"  NLI 모델 로드: {model_name}  (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()
    id2label = model.config.id2label
    entail_idx = next((k for k, v in id2label.items() if "entail" in str(v).lower()), 0)
    print(f"  레이블 매핑: {id2label}  (entailment index={entail_idx})")
    return tokenizer, model, entail_idx, device


def _nli_batch_predict(
    premises: list[str],
    hypotheses: list[str],
    tokenizer,
    model,
    entail_idx: int,
    device,
    batch_size: int = 16,
    label: str = "NLI",
) -> list[int]:
    import torch
    preds = []
    for i in range(0, len(premises), batch_size):
        bp = premises[i : i + batch_size]
        bh = hypotheses[i : i + batch_size]
        enc = tokenizer(bp, bh, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        for label_id in logits.argmax(dim=-1).cpu().tolist():
            preds.append(0 if label_id == entail_idx else 1)
        if (i + batch_size) % 200 == 0:
            print(f"  {label} {min(i + batch_size, len(premises))}/{len(premises)}")
    return preds


# ── KLUE-RoBERTa NLI (한국어 특화) ───────────────────────────────────────────

def run_klue_nli_checker(items: list[dict], batch_size: int = 16) -> list[int]:
    """joeddav/xlm-roberta-large-xnli: 한국어 포함 15개 언어 XNLI 파인튜닝, GPU 호환.

    entailment → 정상, neutral/contradiction → 환각 의심.
    """
    tokenizer, model, entail_idx, device = _load_nli_model("joeddav/xlm-roberta-large-xnli")
    sources = [b["source_text"][:400] for b in items]
    answers = [b["answer"][:200] for b in items]
    return _nli_batch_predict(sources, answers, tokenizer, model, entail_idx, device, batch_size, "XLM-RoBERTa-XNLI")


# ── mDeBERTa-NLI (다국어) ─────────────────────────────────────────────────────

def run_nli_checker(items: list[dict], batch_size: int = 16) -> list[int]:
    """MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7: 다국어 NLI 모델.

    entailment → 정상, neutral/contradiction → 환각 의심.
    """
    tokenizer, model, entail_idx, device = _load_nli_model(
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )
    sources = [b["source_text"][:400] for b in items]
    answers = [b["answer"][:200] for b in items]
    return _nli_batch_predict(sources, answers, tokenizer, model, entail_idx, device, batch_size, "mDeBERTa-NLI")


# ── DeBERTa-NLI Cross-encoder ─────────────────────────────────────────────────

def run_deberta_nli_checker(items: list[dict], batch_size: int = 16) -> list[int]:
    """cross-encoder/nli-deberta-v3-small: Cross-encoder NLI 기반 환각 판단.

    entailment → 정상, neutral/contradiction → 환각 의심.
    """
    tokenizer, model, entail_idx, device = _load_nli_model("cross-encoder/nli-deberta-v3-small")
    sources = [b["source_text"][:400] for b in items]
    answers = [b["answer"][:200] for b in items]
    return _nli_batch_predict(sources, answers, tokenizer, model, entail_idx, device, batch_size, "DeBERTa-NLI")


# ── BERTScore ─────────────────────────────────────────────────────────────────

def run_bertscore_checker(items: list[dict], threshold: float = BERTSCORE_THRESHOLD) -> list[int]:
    """xlm-roberta-large 기반 BERTScore로 환각 판단.

    소스(reference)와 답변(hypothesis) 간 F1이 threshold 미만이면 환각으로 판정.
    threshold=0.85는 고정값 — 논문에서 dev set 기반 조정 가능.
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("  [오류] bert-score 패키지 없음: pip install bert-score")
        return [0] * len(items)

    sources = [item["source_text"][:500] for item in items]
    answers = [item["answer"][:300] for item in items]

    print(f"  BERTScore 계산 중... (model=xlm-roberta-large, threshold={threshold})")
    _P, _R, F1 = bert_score_fn(
        answers,
        sources,
        lang="ko",
        model_type="xlm-roberta-large",
        verbose=False,
        batch_size=16,
    )

    f1_list = F1.tolist()
    preds = [1 if f < threshold else 0 for f in f1_list]
    print(f"  평균 BERTScore F1: {sum(f1_list)/len(f1_list):.4f}")
    return preds


# ── Ablation Study ────────────────────────────────────────────────────────────

async def generate_rag_answer(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    question: str,
    context: str | None,
    model: str,
) -> str:
    async with sem:
        if context:
            system = "당신은 대한민국 법률 상담사입니다. 주어진 참고 자료를 근거로 답변하세요."
            user = f"[참고 자료]\n{context[:800]}\n\n[질문]\n{question}"
        else:
            system = "당신은 대한민국 법률 상담사입니다. 알고 있는 법률 지식으로 답변하세요."
            user = f"[질문]\n{question}"

        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return ""


async def run_ablation_async(
    items: list[dict],
    llm_model: str,
) -> dict[str, dict]:
    """4가지 Ablation 조건으로 환각 탐지율 비교.

    A/B 조건의 생성된 답변 평가는 mDeBERTa NLI를 oracle로 사용 (LLM-free).
    """
    import torch
    from src.modules.ner_checker import NERFactChecker
    from src.modules.consistency_checker import ConsistencyChecker

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(CONCURRENCY)
    gold_labels = [1 if i["label"] == "hallucination" else 0 for i in items]

    # A/B oracle용 NLI 모델 로드 (LLM 대신 판별 모델로 판정)
    print("  [Ablation oracle] mDeBERTa NLI 로드...")
    nli_tok, nli_mod, nli_entail_idx, nli_device = _load_nli_model(
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )
    sources_for_nli = [item["source_text"][:400] for item in items]

    def _nli_oracle(answers: list[str]) -> list[int]:
        return _nli_batch_predict(
            sources_for_nli, answers,
            nli_tok, nli_mod, nli_entail_idx, nli_device,
            label="ablation-NLI",
        )

    # ─ Condition A: RAG 없음 ──────────────────────────────────────────────────
    print("\n[Ablation A] RAG 없음 (LLM 단독)...")
    answers_a = await asyncio.gather(*[
        generate_rag_answer(client, sem, item["question"], None, llm_model)
        for item in items
    ])
    preds_a = _nli_oracle(answers_a)

    # ─ Condition B: RAG 있음 ──────────────────────────────────────────────────
    print("[Ablation B] RAG 있음...")
    answers_b = await asyncio.gather(*[
        generate_rag_answer(client, sem, item["question"], item["source_text"], llm_model)
        for item in items
    ])
    preds_b = _nli_oracle(answers_b)

    # NLI 모델 해제 (GPU 메모리 확보)
    del nli_mod

    # ─ Condition C: RAG + SIM (실제 ConsistencyChecker 사용) ─────────────────
    print("[Ablation C] RAG + SIM (ConsistencyChecker 실행)...")
    sim_sem = asyncio.Semaphore(5)  # 내부 LLM 호출 11개이므로 동시 실행 제한
    sim_checker = ConsistencyChecker()

    async def _run_sim_single(item: dict) -> tuple[int, str]:
        async with sim_sem:
            rag_docs = [{
                "text": item["source_text"],
                "content": item["source_text"],
                "metadata": {},
                "score": 1.0,
            }]
            try:
                is_reliable, original_answer, _score, _ = await sim_checker.run(
                    question=item["question"],
                    rag_docs=rag_docs,
                )
                return (0 if is_reliable else 1), original_answer
            except Exception as e:
                print(f"  [ConsistencyChecker 오류] {e}")
                return 0, ""

    results_c = await asyncio.gather(*[_run_sim_single(item) for item in items])
    preds_c = [r[0] for r in results_c]
    answers_c = [r[1] for r in results_c]
    print(f"  ConsistencyChecker 완료: 환각 탐지 {sum(preds_c)}/{len(preds_c)}")

    # ─ Condition D: RAG + SIM + NER ───────────────────────────────────────────
    print("[Ablation D] RAG + SIM + NER (전체 파이프라인)...")
    ner_checker = NERFactChecker(model_path=NER_MODEL_PATH)
    preds_d = []
    for i, (item, sim_pred, answer_c) in enumerate(zip(items, preds_c, answers_c)):
        if i % 100 == 0:
            print(f"  NER {i}/{len(items)}")
        rag_chunks = [{"chunk_id": item.get("chunk_id", str(i)), "text": item["source_text"]}]
        answer_for_ner = answer_c if answer_c else item["answer"]
        try:
            ner_hallus = ner_checker.find_hallucinations(answer_for_ner, rag_chunks)
            ner_pred = 1 if ner_hallus else 0
        except Exception:
            ner_pred = 0
        preds_d.append(1 if (sim_pred == 1 or ner_pred == 1) else 0)

    return {
        "A_no_rag":     classification_metrics(gold_labels, preds_a),
        "B_rag":        classification_metrics(gold_labels, preds_b),
        "C_rag_sim":    classification_metrics(gold_labels, preds_c),
        "D_rag_sim_ner": classification_metrics(gold_labels, preds_d),
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main(skip_ablation: bool = False):
    items = load_hallu_eval()
    gold_labels = [1 if i["label"] == "hallucination" else 0 for i in items]

    results: dict[str, dict] = {}

    # ── 우리 NER 체커 ──────────────────────────────────────────────────────────
    print("\n[1/5] 우리 NER 체커 평가...")
    ner_preds = run_ner_checker(items)
    results["Ours(NER)"] = classification_metrics(gold_labels, ner_preds)
    print(f"  결과: {results['Ours(NER)']}")

    # NERFactChecker 내부 모델이 GPU를 점유하므로 캐시 해제
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    # ── KLUE-RoBERTa NLI (한국어 특화) ────────────────────────────────────────
    print("\n[2/5] XLM-RoBERTa-XNLI(joeddav) 평가...")
    klue_preds = run_klue_nli_checker(items)
    results["XLM-RoBERTa-XNLI"] = classification_metrics(gold_labels, klue_preds)
    print(f"  결과: {results['XLM-RoBERTa-XNLI']}")

    # ── mDeBERTa NLI (다국어) ─────────────────────────────────────────────────
    print("\n[3/5] NLI(mDeBERTa-v3-xnli-multilingual) 평가...")
    nli_preds = run_nli_checker(items)
    results["NLI(mDeBERTa-xnli)"] = classification_metrics(gold_labels, nli_preds)
    print(f"  결과: {results['NLI(mDeBERTa-xnli)']}")

    # ── DeBERTa-NLI Cross-encoder ──────────────────────────────────────────────
    print("\n[4/5] DeBERTa-NLI(cross-encoder) 평가...")
    deberta_preds = run_deberta_nli_checker(items)
    results["DeBERTa-NLI(cross-encoder)"] = classification_metrics(gold_labels, deberta_preds)
    print(f"  결과: {results['DeBERTa-NLI(cross-encoder)']}")

    # ── BERTScore ─────────────────────────────────────────────────────────────
    print(f"\n[5/5] BERTScore(xlm-roberta-large, threshold={BERTSCORE_THRESHOLD}) 평가...")
    bert_preds = run_bertscore_checker(items)
    results["BERTScore(xlm-roberta)"] = classification_metrics(gold_labels, bert_preds)
    print(f"  결과: {results['BERTScore(xlm-roberta)']}")

    # ── 전체 결과 출력 ────────────────────────────────────────────────────────
    W = 70
    print("\n" + "=" * W)
    print("=== 환각 탐지 성능 비교 ===")
    print("=" * W)
    header = f"{'모델':<30} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}"
    print(header)
    print("-" * W)
    for name, r in results.items():
        marker = " ◀" if name == "Ours(NER)" else ""
        print(
            f"{name:<30} {r['precision']:>10.4f} {r['recall']:>8.4f} "
            f"{r['f1']:>8.4f} {r['accuracy']:>10.4f}{marker}"
        )

    # ── 법률 엔티티 환각 특화 비교 ────────────────────────────────────────────
    LEGAL_ENTITY_TYPES = {"article_number_error", "forbidden_law_injection", "none"}
    legal_idx = [i for i, item in enumerate(items) if item["hallu_type"] in LEGAL_ENTITY_TYPES]
    legal_gold = [gold_labels[i] for i in legal_idx]

    all_preds = {
        "Ours(NER)":                  ner_preds,
        "XLM-RoBERTa-XNLI":          klue_preds,
        "NLI(mDeBERTa-xnli)":         nli_preds,
        "DeBERTa-NLI(cross-encoder)": deberta_preds,
        "BERTScore(xlm-roberta)":     bert_preds,
    }
    legal_results: dict[str, dict] = {
        name: classification_metrics(legal_gold, [preds[i] for i in legal_idx])
        for name, preds in all_preds.items()
    }

    n_art  = sum(1 for item in items if item["hallu_type"] == "article_number_error")
    n_law  = sum(1 for item in items if item["hallu_type"] == "forbidden_law_injection")
    n_none = sum(1 for item in items if item["hallu_type"] == "none")

    print(f"\n{'=' * W}")
    print(f"=== 법률 엔티티 환각 탐지 특화 비교 ===")
    print(f"    (조항번호 오류 {n_art}건 + 법령명 조작 {n_law}건 + 정상 {n_none}건, 계 {n_art+n_law+n_none}건)")
    print(f"{'=' * W}")
    print(header)
    print("-" * W)
    for name, r in legal_results.items():
        marker = " ◀ 우리 모델" if name == "Ours(NER)" else ""
        print(
            f"{name:<30} {r['precision']:>10.4f} {r['recall']:>8.4f} "
            f"{r['f1']:>8.4f} {r['accuracy']:>10.4f}{marker}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined = {"overall": results, "legal_entity_focused": legal_results}
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {OUT_PATH}")

    # ── Ablation ──────────────────────────────────────────────────────────────
    if not skip_ablation:
        print("\n\n" + "=" * 60)
        print("=== Ablation Study ===")
        print("=" * 60)
        ablation_results = asyncio.run(
            run_ablation_async(items, llm_model="gpt-4o-mini")
        )

        header2 = f"{'조건':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}"
        print(header2)
        print("-" * 60)
        labels_map = {
            "A_no_rag":      "A: RAG 없음",
            "B_rag":         "B: RAG",
            "C_rag_sim":     "C: RAG+SIM",
            "D_rag_sim_ner": "D: RAG+SIM+NER",
        }
        for key, name in labels_map.items():
            r = ablation_results[key]
            print(f"{name:<20} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['accuracy']:>10.4f}")

        ABLATION_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ABLATION_OUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(ablation_results, f, ensure_ascii=False, indent=2)
        print(f"\nAblation 결과 저장: {ABLATION_OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ablation", action="store_true", help="Ablation 생략 (빠른 실행)")
    args = parser.parse_args()
    main(skip_ablation=args.skip_ablation)
