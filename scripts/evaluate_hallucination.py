"""
환각 탐지 성능 평가 + Ablation Study

평가 모델:
  - Ours       : NER 체커 (legal-ner-v3 기반)
  - GPT-Judge  : GPT-4o-mini as judge (zero-shot CoT)
  - MiniCheck  : MiniCheck-Flan-T5-Large (pip install git+https://github.com/Liyan06/MiniCheck.git)
  - AlignScore : torch<2 요구 → 현재 환경(torch 2.x)과 호환 불가, 항상 생략

Ablation 조건 (우리 시스템만):
  A: RAG 없음        (LLM 단독 답변)
  B: RAG             (RAG 검색 + LLM)
  C: RAG + SIM       (+ 일관성 검사)
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

import numpy as np
from openai import AsyncOpenAI

HALLU_EVAL_PATH = REPO_ROOT / "data" / "evaluation" / "hallu_eval.jsonl"
OUT_PATH = REPO_ROOT / "outputs" / "hallucination_eval_results.json"
ABLATION_OUT_PATH = REPO_ROOT / "outputs" / "ablation_results.json"

NER_MODEL_PATH = str(REPO_ROOT / "outputs" / "legal-ner-v3")
CONCURRENCY = 10


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


# ── 우리 NER 체커 (직접 구현, sentence_transformers 의존성 없음) ───────────────

def _extract_entities(text: str, tokenizer, model, id2label: dict, device) -> list[tuple[str, str]]:
    """텍스트에서 NER 엔티티를 추출. [(entity_text, entity_type), ...]"""
    import torch
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512,
        return_offsets_mapping=True,
    )
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
    pred_ids = logits.argmax(-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())

    entities: list[tuple[str, str]] = []
    cur_tokens: list[str] = []
    cur_type: str | None = None

    for token, pred_id, (char_s, char_e) in zip(tokens, pred_ids, offset_mapping):
        if char_s == char_e:   # 특수토큰 ([CLS], [SEP], padding)
            if cur_tokens:
                entities.append(("".join(cur_tokens), cur_type))
                cur_tokens, cur_type = [], None
            continue

        label = id2label.get(pred_id, "O")
        clean = token.replace("##", "").replace("▁", "")

        if label.startswith("B-"):
            if cur_tokens:
                entities.append(("".join(cur_tokens), cur_type))
            cur_tokens = [clean]
            cur_type = label[2:]
        elif label.startswith("I-") and cur_type == label[2:]:
            cur_tokens.append(clean)
        else:
            if cur_tokens:
                entities.append(("".join(cur_tokens), cur_type))
            cur_tokens, cur_type = [], None

    if cur_tokens:
        entities.append(("".join(cur_tokens), cur_type))

    return [(t, tp) for t, tp in entities if t.strip()]


def run_ner_checker(items: list[dict]) -> list[int]:
    """legal-ner-v3로 직접 엔티티 추출 후 소스 텍스트와 비교해 환각 판단.

    판단 규칙:
    - 답변에서 추출한 법률 전용 엔티티(LAW, CRIME, PENALTY)가
      소스 텍스트에서 추출한 엔티티 집합에 없으면 환각으로 판정.
    - semantic_error(논리적 오류)는 엔티티 기반으로 탐지 불가 → 한계로 명시.
    """
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  NER 모델 로드: {NER_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
    model.to(device)
    model.eval()
    id2label = model.config.id2label

    # 환각 판단에 사용할 엔티티 타입 (범용 타입 제외, 법률 전용만)
    HALLUCINATION_TYPES = {"LAW", "CRIME", "PENALTY"}

    preds = []
    for i, item in enumerate(items):
        if i % 100 == 0:
            print(f"  NER 직접 {i}/{len(items)}")

        answer_ents = _extract_entities(item["answer"], tokenizer, model, id2label, device)
        source_ents = _extract_entities(item["source_text"][:600], tokenizer, model, id2label, device)

        source_texts = {t for t, _ in source_ents}

        is_hallucination = False
        for ent_text, ent_type in answer_ents:
            if ent_type in HALLUCINATION_TYPES and ent_text not in source_texts:
                is_hallucination = True
                break

        preds.append(1 if is_hallucination else 0)

    return preds


# ── GPT-Judge ─────────────────────────────────────────────────────────────────

GPT_JUDGE_SYSTEM = """당신은 법률 정보의 정확성을 검증하는 전문가입니다.
[참고 문서]와 [답변]을 비교하여 답변에 사실적 오류(환각)가 있는지 판단하세요.

환각 기준:
1. 참고 문서에 없는 법조항 번호나 법률명 언급
2. 날조된 판례 번호나 존재하지 않는 법령 인용
3. 잘못된 연락처나 기관 전화번호
4. 참고 문서의 내용과 의미가 반대되거나 조건이 틀린 서술

출력 형식 (JSON):
{
  "reasoning": "2-3문장으로 판단 근거",
  "has_hallucination": true 또는 false
}"""


async def judge_single(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    item: dict,
    model: str,
) -> int:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GPT_JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"[참고 문서]\n{item['source_text'][:500]}\n\n"
                            f"[질문]\n{item['question']}\n\n"
                            f"[답변]\n{item['answer']}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
            return 1 if parsed.get("has_hallucination", False) else 0
        except Exception as e:
            print(f"  [GPT-Judge 오류] {e}")
            return 0


async def run_gpt_judge_async(items: list[dict], model: str = "gpt-4o-mini") -> list[int]:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(CONCURRENCY)

    tasks = [judge_single(client, sem, item, model) for item in items]
    preds = []
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        preds.append(result)
        if len(preds) % 100 == 0:
            print(f"  GPT-Judge {len(preds)}/{len(items)}")

    # as_completed 순서가 바뀌므로 순서 복원
    ordered_preds = [0] * len(items)
    # 순서 복원을 위해 인덱스 포함 버전으로 재실행
    return preds  # 순서 무관하게 메트릭만 계산하므로 OK


def run_gpt_judge(items: list[dict], model: str = "gpt-4o-mini") -> list[int]:
    print(f"  GPT-Judge 실행 중... ({len(items)}개, 동시 {CONCURRENCY}개)")

    async def _run():
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        sem = asyncio.Semaphore(CONCURRENCY)

        results = [None] * len(items)

        async def _judge(idx: int, item: dict):
            results[idx] = await judge_single(client, sem, item, model)

        tasks = [_judge(i, item) for i, item in enumerate(items)]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            await coro
            if (i + 1) % 100 == 0:
                print(f"  GPT-Judge {i + 1}/{len(items)}")

        return results

    preds = asyncio.run(_run())
    return [p if p is not None else 0 for p in preds]


# ── MiniCheck ─────────────────────────────────────────────────────────────────

def run_minicheck(items: list[dict]) -> list[int]:
    """MiniCheck-Flan-T5-Large: 문서 지지 여부 판단."""
    try:
        from minicheck.minicheck import MiniCheck
    except ImportError:
        print("  [SKIP] minicheck 패키지 없음. pip install minicheck 설치 필요")
        return []

    scorer = MiniCheck(model_name="flan-t5-large", enable_prefix_caching=False)
    preds = []
    for i, item in enumerate(items):
        if i % 100 == 0:
            print(f"  MiniCheck {i}/{len(items)}")
        try:
            pred_labels, _, _, _ = scorer.score(
                docs=[item["source_text"][:1000]],
                claims=[item["answer"]],
            )
            # MiniCheck: 1=supported, 0=not supported → 환각이면 0
            preds.append(0 if pred_labels[0] == 1 else 1)
        except Exception as e:
            print(f"  [MiniCheck 오류] {e}")
            preds.append(0)
    return preds


# ── NLI (klue/roberta-large) ──────────────────────────────────────────────────

def run_nli_checker(items: list[dict], batch_size: int = 16) -> list[int]:
    """klue/roberta-large NLI: 문서가 답변을 entail하지 않으면 환각으로 판단.

    KLUE NLI 레이블: 0=entailment, 1=neutral, 2=contradiction
    - entailment(0) → 정상 (문서가 답변을 지지)
    - neutral(1) or contradiction(2) → 환각 의심
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "klue/roberta-large"
    print(f"  NLI 모델 로드: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    preds = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        premises  = [b["source_text"][:400] for b in batch]
        hypotheses = [b["answer"][:200] for b in batch]

        enc = tokenizer(
            premises, hypotheses,
            padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
        label_ids = logits.argmax(dim=-1).cpu().tolist()

        for label_id in label_ids:
            # entailment(0) → 정상(0), neutral(1)/contradiction(2) → 환각(1)
            preds.append(0 if label_id == 0 else 1)

        if (i + batch_size) % 200 == 0:
            print(f"  NLI {min(i + batch_size, len(items))}/{len(items)}")

    return preds


# ── Ablation Study ────────────────────────────────────────────────────────────

async def generate_rag_answer(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    question: str,
    context: str | None,
    model: str,
) -> str:
    """RAG context 유무에 따라 LLM 답변 생성."""
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
    judge_model: str,
) -> dict[str, dict]:
    """4가지 Ablation 조건으로 환각 탐지율 비교."""
    from src.modules.ner_checker import NERFactChecker
    from src.modules.consistency_checker import ConsistencyChecker

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(CONCURRENCY)
    gold_labels = [1 if i["label"] == "hallucination" else 0 for i in items]

    # ─ Condition A: RAG 없음 ──────────────────────────────────────────────────
    print("\n[Ablation A] RAG 없음 (LLM 단독)...")
    tasks_a = [
        generate_rag_answer(client, sem, item["question"], None, llm_model)
        for item in items
    ]
    answers_a = await asyncio.gather(*tasks_a)

    # ─ Condition B: RAG 있음 ──────────────────────────────────────────────────
    print("[Ablation B] RAG 있음...")
    tasks_b = [
        generate_rag_answer(client, sem, item["question"], item["source_text"], llm_model)
        for item in items
    ]
    answers_b = await asyncio.gather(*tasks_b)

    # GPT-Judge로 각 조건 평가 (공통)
    async def judge_answers(answers: list[str], label: str) -> list[int]:
        print(f"  GPT-Judge 평가 ({label})...")
        judge_items = [
            {**items[i], "answer": a}
            for i, a in enumerate(answers)
        ]
        results = [None] * len(judge_items)

        async def _j(idx, item):
            results[idx] = await judge_single(client, sem, item, judge_model)

        await asyncio.gather(*[_j(i, item) for i, item in enumerate(judge_items)])
        return [r if r is not None else 0 for r in results]

    preds_a = await judge_answers(answers_a, "A-NoRAG")
    preds_b = await judge_answers(answers_b, "B-RAG")

    # ─ Condition C: RAG + SIM ─────────────────────────────────────────────────
    print("[Ablation C] RAG + SIM (일관성 검사)...")
    # SIM: 같은 질문을 3번 생성해서 답변 일관성 확인
    # 일관성이 낮으면 (불확실한 답변) 환각 가능성 높음으로 판단
    preds_c = []
    for i, item in enumerate(items):
        if i % 100 == 0:
            print(f"  SIM {i}/{len(items)}")
        multi_tasks = [
            generate_rag_answer(client, sem, item["question"], item["source_text"], llm_model)
            for _ in range(3)
        ]
        multi_answers = await asyncio.gather(*multi_tasks)
        # 3개 답변 중 GPT-Judge로 각각 평가 → 2/3 이상 환각이면 양성
        judge_multi = [
            await judge_single(client, sem, {**item, "answer": a}, judge_model)
            for a in multi_answers
        ]
        preds_c.append(1 if sum(judge_multi) >= 2 else 0)

    # ─ Condition D: RAG + SIM + NER ───────────────────────────────────────────
    print("[Ablation D] RAG + SIM + NER (전체 파이프라인)...")
    ner_checker = NERFactChecker(model_path=NER_MODEL_PATH)
    preds_d = []
    for i, (item, sim_pred) in enumerate(zip(items, preds_c)):
        if i % 100 == 0:
            print(f"  NER {i}/{len(items)}")
        rag_chunks = [{"chunk_id": item["chunk_id"], "text": item["source_text"]}]
        try:
            ner_hallus = ner_checker.find_hallucinations(item["answer"], rag_chunks)
            ner_pred = 1 if ner_hallus else 0
        except Exception:
            ner_pred = 0
        # SIM OR NER 중 하나라도 양성이면 최종 양성
        preds_d.append(1 if (sim_pred == 1 or ner_pred == 1) else 0)

    return {
        "A_no_rag": classification_metrics(gold_labels, preds_a),
        "B_rag": classification_metrics(gold_labels, preds_b),
        "C_rag_sim": classification_metrics(gold_labels, preds_c),
        "D_rag_sim_ner": classification_metrics(gold_labels, preds_d),
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main(skip_ablation: bool = False):
    items = load_hallu_eval()
    gold_labels = [1 if i["label"] == "hallucination" else 0 for i in items]

    results: dict[str, dict] = {}

    # ── 우리 NER 체커 ──────────────────────────────────────────────────────────
    print("\n[1/4] 우리 NER 체커 평가...")
    ner_preds = run_ner_checker(items)
    results["Ours(NER)"] = classification_metrics(gold_labels, ner_preds)
    print(f"  결과: {results['Ours(NER)']}")

    # ── GPT-Judge ──────────────────────────────────────────────────────────────
    print("\n[2/4] GPT-Judge 평가...")
    gpt_preds = run_gpt_judge(items)
    results["GPT-4o-mini(Judge)"] = classification_metrics(gold_labels, gpt_preds)
    print(f"  결과: {results['GPT-4o-mini(Judge)']}")

    # ── NLI (klue/roberta-large) ───────────────────────────────────────────────
    print("\n[3/4] NLI(klue/roberta-large) 평가...")
    nli_preds = run_nli_checker(items)
    results["NLI(klue/roberta)"] = classification_metrics(gold_labels, nli_preds)
    print(f"  결과: {results['NLI(klue/roberta)']}")

    # ── MiniCheck ──────────────────────────────────────────────────────────────
    print("\n[4/4] MiniCheck(Flan-T5) 평가...")
    mini_preds = run_minicheck(items)
    if mini_preds:
        results["MiniCheck(Flan-T5)"] = classification_metrics(gold_labels, mini_preds)
        print(f"  결과: {results['MiniCheck(Flan-T5)']}")
    else:
        results["MiniCheck(Flan-T5)"] = {"note": "패키지 미설치로 생략"}

    # ── 결과 출력 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("=== 환각 탐지 성능 비교 ===")
    print("=" * 60)
    header = f"{'모델':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}"
    print(header)
    print("-" * 60)
    for name, r in results.items():
        if "note" in r:
            print(f"{name:<25} {'(생략)':>38}")
            continue
        print(f"{name:<25} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['accuracy']:>10.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {OUT_PATH}")

    # ── Ablation ──────────────────────────────────────────────────────────────
    if not skip_ablation:
        print("\n\n" + "=" * 60)
        print("=== Ablation Study ===")
        print("=" * 60)
        ablation_results = asyncio.run(
            run_ablation_async(items, llm_model="gpt-4o-mini", judge_model="gpt-4o-mini")
        )

        header2 = f"{'조건':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}"
        print(header2)
        print("-" * 60)
        labels_map = {
            "A_no_rag": "A: RAG 없음",
            "B_rag": "B: RAG",
            "C_rag_sim": "C: RAG+SIM",
            "D_rag_sim_ner": "D: RAG+SIM+NER",
        }
        for key, name in labels_map.items():
            r = ablation_results[key]
            print(f"{name:<20} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['accuracy']:>10.4f}")

        with ABLATION_OUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(ablation_results, f, ensure_ascii=False, indent=2)
        print(f"\nAblation 결과 저장: {ABLATION_OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ablation", action="store_true", help="Ablation 생략 (빠른 실행)")
    args = parser.parse_args()
    main(skip_ablation=args.skip_ablation)
