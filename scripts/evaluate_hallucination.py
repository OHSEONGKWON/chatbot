"""
환각 탐지 성능 평가 + Ablation Study

평가 모델:
  - Ours             : NER 체커 (legal-ner-v3 기반)
  - GPT-Judge        : GPT-4o-mini as judge (zero-shot CoT)
  - NLI(klue-nli)    : Huffon/klue-roberta-base-nli (한국어 NLI 파인튜닝, torch 2.x 호환)
  - DeBERTa-NLI      : cross-encoder/nli-deberta-v3-small (다국어 NLI, 별도 패키지 불필요)

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
import re
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


_PHONE_PATTERNS = re.compile(
    r"\b\d{2,4}-\d{3,4}-\d{4}\b"          # 02-1234-5678 / 010-1234-5678
    r"|\b\d{4}-\d{4}\b"                     # 1566-0000
    r"|\b(?:119|112|182|117|111|128|129"    # 긴급·상담 단축번호
    r"|1330|1382|1811|1599|1544|1566"
    r"|1661|1600|1670|1644|1666|1688)\b",
)


def _extract_contacts(text: str) -> set[str]:
    """텍스트에서 전화번호·단축번호 패턴을 추출합니다."""
    return set(_PHONE_PATTERNS.findall(text))


def run_ner_checker(items: list[dict]) -> list[int]:
    """legal-ner-v3 + 연락처 규칙으로 환각 판단.

    판단 규칙:
    1. NER: 답변의 법률 전용 엔티티(LAW, CRIME, PENALTY)가 소스에 없으면 환각.
    2. Contact: 답변에 등장한 전화번호·단축번호가 소스에 없으면 환각.
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

    HALLUCINATION_TYPES = {"LAW", "CRIME", "PENALTY"}

    preds = []
    for i, item in enumerate(items):
        if i % 100 == 0:
            print(f"  NER+Contact {i}/{len(items)}")

        answer_ents = _extract_entities(item["answer"], tokenizer, model, id2label, device)
        source_ents = _extract_entities(item["source_text"][:600], tokenizer, model, id2label, device)
        source_texts = {t for t, _ in source_ents}

        # 규칙 1: NER 엔티티 비교
        is_hallucination = any(
            et in HALLUCINATION_TYPES and ev not in source_texts
            for ev, et in answer_ents
        )

        # 규칙 2: 연락처 비교 (소스에 없는 전화번호가 답변에 등장)
        if not is_hallucination:
            answer_contacts = _extract_contacts(item["answer"])
            if answer_contacts:
                source_contacts = _extract_contacts(item["source_text"])
                if answer_contacts - source_contacts:
                    is_hallucination = True

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
    for coro in asyncio.as_completed(tasks):
        result = await coro
        preds.append(result)
        if len(preds) % 100 == 0:
            print(f"  GPT-Judge {len(preds)}/{len(items)}")

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


# ── NLI (Huffon/klue-roberta-base-nli) ───────────────────────────────────────

def run_nli_checker(items: list[dict], batch_size: int = 16) -> list[int]:
    """MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7: 다국어 NLI 모델로 환각 판단.

    한국어 포함 다국어 NLI 파인튜닝, GPU 완전 호환.
    entailment → 정상 (문서가 답변을 지지)
    neutral / contradiction → 환각 의심
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    print(f"  NLI 모델 로드: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    id2label = model.config.id2label
    entail_idx = next(
        (k for k, v in id2label.items() if "entail" in str(v).lower()), 0
    )
    print(f"  레이블 매핑: {id2label}  (entailment index={entail_idx})")

    preds = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        premises   = [b["source_text"][:400] for b in batch]
        hypotheses = [b["answer"][:200]      for b in batch]

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
            preds.append(0 if label_id == entail_idx else 1)

        if (i + batch_size) % 200 == 0:
            print(f"  NLI {min(i + batch_size, len(items))}/{len(items)}")

    return preds


# ── DeBERTa-NLI (cross-encoder/nli-deberta-v3-small) ─────────────────────────

def run_deberta_nli_checker(items: list[dict], batch_size: int = 16) -> list[int]:
    """cross-encoder/nli-deberta-v3-small: NLI 기반 사실성 검증 (MiniCheck 대체).

    torch 2.x 완전 호환, 별도 패키지 불필요.
    entailment → 정상 (문서가 답변을 지지)
    neutral / contradiction → 환각 의심
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "cross-encoder/nli-deberta-v3-small"
    print(f"  DeBERTa-NLI 모델 로드: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    id2label = model.config.id2label
    entail_idx = next(
        (k for k, v in id2label.items() if "entail" in str(v).lower()), 1
    )
    print(f"  레이블 매핑: {id2label}  (entailment index={entail_idx})")

    preds = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        premises   = [b["source_text"][:400] for b in batch]
        hypotheses = [b["answer"][:200]      for b in batch]

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
            preds.append(0 if label_id == entail_idx else 1)

        if (i + batch_size) % 200 == 0:
            print(f"  DeBERTa-NLI {min(i + batch_size, len(items))}/{len(items)}")

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

    # ── NLI (mDeBERTa-v3 multilingual) ────────────────────────────────────────
    print("\n[3/4] NLI(mDeBERTa-v3-xnli-multilingual) 평가...")
    nli_preds = run_nli_checker(items)
    results["NLI(mDeBERTa-xnli)"] = classification_metrics(gold_labels, nli_preds)
    print(f"  결과: {results['NLI(mDeBERTa-xnli)']}")

    # ── DeBERTa-NLI (cross-encoder/nli-deberta-v3-small) ──────────────────────
    print("\n[4/4] DeBERTa-NLI(cross-encoder) 평가...")
    deberta_preds = run_deberta_nli_checker(items)
    results["DeBERTa-NLI(cross-encoder)"] = classification_metrics(gold_labels, deberta_preds)
    print(f"  결과: {results['DeBERTa-NLI(cross-encoder)']}")

    # ── 전체 결과 출력 ────────────────────────────────────────────────────────
    W = 65
    print("\n" + "=" * W)
    print("=== 환각 탐지 성능 비교 (전체 800건) ===")
    print("=" * W)
    header = f"{'모델':<28} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}"
    print(header)
    print("-" * W)
    for name, r in results.items():
        print(f"{name:<28} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['accuracy']:>10.4f}")

    # ── 법률 엔티티 환각 특화 비교 ────────────────────────────────────────────
    # 우리 NER 체커의 설계 목적(조항번호·법령명 환각)에 맞는 공정 비교
    # 대상: article_number_error + forbidden_law_injection + none(정상)
    LEGAL_ENTITY_TYPES = {"article_number_error", "forbidden_law_injection", "none"}
    legal_idx = [i for i, item in enumerate(items) if item["hallu_type"] in LEGAL_ENTITY_TYPES]
    legal_gold = [gold_labels[i] for i in legal_idx]

    all_preds = {
        "Ours(NER+Contact)":          ner_preds,
        "GPT-4o-mini(Judge)":         gpt_preds,
        "NLI(mDeBERTa-xnli)":         nli_preds,
        "DeBERTa-NLI(cross-encoder)": deberta_preds,
    }
    legal_results: dict[str, dict] = {}
    for name, preds in all_preds.items():
        legal_results[name] = classification_metrics(legal_gold, [preds[i] for i in legal_idx])

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
        marker = " ◀ 우리 모델" if name == "Ours(NER+Contact)" else ""
        print(f"{name:<28} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['accuracy']:>10.4f}{marker}")

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
