"""우리 모델: BERTScore (일관성) + NER + NLI"""
import asyncio
import json
import sys
import re
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.modules.ner_checker import NERFactChecker
from src.modules.rag import retriever
from src.modules.llm_client import llm_client
from src.modules.embedder import get_embedder
from load_dataset import load_dataset

# ==================== 설정 ====================
BERTSCORE_THRESHOLD = 0.05  # 공격적 설정 (Recall 최대화)
N_SAMPLES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def split_sentences(text: str) -> list[str]:
    """텍스트를 문장으로 분해"""
    sentences = re.split(r'[.!?]\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
    return sentences


async def generate_samples(question: str, n_samples: int = N_SAMPLES) -> list[str]:
    """샘플 답변 생성"""
    samples = []
    for i in range(n_samples):
        try:
            sample = await llm_client.complete(
                prompt=f"다음 법률 질문에 답변하세요:\n\n{question}",
                temperature=1.0,
                max_tokens=200
            )
            samples.append(sample.strip())
        except Exception as e:
            print(f"[WARNING] 샘플 생성 실패 ({i+1}/{n_samples}): {e}")
            continue
    return samples


async def selfcheck_bertscore(question: str, answer: str) -> tuple[bool, float]:
    """
    SelfCheckGPT-BERTScore (논문 Section 5.1)
    S_BERT(i) = 1 - (1/N) * Σ max_k (B(r_i, s_n^k))
    """
    try:
        # 1. 샘플 생성
        samples = await generate_samples(question, n_samples=N_SAMPLES)
        if len(samples) == 0:
            return False, 0.0

        # 2. 문장 분해
        answer_sentences = split_sentences(answer)
        if len(answer_sentences) == 0:
            return False, 0.0

        # 3. Embedder
        embedder = get_embedder()
        if embedder is None:
            return False, 0.0

        # 4. 각 문장의 hallucination score
        sentence_scores = []

        for r_i in answer_sentences:
            max_similarities = []

            for sample in samples:
                sample_sentences = split_sentences(sample)
                if len(sample_sentences) == 0:
                    continue

                try:
                    r_i_emb = embedder.encode(r_i, convert_to_tensor=True, device=DEVICE)
                    sample_embs = embedder.encode(sample_sentences, convert_to_tensor=True, device=DEVICE)

                    similarities = torch.cosine_similarity(
                        r_i_emb.unsqueeze(0),
                        sample_embs
                    )

                    max_sim = similarities.max().item()
                    max_similarities.append(max_sim)

                except Exception:
                    continue

            if max_similarities:
                avg_max_sim = np.mean(max_similarities)
                s_bert_i = 1 - avg_max_sim
                sentence_scores.append(s_bert_i)

        if not sentence_scores:
            return False, 0.0

        # 5. 전체 평균
        final_score = np.mean(sentence_scores)

        # 6. Threshold
        is_hallucinated = final_score > BERTSCORE_THRESHOLD

        return is_hallucinated, final_score

    except Exception as e:
        print(f"[ERROR] BERTScore 실패: {e}")
        return False, 0.0


async def evaluate_ours(cases):
    """우리 모델: BERTScore + NER + NLI"""
    print("\n" + "="*70)
    print("우리 모델 평가 (BERTScore + NER + NLI)")
    print(f"BERTScore Threshold: {BERTSCORE_THRESHOLD}")
    print(f"Samples: {N_SAMPLES}")
    print(f"Device: {DEVICE}")
    print("="*70)

    # 초기화
    ner_checker = NERFactChecker()
    await retriever.warmup()

    results = []

    for case in tqdm(cases, desc="평가 중"):
        try:
            # 1. BERTScore (일관성)
            is_inconsistent, consistency_score = await selfcheck_bertscore(
                case.question,
                case.answer
            )

            # 2. RAG 검색
            rag_docs = await retriever.retrieve_async(case.question, top_k=5)

            # 3. NER + NLI
            ner_result = await ner_checker.check(case.answer, rag_docs)

            # 4. OR 조건
            is_hallu_detected = (
                is_inconsistent or
                len(ner_result.mismatched_entities) > 0
            )

            results.append({
                "hallu_id": case.hallu_id,
                "true_label": bool(case.is_hallucinated),
                "predicted": bool(is_hallu_detected),
                "consistency_score": float(consistency_score),
                "is_inconsistent": bool(is_inconsistent),
                "ner_hallucinations": int(len(ner_result.mismatched_entities)),
                "hallu_type": case.hallu_type
            })

        except Exception as e:
            print(f"\n[ERROR] {case.hallu_id}: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                "hallu_id": case.hallu_id,
                "true_label": bool(case.is_hallucinated),
                "predicted": False,
                "consistency_score": 0.0,
                "is_inconsistent": False,
                "ner_hallucinations": 0,
                "hallu_type": case.hallu_type
            })

        # 중간 체크포인트
        if len(results) % 100 == 0:
            tp = sum(1 for r in results if r["true_label"] and r["predicted"])
            fp = sum(1 for r in results if not r["true_label"] and r["predicted"])
            fn = sum(1 for r in results if r["true_label"] and not r["predicted"])

            temp_p = tp / (tp + fp) if (tp + fp) > 0 else 0
            temp_r = tp / (tp + fn) if (tp + fn) > 0 else 0
            temp_f1 = 2 * temp_p * temp_r / (temp_p + temp_r) if (temp_p + temp_r) > 0 else 0

            print(f"\n[중간 결과 {len(results)}/800] P={temp_p:.3f}, R={temp_r:.3f}, F1={temp_f1:.3f}")

    # 저장
    output_dir = Path("scripts_hallucination/results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "ours_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[저장] {output_path}")

    # 최종 통계
    tp = sum(1 for r in results if r["true_label"] and r["predicted"])
    fp = sum(1 for r in results if not r["true_label"] and r["predicted"])
    fn = sum(1 for r in results if r["true_label"] and not r["predicted"])
    tn = sum(1 for r in results if not r["true_label"] and not r["predicted"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n=== 최종 결과 ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nTP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    # 단계별 기여도
    consistency_detected = sum(1 for r in results if r["true_label"] and r["is_inconsistent"])
    ner_detected = sum(1 for r in results if r["true_label"] and r["ner_hallucinations"] > 0)

    print(f"\n=== 단계별 기여도 ===")
    print(f"BERTScore로 탐지: {consistency_detected}/{tp} ({consistency_detected/tp*100 if tp > 0 else 0:.1f}%)")
    print(f"NER/NLI로 탐지: {ner_detected}/{tp} ({ner_detected/tp*100 if tp > 0 else 0:.1f}%)")

    return results


if __name__ == "__main__":
    cases = load_dataset()
    asyncio.run(evaluate_ours(cases))
