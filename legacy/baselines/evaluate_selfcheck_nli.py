"""SelfCheckGPT-NLI (논문 Section 5.4)"""
import asyncio
import json
import sys
import re
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.modules.llm_client import llm_client
sys.path.append(str(Path(__file__).resolve().parents[1]))
from alcv.dataset import load_dataset

# ==================== 설정 ====================
NLI_MODEL_NAME = "Huffon/klue-roberta-base-nli"
THRESHOLD = 0.15  # 공격적 설정 (Recall 최대화)
N_SAMPLES = 5
DEVICE = "cpu"  # NLI는 CPU 사용 (GPU 인덱스 오류 회피)

# NLI 모델 전역
_NLI_MODEL = None
_NLI_TOKENIZER = None


def load_nli_model():
    """NLI 모델 로딩"""
    global _NLI_MODEL, _NLI_TOKENIZER

    if _NLI_MODEL is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            print(f"[NLI 모델 로딩] {NLI_MODEL_NAME}")
            _NLI_TOKENIZER = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
            _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
            _NLI_MODEL.to(DEVICE)
            _NLI_MODEL.eval()
            print(f"[완료] Device: {DEVICE}\n")
        except Exception as e:
            print(f"[ERROR] NLI 모델 로딩 실패: {e}")
            return None, None

    return _NLI_MODEL, _NLI_TOKENIZER


def split_sentences(text: str) -> list[str]:
    """문장 분해"""
    sentences = re.split(r'[.!?]\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
    return sentences


async def generate_samples(question: str, n_samples: int = N_SAMPLES) -> list[str]:
    """샘플 생성"""
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


async def selfcheck_nli_score(question: str, answer: str) -> tuple[bool, float]:
    """
    SelfCheckGPT-NLI
    P(contradict) = exp(zc) / (exp(ze) + exp(zc))
    """
    try:
        # 1. 샘플 생성
        samples = await generate_samples(question, n_samples=N_SAMPLES)
        if len(samples) == 0:
            return False, 0.0

        # 2. 문장 분해
        sentences = split_sentences(answer)
        if len(sentences) == 0:
            return False, 0.0

        # 3. NLI 모델
        model, tokenizer = load_nli_model()
        if model is None or tokenizer is None:
            return False, 0.0

        # 4. 각 문장의 모순 점수
        sentence_scores = []

        for sentence in sentences:
            contradiction_probs = []

            for sample in samples:
                try:
                    # Tokenize
                    inputs = tokenizer(
                        sample[:200],      # premise
                        sentence[:200],    # hypothesis
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    )

                    # RoBERTa는 token_type_ids 불필요 (제거)
                    if "token_type_ids" in inputs:
                        del inputs["token_type_ids"]

                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                    # NLI 추론
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits[0]

                    # P(contradict)
                    ze = logits[0]  # entailment
                    zc = logits[2]  # contradiction

                    p_contradict = torch.exp(zc) / (torch.exp(ze) + torch.exp(zc))
                    contradiction_probs.append(p_contradict.item())

                except Exception as e:
                    # 조용히 넘어감 (일부 실패 허용)
                    continue

            if contradiction_probs:
                avg_contradict = np.mean(contradiction_probs)
                sentence_scores.append(avg_contradict)

        if not sentence_scores:
            return False, 0.0

        # 5. 전체 평균
        final_score = np.mean(sentence_scores)

        # 6. Threshold
        is_hallucinated = final_score > THRESHOLD

        return is_hallucinated, final_score

    except Exception as e:
        print(f"[ERROR] SelfCheck-NLI 실패: {e}")
        return False, 0.0


async def evaluate_selfcheck_nli(cases):
    """SelfCheckGPT-NLI 평가"""
    print("\n" + "="*70)
    print("SelfCheckGPT-NLI 평가")
    print(f"Threshold: {THRESHOLD}")
    print(f"Samples: {N_SAMPLES}")
    print(f"Device: {DEVICE}")
    print("="*70)

    results = []

    for case in tqdm(cases, desc="평가 중"):
        try:
            is_hallu_detected, score = await selfcheck_nli_score(
                case.question,
                case.answer
            )

            results.append({
                "hallu_id": case.hallu_id,
                "true_label": bool(case.is_hallucinated),
                "predicted": bool(is_hallu_detected),
                "contradiction_score": float(score),
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
                "contradiction_score": 0.0,
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

    output_path = output_dir / "selfcheck_nli_results.json"
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

    # 점수 분포
    hallu_scores = [r["contradiction_score"] for r in results if r["true_label"]]
    normal_scores = [r["contradiction_score"] for r in results if not r["true_label"]]

    if hallu_scores and normal_scores:
        print(f"\n=== 점수 분포 ===")
        print(f"환각 평균: {np.mean(hallu_scores):.4f}")
        print(f"정상 평균: {np.mean(normal_scores):.4f}")

    return results


if __name__ == "__main__":
    cases = load_dataset()
    asyncio.run(evaluate_selfcheck_nli(cases))
