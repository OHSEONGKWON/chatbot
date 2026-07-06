"""MetaQA 평가 (메타 질문 기반)"""
import asyncio
import json
import sys
import re
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.modules.llm_client import llm_client
sys.path.append(str(Path(__file__).resolve().parents[1]))
from alcv.dataset import load_dataset


async def extract_facts(answer: str) -> list[str]:
    """답변에서 주요 사실 추출"""
    prompt = f"""다음 법률 답변에서 검증 가능한 사실만 추출하세요.

[답변]
{answer}

사실만 JSON 배열로 출력하세요. 예: ["사실1", "사실2", ...]
"""

    try:
        response = await llm_client.complete(prompt=prompt, temperature=0.0)

        # JSON 파싱
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            facts = json.loads(match.group())
            return facts[:5]  # 최대 5개

    except Exception as e:
        print(f"[WARNING] 사실 추출 실패: {e}")

    return []


async def verify_fact(fact: str, answer: str) -> bool:
    """메타 질문으로 사실 검증"""
    prompt = f"""다음 답변에서 이 사실이 정확한지 확인하세요.

[답변]
{answer}

[확인할 사실]
{fact}

이 사실이 정확하면 "예", 부정확하면 "아니오"만 출력하세요:"""

    try:
        response = await llm_client.complete(prompt=prompt, temperature=0.0)

        # "예" 또는 "아니오" 추출
        response_lower = response.lower()

        if "예" in response_lower or "yes" in response_lower:
            return True
        elif "아니" in response_lower or "no" in response_lower:
            return False

    except Exception as e:
        print(f"[WARNING] 검증 실패: {e}")

    return True  # 기본값: 정확하다고 가정


async def evaluate_metaqa(cases):
    """MetaQA로 환각 탐지"""
    print("\n" + "="*70)
    print("MetaQA 평가")
    print("="*70)

    results = []

    for case in tqdm(cases, desc="평가 중"):
        try:
            # 1. 사실 추출
            facts = await extract_facts(case.answer)

            # 2. 각 사실 검증
            verified = []
            for fact in facts:
                is_accurate = await verify_fact(fact, case.answer)
                verified.append(is_accurate)

            # 3. 환각 판정
            if verified:
                accuracy_rate = sum(verified) / len(verified)
                is_hallu_detected = accuracy_rate < 0.8  # 80% 미만이면 환각
            else:
                is_hallu_detected = False  # 사실 없으면 정상

            results.append({
                "hallu_id": case.hallu_id,
                "true_label": case.is_hallucinated,
                "predicted": is_hallu_detected,
                "num_facts": len(facts),
                "accuracy_rate": accuracy_rate if verified else 1.0,
                "hallu_type": case.hallu_type
            })

        except Exception as e:
            print(f"\n[ERROR] {case.hallu_id}: {e}")
            results.append({
                "hallu_id": case.hallu_id,
                "true_label": case.is_hallucinated,
                "predicted": False,
                "num_facts": 0,
                "accuracy_rate": 1.0,
                "hallu_type": case.hallu_type
            })

    # 저장
    output_dir = Path("scripts_hallucination/results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "metaqa_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[저장] {output_path}")

    # 통계
    tp = sum(1 for r in results if r["true_label"] and r["predicted"])
    fp = sum(1 for r in results if not r["true_label"] and r["predicted"])
    fn = sum(1 for r in results if r["true_label"] and not r["predicted"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return results


if __name__ == "__main__":
    cases = load_dataset()
    asyncio.run(evaluate_metaqa(cases))
