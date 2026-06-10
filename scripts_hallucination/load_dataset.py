"""데이터셋 로더"""
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class HalluCase:
    """환각 평가 케이스"""
    hallu_id: str
    question: str
    answer: str
    is_hallucinated: bool  # True면 환각, False면 정상
    hallu_type: str


def load_dataset(data_path: str = "data/evaluation/hallu_eval.jsonl"):
    """데이터셋 로드 (800개)"""
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"데이터셋 없음: {path}")

    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            # hallu_type이 "none"이면 정상, 아니면 환각
            is_hallu = data["hallu_type"] != "none"

            cases.append(HalluCase(
                hallu_id=data["hallu_id"],
                question=data["question"],
                answer=data["answer"],
                is_hallucinated=is_hallu,
                hallu_type=data["hallu_type"]
            ))

    print(f"[로드 완료] 총 {len(cases)}개")

    # 통계
    normal = sum(1 for c in cases if not c.is_hallucinated)
    hallu = sum(1 for c in cases if c.is_hallucinated)

    print(f"  정상: {normal}개")
    print(f"  환각: {hallu}개")

    return cases


if __name__ == "__main__":
    cases = load_dataset()

    # 환각 유형별 통계
    from collections import Counter
    hallu_types = Counter(c.hallu_type for c in cases if c.is_hallucinated)

    print("\n환각 유형:")
    for h_type, count in hallu_types.items():
        print(f"  {h_type}: {count}개")
