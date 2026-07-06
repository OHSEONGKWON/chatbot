"""평가셋 로더 (정상 400 + 환각 400)."""
import json
from dataclasses import dataclass
from pathlib import Path

from .config import EVAL_PATH


@dataclass
class HalluCase:
    hallu_id: str
    question: str
    answer: str
    is_hallucinated: bool
    hallu_type: str


def load_dataset(data_path: Path = EVAL_PATH):
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"데이터셋 없음: {path}")

    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            cases.append(HalluCase(
                hallu_id=d["hallu_id"],
                question=d["question"],
                answer=d["answer"],
                is_hallucinated=(d["hallu_type"] != "none"),
                hallu_type=d["hallu_type"],
            ))

    normal = sum(1 for c in cases if not c.is_hallucinated)
    hallu = len(cases) - normal
    print(f"[로드] 총 {len(cases)}개 (정상 {normal} / 환각 {hallu})")
    return cases
