"""환각 탐지 데이터 증강 스크립트

800개 데이터를 2,000~3,000개로 증강하는 방법:
1. 질문 패러프레이징 (LLM 활용)
2. 정상 답변에 환각 주입 (템플릿 기반)
3. 환각 유형별 합성 데이터 생성
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List
from collections import Counter
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.modules.llm_client import llm_client


def load_hallu_dataset(path: str = "data/evaluation/hallu_eval.jsonl") -> List[Dict]:
    """환각 데이터셋 로드"""
    cases = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def analyze_dataset(cases: List[Dict]):
    """데이터셋 분석"""
    print(f"\n{'='*60}")
    print(f"총 케이스 수: {len(cases)}")

    # label 분포
    labels = Counter(c['label'] for c in cases)
    print(f"\n[Label 분포]")
    for label, count in labels.items():
        print(f"  {label}: {count}개")

    # hallu_type 분포
    hallu_types = Counter(c['hallu_type'] for c in cases)
    print(f"\n[Hallucination Type 분포]")
    for htype, count in sorted(hallu_types.items(), key=lambda x: -x[1]):
        print(f"  {htype}: {count}개")
    print(f"{'='*60}\n")


# ==================== 증강 방법 1: 질문 패러프레이징 ====================

async def paraphrase_question(question: str, style: str = "formal") -> str:
    """질문 패러프레이징"""
    style_prompts = {
        "formal": "다음 질문을 격식 있는 표현으로 바꿔서 다시 작성해주세요. 의미는 동일하게 유지하되, 표현만 달리해주세요.",
        "casual": "다음 질문을 일상적이고 친근한 말투로 바꿔주세요. 의미는 동일하게 유지하되, 표현만 달리해주세요.",
        "detailed": "다음 질문을 더 구체적이고 자세하게 바꿔주세요. 의미는 동일하게 유지하되, 맥락을 추가해주세요.",
        "short": "다음 질문을 더 간결하고 짧게 바꿔주세요. 핵심 의미만 유지하면서 단순화해주세요.",
        "professional": "다음 질문을 법률 전문가가 하는 것처럼 전문적인 용어로 바꿔주세요.",
        "confused": "다음 질문을 법률 지식이 없는 일반인이 혼란스러워하며 묻는 것처럼 바꿔주세요.",
    }

    prompt = f"""{style_prompts[style]}

원본 질문: {question}

바꾼 질문:"""

    try:
        response = await llm_client.complete(prompt=prompt, temperature=0.7, max_tokens=150)
        return response.strip()
    except Exception as e:
        print(f"[ERROR] 패러프레이징 실패: {e}")
        return question


async def augment_by_paraphrasing(cases: List[Dict], target_count: int = 400) -> List[Dict]:
    """질문 패러프레이징으로 증강 (정상 케이스만)"""
    normal_cases = [c for c in cases if c['label'] == 'normal']
    augmented = []

    styles = ["formal", "casual", "detailed", "short", "professional", "confused"]
    count = 0

    # 라운드 계산: 각 케이스를 몇 번씩 반복할지
    rounds = (target_count // len(normal_cases) // len(styles)) + 1

    for round_idx in range(rounds):
        if count >= target_count:
            break

        for case in normal_cases:
            if count >= target_count:
                break

            for style in styles:
                if count >= target_count:
                    break

                new_question = await paraphrase_question(case['question'], style)

                augmented.append({
                    "hallu_id": f"{case['hallu_id']}_para_{style}_r{round_idx}",
                    "chunk_id": case['chunk_id'],
                    "source_text": case['source_text'],
                    "question": new_question,
                    "answer": case['answer'],
                    "label": case['label'],
                    "hallu_type": case['hallu_type'],
                    "augmentation": f"paraphrase_{style}_round{round_idx}",
                })
                count += 1

                if count % 50 == 0:
                    print(f"  패러프레이징 진행 (라운드 {round_idx+1}): {count}/{target_count}")

    return augmented


# ==================== 증강 방법 2: 환각 주입 ====================

HALLUCINATION_TEMPLATES = {
    "article_number_error": [
        lambda text, old, new: text.replace(f"제{old}조", f"제{new}조"),
        lambda text, old, new: text.replace(f"제{old}항", f"제{new}항"),
        lambda text, old, new: text.replace(f"제{old}조", f"제{new}조의2"),
        lambda text, old, new: text.replace(f"{old}조", f"{new}조"),
    ],
    "semantic_error": [
        lambda text: text.replace("할 수 있습니다", "할 수 없습니다"),
        lambda text: text.replace("가능합니다", "불가능합니다"),
        lambda text: text.replace("인정됩니다", "인정되지 않습니다"),
        lambda text: text.replace("허용됩니다", "금지됩니다"),
        lambda text: text.replace("필요합니다", "필요하지 않습니다"),
        lambda text: text.replace("의무입니다", "의무가 아닙니다"),
        lambda text: text.replace("적용됩니다", "적용되지 않습니다"),
        lambda text: text.replace("유효합니다", "무효입니다"),
    ],
    "forbidden_law_injection": [
        lambda text: text + " 또한 대법원 2099도99999 판례에서도 이를 명확히 하고 있습니다.",
        lambda text: text + " 민법 제9999조에 따라 이러한 경우에는 특별한 절차가 필요합니다.",
        lambda text: text + " 형법 제8888조의3에 따라 이는 처벌 대상입니다.",
        lambda text: text + " 헌법재판소 2088헌마1234 결정에서 합헌 판단을 받았습니다.",
        lambda text: text + " 상법 제7777조에 명시된 바와 같이 이는 반드시 준수해야 합니다.",
    ],
    "contact_mismatch": [
        lambda text: text.replace("법률", "법률 (문의: 119)"),
        lambda text: text + " 자세한 사항은 999-9999로 문의하시기 바랍니다.",
        lambda text: text + " 법률 상담은 112로 연락하시면 됩니다.",
        lambda text: text + " 관련 문의는 1234-5678로 하시기 바랍니다.",
    ],
}


def inject_hallucination(case: Dict, hallu_type: str) -> Dict:
    """정상 답변에 환각 주입"""
    if hallu_type not in HALLUCINATION_TEMPLATES:
        return None

    templates = HALLUCINATION_TEMPLATES[hallu_type]
    import random
    template = random.choice(templates)

    try:
        if hallu_type == "article_number_error":
            # 기존 조항 번호 찾기
            import re
            articles = re.findall(r'제(\d+)조', case['answer'])
            if articles:
                old_num = articles[0]
                new_num = str(int(old_num) + 10)  # 10 증가
                hallucinated_answer = template(case['answer'], old_num, new_num)
            else:
                return None
        else:
            hallucinated_answer = template(case['answer'])

        return {
            "hallu_id": f"{case['hallu_id']}_inject_{hallu_type}",
            "chunk_id": case['chunk_id'],
            "source_text": case['source_text'],
            "question": case['question'],
            "answer": hallucinated_answer,
            "label": "hallucination",
            "hallu_type": hallu_type,
            "augmentation": f"injected_{hallu_type}",
            "injected_error": f"Template-based {hallu_type} injection",
        }
    except Exception as e:
        print(f"[ERROR] 환각 주입 실패: {e}")
        return None


def augment_by_injection(cases: List[Dict], target_count: int = 400) -> List[Dict]:
    """환각 주입으로 증강 (정상 → 환각)"""
    normal_cases = [c for c in cases if c['label'] == 'normal']
    augmented = []

    hallu_types = list(HALLUCINATION_TEMPLATES.keys())

    # 각 정상 케이스마다 여러 환각 유형 적용
    rounds = (target_count // len(normal_cases) // len(hallu_types)) + 1

    for round_idx in range(rounds):
        if len(augmented) >= target_count:
            break

        for i, case in enumerate(normal_cases):
            if len(augmented) >= target_count:
                break

            for hallu_type in hallu_types:
                if len(augmented) >= target_count:
                    break

                injected = inject_hallucination(case, hallu_type)
                if injected:
                    # 라운드 번호 추가하여 중복 방지
                    injected['hallu_id'] = f"{case['hallu_id']}_inject_{hallu_type}_r{round_idx}"
                    augmented.append(injected)

            if (i + 1) % 50 == 0:
                print(f"  환각 주입 진행 (라운드 {round_idx+1}): {len(augmented)}/{target_count}")

    return augmented


# ==================== 증강 방법 3: 합성 데이터 생성 ====================

async def generate_synthetic_case(chunk_text: str, hallu_type: str) -> Dict:
    """LLM으로 합성 데이터 생성"""

    if hallu_type == "normal":
        prompt = f"""다음 법률 문서를 읽고, 이 내용과 관련된 질문과 정확한 답변을 생성하세요.

법률 문서:
{chunk_text[:500]}

다음 형식으로 작성하세요:
질문: [사용자가 물어볼 만한 법률 질문]
답변: [위 문서에 근거한 정확한 답변]
"""
    else:
        prompt = f"""다음 법률 문서를 읽고, 질문과 **의도적으로 잘못된** 답변을 생성하세요.
환각 유형: {hallu_type}

법률 문서:
{chunk_text[:500]}

다음 형식으로 작성하세요:
질문: [사용자가 물어볼 만한 법률 질문]
답변: [의도적으로 {hallu_type} 오류를 포함한 답변]
"""

    try:
        response = await llm_client.complete(prompt=prompt, temperature=0.8, max_tokens=300)

        # 질문/답변 파싱
        lines = response.strip().split('\n')
        question = None
        answer = None

        for line in lines:
            if line.startswith('질문:'):
                question = line.replace('질문:', '').strip()
            elif line.startswith('답변:'):
                answer = line.replace('답변:', '').strip()

        if question and answer:
            return {
                "question": question,
                "answer": answer,
                "label": "normal" if hallu_type == "normal" else "hallucination",
                "hallu_type": "none" if hallu_type == "normal" else hallu_type,
            }
        return None

    except Exception as e:
        print(f"[ERROR] 합성 데이터 생성 실패: {e}")
        return None


# ==================== 메인 증강 파이프라인 ====================

async def augment_dataset(
    input_path: str = "data/evaluation/hallu_eval.jsonl",
    output_path: str = "data/evaluation/hallu_eval_augmented.jsonl",
    target_total: int = 5000,  # 800 → 5000 (6.25배)
):
    """데이터셋 증강 메인 함수"""

    print("\n" + "="*60)
    print("환각 탐지 데이터 증강 시작")
    print("="*60)

    # 1. 원본 데이터 로드
    original_cases = load_hallu_dataset(input_path)
    print(f"\n[1] 원본 데이터 로드 완료: {len(original_cases)}개")
    analyze_dataset(original_cases)

    # 2. 증강 목표 계산
    target_augment = target_total - len(original_cases)
    paraphrase_target = target_augment // 2  # 절반은 패러프레이징
    injection_target = target_augment // 2   # 절반은 환각 주입

    print(f"[2] 증강 목표: {target_augment}개 추가")
    print(f"  - 패러프레이징: {paraphrase_target}개")
    print(f"  - 환각 주입: {injection_target}개\n")

    # 3. 패러프레이징 증강
    print("[3] 질문 패러프레이징 시작...")
    paraphrased = await augment_by_paraphrasing(original_cases, paraphrase_target)
    print(f"  완료: {len(paraphrased)}개 생성\n")

    # 4. 환각 주입 증강
    print("[4] 환각 주입 시작...")
    injected = augment_by_injection(original_cases, injection_target)
    print(f"  완료: {len(injected)}개 생성\n")

    # 5. 결합
    augmented_dataset = original_cases + paraphrased + injected
    print(f"[5] 최종 데이터셋: {len(augmented_dataset)}개")

    # 6. 저장
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open('w', encoding='utf-8') as f:
        for case in augmented_dataset:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')

    print(f"[6] 저장 완료: {output_path}")

    # 7. 최종 분석
    print("\n[최종 데이터셋 분석]")
    analyze_dataset(augmented_dataset)

    print("="*60)
    print("증강 완료!")
    print("="*60)


if __name__ == "__main__":
    # 실행
    asyncio.run(augment_dataset(
        target_total=5000  # 800 → 5000 (6.25배 증강)
    ))
