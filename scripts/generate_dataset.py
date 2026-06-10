"""
GPT API를 사용하여 법률 상담 평가 데이터셋 생성

노동 100쌍 + 성폭력 100쌍 = 총 200쌍 생성
"""

import json
import asyncio
from pathlib import Path
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.modules.llm_client import llm_client
from src.modules.rag import retriever


# 각 카테고리별 이슈 유형
LABOR_ISSUES = {
    "wage_unpaid": {
        "name": "임금체불",
        "examples": [
            "편의점 알바 월급을 못 받았습니다.",
            "사장님이 알바비를 계속 미루고 있습니다.",
            "퇴직 후 급여를 안 줍니다."
        ],
        "keywords": ["임금", "체불", "급여", "알바비", "월급"],
        "laws": ["근로기준법 제43조", "근로기준법 제37조"]
    },
    "minimum_wage": {
        "name": "최저임금 위반",
        "examples": [
            "시급이 최저임금보다 적은 것 같습니다.",
            "수습이라며 최저시급보다 적게 줬습니다."
        ],
        "keywords": ["최저임금", "최저시급", "시급"],
        "laws": ["최저임금법 제6조"]
    },
    "dismissal": {
        "name": "부당해고",
        "examples": [
            "갑자기 해고 통보를 받았습니다.",
            "정당한 이유 없이 그만두라고 합니다."
        ],
        "keywords": ["해고", "부당해고", "해고예고"],
        "laws": ["근로기준법 제23조", "근로기준법 제26조"]
    },
    "missing_contract": {
        "name": "근로계약서 미작성",
        "examples": [
            "계약서 없이 일했습니다.",
            "근로조건을 구두로만 들었습니다."
        ],
        "keywords": ["근로계약서", "계약서", "서면", "근로조건"],
        "laws": ["근로기준법 제17조"]
    },
    "workplace_harassment": {
        "name": "직장 내 괴롭힘",
        "examples": [
            "점장이 계속 폭언하고 괴롭힙니다.",
            "직장에서 따돌림을 당합니다."
        ],
        "keywords": ["괴롭힘", "직장 내 괴롭힘", "폭언", "따돌림"],
        "laws": ["근로기준법 제76조의2", "근로기준법 제76조의3"]
    }
}

SEXUAL_VIOLENCE_ISSUES = {
    "sexual_harassment": {
        "name": "성희롱",
        "examples": [
            "교수가 성희롱 발언을 반복합니다.",
            "선배가 성적 농담을 계속합니다."
        ],
        "keywords": ["성희롱", "성적 농담", "외모 평가"],
        "laws": ["양성평등기본법 제30조"]
    },
    "indecent_assault": {
        "name": "강제추행",
        "examples": [
            "동의 없이 몸을 만졌습니다.",
            "술자리에서 허리를 만졌습니다."
        ],
        "keywords": ["강제추행", "성추행", "동의 없이"],
        "laws": ["형법 제298조"]
    },
    "illegal_filming": {
        "name": "불법촬영",
        "examples": [
            "동의 없이 사진을 찍었습니다.",
            "몰카를 찍은 것 같습니다."
        ],
        "keywords": ["불법촬영", "몰카", "카메라", "촬영"],
        "laws": ["성폭력범죄의 처벌 등에 관한 특례법 제14조"]
    }
}


class DatasetGenerator:
    def __init__(self):
        self.llm = llm_client
        self.retriever = retriever
        self.generated_queries = set()

    async def generate_question_variants(
        self,
        issue_type: str,
        issue_info: dict[str, Any],
        category: str,
        count: int = 20
    ) -> list[str]:
        """특정 이슈 유형에 대한 질문 변형 생성"""

        examples = "\n".join(f"- {ex}" for ex in issue_info["examples"])
        keywords = ", ".join(issue_info["keywords"])

        prompt = f"""당신은 법률 상담 챗봇 평가 데이터를 만드는 전문가입니다.

다음 법률 이슈에 대한 자연스러운 사용자 질문을 {count}개 생성하세요.

이슈 유형: {issue_info["name"]}
카테고리: {category}
관련 법률: {", ".join(issue_info["laws"])}
핵심 키워드: {keywords}

기존 예시:
{examples}

요구사항:
1. 실제 사용자가 물을 법한 자연스러운 질문
2. 구체적인 상황 포함 (예: 편의점, 카페, 학교 등)
3. 다양한 표현 방식 사용
4. 각 질문은 1-2문장으로 간결하게
5. 반말/존댓말 섞어서 사용
6. 기존 예시와 너무 유사하지 않게

JSON 배열 형식으로만 출력하세요:
["질문1", "질문2", ...]"""

        try:
            response = await self.llm.complete(
                prompt=prompt,
                temperature=0.9,
                max_tokens=2000
            )

            # JSON 파싱
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
                # 중복 제거
                unique = []
                for q in questions:
                    if q not in self.generated_queries:
                        unique.append(q)
                        self.generated_queries.add(q)
                return unique[:count]
        except Exception as e:
            print(f"[WARNING] 질문 생성 실패: {e}")
            return []

    async def find_golden_docs(
        self,
        query: str,
        category: str,
        expected_laws: list[str]
    ) -> dict[str, int]:
        """질문에 대한 정답 문서 검색"""

        # RAG 검색 수행
        docs = await self.retriever.retrieve_async(
            query,
            top_k=10,
            legal_category=category
        )

        golden_docs = {}
        for doc in docs:
            chunk_id = doc.get("chunk_id", "")
            if not chunk_id:
                continue

            # 정답 법률이 포함되어 있으면 우선순위 2, 아니면 1
            text = doc.get("text", "") + str(doc.get("metadata", {}))
            priority = 2 if any(law in text for law in expected_laws) else 1

            golden_docs[chunk_id] = priority

        return golden_docs

    async def generate_case(
        self,
        case_id: str,
        query: str,
        category: str,
        issue_type: str,
        issue_info: dict[str, Any]
    ) -> dict[str, Any]:
        """단일 평가 케이스 생성"""

        # 정답 문서 검색
        golden_docs = await self.find_golden_docs(
            query,
            category,
            issue_info["laws"]
        )

        # must_have: 반드시 포함되어야 할 키워드
        must_have = issue_info["keywords"][:2]  # 상위 2개

        # top_any: 상위 문서에 있어야 할 키워드
        top_any = issue_info["keywords"] + [law.split()[0] for law in issue_info["laws"]]

        return {
            "id": case_id,
            "query": query,
            "category": category,
            "expected_issues": [issue_type],
            "must_have": must_have,
            "top_any": top_any,
            "golden_doc_ids": golden_docs,
            "golden_laws": issue_info["laws"],
            "source": "gpt_generated",
            "reviewed": False
        }

    async def generate_dataset(
        self,
        output_path: str = "data/evaluation/generated_dataset.jsonl",
        labor_count: int = 100,
        sexual_violence_count: int = 100
    ):
        """전체 데이터셋 생성"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_cases = []
        case_counter = 1

        print("=" * 80)
        print("법률 상담 평가 데이터셋 생성 시작")
        print("=" * 80)

        # 1. 노동 카테고리 생성 (100개)
        print(f"\n[노동] 카테고리 생성 중... (목표: {labor_count}개)")
        per_issue = labor_count // len(LABOR_ISSUES)

        for issue_type, issue_info in LABOR_ISSUES.items():
            print(f"\n  - {issue_info['name']}: {per_issue}개 생성 중...")

            questions = await self.generate_question_variants(
                issue_type,
                issue_info,
                "노동",
                count=per_issue
            )

            for i, query in enumerate(questions, 1):
                case = await self.generate_case(
                    f"labor_{case_counter:03d}",
                    query,
                    "노동",
                    issue_type,
                    issue_info
                )
                all_cases.append(case)
                case_counter += 1
                print(f"    OK [{i}/{per_issue}] {query[:50]}...")

        # 2. 성폭력 카테고리 생성 (100개)
        print(f"\n[성폭력] 카테고리 생성 중... (목표: {sexual_violence_count}개)")
        per_issue = sexual_violence_count // len(SEXUAL_VIOLENCE_ISSUES)

        case_counter = 1
        for issue_type, issue_info in SEXUAL_VIOLENCE_ISSUES.items():
            print(f"\n  - {issue_info['name']}: {per_issue}개 생성 중...")

            questions = await self.generate_question_variants(
                issue_type,
                issue_info,
                "성폭력",
                count=per_issue
            )

            for i, query in enumerate(questions, 1):
                case = await self.generate_case(
                    f"sexual_violence_{case_counter:03d}",
                    query,
                    "성폭력",
                    issue_type,
                    issue_info
                )
                all_cases.append(case)
                case_counter += 1
                print(f"    OK [{i}/{per_issue}] {query[:50]}...")

        # 3. JSONL 파일로 저장
        print(f"\n[저장] 데이터셋 저장 중: {output_path}")
        with output_path.open("w", encoding="utf-8") as f:
            for case in all_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")

        # 4. 통계 출력
        labor_cases = [c for c in all_cases if c["category"] == "노동"]
        sexual_cases = [c for c in all_cases if c["category"] == "성폭력"]

        print("\n" + "=" * 80)
        print("데이터셋 생성 완료")
        print("=" * 80)
        print(f"\n총 생성: {len(all_cases)}개")
        print(f"  - 노동: {len(labor_cases)}개")
        print(f"  - 성폭력: {len(sexual_cases)}개")
        print(f"\n저장 위치: {output_path}")
        print("=" * 80)


async def main():
    generator = DatasetGenerator()

    # RAG 워밍업
    print("RAG 인덱스 로딩 중...")
    await generator.retriever.warmup()

    # 데이터셋 생성
    await generator.generate_dataset(
        output_path="data/evaluation/generated_dataset_200.jsonl",
        labor_count=100,
        sexual_violence_count=100
    )


if __name__ == "__main__":
    asyncio.run(main())
