"""
Phase 2: RAGAS 평가 (End-to-End RAG 평가)

BM25-RAG 전체 파이프라인 평가:
- Faithfulness (충실성)
- Answer Relevancy (답변 관련성)
- Context Precision (컨텍스트 정밀도)
- Context Recall (컨텍스트 재현율)
- Citation Accuracy (법률 조문 정확도)
"""

import json
import asyncio
import argparse
import re
from pathlib import Path
from dataclasses import dataclass
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))  # scripts 폴더 추가

from src.modules.llm_client import llm_client
from src.modules.rag import retriever as bm25_retriever

# 공통 Retriever 클래스
from retriever_classes import DenseRetriever, HybridRetriever


@dataclass
class RAGASMetrics:
    """RAGAS 평가 메트릭"""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    citation_accuracy: float


class RAGASEvaluator:
    """RAGAS 평가기"""

    def __init__(self):
        self.llm = llm_client

    async def generate_answer(self, query: str, contexts: list[str]) -> str:
        """검색된 문서 기반 답변 생성 (공평한 평가)"""
        if not contexts:
            return "검색된 문서가 없어 답변할 수 없습니다."

        # 문맥 결합
        contexts_text = "\n\n---\n\n".join(contexts[:5])  # 상위 5개만

        prompt = f"""당신은 법률 상담 전문가입니다. 아래 법률 문서를 참고하여 질문에 답변하세요.

[참고 법률 문서]
{contexts_text}

[질문]
{query}

답변 지침:
1. 참고 문서의 내용만을 근거로 답변하세요
2. 문서에 없는 내용은 추가하지 마세요
3. 관련 법률 조문을 명시하세요
4. 간결하고 명확하게 답변하세요

[답변]"""

        try:
            response = await self.llm.complete(
                prompt=prompt,
                temperature=0.0,
                max_tokens=1000
            )
            return response.strip()
        except Exception as e:
            print(f"[WARNING] 답변 생성 실패: {e}")
            return ""

    async def evaluate_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """충실성 평가 (환각 측정)"""
        if not answer or not contexts:
            return 0.0

        contexts_text = "\n---\n".join(contexts)

        prompt = f"""다음 답변이 제공된 문맥에만 근거했는지 평가하세요.

[문맥]
{contexts_text}

[답변]
{answer}

평가 기준:
- 답변의 모든 사실이 문맥에 존재하는가?
- 문맥에 없는 정보를 추가하지 않았는가?

1-5점으로 평가:
5점: 모두 문맥에 근거
4점: 대부분 근거
3점: 절반 정도
2점: 일부만
1점: 거의 없음

점수만 출력 (1-5):"""

        try:
            response = await self.llm.complete(prompt=prompt, temperature=0.0)
            match = re.search(r'[1-5]', response)
            if match:
                return int(match.group()) / 5.0
        except Exception as e:
            print(f"[WARNING] Faithfulness 평가 실패: {e}")

        return 0.5

    async def evaluate_relevancy(self, question: str, answer: str) -> float:
        """답변 관련성 평가"""
        if not answer:
            return 0.0

        prompt = f"""질문과 답변을 평가하세요.

[질문]
{question}

[답변]
{answer}

평가 기준:
- 질문에 직접적으로 답하는가?
- 핵심만 다루는가?

1-5점으로 평가:
5점: 완벽
4점: 좋음
3점: 보통
2점: 약함
1점: 무관

점수만 출력 (1-5):"""

        try:
            response = await self.llm.complete(prompt=prompt, temperature=0.0)
            match = re.search(r'[1-5]', response)
            if match:
                return int(match.group()) / 5.0
        except Exception as e:
            print(f"[WARNING] Relevancy 평가 실패: {e}")

        return 0.5

    async def is_relevant_context(self, query: str, context: str, golden_laws: list) -> bool:
        """LLM으로 컨텍스트가 관련있는지 평가"""
        laws_text = ", ".join(golden_laws) if golden_laws else "관련 법률"

        prompt = f"""다음 문서가 질문에 대한 정답인지 판단하세요.

[질문]
{query}

[관련 법률]
{laws_text}

[문서]
{context[:500]}

이 문서가 질문에 답하는 데 도움이 되는가?
"예" 또는 "아니오"만 출력:"""

        try:
            response = await self.llm.complete(prompt=prompt, temperature=0.0, max_tokens=10)
            return "예" in response.lower() or "yes" in response.lower()
        except:
            return False

    async def evaluate_citation_accuracy(self, answer: str, golden_laws: list[str]) -> float:
        """법률 조문 정확도"""
        if not golden_laws:
            return 1.0

        mentioned = 0
        for law in golden_laws:
            law_norm = law.replace(" ", "")
            answer_norm = answer.replace(" ", "")
            if law_norm in answer_norm:
                mentioned += 1

        return mentioned / len(golden_laws)

    async def evaluate_dataset(self, retriever, dataset: list, name: str) -> RAGASMetrics:
        """전체 데이터셋 평가"""
        print("="*70)
        print(f"RAGAS 평가 ({name})")
        print("="*70)

        all_faithfulness = []
        all_relevancy = []
        all_precision = []
        all_recall = []
        all_citation = []

        for i, case in enumerate(dataset, 1):
            query = case["query"]
            category = case["category"]
            golden_laws = case.get("golden_laws", [])
            golden_doc_ids = set(case.get("golden_doc_ids", {}).keys())

            # 검색
            if hasattr(retriever, 'retrieve_async'):
                retrieved_docs = await retriever.retrieve_async(
                    query, top_k=5, legal_category=category
                )
            else:
                retrieved_docs = await retriever.retrieve(
                    query, top_k=5, legal_category=category
                )

            contexts = [doc.get("text", "") for doc in retrieved_docs]

            # 답변 생성 (검색된 문서 직접 사용 - 공평한 평가!)
            answer = await self.generate_answer(query, contexts)

            # RAGAS 평가
            faithfulness = await self.evaluate_faithfulness(answer, contexts)
            relevancy = await self.evaluate_relevancy(query, answer)

            # Context 평가 (LLM 기반)
            relevance_flags = []
            for context in contexts:
                is_relevant = await self.is_relevant_context(query, context, golden_laws)
                relevance_flags.append(is_relevant)

            # Precision & Recall
            if relevance_flags:
                precision = sum(relevance_flags) / len(relevance_flags)
                recall = sum(relevance_flags) / max(len(relevance_flags), 1)
            else:
                precision = 0.0
                recall = 0.0

            # Citation
            citation = await self.evaluate_citation_accuracy(answer, golden_laws)

            all_faithfulness.append(faithfulness)
            all_relevancy.append(relevancy)
            all_precision.append(precision)
            all_recall.append(recall)
            all_citation.append(citation)

            if i % 10 == 0:
                print(f"  진행: {i}/{len(dataset)}")

        # 평균
        avg = RAGASMetrics(
            faithfulness=sum(all_faithfulness) / len(all_faithfulness),
            answer_relevancy=sum(all_relevancy) / len(all_relevancy),
            context_precision=sum(all_precision) / len(all_precision),
            context_recall=sum(all_recall) / len(all_recall),
            citation_accuracy=sum(all_citation) / len(all_citation)
        )

        print("[완료]\n")
        return avg


def print_results(results: dict):
    """결과 출력"""
    print("\n" + "="*80)
    print("RAGAS 평가 결과")
    print("="*80 + "\n")

    try:
        from tabulate import tabulate

        table = []
        for name, metrics in results.items():
            table.append([
                name,
                f"{metrics.faithfulness:.4f}",
                f"{metrics.answer_relevancy:.4f}",
                f"{metrics.context_precision:.4f}",
                f"{metrics.context_recall:.4f}",
                f"{metrics.citation_accuracy:.4f}"
            ])

        headers = ["방법", "Faithfulness", "Answer Rel.", "Context Prec.", "Context Rec.", "Citation Acc."]
        print(tabulate(table, headers=headers, tablefmt="grid"))

    except ImportError:
        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  Faithfulness: {metrics.faithfulness:.4f}")
            print(f"  Answer Relevancy: {metrics.answer_relevancy:.4f}")
            print(f"  Context Precision: {metrics.context_precision:.4f}")
            print(f"  Context Recall: {metrics.context_recall:.4f}")
            print(f"  Citation Accuracy: {metrics.citation_accuracy:.4f}")

    print("\n* Faithfulness: 높을수록 환각 적음")
    print("="*80)


async def main(method=None):
    """메인 실행"""
    try:
        # 데이터셋 로드
        dataset_path = Path("data/evaluation/generated_dataset_200.jsonl")

        if not dataset_path.exists():
            print(f"[ERROR] 데이터셋 없음: {dataset_path}")
            return

        print("[데이터셋 로딩]")
        dataset = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))

        print(f"  총 {len(dataset)}개 케이스\n")

        # 평가
        evaluator = RAGASEvaluator()
        results = {}

        # 방법별 평가
        if method is None or method == "bm25":
            print("[BM25 워밍업]")
            await bm25_retriever.warmup()
            print("[완료]\n")

            bm25_metrics = await evaluator.evaluate_dataset(bm25_retriever, dataset, "BM25-RAG")
            results["BM25-RAG"] = bm25_metrics

        if method is None or method == "dense":
            print("[Dense Retriever 로딩]")
            dense = DenseRetriever()
            await dense.warmup()
            print("[완료]\n")

            dense_metrics = await evaluator.evaluate_dataset(dense, dataset, "Dense-RAG (E5-base)")
            results["Dense-RAG (E5-base)"] = dense_metrics

        if method is None or method == "hybrid":
            if method == "hybrid":
                print("[Dense Retriever 로딩]")
                dense = DenseRetriever()
                await dense.warmup()
                print("[완료]\n")

            hybrid = HybridRetriever(alpha=0.7, dense=dense)
            hybrid_metrics = await evaluator.evaluate_dataset(hybrid, dataset, "Hybrid-RAG (α=0.7)")
            results["Hybrid-RAG (α=0.7)"] = hybrid_metrics

        # 결과 출력
        if results:
            print_results(results)

        # 저장
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        output = {name: {
            "faithfulness": m.faithfulness,
            "answer_relevancy": m.answer_relevancy,
            "context_precision": m.context_precision,
            "context_recall": m.context_recall,
            "citation_accuracy": m.citation_accuracy
        } for name, m in results.items()}

        # 방법별 파일명
        if method:
            output_path = output_dir / f"ragas_eval_{method}.json"
        else:
            output_path = output_dir / "ragas_evaluation_results.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n[저장 완료] {output_path}\n")

        # GPU 정리
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[GPU 메모리 정리 완료]\n")

    except Exception as e:
        print(f"\n[ERROR] 실행 중 오류:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], help="평가 방법")
    args = parser.parse_args()

    asyncio.run(main(args.method))
