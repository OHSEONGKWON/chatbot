"""
RAG 성능 평가 (BM25 vs Dense vs Hybrid)

Dense Model: multilingual-e5-base
- BM25와 비교 의미 있는 성능
- 한국어 지원
- 안정적인 메모리 사용
"""

import json
import asyncio
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import sys

# 경로 설정
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))  # scripts 폴더 추가

from src.modules.rag import retriever as bm25_retriever
from src.modules.llm_client import llm_client

# 공통 Retriever 클래스
from retriever_classes import DenseRetriever, HybridRetriever, DEVICE

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as e:
    print(f"[ERROR] 라이브러리 설치 필요: {e}")
    print("pip install sentence-transformers torch")
    sys.exit(1)

print("\n" + "="*70)
print("RAG 성능 평가 시작")
print("="*70)
print(f"GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - 디바이스: {torch.cuda.get_device_name(0)}")
    print(f"  - 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("="*70 + "\n")


@dataclass
class Metrics:
    """검색 성능 메트릭"""
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float


async def is_relevant_document(query: str, doc_text: str, golden_laws: list) -> bool:
    """LLM으로 문서가 질문의 정답인지 평가"""
    laws_text = ", ".join(golden_laws) if golden_laws else "관련 법률"

    prompt = f"""다음 법률 문서가 질문에 대한 정답인지 판단하세요.

[질문]
{query}

[관련 법률]
{laws_text}

[문서]
{doc_text[:500]}

이 문서가 질문에 답하는 데 도움이 되는가?
"예" 또는 "아니오"만 출력:"""

    try:
        response = await llm_client.complete(prompt=prompt, temperature=0.0, max_tokens=10)
        return "예" in response.lower() or "yes" in response.lower()
    except:
        return False


class Evaluator:
    """평가기"""

    @staticmethod
    def calculate_dcg(relevances):
        """DCG 계산"""
        return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))

    def evaluate_single(self, retrieved_docs, relevance_flags):
        """단일 쿼리 평가 (LLM 평가 기반)"""
        # relevance_flags: [True, False, True, ...] (각 문서가 관련있는지)

        if not relevance_flags:
            return {
                "precision_at_1": 0,
                "precision_at_3": 0,
                "precision_at_5": 0,
                "recall_at_5": 0,
                "mrr": 0,
                "ndcg_at_5": 0
            }

        # Precision@K
        relevant_at_k = {}
        for k in [1, 3, 5]:
            top_k = relevance_flags[:k]
            relevant_at_k[k] = sum(top_k)

        total_relevant = sum(relevance_flags)

        precision_at_1 = relevant_at_k[1] / 1 if len(relevance_flags) >= 1 else 0
        precision_at_3 = relevant_at_k[3] / 3 if len(relevance_flags) >= 3 else 0
        precision_at_5 = relevant_at_k[5] / 5 if len(relevance_flags) >= 5 else 0
        recall_at_5 = relevant_at_k[5] / max(total_relevant, 1)

        # MRR
        mrr = 0.0
        for idx, is_relevant in enumerate(relevance_flags, 1):
            if is_relevant:
                mrr = 1.0 / idx
                break

        # NDCG@5
        relevances = [1.0 if r else 0.0 for r in relevance_flags[:5]]
        dcg = self.calculate_dcg(relevances)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = self.calculate_dcg(ideal_relevances)
        ndcg_at_5 = dcg / idcg if idcg > 0 else 0.0

        return {
            "precision_at_1": precision_at_1,
            "precision_at_3": precision_at_3,
            "precision_at_5": precision_at_5,
            "recall_at_5": recall_at_5,
            "mrr": mrr,
            "ndcg_at_5": ndcg_at_5
        }

    async def evaluate(self, retriever, dataset, name):
        """전체 평가 (LLM 기반)"""
        print("="*70)
        print(f"{name} 평가")
        print("="*70)

        all_metrics = []

        for i, case in enumerate(dataset, 1):
            query = case["query"]
            category = case["category"]
            golden_laws = case.get("golden_laws", [])

            # 검색
            if hasattr(retriever, 'retrieve_async'):
                docs = await retriever.retrieve_async(
                    query, top_k=5, legal_category=category
                )
            else:
                docs = await retriever.retrieve(
                    query, top_k=5, legal_category=category
                )

            # LLM으로 각 문서 평가
            relevance_flags = []
            for doc in docs:
                doc_text = doc.get("text", "")
                is_relevant = await is_relevant_document(query, doc_text, golden_laws)
                relevance_flags.append(is_relevant)

            # 평가
            metrics = self.evaluate_single(docs, relevance_flags)
            all_metrics.append(metrics)

            if i % 20 == 0:
                print(f"  진행: {i}/{len(dataset)}")

        # 평균 계산
        avg = Metrics(
            precision_at_1=sum(m["precision_at_1"] for m in all_metrics) / len(all_metrics),
            precision_at_3=sum(m["precision_at_3"] for m in all_metrics) / len(all_metrics),
            precision_at_5=sum(m["precision_at_5"] for m in all_metrics) / len(all_metrics),
            recall_at_5=sum(m["recall_at_5"] for m in all_metrics) / len(all_metrics),
            mrr=sum(m["mrr"] for m in all_metrics) / len(all_metrics),
            ndcg_at_5=sum(m["ndcg_at_5"] for m in all_metrics) / len(all_metrics)
        )

        print(f"[완료]\n")
        return avg


def print_results(results):
    """결과 출력"""
    print("\n" + "="*80)
    print("RAG 평가 결과")
    print("="*80 + "\n")

    try:
        from tabulate import tabulate

        table = []
        for name, metrics in results.items():
            table.append([
                name,
                f"{metrics.precision_at_1:.4f}",
                f"{metrics.precision_at_3:.4f}",
                f"{metrics.precision_at_5:.4f}",
                f"{metrics.recall_at_5:.4f}",
                f"{metrics.mrr:.4f}",
                f"{metrics.ndcg_at_5:.4f}"
            ])

        headers = ["방법", "P@1", "P@3", "P@5", "R@5", "MRR", "NDCG@5"]
        print(tabulate(table, headers=headers, tablefmt="grid"))

    except ImportError:
        print("방법별 성능:")
        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  P@1: {metrics.precision_at_1:.4f}")
            print(f"  P@3: {metrics.precision_at_3:.4f}")
            print(f"  P@5: {metrics.precision_at_5:.4f}")
            print(f"  R@5: {metrics.recall_at_5:.4f}")
            print(f"  MRR: {metrics.mrr:.4f}")
            print(f"  NDCG@5: {metrics.ndcg_at_5:.4f}")

    print("\n" + "="*80)


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

        print(f"  총 {len(dataset)}개 케이스 로드\n")

        # 평가기 초기화
        evaluator = Evaluator()
        results = {}

        # 방법별 평가
        if method is None or method == "bm25":
            print("[BM25 워밍업]")
            await bm25_retriever.warmup()
            print("[완료]\n")

            bm25_metrics = await evaluator.evaluate(bm25_retriever, dataset, "BM25")
            results["BM25"] = bm25_metrics

        if method is None or method == "dense":
            dense = DenseRetriever()
            await dense.warmup()

            dense_metrics = await evaluator.evaluate(dense, dataset, "Dense (E5-base)")
            results["Dense (E5-base)"] = dense_metrics

        if method is None or method == "hybrid":
            if method == "hybrid":
                dense = DenseRetriever()
                await dense.warmup()

            hybrid = HybridRetriever(alpha=0.7, dense=dense)
            hybrid_metrics = await evaluator.evaluate(hybrid, dataset, "Hybrid (α=0.7)")
            results["Hybrid (α=0.7)"] = hybrid_metrics

        # 결과 출력
        if results:
            print_results(results)

        # 저장
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        output = {name: {
            "precision_at_1": m.precision_at_1,
            "precision_at_3": m.precision_at_3,
            "precision_at_5": m.precision_at_5,
            "recall_at_5": m.recall_at_5,
            "mrr": m.mrr,
            "ndcg_at_5": m.ndcg_at_5
        } for name, m in results.items()}

        # 방법별 파일명
        if method:
            output_path = output_dir / f"rag_eval_{method}.json"
        else:
            output_path = output_dir / "rag_evaluation_results.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n[저장 완료] {output_path}\n")

        # GPU 정리
        if DEVICE == "cuda":
            import torch
            torch.cuda.empty_cache()
            print("[GPU 메모리 정리 완료]")

    except Exception as e:
        print(f"\n[ERROR] 실행 중 오류 발생:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], help="평가 방법")
    args = parser.parse_args()

    asyncio.run(main(args.method))
