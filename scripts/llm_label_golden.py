"""
LLM 자동 레이블링: golden_candidates.jsonl → rag_eval_golden.jsonl

Claude Haiku를 사용해 각 (쿼리, 문서) 쌍을 rel=0/1/2로 분류.

최적화:
  - 동일 (issue_key, chunk_id) 쌍은 한 번만 API 호출 (캐시 재사용)
  - 시스템 프롬프트 프롬프트 캐싱으로 비용 절감
  - 진행 상황 자동 저장 → 중단 후 재개 가능

실행:
  python scripts/llm_label_golden.py
  python scripts/llm_label_golden.py --dry-run   # API 미호출, 구조 확인만
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# .env 자동 로드
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

REPO_ROOT  = Path(__file__).resolve().parents[1]
GOLDEN_IN  = REPO_ROOT / "data" / "evaluation" / "golden_candidates.jsonl"
CORPUS_DIR = REPO_ROOT / "data" / "real_data" / "New_Dataset"
PROGRESS_F = REPO_ROOT / "data" / "evaluation" / "llm_label_progress.json"
OUT_JSONL  = REPO_ROOT / "data" / "evaluation" / "rag_eval_golden.jsonl"

MODEL = "claude-haiku-4-5"
MAX_DOC_CHARS = 800  # 문서 미리보기 최대 길이

SYSTEM_PROMPT = """\
당신은 한국 법률 RAG 시스템의 검색 품질을 평가하는 전문가입니다.

주어진 사용자 질문과 법률 문서 청크를 읽고, 해당 문서가 질문에 대한 답변에 얼마나 유용한지 3단계로 평가하세요.

평가 기준:
  rel=2 (핵심근거): 이 문서를 직접 인용하면 질문에 정확히 답변할 수 있다. (관련 법령 조문, 핵심 판례 요지 등)
  rel=1 (보조근거): 답변을 위한 맥락 이해에 도움이 되지만 직접 인용하기는 어렵다. (배경 정보, 간접 관련 판례)
  rel=0 (제외):    키워드는 걸렸지만 이 질문의 답변과 실질적으로 무관하다.

반드시 JSON 형식으로만 응답하세요: {"rel": 0} 또는 {"rel": 1} 또는 {"rel": 2}
JSON 외 다른 텍스트는 절대 포함하지 마세요.\
"""


# ── 코퍼스 로더 ───────────────────────────────────────────────────────────────

def load_corpus(target_ids: set[str]) -> dict[str, str]:
    corpus: dict[str, str] = {}
    for fname in ("rag_law_chunks.jsonl", "rag_case_chunks.jsonl", "rag_manual_chunks.jsonl"):
        path = CORPUS_DIR / fname
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cid = row.get("chunk_id", "")
                if cid in target_ids and cid not in corpus:
                    corpus[cid] = row.get("text", "")
        if len(corpus) >= len(target_ids):
            break
    return corpus


# ── LLM 분류 ─────────────────────────────────────────────────────────────────

def classify(client, query: str, doc_text: str, retries: int = 3) -> int:
    """(query, doc_text) 쌍에 대한 rel=0/1/2 반환. 실패 시 -1."""
    import anthropic

    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=20,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": (
                        f"질문: {query}\n\n"
                        f"문서 청크:\n{doc_text[:MAX_DOC_CHARS]}"
                    ),
                }],
            )
            text = resp.content[0].text.strip()
            # 마크다운 코드블록 제거 (```json ... ```)
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            rel = json.loads(text)["rel"]
            return int(rel)
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"    [rate_limit] {wait}s 대기 후 재시도...")
            time.sleep(wait)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"    [parse_error] attempt={attempt+1}: {e} / raw={text!r}")
            if attempt == retries - 1:
                return -1
        except Exception as e:
            print(f"    [api_error] attempt={attempt+1}: {e}")
            if attempt == retries - 1:
                return -1
            time.sleep(2 ** attempt)
    return -1


# ── 진행 상황 저장/로드 ───────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_F.exists():
        return json.loads(PROGRESS_F.read_text(encoding="utf-8"))
    return {"cache": {}, "completed_cases": []}


def save_progress(progress: dict) -> None:
    PROGRESS_F.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="API 호출 없이 구조 확인만")
    parser.add_argument("--reset", action="store_true",
                        help="진행 상황 초기화 후 처음부터 시작")
    args = parser.parse_args()

    cases = [json.loads(l) for l in GOLDEN_IN.open("r", encoding="utf-8") if l.strip()]
    print(f"케이스: {len(cases)}개")

    if args.dry_run:
        all_ids = set(cid for c in cases for cid in c.get("golden_doc_ids", {}))
        pairs = set()
        for c in cases:
            ik = "|".join(sorted(c.get("expected_issues", [])))
            for cid in c.get("golden_doc_ids", {}):
                pairs.add((ik, cid))
        print(f"전체 (케이스, 청크) 쌍: {sum(len(c['golden_doc_ids']) for c in cases)}")
        print(f"유니크 chunk_id: {len(all_ids)}")
        print(f"유니크 (issue_type, chunk_id): {len(pairs)}")
        print("[dry-run] 종료.")
        return

    import anthropic
    client = anthropic.Anthropic()

    # 진행 상황 로드
    if args.reset and PROGRESS_F.exists():
        PROGRESS_F.unlink()
        print("[reset] 진행 상황 초기화.")
    progress    = load_progress()
    cache       = progress.get("cache", {})        # {issue_key|chunk_id: rel}
    completed   = set(progress.get("completed_cases", []))

    # 코퍼스 로드 (필요한 chunk_id만)
    all_chunk_ids = set(cid for c in cases for cid in c.get("golden_doc_ids", {}))
    print(f"코퍼스 로딩 중 ({len(all_chunk_ids)}개 청크)...")
    corpus = load_corpus(all_chunk_ids)
    missing = all_chunk_ids - set(corpus.keys())
    if missing:
        print(f"  [경고] 코퍼스에 없는 청크 {len(missing)}개 (rel=0 처리됨)")

    pending = [c for c in cases if c["id"] not in completed]
    total   = len(cases)
    print(f"분류 시작: {len(pending)}/{total}개 케이스 남음\n")

    api_calls = 0
    cache_hits = 0

    for case in pending:
        case_id    = case["id"]
        query      = case["query"]
        issue_key  = "|".join(sorted(case.get("expected_issues", [])))
        candidates = case.get("golden_doc_ids", {})

        print(f"[{case_id}] {query[:50]} ({issue_key})")

        for chunk_id in candidates:
            cache_key = f"{issue_key}|{chunk_id}"

            if cache_key in cache:
                cache_hits += 1
                continue  # 이미 분류됨

            doc_text = corpus.get(chunk_id, "")
            if not doc_text:
                cache[cache_key] = 0
                continue

            rel = classify(client, query, doc_text)
            if rel == -1:
                print(f"  [SKIP] {chunk_id[:50]} - 분류 실패, rel=0 처리")
                rel = 0
            else:
                print(f"  {chunk_id[:50]} → rel={rel}")
            cache[cache_key] = rel
            api_calls += 1

        completed.add(case_id)
        progress = {"cache": cache, "completed_cases": list(completed)}
        save_progress(progress)
        print(f"  → 완료 ({len(completed)}/{total})\n")

    # 결과 파일 생성
    results = []
    for case in cases:
        issue_key = "|".join(sorted(case.get("expected_issues", [])))
        golden_raw = {
            cid: cache.get(f"{issue_key}|{cid}", 0)
            for cid in case.get("golden_doc_ids", {})
        }
        golden_filtered = {k: v for k, v in golden_raw.items() if v >= 1}
        results.append({
            **case,
            "golden_doc_ids":  golden_filtered,
            "reviewed":        True,
            "review_method":   "llm_haiku",
        })

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rel2 = sum(1 for r in results for v in r["golden_doc_ids"].values() if v == 2)
    rel1 = sum(1 for r in results for v in r["golden_doc_ids"].values() if v == 1)
    total_docs = sum(len(r["golden_doc_ids"]) for r in results)
    print("=" * 60)
    print(f"골든셋 저장: {OUT_JSONL}")
    print(f"케이스 {len(results)}개 | 문서 {total_docs}개 (rel=2: {rel2}, rel=1: {rel1})")
    print(f"API 호출: {api_calls}회 | 캐시 재사용: {cache_hits}회")
    print("=" * 60)
    print("\n다음 단계: python scripts/evaluate_rag_comprehensive.py --eval golden_reviewed --skip-zero")


if __name__ == "__main__":
    main()
