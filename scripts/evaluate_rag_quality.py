import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.rag import retriever


CASES = [
    # 임금체불
    {
        "name": "임금체불/알바비 미지급",
        "query": "편의점 알바 월급 80만원을 아직 못 받았습니다.",
        "category": "노동",
        "expected_issues": ["wage_unpaid"],
        "must_have": ["근로기준법", "제43조", "임금"],
        "top_any": ["제43조", "임금"],
    },
    {
        "name": "임금체불/퇴사 후 미지급",
        "query": "카페 알바를 그만둔 뒤 급여를 안 줬습니다.",
        "category": "노동",
        "expected_issues": ["wage_unpaid"],
        "must_have": ["임금", "지급"],
        "top_any": ["제43조", "임금"],
    },
    {
        "name": "임금체불/카톡 증거",
        "query": "사장님이 알바비를 다음 달에 준다고만 하고 계속 체불합니다. 카톡은 있습니다.",
        "category": "노동",
        "expected_issues": ["wage_unpaid"],
        "must_have": ["임금", "체불"],
        "top_any": ["제43조", "임금"],
    },
    {
        "name": "임금체불/근로계약서 없음",
        "query": "알바비를 못 받았고 근로계약서도 없습니다.",
        "category": "노동",
        "expected_issues": ["wage_unpaid", "missing_contract"],
        "must_have": ["근로기준법", "임금"],
        "top_any": ["제43조", "제17조", "임금"],
    },
    # 최저임금
    {
        "name": "최저임금/시급 미달",
        "query": "시급이 최저임금보다 적은 것 같습니다.",
        "category": "노동",
        "expected_issues": ["minimum_wage"],
        "must_have": ["최저임금"],
        "top_any": ["최저임금법", "제6조"],
    },
    {
        "name": "최저임금/최저시급",
        "query": "최저시급보다 낮게 받고 일했습니다.",
        "category": "노동",
        "expected_issues": ["minimum_wage"],
        "must_have": ["최저임금"],
        "top_any": ["최저임금법", "제6조"],
    },
    {
        "name": "최저임금/알바 시급",
        "query": "알바 시급이 너무 적은데 최저임금 위반인지 궁금합니다.",
        "category": "노동",
        "expected_issues": ["minimum_wage"],
        "must_have": ["최저임금"],
        "top_any": ["최저임금법", "제6조"],
    },
    {
        "name": "최저임금/수습 핑계",
        "query": "수습이라면서 최저임금보다 적게 줬습니다.",
        "category": "노동",
        "expected_issues": ["minimum_wage"],
        "must_have": ["최저임금"],
        "top_any": ["최저임금법", "제6조"],
    },
    # 해고
    {
        "name": "해고/갑작스러운 해고",
        "query": "갑자기 내일부터 나오지 말라고 해고당했습니다.",
        "category": "노동",
        "expected_issues": ["dismissal"],
        "must_have": ["해고"],
        "top_any": ["제23조", "부당해고", "정당한 이유"],
    },
    {
        "name": "해고/해고예고",
        "query": "사장이 당일에 그만 나오라고 했는데 해고예고수당을 받을 수 있나요?",
        "category": "노동",
        "expected_issues": ["dismissal"],
        "must_have": ["해고"],
        "top_any": ["제23조", "제26조", "해고예고"],
    },
    {
        "name": "해고/서면 통지 없음",
        "query": "문자로만 해고 통보를 받았고 서면 통지는 없었습니다.",
        "category": "노동",
        "expected_issues": ["dismissal"],
        "must_have": ["해고"],
        "top_any": ["제23조", "제27조", "서면"],
    },
    {
        "name": "해고/부당해고",
        "query": "아무 이유 없이 부당해고를 당한 것 같습니다.",
        "category": "노동",
        "expected_issues": ["dismissal"],
        "must_have": ["부당해고", "해고"],
        "top_any": ["제23조", "부당해고"],
    },
    # 근로계약서 미작성
    {
        "name": "근로계약서/미작성",
        "query": "근로계약서를 안 쓰고 알바를 시작했습니다.",
        "category": "노동",
        "expected_issues": ["missing_contract"],
        "must_have": ["근로계약서", "근로조건"],
        "top_any": ["제17조", "근로계약서"],
    },
    {
        "name": "근로계약서/계약서 없음",
        "query": "계약서 없이 일했는데 문제가 되나요?",
        "category": "노동",
        "expected_issues": ["missing_contract"],
        "must_have": ["근로계약서", "서면"],
        "top_any": ["제17조", "근로조건"],
    },
    {
        "name": "근로계약서/근로조건 미명시",
        "query": "시급과 근무시간을 서면으로 받은 적이 없습니다.",
        "category": "노동",
        "expected_issues": ["missing_contract"],
        "must_have": ["근로조건", "서면"],
        "top_any": ["제17조", "근로조건"],
    },
    {
        "name": "근로계약서/알바 계약서",
        "query": "알바 계약서를 작성하지 않았고 구두로만 조건을 들었습니다.",
        "category": "노동",
        "expected_issues": ["missing_contract"],
        "must_have": ["근로계약서"],
        "top_any": ["제17조", "근로계약서"],
    },
    # 직장 내 괴롭힘
    {
        "name": "직장 내 괴롭힘/폭언",
        "query": "점장이 계속 폭언하고 괴롭힙니다.",
        "category": "노동",
        "expected_issues": ["workplace_harassment"],
        "must_have": ["직장 내 괴롭힘"],
        "top_any": ["제76조의2", "직장 내 괴롭힘"],
    },
    {
        "name": "직장 내 괴롭힘/따돌림",
        "query": "직장에서 계속 따돌림을 당하고 업무에서 배제됩니다.",
        "category": "노동",
        "expected_issues": ["workplace_harassment"],
        "must_have": ["괴롭힘"],
        "top_any": ["제76조의2", "직장 내 괴롭힘"],
    },
    {
        "name": "직장 내 괴롭힘/갑질",
        "query": "상사가 갑질하고 모욕적인 말을 반복합니다.",
        "category": "노동",
        "expected_issues": ["workplace_harassment"],
        "must_have": ["괴롭힘"],
        "top_any": ["제76조의2", "직장 내 괴롭힘"],
    },
    {
        "name": "직장 내 괴롭힘/조치",
        "query": "직장 내 괴롭힘을 신고하면 회사가 어떤 조치를 해야 하나요?",
        "category": "노동",
        "expected_issues": ["workplace_harassment"],
        "must_have": ["직장 내 괴롭힘", "조치"],
        "top_any": ["제76조의2", "제76조의3", "직장 내 괴롭힘"],
    },
    # 성희롱
    {
        "name": "성희롱/교수 발언",
        "query": "교수가 성희롱 발언을 반복했고 학교에 신고하고 싶습니다.",
        "category": "성폭력",
        "expected_issues": ["sexual_harassment"],
        "must_have": ["성희롱", "신고"],
        "top_any": ["성희롱", "제30조", "상담 및 신고 접수"],
    },
    {
        "name": "성희롱/성적 농담",
        "query": "선배가 성적 농담을 계속해서 불쾌합니다.",
        "category": "성폭력",
        "expected_issues": ["sexual_harassment"],
        "must_have": ["성희롱"],
        "top_any": ["성희롱", "상담"],
    },
    {
        "name": "성희롱/외모 평가",
        "query": "조교가 외모 평가와 성적 발언을 반복합니다.",
        "category": "성폭력",
        "expected_issues": ["sexual_harassment"],
        "must_have": ["성희롱"],
        "top_any": ["성희롱", "신고", "상담"],
    },
    {
        "name": "성희롱/신고 절차",
        "query": "대학 내 성희롱 신고 절차가 궁금합니다.",
        "category": "성폭력",
        "expected_issues": ["sexual_harassment"],
        "must_have": ["성희롱", "신고"],
        "top_any": ["상담 및 신고 접수", "제30조", "성희롱"],
    },
    # 강제추행
    {
        "name": "강제추행/동의 없는 신체접촉",
        "query": "술자리에서 동의 없이 몸을 만졌습니다.",
        "category": "성폭력",
        "expected_issues": ["indecent_assault"],
        "must_have": ["강제추행"],
        "top_any": ["제298조", "강제추행"],
    },
    {
        "name": "강제추행/허리 접촉",
        "query": "동아리 선배가 거절했는데도 허리를 만졌습니다.",
        "category": "성폭력",
        "expected_issues": ["indecent_assault"],
        "must_have": ["강제추행"],
        "top_any": ["제298조", "강제추행"],
    },
    {
        "name": "강제추행/입맞춤",
        "query": "상대가 갑자기 입맞춤을 해서 성추행으로 신고하고 싶습니다.",
        "category": "성폭력",
        "expected_issues": ["indecent_assault"],
        "must_have": ["강제추행", "성폭력"],
        "top_any": ["제298조", "강제추행"],
    },
    {
        "name": "강제추행/반복 접촉",
        "query": "거절했는데도 계속 어깨와 몸을 만져서 무섭습니다.",
        "category": "성폭력",
        "expected_issues": ["indecent_assault"],
        "must_have": ["강제추행"],
        "top_any": ["제298조", "강제추행"],
    },
    # 불법촬영
    {
        "name": "불법촬영/동의 없는 사진",
        "query": "동의 없이 사진을 찍었습니다.",
        "category": "성폭력",
        "expected_issues": ["illegal_filming"],
        "must_have": ["촬영"],
        "top_any": ["제14조", "카메라", "촬영"],
    },
    {
        "name": "불법촬영/몰카",
        "query": "누군가 몰카를 찍은 것 같습니다.",
        "category": "성폭력",
        "expected_issues": ["illegal_filming"],
        "must_have": ["촬영"],
        "top_any": ["제14조", "카메라", "촬영"],
    },
    {
        "name": "불법촬영/영상 유포",
        "query": "동의 없이 찍은 영상을 단톡방에 올렸습니다.",
        "category": "성폭력",
        "expected_issues": ["illegal_filming"],
        "must_have": ["촬영물"],
        "top_any": ["제14조", "촬영물", "반포"],
    },
    {
        "name": "불법촬영/카메라",
        "query": "화장실에서 카메라로 몰래 촬영한 것 같아요.",
        "category": "성폭력",
        "expected_issues": ["illegal_filming"],
        "must_have": ["카메라", "촬영"],
        "top_any": ["제14조", "카메라"],
    },
]


def merge_docs(docs: list[dict], limit: int = 5) -> str:
    return "\n".join(
        f"{doc.get('text', '')} {doc.get('metadata', {})}" for doc in docs[:limit]
    )


def case_passed(case: dict, docs: list[dict], inferred_issues: list[str]) -> tuple[bool, dict]:
    top_text = merge_docs(docs, limit=3)
    all_text = merge_docs(docs, limit=5)
    issue_hits = [issue for issue in case["expected_issues"] if issue in inferred_issues]
    must_hits = [term for term in case["must_have"] if term in all_text]
    top_hits = [term for term in case["top_any"] if term in top_text]
    ok = (
        len(issue_hits) == len(case["expected_issues"])
        and len(must_hits) >= max(1, min(2, len(case["must_have"])))
        and len(top_hits) >= 1
    )
    return ok, {"issue_hits": issue_hits, "must_hits": must_hits, "top_hits": top_hits}


def main():
    failures = []
    for case in CASES:
        inferred_issues = retriever.infer_issues(case["query"], category=case["category"])
        docs = retriever.retrieve(case["query"], top_k=5, legal_category=case["category"])
        ok, detail = case_passed(case, docs, inferred_issues)
        if not ok:
            failures.append((case, detail, inferred_issues, docs))

        print(f"\n[{case['name']}] {'PASS' if ok else 'FAIL'}")
        print(f"issues: expected={case['expected_issues']} inferred={inferred_issues}")
        print(f"hits: {detail}")
        for i, doc in enumerate(docs[:3], 1):
            metadata = doc.get("metadata") or {}
            title = metadata.get("law_name") or metadata.get("source_file") or "근거 문서"
            article = metadata.get("article_id") or metadata.get("section_label") or ""
            print(f"{i}. {round(float(doc.get('score', 0.0)), 2)} | {title} | {article}")

    passed = len(CASES) - len(failures)
    print(f"\nRAG quality: {passed}/{len(CASES)} passed")
    if failures:
        names = ", ".join(case["name"] for case, *_ in failures)
        raise SystemExit(f"{len(failures)} RAG quality case(s) failed: {names}")


if __name__ == "__main__":
    main()
