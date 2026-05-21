"""
파이프라인 4-지표 평가 시스템

지표 1 — SIM (유사성 지표)
    동일 RAG 근거로 유사 질문들을 답변했을 때의 내용 일치도.
    측정 신뢰성은 질문 품질(법률 슬롯 보존 + 다양성)로 조정.
    SIM = raw_consistency × (0.65 + 0.35 × question_quality)

지표 2 — RAG (RAG 품질 지표)
    법령 근거 커버리지를 기존 RRS보다 강화한 검색 품질.
    RAG = 0.25×top_doc + 0.25×issue + 0.35×law + 0.15×domain
    (법령 근거 없으면 doc_count 기반 캡핑)

지표 3 — NER (NER 지표)
    답변 개체명의 RAG 근거 일치도 — 곱셈 감쇠 방식.
    고위험 ×0.70 / 중위험 ×0.85 / 경고 −0.03 / 엔티티 없음 → 0.70

지표 4 — FINAL (최종 지표)
    기하평균: SIM^0.35 × RAG^0.40 × NER^0.25
    한 지표라도 크게 하락하면 최종도 연동 하락.
    RAG 가중치가 가장 높은 이유: 검색 실패는 이후 단계 전체에 영향을 미침.

설계 근거:
    System 1 (운영 지표): SQQS, raw_consistency_score, RRS, NER mismatch → 런타임 산정
    System 2 (평가 철학): AUC 분리도, Entity F1, NDCG → 측정 방향 차용
    통합 결과: 정답 레이블 없이도 런타임에 산정 가능하면서 System 2 수준의 민감도 확보
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """파이프라인 1회 실행에 대한 4개 지표 결과."""
    sim_score: float
    rag_score: float
    ner_score: float
    final_score: float
    sim_detail: dict[str, Any] = field(default_factory=dict)
    rag_detail: dict[str, Any] = field(default_factory=dict)
    ner_detail: dict[str, Any] = field(default_factory=dict)
    final_detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 지표 1: SIM (유사성 지표)
# ---------------------------------------------------------------------------

def compute_sim_score(
    raw_consistency_score: float,
    sqqs_detail: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """
    유사성 지표 (SIM) — [0, 1]

    primary:  raw_consistency_score (유사 질문 답변들의 임베딩 코사인 유사도 중앙값)
    modifier: question_quality (법률 슬롯 보존율 × 다양성) — 측정 자체의 신뢰성

    질문 품질이 낮으면 일관성 측정이 의미 없으므로 SIM을 최대 35% 하향 조정한다.
    질문 품질이 완벽(1.0)이면 SIM = raw_consistency 그대로.
    질문 품질이 최악(0.0)이면 SIM = raw_consistency × 0.65.

    SIM = raw_consistency × (0.65 + 0.35 × question_quality)
    where question_quality = 0.55 × legal_core_preservation + 0.45 × diversity_non_duplication
    """
    raw = max(0.0, min(1.0, float(raw_consistency_score or 0.0)))
    legal_core = max(0.0, min(1.0, float(sqqs_detail.get("legal_core_preservation") or 0.5)))
    diversity = max(0.0, min(1.0, float(sqqs_detail.get("diversity_non_duplication") or 0.5)))

    question_quality = 0.55 * legal_core + 0.45 * diversity
    sim_raw = raw * (0.65 + 0.35 * question_quality)
    sim_score = round(max(0.0, min(1.0, sim_raw)), 3)

    return sim_score, {
        "score": sim_score,
        "answer_consistency": round(raw, 3),
        "question_quality": round(question_quality, 3),
        "legal_core_preservation": round(legal_core, 3),
        "diversity_non_duplication": round(diversity, 3),
    }


# ---------------------------------------------------------------------------
# 지표 2: RAG (RAG 품질 지표)
# ---------------------------------------------------------------------------

def compute_rag_score(
    rrs_report: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """
    RAG 품질 지표 (RAG) — [0, 1]

    기존 RRS 공식 대비 법령 근거(legal_basis_coverage) 가중치를 강화.
    이유: 법령 조문 없이 검색된 문서는 환각 위험이 높고, System 2의 NDCG에서
    관련 문서의 법령 적합성이 핵심 평가 기준임을 반영.

         기존 RRS    신규 RAG
    top_doc   0.30  →   0.25
    issue     0.25  →   0.25
    law       0.25  →   0.35  (법령 근거 강화)
    domain    0.20  →   0.15

    doc_count 캡핑: 근거 문서가 없거나 1개뿐이면 상한을 제한.
    """
    top_doc = max(0.0, min(1.0, float(rrs_report.get("top_doc_relevance") or 0.0)))
    issue = max(0.0, min(1.0, float(rrs_report.get("issue_match") or 0.0)))
    law = max(0.0, min(1.0, float(rrs_report.get("legal_basis_coverage") or 0.0)))
    domain = max(0.0, min(1.0, float(rrs_report.get("domain_consistency") or 1.0)))
    doc_count = int(rrs_report.get("doc_count") or 0)

    base = 0.25 * top_doc + 0.25 * issue + 0.35 * law + 0.15 * domain

    if doc_count == 0:
        rag_score = min(base, 0.20)
    elif doc_count == 1:
        rag_score = min(base, 0.72)
    else:
        rag_score = base

    rag_score = round(max(0.0, min(1.0, rag_score)), 3)

    return rag_score, {
        "score": rag_score,
        "top_doc_relevance": round(top_doc, 3),
        "issue_match": round(issue, 3),
        "legal_basis_coverage": round(law, 3),
        "domain_consistency": round(domain, 3),
        "doc_count": doc_count,
        "rrs_original": round(float(rrs_report.get("score") or 0.0), 3),
    }


# ---------------------------------------------------------------------------
# 지표 3: NER (NER 지표)
# ---------------------------------------------------------------------------

def compute_ner_score(
    found_entities: list[dict[str, Any]],
    mismatches: list[dict[str, Any]],
    mid_confidence_warnings: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """
    NER 지표 (NER) — [0, 1]

    답변 내 개체명이 RAG 근거와 일치하는지를 환각 심각도 기반 곱셈 감쇠로 측정.
    System 2의 Entity-level F1 철학을 런타임에 근사: '지지된 엔티티 비율'을
    직접 계산하는 대신 '불일치의 누적 심각도'로 역산.

    - 고위험(high risk):  × 0.70 per mismatch  (조문번호 오기, 금지 법령 등)
    - 중위험(medium risk): × 0.85 per mismatch
    - 경고(warning):      − 0.03 per warning (최대 3건 적용)
    - 엔티티 없음:        0.70 (검증 불가 → 보수적 중립)

    예시:
      고위험 1건 → 0.70
      고위험 2건 → 0.49
      중위험 1건 → 0.85
      중위험 2건 → 0.72
      경고 3건    → 0.91
    """
    found_entities = list(found_entities or [])
    mismatches = list(mismatches or [])
    mid_confidence_warnings = list(mid_confidence_warnings or [])

    high_risk = [m for m in mismatches if m.get("risk_level") == "high"]
    medium_risk = [m for m in mismatches if m.get("risk_level") == "medium"]

    if not found_entities:
        return 0.70, {
            "score": 0.70,
            "reason": "no_entities_found",
            "total_entities": 0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "warning_count": 0,
        }

    base = 1.0
    for _ in high_risk:
        base *= 0.70
    for _ in medium_risk:
        base *= 0.85

    warning_penalty = 0.03 * min(3, len(mid_confidence_warnings))
    ner_score = round(max(0.0, base - warning_penalty), 3)

    return ner_score, {
        "score": ner_score,
        "total_entities": len(found_entities),
        "high_risk_count": len(high_risk),
        "medium_risk_count": len(medium_risk),
        "warning_count": len(mid_confidence_warnings),
        "high_risk_labels": [m.get("label") for m in high_risk],
        "medium_risk_labels": [m.get("label") for m in medium_risk],
    }


# ---------------------------------------------------------------------------
# 지표 4: FINAL (최종 지표)
# ---------------------------------------------------------------------------

W_SIM = 0.35
W_RAG = 0.40
W_NER = 0.25


def compute_final_score(
    sim: float,
    rag: float,
    ner: float,
) -> tuple[float, dict[str, Any]]:
    """
    최종 지표 (FINAL) — [0, 1]

    가중합: FINAL = 0.35 × SIM + 0.40 × RAG + 0.25 × NER

    가중치 근거:
      RAG (0.40): 검색 실패는 이후 일관성 검사·NER 교정 전체에 영향을 미침.
                  잘못된 근거 → 틀린 답변 → 어떤 교정도 의미 없어짐.
      SIM (0.35): 답변 일관성은 환각 탐지의 핵심 신호.
                  같은 질문을 다르게 물었을 때 답변이 흔들리면 신뢰 불가.
      NER (0.25): 이미 파이프라인 내부에서 교정이 완료된 상태의 평가.
                  교정 후 잔여 환각 위험을 반영하는 보조 신호로 기여.

    민감도 분석 (기준선 SIM=0.79, RAG=0.78, NER=1.00 → FINAL≈0.839):
      SIM 0.79 → 0.40 : FINAL ≈ −0.137  (−16%)
      RAG 0.78 → 0.33 : FINAL ≈ −0.180  (−21%)
      NER 1.00 → 0.70 : FINAL ≈ −0.075  (−9%)   고위험 1건
      NER 1.00 → 0.49 : FINAL ≈ −0.128  (−15%)  고위험 2건
    """
    s = max(0.0, min(1.0, float(sim or 0.0)))
    r = max(0.0, min(1.0, float(rag or 0.0)))
    n = max(0.0, min(1.0, float(ner or 0.0)))

    final = W_SIM * s + W_RAG * r + W_NER * n
    final_score = round(max(0.0, min(1.0, final)), 3)

    if final_score >= 0.90:
        grade = "A"
    elif final_score >= 0.80:
        grade = "B"
    elif final_score >= 0.70:
        grade = "C"
    elif final_score >= 0.55:
        grade = "D"
    else:
        grade = "F"

    return final_score, {
        "score": final_score,
        "grade": grade,
        "weights": {"sim": W_SIM, "rag": W_RAG, "ner": W_NER},
        "components": {
            "sim_contrib": round(W_SIM * s, 3),
            "rag_contrib": round(W_RAG * r, 3),
            "ner_contrib": round(W_NER * n, 3),
        },
    }


# ---------------------------------------------------------------------------
# 통합 진입점
# ---------------------------------------------------------------------------

def evaluate_pipeline_output(
    raw_consistency_score: float,
    sqqs_detail: dict[str, Any],
    rrs_report: dict[str, Any],
    found_entities: list[dict[str, Any]],
    mismatches: list[dict[str, Any]],
    mid_confidence_warnings: list[dict[str, Any]],
) -> EvaluationResult:
    """파이프라인 1회 실행 결과로 4개 지표를 산정한다."""
    sim_score, sim_detail = compute_sim_score(raw_consistency_score, sqqs_detail)
    rag_score, rag_detail = compute_rag_score(rrs_report)
    ner_score, ner_detail = compute_ner_score(found_entities, mismatches, mid_confidence_warnings)
    final_score, final_detail = compute_final_score(sim_score, rag_score, ner_score)

    return EvaluationResult(
        sim_score=sim_score,
        rag_score=rag_score,
        ner_score=ner_score,
        final_score=final_score,
        sim_detail=sim_detail,
        rag_detail=rag_detail,
        ner_detail=ner_detail,
        final_detail=final_detail,
    )
