"""
NER 개선 효과 검증 스크립트 (단계 1-3)

평가:
- 단계 1: 법률용어 정규화 강화
- 단계 2: Entity Co-occurrence Pattern  
- 단계 3: Confidence Score 시스템

기존 10개 평가셋으로 개선 효과를 검증합니다.
"""

import sys
import json
from pathlib import Path

# 프로젝트 루트 경로 설정
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.modules.ner_checker import NERFactChecker


def load_eval_pairs(eval_file):
    """평가셋 로드"""
    pairs = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def evaluate_improvements():
    """개선 효과 검증"""
    
    eval_file = REPO_ROOT / "data" / "mock_data" / "ner_hybrid_eval_pairs.jsonl"
    
    print("=" * 80)
    print("NER 개체명 검증 개선 효과 검증")
    print("=" * 80)
    print()
    
    # NER 체커 초기화
    ner_checker = NERFactChecker()
    pairs = load_eval_pairs(eval_file)
    
    print(f"평가셋 로드: {len(pairs)}개 케이스")
    print()
    
    # 결과 저장
    results = []
    correct_count = 0
    
    print("=" * 80)
    print(f"{'케이스':<5} | {'레이블':<8} | {'단어':<25} | {'결과':<10} | {'신뢰도':<8} | {'도메인':<6}")
    print("=" * 80)
    
    for idx, pair in enumerate(pairs, 1):
        label = pair["label"]
        word = pair["word"]
        candidates = pair["candidates"]
        expected = pair["expected_supported"]
        
        # 매칭 수행 (context_law 없이 기본 테스트)
        match_result = ner_checker._match_entity_against_candidates(
            label=label,
            word=word,
            candidates=candidates,
            context_law=None  # 기본 테스트
        )
        
        is_supported = match_result["is_supported"]
        confidence = match_result.get("confidence", 0.0)
        combo_score = match_result.get("combo_score", 1.0)
        method = match_result.get("method", "none")
        
        # 정확도 판정
        is_correct = is_supported == expected
        correct_count += is_correct
        result_str = "✓ PASS" if is_correct else "✗ FAIL"
        
        results.append({
            "idx": idx,
            "label": label,
            "word": word,
            "expected": expected,
            "got": is_supported,
            "method": method,
            "confidence": confidence,
            "combo_score": combo_score,
            "correct": is_correct,
            "candidate": match_result.get("candidate"),
        })
        
        print(f"{idx:<5} | {label:<8} | {word:<25} | {result_str:<10} | {confidence:>6.1%} | {combo_score:>5.1%}")
    
    print("=" * 80)
    print()
    
    # 정확도 요약
    accuracy = correct_count / len(pairs) * 100
    print(f"정확도: {correct_count}/{len(pairs)} = {accuracy:.1f}%")
    print()
    
    # 상세 분석
    print("=" * 80)
    print("상세 분석 (실패 케이스)")
    print("=" * 80)
    
    failures = [r for r in results if not r["correct"]]
    if not failures:
        print("모든 케이스 통과! ✨")
    else:
        for r in failures:
            print(f"\n[케이스 {r['idx']}] {r['label']} - {r['word']}")
            print(f"  예상: {r['expected']} | 결과: {r['got']}")
            print(f"  방법: {r['method']} | 신뢰도: {r['confidence']:.3f} | 후보: {r['candidate']}")
    
    print()
    print("=" * 80)
    print("개선 사항 요약")
    print("=" * 80)
    print("""
✅ 단계 1: 법률용어 정규화 강화
   - "형법 제298조" -> "형법" 정규화
   - 약칭 매핑 (근기 -> 근로기준법)
   - 제조항 번호 자동 제거

✅ 단계 2: Entity Co-occurrence Pattern
   - 법령 + 엔티티 도메인 호환성 검증
   - 도메인 불일치 시 신뢰도 감소 (0.3~1.0)
   - CRIME/PENALTY와 법령 조합 검증

✅ 단계 3: Confidence Score 시스템
   - 최종 신뢰도 스코어 (0~1 범위)
   - Weighted averaging: 정확(50%) + 퍼지(30%) + 의미(20%)
   - 도메인 호환성 점수 통합
   - 임계값 기반 판정:
     * confidence > 0.95: 자동 교정 (신뢰도 높음)
     * 0.80~0.95: 경고 (중간 신뢰)
     * < 0.80: 무시 (낮은 신뢰)

📊 평가 결과
   - 정확도: {:.1f}% ({}/{})
   - 신뢰도 스코어 추가로 거짓 양성 필터링 가능
    """.format(accuracy, correct_count, len(pairs)))
    
    # Confidence 분포
    confidences = [r["confidence"] for r in results]
    high_conf = sum(1 for c in confidences if c > 0.95)
    mid_conf = sum(1 for c in confidences if 0.80 <= c <= 0.95)
    low_conf = sum(1 for c in confidences if c < 0.80)
    
    print(f"신뢰도 분포:")
    print(f"  - 높음 (> 0.95): {high_conf}개 (자동 교정)")
    print(f"  - 중간 (0.80~0.95): {mid_conf}개 (경고)")
    print(f"  - 낮음 (< 0.80): {low_conf}개 (무시)")
    
    # 도메인 호환성 분석
    print(f"\n도메인 호환성 분석:")
    domain_issues = [r for r in results if r["combo_score"] < 1.0]
    if domain_issues:
        print(f"  도메인 불일치 감지: {len(domain_issues)}개")
        for r in domain_issues[:3]:  # 처음 3개만 표시
            print(f"    - {r['label']}: {r['word']}")
    else:
        print(f"  모든 케이스 도메인 호환성 OK")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    evaluate_improvements()
