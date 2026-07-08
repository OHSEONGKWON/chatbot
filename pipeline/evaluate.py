"""최종 평가: 우리 방법(detect_final.jsonl) vs 베이스라인 4종 (test 분할만, 그룹핑 A·B)

사용:
  python pipeline\\evaluate.py

산출: results_table.md (논문 Table 1 재료)
"""
import hashlib
import json
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COLL = ROOT / "data" / "collection"


def is_test(q_id: str) -> bool:
    """test 분할 판정 (HANDOVER §5)"""
    return int(hashlib.md5(q_id.encode()).hexdigest(), 16) % 10 >= 3


def extract_q_id(c_id: str) -> str:
    """c_id에서 q_id 추출: q_xxx__condition__cNN → q_xxx"""
    return c_id.split("__")[0]


def extract_condition(c_id: str) -> str:
    """c_id에서 condition 추출: q_xxx__condition__cNN → condition"""
    parts = c_id.split("__")
    return parts[1] if len(parts) >= 3 else "unknown"


def extract_claim_type(claims_map, c_id: str) -> str:
    """claims.jsonl에서 claim_type 추출"""
    return claims_map.get(c_id, {}).get("claim_type", "unknown")


def load_jsonl(path, skip_errors=False):
    """JSONL 로드 (손상 라인 무시 옵션)"""
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                if skip_errors:
                    print(f"  [경고] {path.name}:{i} 파싱 실패 (무시): {e}")
                else:
                    raise
    return records


def compute_metrics(y_true, y_pred, grouping="A"):
    """
    그룹핑 A: 환각 = NOT_SUPPORTED (이진)
    그룹핑 B: 미지지 = NOT_SUPPORTED + NEI (이진)
    """
    if grouping == "A":
        y_true_bin = [1 if y == "NOT_SUPPORTED" else 0 for y in y_true]
        y_pred_bin = [1 if y == "NOT_SUPPORTED" else 0 for y in y_pred]
    else:  # B
        y_true_bin = [1 if y in ("NOT_SUPPORTED", "NEI") else 0 for y in y_true]
        y_pred_bin = [1 if y in ("NOT_SUPPORTED", "NEI") else 0 for y in y_pred]

    p, r, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin,
                                                    average="binary", pos_label=1, zero_division=0)
    # 3분류 정확도도 함께 계산
    acc3 = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true) if y_true else 0

    return {"precision": p, "recall": r, "f1": f1, "accuracy_3class": acc3}


def compute_pr_auc(y_true, y_scores, grouping="A"):
    """
    SelfCheck용: 연속 점수로 PR-AUC 계산
    y_scores: hallu_score (높을수록 환각)
    """
    if grouping == "A":
        y_true_bin = [1 if y == "NOT_SUPPORTED" else 0 for y in y_true]
    else:  # B
        y_true_bin = [1 if y in ("NOT_SUPPORTED", "NEI") else 0 for y in y_true]

    if sum(y_true_bin) == 0 or sum(y_true_bin) == len(y_true_bin):
        return 0.0  # 단일 클래스만 존재

    return average_precision_score(y_true_bin, y_scores)


def find_best_threshold(y_true, y_scores, grouping="A"):
    """dev 분할에서 최적 임계값 찾기 (F1 최대화)"""
    if grouping == "A":
        y_true_bin = [1 if y == "NOT_SUPPORTED" else 0 for y in y_true]
    else:
        y_true_bin = [1 if y in ("NOT_SUPPORTED", "NEI") else 0 for y in y_true]

    thresholds = np.linspace(0, 1, 101)
    best_f1, best_th = 0, 0.5

    for th in thresholds:
        y_pred_bin = [1 if s >= th else 0 for s in y_scores]
        _, _, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin,
                                                        average="binary", pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th

    return best_th, best_f1


def bootstrap_f1_diff(y_true_map, pred_a, pred_b, cids, grouping, n_iter=1000, seed=42):
    """질문 단위 부트스트랩: 방법 a - 방법 b 의 F1 차이 95% CI와 p-value(Δ≤0 비율)."""
    rng = np.random.default_rng(seed)
    by_q = defaultdict(list)
    for c in cids:
        by_q[extract_q_id(c)].append(c)
    qids = list(by_q)

    def f1_of(pred, sample_cids):
        yt = [y_true_map[c] for c in sample_cids]
        yp = [pred[c] for c in sample_cids]
        return compute_metrics(yt, yp, grouping)["f1"]

    diffs = []
    for _ in range(n_iter):
        qs = rng.choice(qids, size=len(qids), replace=True)
        sample = [c for q in qs for c in by_q[q]]
        diffs.append(f1_of(pred_a, sample) - f1_of(pred_b, sample))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), float((diffs <= 0).mean())


def main():
    print("=" * 80)
    print("최종 평가: 우리 방법 vs 베이스라인 (test 분할, 그룹핑 A·B)")
    print("=" * 80)

    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    labels = load_jsonl(COLL / "labels_final.jsonl")
    claims_raw = load_jsonl(COLL / "claims.jsonl")
    detect = load_jsonl(COLL / "detect_final.jsonl", skip_errors=True)

    claims_map = {c["c_id"]: c for c in claims_raw}
    labels_map = {r["c_id"]: r["label"] for r in labels}
    detect_map = {r["c_id"]: r["verdict"] for r in detect}

    # 베이스라인 로드
    baselines = {}
    for method in ["vanilla", "ragjudge", "metaqa", "selfcheck"]:
        path = COLL / f"baseline_{method}.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            print(f"  [경고] {method}: 파일 없음 또는 비어있음 → 스킵")
            continue
        records = load_jsonl(path)
        if method == "selfcheck":
            baselines[method] = {r["c_id"]: r["hallu_score"] for r in records}
        else:
            baselines[method] = {r["c_id"]: r["verdict"] for r in records}

    # 2. test/dev 분할
    print("\n[2] test/dev 분할...")
    all_cids = set(labels_map.keys())
    test_cids = {c for c in all_cids if is_test(extract_q_id(c))}
    dev_cids = all_cids - test_cids

    print(f"  전체: {len(all_cids):,} | dev: {len(dev_cids):,} | test: {len(test_cids):,}")

    # 3. 무결성 확인
    print("\n[3] 산출물 무결성 확인...")
    print(f"  labels_final: {len(labels):,}")
    print(f"  detect_final: {len(detect_map):,} (손상 라인 {len(labels) - len(detect_map)}개 무시됨)")

    for method, data in baselines.items():
        print(f"  baseline_{method}: {len(data):,}")

    # 레이블 분포 (test)
    test_labels = [labels_map[c] for c in test_cids if c in labels_map]
    label_dist = Counter(test_labels)
    print(f"\n  test 레이블 분포: {dict(label_dist)}")

    # 4. SelfCheck 임계값 튜닝 (dev)
    selfcheck_thresholds = {}
    if "selfcheck" in baselines:
        print("\n[4] SelfCheck 임계값 튜닝 (dev 분할)...")
        dev_labels = [labels_map[c] for c in dev_cids if c in labels_map and c in baselines["selfcheck"]]
        dev_scores = [baselines["selfcheck"][c] for c in dev_cids if c in labels_map and c in baselines["selfcheck"]]

        for grp in ["A", "B"]:
            th, f1 = find_best_threshold(dev_labels, dev_scores, grp)
            selfcheck_thresholds[grp] = th
            print(f"  그룹핑 {grp}: threshold={th:.3f}, dev F1={f1:.3f}")

    # 5. 전체 비교표 (test)
    print("\n[5] 전체 비교표 산출 (test 분할)...")

    results = {}

    # 우리 방법
    test_y_true = [labels_map[c] for c in test_cids if c in labels_map and c in detect_map]
    test_y_pred = [detect_map[c] for c in test_cids if c in labels_map and c in detect_map]

    for grp in ["A", "B"]:
        results[f"ALCV-v2 (ours)_{grp}"] = compute_metrics(test_y_true, test_y_pred, grp)

    # 베이스라인 (verdict 기반)
    for method in ["vanilla", "ragjudge", "metaqa"]:
        if method not in baselines:
            continue
        test_y_pred_bl = [baselines[method][c] for c in test_cids if c in labels_map and c in baselines[method]]
        test_y_true_bl = [labels_map[c] for c in test_cids if c in labels_map and c in baselines[method]]

        for grp in ["A", "B"]:
            results[f"{method}_{grp}"] = compute_metrics(test_y_true_bl, test_y_pred_bl, grp)

    # SelfCheck (연속 점수 → PR-AUC + 임계값 기반 F1)
    if "selfcheck" in baselines:
        test_y_true_sc = [labels_map[c] for c in test_cids if c in labels_map and c in baselines["selfcheck"]]
        test_y_scores_sc = [baselines["selfcheck"][c] for c in test_cids if c in labels_map and c in baselines["selfcheck"]]

        for grp in ["A", "B"]:
            pr_auc = compute_pr_auc(test_y_true_sc, test_y_scores_sc, grp)
            th = selfcheck_thresholds[grp]
            test_y_pred_sc = ["NOT_SUPPORTED" if s >= th else "SUPPORTED" for s in test_y_scores_sc]
            metrics = compute_metrics(test_y_true_sc, test_y_pred_sc, grp)
            metrics["pr_auc"] = pr_auc
            metrics["threshold"] = th
            results[f"selfcheck_{grp}"] = metrics

    # 5-1. trivial 기준선 (전부 양성 판정 — 비교의 정직성 참조점)
    test_y_true_all = [labels_map[c] for c in test_cids if c in labels_map]
    for grp in ["A", "B"]:
        results[f"trivial_{grp}"] = compute_metrics(
            test_y_true_all, ["NOT_SUPPORTED"] * len(test_y_true_all), grp)

    # 5-2. 부트스트랩 유의성 (질문 단위 리샘플링, ours vs ragjudge)
    boot = {}
    if "ragjudge" in baselines:
        print("\n[5-2] 부트스트랩 유의성 검정 (1,000회)...")
        common = [c for c in test_cids
                  if c in labels_map and c in detect_map and c in baselines["ragjudge"]]
        for grp in ["A", "B"]:
            boot[grp] = bootstrap_f1_diff(labels_map, detect_map, baselines["ragjudge"],
                                          common, grp)
            print(f"  그룹핑 {grp}: ΔF1 95% CI [{boot[grp][0]:+.3f}, {boot[grp][1]:+.3f}], "
                  f"p(Δ≤0)={boot[grp][2]:.3f}")

    # 6. 분해표: claim_type별, condition별
    print("\n[6] 분해표 산출...")

    breakdown = {"claim_type": defaultdict(lambda: {"y_true": [], "y_pred": []}),
                 "condition": defaultdict(lambda: {"y_true": [], "y_pred": []})}

    for c in test_cids:
        if c not in labels_map or c not in detect_map:
            continue

        ct = extract_claim_type(claims_map, c)
        cond = extract_condition(c)

        breakdown["claim_type"][ct]["y_true"].append(labels_map[c])
        breakdown["claim_type"][ct]["y_pred"].append(detect_map[c])
        breakdown["condition"][cond]["y_true"].append(labels_map[c])
        breakdown["condition"][cond]["y_pred"].append(detect_map[c])

    breakdown_results = {}
    for dim in ["claim_type", "condition"]:
        for key, data in breakdown[dim].items():
            if len(data["y_true"]) == 0:
                continue
            for grp in ["A", "B"]:
                metrics = compute_metrics(data["y_true"], data["y_pred"], grp)
                breakdown_results[f"{dim}_{key}_{grp}"] = {**metrics, "count": len(data["y_true"])}

    # 7. 결과 저장
    print("\n[7] 결과 저장 중...")

    with open(ROOT / "results_table.md", "w", encoding="utf-8") as f:
        f.write("# 최종 평가 결과 (test 분할)\n\n")
        f.write(f"- 평가 대상: {len(test_cids):,} 클레임\n")
        f.write(f"- 레이블 분포: {dict(label_dist)}\n")
        f.write(f"- 그룹핑 A: 환각 = NOT_SUPPORTED\n")
        f.write(f"- 그룹핑 B: 미지지 = NOT_SUPPORTED + NEI\n\n")

        f.write("## 1. 전체 비교표\n\n")
        f.write("| Method | Grouping | Precision | Recall | F1 | Acc (3-class) | PR-AUC | Notes |\n")
        f.write("|--------|----------|-----------|--------|-----|---------------|--------|-------|\n")

        # 우리 방법 먼저
        for grp in ["A", "B"]:
            key = f"ALCV-v2 (ours)_{grp}"
            m = results[key]
            f.write(f"| **ALCV-v2 (ours)** | {grp} | {m['precision']:.3f} | {m['recall']:.3f} | "
                   f"**{m['f1']:.3f}** | {m['accuracy_3class']:.3f} | - | - |\n")

        # 베이스라인
        for method in ["vanilla", "ragjudge", "metaqa", "selfcheck"]:
            if f"{method}_A" not in results:
                continue

            for grp in ["A", "B"]:
                key = f"{method}_{grp}"
                m = results[key]

                if method == "selfcheck":
                    notes = f"th={m['threshold']:.3f}"
                    f.write(f"| {method} | {grp} | {m['precision']:.3f} | {m['recall']:.3f} | "
                           f"{m['f1']:.3f} | {m['accuracy_3class']:.3f} | {m['pr_auc']:.3f} | {notes} |\n")
                else:
                    f.write(f"| {method} | {grp} | {m['precision']:.3f} | {m['recall']:.3f} | "
                           f"{m['f1']:.3f} | {m['accuracy_3class']:.3f} | - | - |\n")

        # trivial 기준선 행 + selfcheck 각주
        for grp in ["A", "B"]:
            m = results[f"trivial_{grp}"]
            f.write(f"| trivial (전부 양성 판정) | {grp} | {m['precision']:.3f} | {m['recall']:.3f} | "
                   f"{m['f1']:.3f} | - | - | 참조점 |\n")
        f.write("\n주: selfcheck 그룹핑 B의 최적 임계값이 0 부근이면 'trivial(전부 양성)'과 사실상 동일한 "
                "퇴화 분류기이므로, F1이 아닌 PR-AUC로 해석할 것.\n")

        if boot:
            f.write("\n### 통계적 유의성 — 질문 단위 부트스트랩 1,000회 (ours − ragjudge)\n\n")
            for grp in ["A", "B"]:
                lo, hi, pv = boot[grp]
                sig = "유의" if lo > 0 else "비유의(CI가 0 포함)"
                f.write(f"- 그룹핑 {grp}: ΔF1 95% CI [{lo:+.3f}, {hi:+.3f}], p(Δ≤0)={pv:.3f} → {sig}\n")

        f.write("\n## 2. 분해표: claim_type별 (우리 방법)\n\n")
        f.write("| Claim Type | Grouping | Count | Precision | Recall | F1 | Acc (3-class) |\n")
        f.write("|------------|----------|-------|-----------|--------|-----|---------------|\n")

        for ct in sorted(breakdown["claim_type"].keys()):
            for grp in ["A", "B"]:
                key = f"claim_type_{ct}_{grp}"
                if key not in breakdown_results:
                    continue
                m = breakdown_results[key]
                f.write(f"| {ct} | {grp} | {m['count']} | {m['precision']:.3f} | {m['recall']:.3f} | "
                       f"{m['f1']:.3f} | {m['accuracy_3class']:.3f} |\n")

        f.write("\n## 3. 분해표: condition별 (우리 방법)\n\n")
        f.write("| Condition | Grouping | Count | Precision | Recall | F1 | Acc (3-class) |\n")
        f.write("|-----------|----------|-------|-----------|--------|-----|---------------|\n")

        for cond in ["closed_book", "weak_rag"]:
            for grp in ["A", "B"]:
                key = f"condition_{cond}_{grp}"
                if key not in breakdown_results:
                    continue
                m = breakdown_results[key]
                f.write(f"| {cond} | {grp} | {m['count']} | {m['precision']:.3f} | {m['recall']:.3f} | "
                       f"{m['f1']:.3f} | {m['accuracy_3class']:.3f} |\n")

        f.write("\n## 4. 해석\n\n")

        # 주요 발견 자동 생성
        ours_a = results["ALCV-v2 (ours)_A"]["f1"]
        ours_b = results["ALCV-v2 (ours)_B"]["f1"]

        baseline_best_a = max([results[f"{m}_A"]["f1"] for m in ["vanilla", "ragjudge", "metaqa", "selfcheck"]
                               if f"{m}_A" in results], default=0)
        baseline_best_b = max([results[f"{m}_B"]["f1"] for m in ["vanilla", "ragjudge", "metaqa", "selfcheck"]
                               if f"{m}_B" in results], default=0)

        f.write(f"### 주요 발견\n\n")
        f.write(f"1. **그룹핑 A (환각=NOT_SUPPORTED)**: 우리 방법 F1={ours_a:.3f}, 최고 베이스라인 F1={baseline_best_a:.3f}\n")
        f.write(f"2. **그룹핑 B (미지지=NOT_SUPPORTED+NEI)**: 우리 방법 F1={ours_b:.3f}, 최고 베이스라인 F1={baseline_best_b:.3f}\n")

        if ours_a > baseline_best_a:
            f.write(f"3. 우리 방법이 그룹핑 A에서 베이스라인 대비 **{ours_a - baseline_best_a:.3f}p 우위** (논문 주장 입증)\n")
        else:
            f.write(f"3. ⚠️ 우리 방법이 그룹핑 A에서 베이스라인에 **미달** (추가 분석 필요)\n")

        if ours_b > baseline_best_b:
            f.write(f"4. 우리 방법이 그룹핑 B에서 베이스라인 대비 **{ours_b - baseline_best_b:.3f}p 우위**\n")

        f.write(f"\n### 베이스라인 비교\n\n")
        for method in ["vanilla", "ragjudge", "selfcheck"]:
            if f"{method}_A" not in results:
                continue
            f.write(f"- **{method}**: ")
            if method == "selfcheck":
                f.write(f"PR-AUC={results[f'{method}_A']['pr_auc']:.3f} (A), {results[f'{method}_B']['pr_auc']:.3f} (B) | ")
            f.write(f"F1={results[f'{method}_A']['f1']:.3f} (A), {results[f'{method}_B']['f1']:.3f} (B)\n")

    print(f"\n✅ 평가 완료 → {ROOT / 'results_table.md'}")
    print(f"\n주요 지표 (test):")
    print(f"  ALCV-v2 (ours) | 그룹핑 A: F1={ours_a:.3f} | 그룹핑 B: F1={ours_b:.3f}")
    print(f"  최고 베이스라인 | 그룹핑 A: F1={baseline_best_a:.3f} | 그룹핑 B: F1={baseline_best_b:.3f}")


if __name__ == "__main__":
    main()
