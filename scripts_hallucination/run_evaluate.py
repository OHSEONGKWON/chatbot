"""3단계: ALCV 판정/평가/비교 (오프라인, numpy만).

alcv_features.json 을 입력으로:
  - 층화 train/test 분할 + 5-겹 CV 로 ALCV 모델(하드게이트+로지스틱) 평가
  - 절제실험: 하드게이트 단독 / 소프트 근거검증 단독 / 전체
  - 베이스라인(MetaQA per-case, SelfCheck 요약) 동일 test 분할 비교
  - 환각 유형별 탐지율

실행:  venv\Scripts\python.exe scripts_hallucination\run_evaluate.py
출력(results/): alcv_results.json / alcv_results.md / alcv_config.json / fig_alcv.png
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alcv.config import RECALL_FLOOR, RESULTS_DIR
from alcv.aggregate import (FEATURE_NAMES, AlcvModel, features_to_matrix,
                            metrics, best_threshold)

FEATURES_PATH = RESULTS_DIR / "alcv_features.json"
TEST_FRACTION = 0.4
SEED = 42
N_FOLDS = 5


def load_features():
    if not FEATURES_PATH.exists():
        sys.exit(f"[오류] {FEATURES_PATH} 없음. 먼저 run_collect_features.py 실행")
    rows = [r for r in json.loads(FEATURES_PATH.read_text(encoding="utf-8")) if "error" not in r]
    ids = [r["hallu_id"] for r in rows]
    y = np.array([1 if r["true_label"] else 0 for r in rows])
    X = features_to_matrix([r["features"] for r in rows])
    htype = [r["hallu_type"] for r in rows]
    return ids, X, y, htype, rows


def stratified_split(y, frac, seed):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in (0, 1):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        n = int(round(len(idx) * frac))
        te += idx[:n].tolist(); tr += idx[n:].tolist()
    return np.array(sorted(tr)), np.array(sorted(te))


def stratified_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), int)
    for c in (0, 1):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        for i, j in enumerate(idx):
            fold[j] = i % k
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)]


def cv_f1(X, y):
    f1s = []
    for tr, te in stratified_folds(y, N_FOLDS, SEED):
        m = AlcvModel(RECALL_FLOOR).fit(X[tr], y[tr])
        f1s.append(metrics(y[te], m.predict(X[te]))["f1"])
    return float(np.mean(f1s)), float(np.std(f1s))


def baselines_on(ids_te, y_te):
    out = {}
    mq = RESULTS_DIR / "metaqa_results.json"
    if mq.exists():
        rows = {r["hallu_id"]: r for r in json.loads(mq.read_text(encoding="utf-8"))}
        yp = np.array([1 if (rows.get(h) and rows[h].get("predicted")) else 0 for h in ids_te])
        out["MetaQA"] = metrics(y_te, yp)
    sc = RESULTS_DIR / "selfcheck_nli_summary.json"
    if sc.exists():
        s = json.loads(sc.read_text(encoding="utf-8")).get("metrics", {})
        out["SelfCheck-NLI(전체셋 요약)"] = {"precision": s.get("precision"),
                                            "recall": s.get("recall"), "f1": s.get("f1")}
    return out


def main():
    ids, X, y, htype, rows = load_features()
    print(f"[로드] {len(y)}건 (환각 {int(y.sum())}/정상 {int((1-y).sum())}) | recall_floor={RECALL_FLOOR}\n")

    tr, te = stratified_split(y, TEST_FRACTION, SEED)
    model = AlcvModel(RECALL_FLOOR).fit(X[tr], y[tr])
    pred_te = model.predict(X[te])
    m_full = metrics(y[te], pred_te)
    cvm, cvs = cv_f1(X, y)

    sf = X[:, FEATURE_NAMES.index("n_hard_fail")]
    fu = X[:, FEATURE_NAMES.index("frac_soft_fail")]
    gate_only = metrics(y[te], (sf[te] > 0).astype(int))
    thr_fu = best_threshold(fu[tr], y[tr], RECALL_FLOOR)
    soft_only = metrics(y[te], (fu[te] > thr_fu).astype(int))

    table = [
        ("ALCV 전체(하드게이트+로지스틱)", m_full, (cvm, cvs)),
        ("  하드게이트 단독(결정적)", gate_only, None),
        ("  소프트 근거검증 단독", soft_only, None),
    ]
    print("=== ALCV 결과 (test 분할) ===")
    for name, m, cv in table:
        cvtxt = f" | CV F1={cv[0]:.3f}±{cv[1]:.3f}" if cv else ""
        print(f"{name:30s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} Acc={m['accuracy']:.3f}{cvtxt}")

    base = baselines_on([ids[i] for i in te], y[te])
    print("\n=== 베이스라인(동일 test 분할) ===")
    for k, v in base.items():
        print(f"{k:30s} P={v['precision']} R={v['recall']} F1={v['f1']}")

    by_type = defaultdict(lambda: [0, 0])
    for pos, i in enumerate(te):
        if y[i] == 1:
            by_type[htype[i]][1] += 1
            if pred_te[pos] == 1:
                by_type[htype[i]][0] += 1
    print("\n=== 환각 유형별 탐지율(ALCV 전체) ===")
    for t, (d, n) in by_type.items():
        print(f"  {t:26s} {d}/{n} = {d/n if n else 0:.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison = {
        "settings": {"recall_floor": RECALL_FLOOR, "test_fraction": TEST_FRACTION,
                     "seed": SEED, "n_folds": N_FOLDS, "n_total": len(y)},
        "alcv": {"full": m_full, "cv_f1_mean": cvm, "cv_f1_std": cvs,
                 "hard_gate_only": gate_only, "soft_only": soft_only,
                 "config": model.config()},
        "baselines_on_test": base,
        "detection_by_type": {t: {"detected": d, "total": n} for t, (d, n) in by_type.items()},
    }
    (RESULTS_DIR / "alcv_results.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "alcv_config.json").write_text(
        json.dumps(model.config(), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(m_full, cvm, cvs, gate_only, soft_only, base, by_type)
    _write_fig(m_full, base)
    print(f"\n[저장] {RESULTS_DIR/'alcv_results.md'} (논문 표용)")


def _write_md(full, cvm, cvs, gate, soft, base, by_type):
    L = ["# ALCV 환각 탐지 결과 (test 분할)", "",
         "| 방법 | Precision | Recall | F1 | Accuracy |",
         "|------|-----------|--------|----|----------|",
         f"| ALCV 전체 | {full['precision']:.3f} | {full['recall']:.3f} | {full['f1']:.3f} | {full['accuracy']:.3f} |",
         f"| 하드게이트 단독 | {gate['precision']:.3f} | {gate['recall']:.3f} | {gate['f1']:.3f} | {gate['accuracy']:.3f} |",
         f"| 소프트 근거검증 단독 | {soft['precision']:.3f} | {soft['recall']:.3f} | {soft['f1']:.3f} | {soft['accuracy']:.3f} |"]
    for k, v in base.items():
        def f(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "-"
        L.append(f"| {k} | {f(v['precision'])} | {f(v['recall'])} | {f(v['f1'])} | - |")
    L += ["", f"ALCV 전체 CV F1 = {cvm:.3f} +/- {cvs:.3f}", "", "## 환각 유형별 탐지율(ALCV)", "",
          "| 유형 | 탐지/전체 | 비율 |", "|------|-----------|------|"]
    for t, (d, n) in by_type.items():
        L.append(f"| {t} | {d}/{n} | {d/n if n else 0:.2f} |")
    (RESULTS_DIR / "alcv_results.md").write_text("\n".join(L), encoding="utf-8")


def _write_fig(full, base):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[알림] matplotlib 없음 - 그림 생략")
        return
    ascii_name = {"SelfCheck-NLI(전체셋 요약)": "SelfCheck-NLI"}
    names = ["ALCV"] + [ascii_name.get(k, k) for k in base.keys()]
    P = [full["precision"]] + [base[k]["precision"] or 0 for k in base]
    R = [full["recall"]] + [base[k]["recall"] or 0 for k in base]
    F = [full["f1"]] + [base[k]["f1"] or 0 for k in base]
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, P, w, label="Precision"); ax.bar(x, R, w, label="Recall"); ax.bar(x + w, F, w, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.05)
    ax.set_title("ALCV vs Baselines (legal hallucination detection)"); ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_alcv.png", dpi=150)
    print(f"[저장] {RESULTS_DIR/'fig_alcv.png'}")


if __name__ == "__main__":
    main()
