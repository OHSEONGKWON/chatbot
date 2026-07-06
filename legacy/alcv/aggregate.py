"""개체 검증 결과 -> 답변 단위 특징 + 판정 모델.

설계:
  (1) 하드 게이트: 결정적 위반(조문부재/금지법령/수치모순)이 하나라도 있으면 즉시 환각
  (2) 그 외: 소프트 특징에 로지스틱 회귀 + 재현율 하한 제약 임계값
"""
import numpy as np

# 답변 단위 특징 순서
FEATURE_NAMES = [
    "n_entities",     # 추출된 개체 수
    "n_hard_fail",    # 하드 위반 수 (조문부재/금지법령/수치모순) -> 하드 게이트
    "n_soft_fail",    # 소프트 위반 수 (기관/날짜/금액 미근거)
    "frac_soft_fail", # 소프트 위반 비율
]

_HARD_CODES = {"nonexistent_article", "forbidden_law", "numeric_mismatch", "wrong_article_for_crime"}


def case_features(verdicts: list) -> dict:
    """per-entity verdict -> 답변 단위 특징 dict."""
    n = len(verdicts)
    if n == 0:
        return {"n_entities": 0, "n_hard_fail": 0, "n_soft_fail": 0, "frac_soft_fail": 0.0}
    n_hard = sum(1 for v in verdicts if v.get("verdict") == "hallu" and v.get("hard"))
    n_soft = sum(1 for v in verdicts if v.get("verdict") == "hallu" and not v.get("hard"))
    return {"n_entities": n, "n_hard_fail": n_hard, "n_soft_fail": n_soft, "frac_soft_fail": n_soft / n}


def features_to_matrix(feat_dicts: list) -> np.ndarray:
    return np.array([[d[k] for k in FEATURE_NAMES] for d in feat_dicts], dtype=float)


# ==================== 지표 / 임계값 ====================
def metrics(y_true, y_pred) -> dict:
    y_true = y_true.astype(bool); y_pred = y_pred.astype(bool)
    tp = int(np.sum(y_true & y_pred)); fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred)); tn = int(np.sum(~y_true & ~y_pred))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "accuracy": (tp + tn) / max(len(y_true), 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_threshold(scores, y, recall_floor):
    cands = np.concatenate([[-np.inf], np.unique(scores), [np.inf]])
    best, bf1, bthr = None, -1.0, 0.0
    for thr in cands:
        m = metrics(y, scores > thr)
        if m["f1"] > bf1:
            bf1, bthr = m["f1"], thr
        if m["recall"] >= recall_floor and (best is None or m["f1"] > best[0]):
            best = (m["f1"], thr)
    return best[1] if best else bthr


# ==================== 로지스틱 회귀 (numpy) ====================
class LogReg:
    def __init__(self, lr=0.1, epochs=3000, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        self.mu = X.mean(axis=0)
        self.sd = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        self.w = np.zeros(d); self.b = 0.0
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-(Xs @ self.w + self.b)))
            g = p - y
            self.w -= self.lr * (Xs.T @ g / n + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    def proba(self, X):
        Xs = (X - self.mu) / self.sd
        return 1 / (1 + np.exp(-(Xs @ self.w + self.b)))


class AlcvModel:
    """하드 게이트(결정적 위반) + 로지스틱(소프트 특징) 결합 판정기."""

    def __init__(self, recall_floor: float):
        self.recall_floor = recall_floor

    def fit(self, X, y):
        sf = X[:, FEATURE_NAMES.index("n_hard_fail")]
        gate = sf > 0
        rest = ~gate
        self.lr = LogReg().fit(X[rest], y[rest]) if rest.sum() > 5 else LogReg().fit(X, y)
        prob = self.lr.proba(X)
        score = np.where(gate, 1.0, prob)
        self.threshold = best_threshold(score, y, self.recall_floor)
        return self

    def decision_score(self, X):
        sf = X[:, FEATURE_NAMES.index("n_hard_fail")]
        gate = sf > 0
        return np.where(gate, 1.0, self.lr.proba(X))

    def predict(self, X):
        return (self.decision_score(X) > self.threshold).astype(int)

    def config(self) -> dict:
        return {"type": "ALCV (hard-gate + logistic, no-LLM)",
                "recall_floor": self.recall_floor,
                "coef": [round(float(c), 3) for c in self.lr.w],
                "intercept": round(float(self.lr.b), 3),
                "threshold": round(float(self.threshold), 4),
                "feature_order": FEATURE_NAMES}
