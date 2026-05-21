"""
환각 탐지 성능 평가 스크립트

NLI 단독, NER 단독, 3계층 앙상블, 외부 NER 모델을 비교합니다.

비교 시스템:
  1. NLI 단독: klue/roberta-large
  2. NER 단독: 우리 NER 모델 (LawsGuard-NER-v2)
  3. 3계층 앙상블: 현재 시스템 (NER + NLI + 법적추론)
  4. 외부 NER: KoELECTRA-naver 기반 (동일 방법론, 외부 모델 백엔드)
  5. 외부 NER: KoELECTRA-modu 기반
  6. 외부 NER: WikiNEural-Multilingual 기반

평가 메트릭: Accuracy, Precision, Recall, F1, AUC-ROC

사용법:
  python scripts/evaluate_hallucination.py
  python scripts/evaluate_hallucination.py --eval-data data/evaluation/hallu_eval.jsonl
  python scripts/evaluate_hallucination.py --systems ner_only ext_naver ext_modu ext_wiki
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EVAL_DATA_PATH = REPO_ROOT / "data" / "evaluation" / "hallu_eval.jsonl"
OUT_DIR = REPO_ROOT / "data" / "evaluation" / "results"

NER_MODEL_PATH = str(REPO_ROOT / "outputs" / "legal-ner-lawsguard-v2-30k")
NLI_MODEL_ID = "klue/roberta-large"
VECTARA_MODEL_ID = "vectara/hallucination_evaluation_model"

EXTERNAL_NER_MODELS = {
    "ext_naver": {
        "name": "KoELECTRA-naver (외부 NER)",
        "path": str(REPO_ROOT / "outputs" / "baselines" / "monologg_koelectra-base-v3-naver-ner"),
    },
    "ext_modu": {
        "name": "KoELECTRA-modu (외부 NER)",
        "path": str(REPO_ROOT / "outputs" / "baselines" / "Leo97_KoELECTRA-small-v3-modu-ner"),
    },
    "ext_wiki": {
        "name": "WikiNEural-Multilingual (외부 NER)",
        "path": str(REPO_ROOT / "outputs" / "baselines" / "Babelscape_wikineural-multilingual-ner"),
    },
}

# 외부 NER 레이블 → 우리 법률 엔티티 카테고리 매핑
_EXT_LABEL_MAP: dict[str, str] = {
    # babelscape/wikineural (B-PER/B-ORG/B-LOC/B-MISC)
    "B-ORG": "ORG",   "I-ORG": "ORG",
    "B-MISC": "LAW",  "I-MISC": "LAW",
    "B-LOC": "ORG",   "I-LOC": "ORG",
    # monologg/koelectra-base-v3-naver-ner (B-ORG/B-DAT/B-NUM/B-CVL/B-EVN/B-AFW)
    "B-DAT": "DATE",  "I-DAT": "DATE",
    "B-TIM": "DATE",  "I-TIM": "DATE",
    "B-NUM": "AMOUNT","I-NUM": "AMOUNT",
    "B-CVL": "LAW",   "I-CVL": "LAW",
    "B-EVN": "CRIME", "I-EVN": "CRIME",
    "B-AFW": "PENALTY","I-AFW": "PENALTY",
    # Leo97/KoELECTRA-small-v3-modu-ner (prefix 없는 형태)
    "ORG": "ORG",
    "DAT": "DATE",    "TIM": "DATE",
    "NUM": "AMOUNT",
    "CVL": "LAW",
    "EVN": "CRIME",
    "AFW": "PENALTY",
}


def load_eval_data(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ──────────────────────────────
# 1. NLI 단독 시스템
# ──────────────────────────────

def build_nli_detector(model_id: str = NLI_MODEL_ID):
    from transformers import pipeline as hf_pipeline

    device = 0 if torch.cuda.is_available() else -1
    print(f"  NLI 모델 로딩: {model_id}")
    nli = hf_pipeline("text-classification", model=model_id, device=device, truncation=True, max_length=512)

    def detect(record: dict) -> tuple[bool, float]:
        answer = record.get("answer", "")
        rag_docs = record.get("rag_docs", [])
        if not rag_docs or not answer:
            return False, 0.5

        scores = []
        for doc in rag_docs[:3]:
            premise = doc.get("text", "")[:400]
            hypothesis = answer[:200]
            try:
                result = nli(f"{premise} [SEP] {hypothesis}")
                # KLUE RoBERTa NLI: CONTRADICTION → 환각
                label = result[0]["label"]
                score = result[0]["score"]
                if "contradiction" in label.lower():
                    scores.append(score)
                elif "entailment" in label.lower():
                    scores.append(1 - score)
                else:
                    scores.append(0.5)
            except Exception:
                scores.append(0.5)

        avg_score = sum(scores) / len(scores) if scores else 0.5
        return avg_score > 0.5, avg_score

    return detect


# ──────────────────────────────
# 2. NER 단독 시스템
# ──────────────────────────────

def build_ner_detector(model_path: str = NER_MODEL_PATH):
    if not Path(model_path).exists():
        print(f"  [WARN] NER 모델 없음: {model_path}", file=sys.stderr)
        return None

    from src.modules.ner_checker import NERFactChecker
    from difflib import SequenceMatcher

    print(f"  NER 모델 로딩: {model_path}")
    checker = NERFactChecker(model_path=model_path)

    LEGAL_LABELS = {"LAW", "AMOUNT", "DATE", "ORG", "CRIME", "PENALTY"}

    def _fuzzy(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def detect(record: dict) -> tuple[bool, float]:
        answer = record.get("answer", "")
        rag_docs = record.get("rag_docs", [])
        if not answer or not rag_docs:
            return False, 0.0

        # 우리 NER 모델로 answer에서 법률 엔티티 추출
        ans_entities = checker.extract_entities(answer)
        if not ans_entities:
            return False, 0.0

        # RAG 문서에서도 동일 모델로 엔티티 추출 + 원문 텍스트 보관
        rag_text = " ".join(doc.get("text", "")[:400] for doc in rag_docs[:3])
        rag_entities = checker.extract_entities(rag_text)
        rag_by_label: dict[str, list[str]] = {}
        for ent in rag_entities:
            lbl = ent.get("entity_group") or ent.get("label", "")
            word = ent.get("word", "")
            if lbl and word:
                rag_by_label.setdefault(lbl, []).append(word)

        # 외부 NER 탐지기와 동일한 단순 매칭: 법률 엔티티가 RAG 문서에 없으면 불일치
        mismatch_count = 0
        for ent in ans_entities:
            lbl = ent.get("entity_group") or ent.get("label", "")
            word = ent.get("word", "")
            if not lbl or not word or lbl not in LEGAL_LABELS:
                continue
            rag_words = rag_by_label.get(lbl, [])
            supported = any(_fuzzy(word, rw) >= 0.85 for rw in rag_words) if rag_words else False
            if not supported and word in rag_text:
                supported = True
            if not supported:
                mismatch_count += 1

        score = min(1.0, mismatch_count * 0.35)
        return score > 0.3, score

    return detect


# ──────────────────────────────
# 3. 3계층 파이프라인 (현재 시스템)
# ──────────────────────────────

def build_pipeline_detector():
    try:
        from src.modules.ner_checker import NERFactChecker
        from src.modules.consistency_checker import ConsistencyChecker

        ner_checker = NERFactChecker()
        consistency_checker = ConsistencyChecker()

        def detect(record: dict) -> tuple[bool, float]:
            answer = record.get("answer", "")
            rag_docs = record.get("rag_docs", [])
            if not answer:
                return False, 0.0

            scores = []

            # Layer 1: NER
            try:
                mismatches = ner_checker.find_hallucinations(answer, rag_docs)
                ner_score = min(1.0, len(mismatches) * 0.4)
                scores.append(ner_score)
            except Exception:
                scores.append(0.0)

            # Layer 2: NLI 일관성 (ConsistencyChecker)
            try:
                is_consistent, cons_score = consistency_checker.check(answer, rag_docs)
                inconsistency_score = 1.0 - cons_score
                scores.append(inconsistency_score)
            except Exception:
                scores.append(0.0)

            avg = sum(scores) / len(scores) if scores else 0.0
            return avg > 0.4, avg

        return detect

    except Exception as e:
        print(f"  [WARN] 파이프라인 구성 실패: {e}", file=sys.stderr)
        return None


# ──────────────────────────────
# 4. 외부 NER 기반 탐지기 (공통 빌더)
# ──────────────────────────────

def build_external_ner_detector(model_path: str, display_name: str):
    """외부 NER 모델을 백엔드로 사용하는 환각 탐지기.

    우리 NERFactChecker와 동일한 방법론:
      1. 외부 NER 모델로 answer/RAG 문서에서 엔티티 추출
      2. _EXT_LABEL_MAP으로 법률 카테고리 변환
      3. answer 엔티티가 RAG 문서 엔티티에 없으면 → 환각 의심
    """
    if not Path(model_path).exists():
        print(f"  [WARN] 외부 NER 모델 없음: {model_path}", file=sys.stderr)
        return None

    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline as hf_pipeline
    from difflib import SequenceMatcher

    device = 0 if torch.cuda.is_available() else -1
    print(f"  외부 NER 모델 로딩: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        model = AutoModelForTokenClassification.from_pretrained(model_path, trust_remote_code=False)
        ner_pipe = hf_pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            aggregation_strategy="none",
        )
    except Exception as e:
        print(f"  [WARN] 외부 NER 로딩 실패: {e}", file=sys.stderr)
        return None

    def _extract(text: str) -> dict[str, list[str]]:
        """텍스트에서 카테고리별 엔티티 집합 반환."""
        entities: dict[str, list[str]] = {}
        if not text.strip():
            return entities
        try:
            outputs = ner_pipe(text[:512])
        except Exception:
            return entities
        current_label: str | None = None
        current_tokens: list[str] = []
        for tok in outputs:
            raw = tok.get("entity", "O")
            # prefix/label 분리
            if raw.startswith("B-") or raw.startswith("I-"):
                pfx, label = raw[:2], raw[2:]
            elif raw.endswith("-B") or raw.endswith("-I"):
                pfx, label = raw[-1] + "-", raw[:-2]
            else:
                pfx, label = "B-", raw
            mapped = _EXT_LABEL_MAP.get(raw) or _EXT_LABEL_MAP.get(f"B-{label}") or _EXT_LABEL_MAP.get(label)
            if mapped is None:
                if current_label and current_tokens:
                    entities.setdefault(current_label, []).append("".join(current_tokens).strip())
                current_label, current_tokens = None, []
                continue
            word = tok.get("word", "")
            if pfx == "B-" or current_label != mapped:
                if current_label and current_tokens:
                    entities.setdefault(current_label, []).append("".join(current_tokens).strip())
                current_label, current_tokens = mapped, [word.lstrip("##")]
            else:
                current_tokens.append(word.lstrip("##"))
        if current_label and current_tokens:
            entities.setdefault(current_label, []).append("".join(current_tokens).strip())
        return entities

    def _fuzzy(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def detect(record: dict) -> tuple[bool, float]:
        answer = record.get("answer", "")[:400]
        rag_docs = record.get("rag_docs", [])
        if not answer or not rag_docs:
            return False, 0.0

        ans_ents = _extract(answer)
        if not ans_ents:
            return False, 0.0

        # RAG 문서 전체 텍스트 합쳐서 엔티티 추출
        rag_text = " ".join(doc.get("text", "")[:300] for doc in rag_docs[:3])
        rag_ents = _extract(rag_text)

        mismatch_count = 0
        for label, words in ans_ents.items():
            rag_words = rag_ents.get(label, [])
            for word in words:
                if not word:
                    continue
                # RAG 문서에 같은 레이블로 충분히 유사한 엔티티가 있으면 지지됨
                supported = any(_fuzzy(word, rw) >= 0.85 for rw in rag_words) if rag_words else False
                # RAG 문서 전체 텍스트에 단어가 직접 포함되는지도 확인
                if not supported and word in rag_text:
                    supported = True
                if not supported:
                    mismatch_count += 1

        score = min(1.0, mismatch_count * 0.35)
        return score > 0.3, score

    return detect


# ──────────────────────────────
# 5. Vectara HEM
# ──────────────────────────────

def build_vectara_detector():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_id = VECTARA_MODEL_ID
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Vectara 모델 로딩: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
        model.eval()

        def detect(record: dict) -> tuple[bool, float]:
            answer = record.get("answer", "")
            rag_docs = record.get("rag_docs", [])
            if not rag_docs or not answer:
                return False, 0.5

            scores = []
            for doc in rag_docs[:2]:
                source = doc.get("text", "")[:512]
                try:
                    inputs = tokenizer(
                        source,
                        answer[:256],
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    # Vectara HEM: label 0 = hallucinated, 1 = factual
                    probs = torch.softmax(outputs.logits, dim=-1)
                    hallu_prob = probs[0][0].item()
                    scores.append(hallu_prob)
                except Exception:
                    scores.append(0.5)

            avg = sum(scores) / len(scores) if scores else 0.5
            return avg > 0.5, avg

        return detect

    except Exception as e:
        print(f"  [WARN] Vectara 모델 로딩 실패: {e}", file=sys.stderr)
        return None


# ──────────────────────────────
# 메트릭 계산
# ──────────────────────────────

def compute_metrics(labels: list[bool], preds: list[bool], scores: list[float]) -> dict:
    tp = fp = tn = fn = 0
    for gold, pred in zip(labels, preds):
        if gold and pred:
            tp += 1
        elif not gold and pred:
            fp += 1
        elif not gold and not pred:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0

    # AUC-ROC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc_roc": round(auc, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": len(labels),
    }


def print_table(all_results: dict[str, dict]) -> None:
    print("\n" + "=" * 80)
    print("환각 탐지 성능 평가 결과")
    print("=" * 80)
    header = f"{'시스템':<30} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'N':>5}"
    print(header)
    print("-" * 80)

    for sys_name, metrics in all_results.items():
        row = (
            f"{sys_name:<30}"
            f" {metrics.get('accuracy', 0.0):>6.3f}"
            f" {metrics.get('precision', 0.0):>6.3f}"
            f" {metrics.get('recall', 0.0):>6.3f}"
            f" {metrics.get('f1', 0.0):>6.3f}"
            f" {metrics.get('auc_roc', 0.0):>6.3f}"
            f" {metrics.get('support', 0):>5}"
        )
        print(row)

    print("=" * 80)


def save_csv(all_results: dict[str, dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["시스템", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "TP", "FP", "TN", "FN", "N"])
        for sys_name, metrics in all_results.items():
            writer.writerow([
                sys_name,
                metrics.get("accuracy", 0.0),
                metrics.get("precision", 0.0),
                metrics.get("recall", 0.0),
                metrics.get("f1", 0.0),
                metrics.get("auc_roc", 0.0),
                metrics.get("tp", 0),
                metrics.get("fp", 0),
                metrics.get("tn", 0),
                metrics.get("fn", 0),
                metrics.get("support", 0),
            ])
    print(f"CSV 저장: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="환각 탐지 성능 비교 평가")
    parser.add_argument("--eval-data", type=Path, default=EVAL_DATA_PATH)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["all"],
        help="평가할 시스템 (nli_only, ner_only, pipeline, vectara) 또는 all",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if not args.eval_data.exists():
        print(f"[ERROR] 평가 데이터 없음: {args.eval_data}", file=sys.stderr)
        print("먼저 generate_hallu_eval_data.py 를 실행하세요.", file=sys.stderr)
        sys.exit(1)

    records = load_eval_data(args.eval_data)
    labels = [rec["is_hallucination"] for rec in records]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"평가 데이터: {len(records)}개 (positive={n_pos}, negative={n_neg})")

    run_all = "all" in args.systems
    all_results = {}

    systems_to_run = {
        "nli_only": ("NLI 단독 (klue/roberta-large)", build_nli_detector),
        "ner_only": ("NER 단독 (우리 모델)", build_ner_detector),
        "pipeline": ("3계층 앙상블 (우리 시스템)", build_pipeline_detector),
        "vectara": ("Vectara HEM", build_vectara_detector),
        "ext_naver": (
            EXTERNAL_NER_MODELS["ext_naver"]["name"],
            lambda: build_external_ner_detector(
                EXTERNAL_NER_MODELS["ext_naver"]["path"],
                EXTERNAL_NER_MODELS["ext_naver"]["name"],
            ),
        ),
        "ext_modu": (
            EXTERNAL_NER_MODELS["ext_modu"]["name"],
            lambda: build_external_ner_detector(
                EXTERNAL_NER_MODELS["ext_modu"]["path"],
                EXTERNAL_NER_MODELS["ext_modu"]["name"],
            ),
        ),
        "ext_wiki": (
            EXTERNAL_NER_MODELS["ext_wiki"]["name"],
            lambda: build_external_ner_detector(
                EXTERNAL_NER_MODELS["ext_wiki"]["path"],
                EXTERNAL_NER_MODELS["ext_wiki"]["name"],
            ),
        ),
    }

    for sys_key, (sys_name, builder) in systems_to_run.items():
        if not run_all and sys_key not in args.systems:
            continue

        print(f"\n--- {sys_name} 평가 중 ---")
        try:
            detector = builder()
            if detector is None:
                print(f"  [SKIP] 빌드 실패")
                continue

            preds = []
            scores = []
            for i, rec in enumerate(records):
                if i % 50 == 0:
                    print(f"  진행: {i}/{len(records)}")
                is_hallu, score = detector(rec)
                preds.append(is_hallu)
                scores.append(score)

            metrics = compute_metrics(labels, preds, scores)
            all_results[sys_name] = metrics
            print(
                f"  Acc={metrics['accuracy']:.3f} P={metrics['precision']:.3f} "
                f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f} AUC={metrics['auc_roc']:.3f}"
            )

        except Exception as e:
            print(f"  [ERROR] {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    if not all_results:
        print("[ERROR] 평가된 시스템이 없습니다.", file=sys.stderr)
        sys.exit(1)

    print_table(all_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "hallucination_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 저장: {json_path}")

    save_csv(all_results, args.output_dir / "hallucination_table.csv")


if __name__ == "__main__":
    main()
