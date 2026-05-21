"""
NER 성능 평가 스크립트

우리 모델과 베이스라인 모델들을 동일한 평가 데이터로 비교합니다.

비교 모델:
  - 우리 모델: outputs/legal-ner-lawsguard-v2-30k/
  - 베이스라인: outputs/baselines/ 내 3개 모델
  - 추가 HuggingFace: klue/bert-base (NER fine-tuned), snunlp/KR-ELECTRA-discriminator

평가 데이터: data/evaluation/ner_eval.jsonl
  형식: {"chunk_id": ..., "tokens": [...], "ner_tags": [...]}

출력:
  - data/evaluation/results/ner_results.json  (상세 결과)
  - data/evaluation/results/ner_table.csv     (논문용 표)

사용법:
  python scripts/evaluate_ner.py
  python scripts/evaluate_ner.py --eval-data data/evaluation/ner_eval.jsonl
  python scripts/evaluate_ner.py --models our_model babelscape  # 특정 모델만
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EVAL_DATA_PATHS = [
    REPO_ROOT / "data" / "evaluation" / "ner_eval.jsonl",
    REPO_ROOT / "data" / "hallucination_data" / "law_hallu_bio.jsonl",
    REPO_ROOT / "data" / "hallucination_data" / "case_hallu_bio.jsonl",
]

OUT_DIR = REPO_ROOT / "data" / "evaluation" / "results"

# 모델 레지스트리
MODELS: dict[str, dict] = {
    "our_model": {
        "name": "LawsGuard-NER-v2 (ours)",
        "path": str(REPO_ROOT / "outputs" / "legal-ner-lawsguard-v2-30k"),
        "local": True,
        "tag_set": "legal",  # B-LAW, I-LAW, etc.
    },
    "babelscape": {
        "name": "WikiNEural-Multilingual",
        "path": str(REPO_ROOT / "outputs" / "baselines" / "Babelscape_wikineural-multilingual-ner"),
        "local": True,
        "tag_set": "general",
    },
    "koelectra_modu": {
        "name": "KoELECTRA-modu-NER",
        "path": str(REPO_ROOT / "outputs" / "baselines" / "Leo97_KoELECTRA-small-v3-modu-ner"),
        "local": True,
        "tag_set": "modu",
    },
    "koelectra_naver": {
        "name": "KoELECTRA-naver-NER",
        "path": str(REPO_ROOT / "outputs" / "baselines" / "monologg_koelectra-base-v3-naver-ner"),
        "local": True,
        "tag_set": "klue",
    },
    "klue_bert": {
        "name": "KLUE-BERT-NER",
        "path": "snunlp/KR-ELECTRA-discriminator",
        "local": False,
        "tag_set": "klue",
    },
}

# 우리 태그 → 범용 카테고리 매핑
LEGAL_TAG_MAP = {
    "LAW": "LAW",
    "ORG": "ORG",
    "DATE": "DATE",
    "AMOUNT": "AMOUNT",
    "CRIME": "CRIME",
    "PENALTY": "PENALTY",
}

# 외부 모델 태그 → 우리 카테고리 매핑
GENERAL_TAG_MAP = {
    # ── babelscape/wikineural-multilingual (B-PER/B-ORG/B-LOC/B-MISC) ──
    "B-ORG": "ORG",   "I-ORG": "ORG",
    "B-MISC": "LAW",  "I-MISC": "LAW",   # 법령명 = miscellaneous
    "B-LOC": "ORG",   "I-LOC": "ORG",    # 법원·기관 위치 기반 표기
    # ── monologg/koelectra-base-v3-naver-ner (PER/FLD/AFW/ORG/LOC/CVL/DAT/TIM/NUM/EVN) ──
    "B-ORG": "ORG",   "I-ORG": "ORG",
    "B-DAT": "DATE",  "I-DAT": "DATE",
    "B-TIM": "DATE",  "I-TIM": "DATE",
    "B-NUM": "AMOUNT","I-NUM": "AMOUNT",
    "B-CVL": "LAW",   "I-CVL": "LAW",    # 문화·제도 → 법령
    "B-EVN": "CRIME", "I-EVN": "CRIME",  # 사건·사고 → 범죄
    "B-AFW": "PENALTY","I-AFW": "PENALTY", # 인공물·처분 → 형벌
    # ── Leo97/KoELECTRA-small-v3-modu-ner (prefix 없는 형태) ──
    "ORG": "ORG",
    "DAT": "DATE",  "TIM": "DATE",
    "NUM": "AMOUNT",
    "CVL": "LAW",
    "EVN": "CRIME",
    "AFW": "PENALTY",
    # ── KLUE NER 형식 (OG/LC/PS/DT/TI/QT/FD/TR/AF/CV/AM/PT/MT/TM) ──
    "B-OG": "ORG",  "I-OG": "ORG",
    "B-LC": "ORG",  "I-LC": "ORG",    # 법원 등 기관명이 지명 태그로 나오는 경우
    "B-DT": "DATE", "I-DT": "DATE",
    "B-TI": "DATE", "I-TI": "DATE",
    "B-QT": "AMOUNT","I-QT": "AMOUNT",
    "B-CV": "LAW",  "I-CV": "LAW",    # 문화·제도 → 법령
    "B-TR": "LAW",  "I-TR": "LAW",    # 이론·제도
    "B-AF": "PENALTY","I-AF": "PENALTY", # 인공물·처분
    # ── 접두사 없는 레이블 (일부 모델 출력 형식) ──
    "ORG-B": "ORG", "ORG-I": "ORG",
    "DAT-B": "DATE","DAT-I": "DATE",
    "TIM-B": "DATE","TIM-I": "DATE",
    "NUM-B": "AMOUNT","NUM-I": "AMOUNT",
}

EVAL_ENTITY_TYPES = ["LAW", "ORG", "DATE", "AMOUNT", "CRIME", "PENALTY"]


def load_eval_data(path: Path, max_records: int = 2000) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if "tokens" in rec and "ner_tags" in rec:
                    records.append(rec)
                if len(records) >= max_records:
                    break
    return records


def run_inference(model_key: str, model_info: dict, records: list[dict]) -> list[list[str]]:
    """모델 추론 실행 → 예측 태그 리스트 반환."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    model_path = model_info["path"]
    print(f"  모델 로딩: {model_path}")

    device = 0 if torch.cuda.is_available() else -1

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        model = AutoModelForTokenClassification.from_pretrained(model_path, trust_remote_code=False)
        ner_pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            aggregation_strategy="none",  # 토큰 레벨 출력
        )
    except Exception as e:
        print(f"  [WARN] 모델 로딩 실패: {e}", file=sys.stderr)
        return [["O"] * len(rec["tokens"]) for rec in records]

    all_preds = []
    batch_size = 16

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        for rec in batch:
            text = "".join(rec["tokens"])
            try:
                outputs = ner_pipeline(text)
                # 토큰 레벨 예측을 문자 레벨로 변환
                preds = _align_char_predictions(text, outputs, model_info["tag_set"])
            except Exception:
                preds = ["O"] * len(rec["tokens"])
            all_preds.append(preds)

    return all_preds


def _align_char_predictions(text: str, ner_outputs: list[dict], tag_set: str) -> list[str]:
    """HuggingFace pipeline 출력을 문자 레벨 BIO 태그로 변환."""
    preds = ["O"] * len(text)

    for ent in ner_outputs:
        raw_label = ent.get("entity", "O")
        start = ent.get("start", 0)
        end = ent.get("end", 0)

        # 카테고리 정규화
        if raw_label.startswith("B-") or raw_label.startswith("I-"):
            prefix, label = raw_label[:2], raw_label[2:]
        elif raw_label.endswith("-B") or raw_label.endswith("-I"):
            prefix, label = raw_label[-1] + "-", raw_label[:-2]
        else:
            prefix, label = "B-", raw_label

        # 태그 매핑
        mapped = LEGAL_TAG_MAP.get(label) or GENERAL_TAG_MAP.get(raw_label) or GENERAL_TAG_MAP.get(label)
        if mapped is None:
            continue

        for i in range(start, min(end, len(text))):
            if i == start:
                preds[i] = f"B-{mapped}"
            else:
                preds[i] = f"I-{mapped}"

    return preds


def extract_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    """BIO 태그에서 (start, end, label) span 집합 추출."""
    spans = set()
    i = 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            label = tags[i][2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{label}":
                j += 1
            spans.add((i, j, label))
            i = j
        else:
            i += 1
    return spans


def compute_metrics(gold_list: list[list[str]], pred_list: list[list[str]]) -> dict:
    """전체 및 엔티티별 Precision/Recall/F1 계산."""
    tp_all = fp_all = fn_all = 0
    per_type: dict[str, dict] = {t: {"tp": 0, "fp": 0, "fn": 0} for t in EVAL_ENTITY_TYPES}

    for gold_tags, pred_tags in zip(gold_list, pred_list):
        # 길이 맞추기
        min_len = min(len(gold_tags), len(pred_tags))
        gold_spans = extract_spans(gold_tags[:min_len])
        pred_spans = extract_spans(pred_tags[:min_len])

        # 평가 대상 엔티티 필터
        gold_spans = {s for s in gold_spans if s[2] in EVAL_ENTITY_TYPES}
        pred_spans = {s for s in pred_spans if s[2] in EVAL_ENTITY_TYPES}

        matched = gold_spans & pred_spans
        tp = len(matched)
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)

        tp_all += tp
        fp_all += fp
        fn_all += fn

        for span in matched:
            per_type[span[2]]["tp"] += 1
        for span in pred_spans - gold_spans:
            per_type[span[2]]["fp"] += 1
        for span in gold_spans - pred_spans:
            per_type[span[2]]["fn"] += 1

    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    overall_p, overall_r, overall_f = _prf(tp_all, fp_all, fn_all)

    result = {
        "overall": {
            "precision": overall_p,
            "recall": overall_r,
            "f1": overall_f,
            "tp": tp_all,
            "fp": fp_all,
            "fn": fn_all,
        },
        "per_entity": {},
    }

    for etype, counts in per_type.items():
        p, r, f = _prf(counts["tp"], counts["fp"], counts["fn"])
        result["per_entity"][etype] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "support": counts["tp"] + counts["fn"],
        }

    # macro F1
    macro_f = sum(result["per_entity"][t]["f1"] for t in EVAL_ENTITY_TYPES) / len(EVAL_ENTITY_TYPES)
    result["macro_f1"] = round(macro_f, 4)

    return result


def print_table(all_results: dict[str, dict]) -> None:
    """논문용 표 형식으로 출력."""
    print("\n" + "=" * 90)
    print("NER 성능 평가 결과 (Entity-level Span F1)")
    print("=" * 90)

    header = f"{'모델':<30} {'LAW':>6} {'ORG':>6} {'DATE':>6} {'AMOUNT':>6} {'CRIME':>6} {'PENALTY':>6} {'Macro':>7} {'Overall':>8}"
    print(header)
    print("-" * 90)

    for model_key, result in all_results.items():
        name = MODELS.get(model_key, {}).get("name", model_key)
        pe = result.get("per_entity", {})
        row = (
            f"{name:<30}"
            f" {pe.get('LAW', {}).get('f1', 0.0):>6.3f}"
            f" {pe.get('ORG', {}).get('f1', 0.0):>6.3f}"
            f" {pe.get('DATE', {}).get('f1', 0.0):>6.3f}"
            f" {pe.get('AMOUNT', {}).get('f1', 0.0):>6.3f}"
            f" {pe.get('CRIME', {}).get('f1', 0.0):>6.3f}"
            f" {pe.get('PENALTY', {}).get('f1', 0.0):>6.3f}"
            f" {result.get('macro_f1', 0.0):>7.3f}"
            f" {result.get('overall', {}).get('f1', 0.0):>8.3f}"
        )
        print(row)

    print("=" * 90)


def save_csv(all_results: dict[str, dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = ["모델", "LAW_F1", "ORG_F1", "DATE_F1", "AMOUNT_F1", "CRIME_F1", "PENALTY_F1",
                  "Macro_F1", "Overall_P", "Overall_R", "Overall_F1", "TP", "FP", "FN"]
        writer.writerow(header)

        for model_key, result in all_results.items():
            name = MODELS.get(model_key, {}).get("name", model_key)
            pe = result.get("per_entity", {})
            ov = result.get("overall", {})
            row = [
                name,
                pe.get("LAW", {}).get("f1", 0.0),
                pe.get("ORG", {}).get("f1", 0.0),
                pe.get("DATE", {}).get("f1", 0.0),
                pe.get("AMOUNT", {}).get("f1", 0.0),
                pe.get("CRIME", {}).get("f1", 0.0),
                pe.get("PENALTY", {}).get("f1", 0.0),
                result.get("macro_f1", 0.0),
                ov.get("precision", 0.0),
                ov.get("recall", 0.0),
                ov.get("f1", 0.0),
                ov.get("tp", 0),
                ov.get("fp", 0),
                ov.get("fn", 0),
            ]
            writer.writerow(row)

    print(f"CSV 저장: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NER 모델 성능 비교 평가")
    parser.add_argument("--eval-data", type=Path, default=None, help="평가 데이터 JSONL 경로")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="평가할 모델 키 (기본: all)",
    )
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    # 평가 데이터 로드
    eval_path = args.eval_data
    if eval_path is None:
        for p in EVAL_DATA_PATHS:
            if p.exists():
                eval_path = p
                break
    if eval_path is None or not eval_path.exists():
        print(f"[ERROR] 평가 데이터 없음. 먼저 generate_ner_eval_data.py 실행하세요.", file=sys.stderr)
        sys.exit(1)

    print(f"평가 데이터: {eval_path}")
    records = load_eval_data(eval_path, args.max_records)
    print(f"로드된 레코드: {len(records)}개")

    gold_tags = [rec["ner_tags"] for rec in records]

    # 평가할 모델 선택
    model_keys = list(MODELS.keys()) if "all" in args.models else args.models

    all_results = {}
    for key in model_keys:
        if key not in MODELS:
            print(f"[WARN] 알 수 없는 모델: {key}", file=sys.stderr)
            continue

        print(f"\n--- {MODELS[key]['name']} 평가 중 ---")
        model_path = MODELS[key]["path"]

        # local 모델 경로 존재 확인
        if MODELS[key]["local"] and not Path(model_path).exists():
            print(f"  [SKIP] 모델 경로 없음: {model_path}")
            continue

        try:
            pred_tags = run_inference(key, MODELS[key], records)
            metrics = compute_metrics(gold_tags, pred_tags)
            all_results[key] = metrics
            ov = metrics["overall"]
            print(f"  P={ov['precision']:.3f} R={ov['recall']:.3f} F1={ov['f1']:.3f} (macro={metrics['macro_f1']:.3f})")
        except Exception as e:
            print(f"  [ERROR] 평가 실패: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    if not all_results:
        print("[ERROR] 평가 완료된 모델이 없습니다.", file=sys.stderr)
        sys.exit(1)

    # 결과 출력 및 저장
    print_table(all_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "ner_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 저장: {json_path}")

    save_csv(all_results, args.output_dir / "ner_table.csv")


if __name__ == "__main__":
    main()
