import re


class AnswerCorrector:
    def __init__(self):
        pass

    def fix_answer(self, llm_answer, hallucinations):
        final_answer = llm_answer

        # 분류: high(자동 교정), mid(권고), low(무시)
        high_conf_threshold = 0.95
        mid_conf_threshold = 0.80

        high_hals = []
        mid_hals = []

        for hal in hallucinations:
            conf = hal.get("confidence")
            try:
                conf = float(conf) if conf is not None else 0.0
            except Exception:
                conf = 0.0
            if conf >= high_conf_threshold:
                high_hals.append(hal)
            elif conf >= mid_conf_threshold:
                mid_hals.append(hal)
            else:
                # low confidence: ignore
                continue

        # 1) high confidence: span 기반 안전 치환(뒤에서부터)
        span_replacements = []
        for hal in high_hals:
            wrong_word = hal.get("wrong_word")
            correct_word = hal.get("correct_word")
            start = hal.get("start")
            end = hal.get("end")

            if not wrong_word or not correct_word:
                continue
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if start < 0 or end > len(final_answer) or start >= end:
                continue
            if final_answer[start:end] != wrong_word:
                continue
            span_replacements.append((start, end, correct_word))

        used_spans = set()
        for start, end, correct_word in sorted(span_replacements, key=lambda x: x[0], reverse=True):
            if (start, end) in used_spans:
                continue
            final_answer = final_answer[:start] + correct_word + final_answer[end:]
            used_spans.add((start, end))

        # 2) high confidence (span 정보 없는 경우): 경계 기반 1회 치환
        for hal in high_hals:
            wrong_word = hal.get("wrong_word")
            correct_word = hal.get("correct_word")
            start = hal.get("start")
            end = hal.get("end")
            if not wrong_word or not correct_word:
                continue
            if isinstance(start, int) and isinstance(end, int) and (start, end) in used_spans:
                continue
            boundary_pattern = rf"(?<![0-9A-Za-z가-힣]){re.escape(wrong_word)}(?![0-9A-Za-z가-힣])"
            updated, n = re.subn(boundary_pattern, correct_word, final_answer, count=1)
            if n == 0:
                updated = final_answer.replace(wrong_word, correct_word, 1)
            final_answer = updated

        # 3) mid confidence: 권고문 추가(치환하지 않음)
        if mid_hals:
            suggestions = []
            for i, hal in enumerate(mid_hals, 1):
                wrong = hal.get("wrong_word")
                cand = hal.get("correct_word")
                conf = float(hal.get("confidence") or 0.0)
                start = hal.get("start")
                end = hal.get("end")
                loc = f"[{start}:{end}]" if isinstance(start, int) and isinstance(end, int) else "[pos:?]"
                suggestions.append({
                    "index": i,
                    "wrong": wrong,
                    "suggestion": cand,
                    "confidence": round(conf, 3),
                    "span": loc,
                })

            # human-friendly block
            human_lines = ["\n\n[권고] 자동 치환 후보(중간 신뢰도) — 검토 후 수동 적용하세요:\n"]
            for s in suggestions:
                human_lines.append(f"{s['index']}. {s['wrong']} -> {s['suggestion']} (신뢰도 {s['confidence']:.2%}) {s['span']}")

            human_block = "\n".join(human_lines)
            # machine-friendly JSON block (separated) for easy parsing by downstream tools
            import json
            json_block = "\n\n[SUGGESTIONS_JSON]\n" + json.dumps(suggestions, ensure_ascii=False, indent=2)

            final_answer = final_answer + human_block + json_block

        return final_answer