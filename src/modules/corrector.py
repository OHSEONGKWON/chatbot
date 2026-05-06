import re


class AnswerCorrector:
    def fix_answer(self, llm_answer, hallucinations):
        final_answer = llm_answer
        high_hals = []
        mid_hals = []

        for hal in hallucinations:
            try:
                confidence = float(hal.get("confidence") or 0.0)
            except Exception:
                confidence = 0.0
            if confidence >= 0.95:
                high_hals.append(hal)
            elif confidence >= 0.80:
                mid_hals.append(hal)

        used_spans = set()
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

        for start, end, correct_word in sorted(span_replacements, key=lambda x: x[0], reverse=True):
            if (start, end) in used_spans:
                continue
            final_answer = final_answer[:start] + correct_word + final_answer[end:]
            used_spans.add((start, end))

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
            final_answer = updated if n else final_answer.replace(wrong_word, correct_word, 1)

        # Mid-confidence findings are useful for logs/evaluation, but should not be
        # appended to user-facing legal advice.

        return final_answer
