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
            action = hal.get("action") or "replace"
            wrong_word = hal.get("wrong_word")
            correct_word = hal.get("correct_word")
            start = hal.get("start")
            end = hal.get("end")
            if action == "log_only":
                continue
            if action == "remove_sentence":
                if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(final_answer):
                    sent_start, sent_end = self._sentence_bounds(final_answer, start, end)
                    if sent_start < sent_end:
                        span_replacements.append((sent_start, sent_end, ""))
                continue
            if action == "soften":
                if not wrong_word:
                    continue
                softened = self._soften_phrase(wrong_word)
                if not softened:
                    continue
                if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(final_answer):
                    if final_answer[start:end] != wrong_word:
                        continue
                    span_replacements.append((start, end, softened))
                    continue
                span_replacements.append((None, None, softened, wrong_word))
                continue
            if not wrong_word or not correct_word:
                continue
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if start < 0 or end > len(final_answer) or start >= end:
                continue
            if final_answer[start:end] != wrong_word:
                continue
            span_replacements.append((start, end, correct_word))

        explicit_spans = [item for item in span_replacements if len(item) == 3]
        soft_spans = [item for item in span_replacements if len(item) == 4]

        for start, end, correct_word in sorted(explicit_spans, key=lambda x: x[0], reverse=True):
            if start is None or end is None:
                continue
            if (start, end) in used_spans:
                continue
            final_answer = final_answer[:start] + correct_word + final_answer[end:]
            used_spans.add((start, end))

        for _, _, softened, wrong_word in soft_spans:
            boundary_pattern = rf"(?<![0-9A-Za-z가-힣]){re.escape(wrong_word)}(?![0-9A-Za-z가-힣])"
            final_answer = re.sub(boundary_pattern, softened, final_answer, count=1)

        for hal in high_hals:
            action = hal.get("action") or "replace"
            if action in {"remove_sentence", "soften", "log_only"}:
                continue
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

    def _sentence_bounds(self, text, start, end):
        left = text.rfind(".", 0, start)
        left_nl = text.rfind("\n", 0, start)
        left = max(left, left_nl)
        right_candidates = [text.find(sep, end) for sep in [".", "\n", "?", "!"]]
        right_candidates = [idx for idx in right_candidates if idx != -1]
        right = min(right_candidates) if right_candidates else len(text)
        if left == -1:
            left = 0
        else:
            left += 1
        if right < len(text):
            right += 1
        return left, right

    def _soften_phrase(self, word):
        if not word:
            return ""
        replacements = {
            # 기존
            "해당합니다": "해당할 가능성이 있습니다",
            "성립합니다": "성립할 가능성이 있습니다",
            "위반입니다": "문제가 될 가능성이 있습니다",
            "처벌됩니다": "처벌 대상이 될 가능성이 있습니다",
            # 법적 확정 표현
            "불법입니다": "법적 문제가 될 수 있습니다",
            "처벌받습니다": "처벌 대상이 될 수 있습니다",
            "인정됩니다": "인정될 수 있습니다",
            "적용됩니다": "적용될 수 있습니다",
            "법적 책임이 있습니다": "법적 책임이 있을 수 있습니다",
            "무효입니다": "무효가 될 수 있습니다",
            "취소됩니다": "취소될 수 있습니다",
            # 결론 단정 표현
            "가능합니다": "가능할 수 있습니다",
            "받을 수 있습니다": "받을 가능성이 있습니다",
            "청구할 수 있습니다": "청구를 검토해 볼 수 있습니다",
        }
        return replacements.get(word, "")
