import re


class AnswerCorrector:
    def __init__(self):
        pass

    def fix_answer(self, llm_answer, hallucinations):
        final_answer = llm_answer

        # 1) NER span이 신뢰 가능한 경우, 해당 구간만 뒤에서부터 치환해 인덱스 꼬임 방지
        span_replacements = []
        for hal in hallucinations:
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

            # span 불일치 시(토크나이저 후처리 차이)에는 fallback 단계에서 처리
            if final_answer[start:end] != wrong_word:
                continue

            span_replacements.append((start, end, correct_word))

        used_spans = set()
        for start, end, correct_word in sorted(span_replacements, key=lambda x: x[0], reverse=True):
            if (start, end) in used_spans:
                continue
            final_answer = final_answer[:start] + correct_word + final_answer[end:]
            used_spans.add((start, end))

        # 2) span 정보가 없거나 매칭 실패한 항목은 경계 조건 기반으로 1회만 치환
        for hal in hallucinations:
            wrong_word = hal.get("wrong_word")
            correct_word = hal.get("correct_word")

            if not wrong_word or not correct_word:
                continue

            start = hal.get("start")
            end = hal.get("end")
            if isinstance(start, int) and isinstance(end, int) and (start, end) in used_spans:
                continue

            boundary_pattern = rf"(?<![0-9A-Za-z가-힣]){re.escape(wrong_word)}(?![0-9A-Za-z가-힣])"
            updated, n = re.subn(boundary_pattern, correct_word, final_answer, count=1)
            if n == 0:
                updated = final_answer.replace(wrong_word, correct_word, 1)
            final_answer = updated

        return final_answer