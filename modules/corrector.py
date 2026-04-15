class AnswerCorrector:
    def __init__(self):
        pass

    def fix_answer(self, llm_answer, hallucinations):
        final_answer = llm_answer
        for hal in hallucinations:
            if hal.get('correct_word'):
                # 타겟 단어만 정밀하게 교체
                final_answer = final_answer.replace(hal['wrong_word'], hal['correct_word'])
        return final_answer