from modules.corrector import AnswerCorrector
s = "피고인은 형법 제298조에 따라 처벌을 받는다. 대법원 판결을 인용한다."
halls = [
    {"wrong_word":"형법 제298조","correct_word":"형법","start":4,"end":12,"confidence":0.98},
    {"wrong_word":"대법원","correct_word":"대법원(정식)","start":32,"end":35,"confidence":0.85},
    {"wrong_word":"존재안함","correct_word":"무효","start":100,"end":106,"confidence":0.4},
]
ac = AnswerCorrector()
print(ac.fix_answer(s,halls))
