# 법률 도메인 환각 탐지 — ALCV (결정적 검증)

답변에서 법률 개체를 추출하고, **규칙·법령 DB·수치비교만으로** 환각을 판정한다.
**판정에 LLM/NLI 가 전혀 개입하지 않으므로** "LLM이 LLM을 검사한다"는 순환 문제에서
자유롭고, 모든 판정이 사람이 검증 가능(auditable)하다.

> 배경: 기존 NER+NLI 신호는 거의 무작위(AUC 0.56/0.52)였고, entailment(NLI)도 두 번
> 연속 무력했다. 그래서 신호를 **결정적 규칙 검증**으로 교체했다.
> 설계 근거: `../환각탐지_방법론_제안_v2.md`

## 검증 규칙 (4개 환각 유형 ↔ 결정적 검사)

| 환각 유형 | 검사 | 강도 |
|---|---|---|
| article_number_error (제999조) | 법령 DB 조문 **존재** 조회 | 하드 |
| forbidden_law_injection (임금질문에 형법) | 사안↔법령 **도메인 정합성** | 하드 |
| semantic_error (3년→30년) | RAG 근거와 **수치/방향 정합성** | 하드 |
| contact_mismatch (틀린 기관) | 기관·날짜의 RAG 근거 **등장 여부** | 소프트 |

집계: 하드 위반 1개라도 → 환각(고정밀 게이트). 소프트(미근거)는 RAG 불완전성을 감안해
재현율 하한(0.90) 제약 로지스틱으로 보정.

## 폴더 구조
```
scripts_hallucination/
├─ alcv/
│  ├─ config.py        # 임계값·사안 키워드·위험가중치
│  ├─ dataset.py       # 평가셋 로더
│  ├─ law_index.py     # 법령 DB 조회 (조문 존재)
│  ├─ extract.py       # NER 개체 추출 (legal-ner-v3, LLM 미사용)
│  ├─ verify.py        # 유형별 규칙 검증 (근거=RAG)
│  └─ aggregate.py     # 하드게이트+로지스틱 판정
├─ run_build_law_index.py   # ① 법령 인덱스
├─ run_collect_features.py  # ② 특징 수집 (--limit N 으로 스모크)
├─ run_evaluate.py          # ③ 평가·비교
├─ baselines/               # MetaQA, SelfCheck
└─ results/
```

## 실행 순서
```powershell
venv\Scripts\python.exe scripts_hallucination\run_build_law_index.py
venv\Scripts\python.exe scripts_hallucination\run_collect_features.py --limit 50   # 스모크 먼저
venv\Scripts\python.exe scripts_hallucination\run_collect_features.py               # 전체
venv\Scripts\python.exe scripts_hallucination\run_evaluate.py
```

## 목표 / 평가 포인트
- 결정적이라 LLM 비용 없음(추출 NER 1회뿐). 
- **하드게이트의 정밀도(거의 1.0)** 와 유형별 탐지율(특히 조문번호·금지법령)을 본다.
- 기여는 "구조적·인용 환각의 고정밀·해석가능 탐지"(Path B). F1 1등이 아니어도
  인용 환각을 결정적으로 잡는다는 점이 법률 도메인 기여.

## 정직한 한계
RAG 근거에 의존하는 contact/semantic 유형은 검색 품질(정밀도 0.33~0.42)에 영향받아
재현율이 제한될 수 있다. 올바른 조문을 인용하며 추론만 틀린 환각은 범위 밖이다.
