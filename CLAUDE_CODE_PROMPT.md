# Claude Code 인수인계 프롬프트 — 법률 환각 탐지 (ALCV, 결정적 검증)

> 사용법: VSCode Claude Code 채팅에 이 파일을 붙여넣거나,
> "CLAUDE_CODE_PROMPT.md 읽고 현재 상태 파악한 뒤 다음 작업 진행해줘" 라고 지시.

---

## 0. 역할과 규칙
법률 도메인 LLM 답변의 **환각 탐지** 연구를 잇는다. 현재 방법론은 **ALCV(결정적 검증)**.
- 추측 금지. 불확실하면 가정을 명시하고 질문한다.
- **성능 수치를 지어내지 않는다.** 수치는 스크립트 실행 결과에서만 가져온다.
- 최소 수정·검증 가능한 목표 단위로 진행한다.
- **논문은 실험이 끝난 뒤** 작성한다(지금은 코드·실험 단계).

---

## 1. 핵심 원칙 (왜 이 설계인가)
- 기존 NER+NLI 신호는 거의 무작위(AUC 0.56 / 0.52)였고, entailment(NLI)도 두 번 무력했다.
- **LLM-judge는 쓰지 않는다.** "LLM이 만든 환각을 LLM이 판정"하는 순환을 피하기 위함.
- 그래서 **판정에 LLM/NLI가 전혀 개입하지 않는** 결정적 규칙 검증으로 교체했다.
  - 개체 추출만 NER(legal-ner-v3), **판정은 규칙·법령 DB·수치비교**.
  - 모든 판정이 사람이 검증 가능(auditable).
- 기여 방향은 **Path B**: F1 1등이 아니라 "구조적·인용 환각의 고정밀·해석가능 탐지".

---

## 2. 방법론 (4개 환각 유형 ↔ 결정적 검사)
| 환각 유형 | 검사 | 강도 |
|---|---|---|
| article_number_error (제999조) | 법령 DB 조문 **존재** 조회 | 하드 |
| forbidden_law_injection (임금질문에 형법) | 사안↔법령 **도메인 정합성** | 하드 |
| semantic_error (3년→30년) | RAG 근거와 **수치/방향 정합성** | 하드 |
| contact_mismatch (틀린 기관) | 기관·날짜의 RAG 근거 **등장 여부** | 소프트 |

집계: 하드 위반 1개라도 → 환각(고정밀 게이트). 소프트는 RAG 불완전성 감안해
재현율 하한(0.90) 제약 로지스틱으로 보정.

---

## 3. 폴더 지도 (`scripts_hallucination/`)
```
alcv/
  config.py        # 임계값·사안 키워드·위험가중치
  dataset.py       # 평가셋 로더
  law_index.py     # 법령 DB 조회(조문 존재) ★핵심
  extract.py       # NER 개체 추출 (LLM 미사용)
  verify.py        # 유형별 규칙 검증 (근거=RAG)
  aggregate.py     # 하드게이트+로지스틱 판정
run_build_law_index.py   # ① 법령 인덱스 (가벼움)
run_collect_features.py  # ② 특징 수집 (--limit N 스모크 지원)
run_evaluate.py          # ③ 평가·비교 (가벼움)
baselines/               # MetaQA, SelfCheck
results/                 # law_index.json, metaqa_results.json(유지) + 산출물
README.md
```

---

## 4. 현재 상태
| 항목 | 상태 |
|------|------|
| ALCV 결정적 검증 구현 | ✅ 완료 (전체 컴파일 OK) |
| 법령 인덱스 구축·검증 | ✅ (법령 50·조문 3,416 / 제43조=존재, 제999조=부재) |
| 4유형 규칙 단위 검증 | ✅ 실데이터로 정상/환각 정확 분류 확인 |
| **실제 특징 수집(②) 미실행** | ⛔ **여기부터 시작** |
| 평가(③) / 논문 | ⛔ 대기 |

> `results/alcv_features.json`, `alcv_results.*`, `alcv_config.json` 은 **플레이스홀더**.
> ②③ 실행 시 덮어써진다. `law_index.json`·`metaqa_results.json` 은 실제 파일.

---

## 5. 여기부터 시작 (Next Actions)

### TASK 0. 환경 점검 → 검증: 임포트
```powershell
venv\Scripts\python.exe -c "import torch, transformers; print('cuda', torch.cuda.is_available())"
```

### TASK 1. 법령 인덱스 (재실행 OK) → 검증: 스모크 3건 OK
```powershell
venv\Scripts\python.exe scripts_hallucination\run_build_law_index.py
```

### TASK 2. ★스모크 50건 먼저 → 검증: 하드게이트가 고정밀인가
```powershell
venv\Scripts\python.exe scripts_hallucination\run_collect_features.py --limit 50
```
- 출력 `results/alcv_features_smoke.json` 의 `verdicts`·`features` 를 눈으로 확인.
- 점검 포인트:
  1. 환각(특히 article_number_error)에서 `n_hard_fail > 0` 가 잡히는가?
  2. **정상 케이스에서 `n_hard_fail` 오탐이 거의 없는가?**(하드게이트 정밀도 핵심)
  3. NER 추출이 LAW의 법령명/조문을 제대로 잡는가? (못 잡으면 추출이 1차 병목)
- 어긋나면 **코드 구조는 두고** 다음만 조정:
  - 수치 허용오차 `config.NUMERIC_TOLERANCE`
  - 사안 키워드 `config.ISSUE_KEYWORDS` (질문→issue 추론용)
  - 검색 수 `config.RETRIEVE_TOP_K`

### TASK 3. 전체 특징 수집 → 검증: alcv_features.json 800건
```powershell
venv\Scripts\python.exe scripts_hallucination\run_collect_features.py
```

### TASK 4. 평가·비교 → 검증: 하드게이트 정밀도 + 유형별 탐지율
```powershell
venv\Scripts\python.exe scripts_hallucination\run_evaluate.py
```
- 출력 `results/alcv_results.md`. 핵심은 **하드게이트 단독 Precision(≈1.0 목표)** 과
  유형별 탐지율(조문번호·금지법령이 높아야 함).
- contact/semantic 재현율이 낮으면: verify.py 의 근거를 RAG 대신 케이스 `source_text`
  로 바꾸는 옵션을 추가해 비교(상한 측정용).

### TASK 5. (실험 성공 후) 논문
결과가 Path B 서사(고정밀·해석가능)를 뒷받침하면 그때 논문 작성. 그 전엔 작성 안 함.

---

## 6. 주의점
1. **판정에 LLM 금지.** 추출(NER)만 학습모델이고, verdict 는 규칙·DB·수치비교여야 한다.
   이 원칙이 방법론 정체성이므로 깨지 말 것.
2. **하드 게이트는 고정밀 전제.** 정상 오탐이 늘면 `law_index._resolve_law_key`(법령명 매칭)
   또는 numeric/도메인 규칙을 먼저 점검.
3. **추출 품질이 1차 병목.** NER이 LAW 법령명+조문, PENALTY/AMOUNT 수치를 못 잡으면
   그 케이스는 검증 자체가 안 된다. 스모크에서 꼭 확인.
4. baseline: MetaQA 는 `results/metaqa_results.json` 으로 동일분할 자동 재계산.
5. 큰 파일 저장 후 드물게 잘림 → 저장 즉시 `python -m py_compile` 로 구문 확인 권장.

---

## 7. 한 줄 요약
> **인덱스(done) → 스모크 50건으로 하드게이트 정밀도부터 확인 → 전체 수집 → 평가.
> 판정은 끝까지 LLM 없이 규칙으로. 기여는 정밀도·해석가능성(Path B). 논문은 실험 후.**
