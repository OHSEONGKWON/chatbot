# 인수인계: 법률 LLM 환각 탐지 연구 (methodology-v3)

> 이 문서는 Claude Code 등 AI 어시스턴트가 프로젝트를 이어받기 위한 컨텍스트다.
> 상세 설계는 `환각탐지_방법론_제안_v3.md`, 일정은 `README.md` 참고.

## 1. 프로젝트 목표

한국어 법률 QA에서 LLM 답변의 환각을 클레임 단위로 탐지하는 방법론(ALCV-v2, 3단 하이브리드)을
구현·평가하고 **8월 초까지 논문 초고**를 완성한다. 성공 기준: 자체 구축 평가셋(test 분할)에서
MetaQA·vanilla LLM judge 등 베이스라인을 상회.

v2의 치명적 문제(규칙 주입 평가셋 ↔ 규칙 역설계 탐지기의 순환성)를 해결하기 위해
**실제 LLM 환각을 수집**해 평가셋을 새로 만들었다 (규칙 주입 없음).

## 2. 핵심 설계 (합의 사항)

- 생성·라벨·탐지 3자 분리: 답변은 GPT-4o-mini 자유 생성(2조건: closed_book / weak_rag),
  **라벨러는 gold 소스 청크를 제공받고**, **탐지기는 gold 없이 검색부터** 수행 (정보 비대칭).
- 라벨: L1 결정적(법령 DB 조문 존재) → L2 silver(gold+BM25+조문 근거, 4o-mini 3표결) →
  수동 스팟체크는 생략하고 Claude(강모델)가 40건 독립 검수로 대체 (이진 일치 82.5~90%).
- 탐지기: Stage1 결정적(법령 DB) → Stage2 로컬 NLI(mDeBERTa) → Stage3 LLM 근거-대조(4o-mini).
- 라벨 정의 주의: NOT_SUPPORTED = "코퍼스 근거 기준 미지지·모순"이지 "객관적 거짓"이 아님.
  평가는 두 그룹핑 병행: (A) 환각=NOT_SUPPORTED, (B) 미지지=NOT_SUPPORTED+NEI.

## 3. 현재 상태 (2026-07-06 기준)

### 완료
- Track A 평가셋 완성: 질문 391 → 답변 782 → **클레임 7,935개** (전량 라벨 완료)
  - 라벨 분포: SUPPORTED 62% / NEI 29% / NOT_SUPPORTED 9.6% (761건, 자연 분포)
  - dev/test 분할: `hash(q_id) % 10 < 3` 이 dev (질문 단위, 클레임 dev 2,231 / test 5,704)
- 탐지기 구현·실행 완료. **현재 성능 (test)**: 3분류 68.3% |
  그룹핑A P=0.317 R=0.646 F1=0.425 | 그룹핑B P=0.665 R=0.683 F1=0.674
- 실증 발견 2개 (논문 재료):
  1. Stage2 NLI는 판별력 없음 (ent_max AUC 0.649) → 전량 Stage3로 라우팅, NLI는 ablation
  2. Stage3 프롬프트 개선(v1→v2: "근거에 없으면 NEI, 같은 대상을 다르게 규정할 때만 반박")으로
     F1 0.403→0.425. v1 결과는 `detect_final_v1.jsonl`에 보존
- SelfCheck용 재생성 샘플 2,346개 생성 완료

### 진행 중 (사용자가 로컬 실행)
- 베이스라인 4종 채점: vanilla / ragjudge / metaqa (`baselines_run.py`), selfcheck (`selfcheck_score.py`)

### 미착수
- 최종 비교표 산출 (test 분할, 그룹핑 A·B, 유형·조건별 분해) ← 베이스라인 완료 직후
- Track B 외부 벤치마크: TriBench-Ko (github.com/holi-lab/TriBench-Ko, 공개 여부 재확인 필요),
  K-HALU (보조). 우리 탐지기를 외부셋에 적용해 일반화 증명
- 기존 800건(`data/evaluation/hallu_eval.jsonl`) 통제 실험 (부록용)
- 답변 단위 점수 집계·보정, ablation 표, 논문 작성

## 4. 파일 지도

```
pipeline/                      # v3 파이프라인 (실행 순서대로)
  llm_utils.py                 # 비동기 4o-mini 호출 공용 (재시도·usage 집계). .env의 OPENAI_API_KEY 사용
  law_db.py                    # 법령 DB 조회 + 인용 추출 (legacy/law_index.json, 50개 법령 3,416조문)
  generate_questions.py        # 1단계: 코퍼스 → 질문 391 (gold 기록)
  generate_answers.py          # 2a: 답변 782 (closed_book/weak_rag) + BM25 구현(tokenize/build_bm25)
  extract_claims.py            # 2b: 클레임 7,935 (6타입 태깅)
  label_l1.py                  # 3-L1: 결정적 라벨 (조문 부재 35건 확정)
  label_l2.py                  # 3-L2: silver 라벨 (gold 제공 + 3표결)
  detect.py                    # 탐지기 S1+S2: DB검증 + BM25 top5 + NLI 점수 저장 (gold 미접근!)
  detect_stage3.py             # 탐지기 S3: LLM 근거-대조 → 최종 verdict
  generate_samples.py          # SelfCheck용 재생성 샘플
  baselines_run.py             # 베이스라인: vanilla/ragjudge/metaqa
  selfcheck_score.py           # 베이스라인: SelfCheck-NLI (로컬)

data/collection/               # 산출물 (전부 jsonl, c_id/a_id/q_id로 조인)
  questions.jsonl              # q_id, question, gold_chunk_id, gold_text, gold_metadata
  answers.jsonl                # a_id = {q_id}__{condition}, answer, rag_context
  claims.jsonl                 # c_id = {a_id}__cNN, claim_text, claim_type
  labels_l1/l2/final.jsonl     # 라벨 (final = L1+L2 병합; label ∈ SUPPORTED/NOT_SUPPORTED/NEI)
  spotcheck_reviewer.jsonl     # Claude 독립 검수 40건
  detect_scores.jsonl          # 탐지기 S1/S2 원점수 + top_evidence(BM25 top3)
  detect_final.jsonl           # 탐지기 최종 verdict (v2 프롬프트; 손상 라인 1개 → 파싱 시 무시)
  detect_final_v1.jsonl        # v1 프롬프트 결과 (ablation용)
  selfcheck_samples.jsonl      # 재생성 샘플 (a_id당 3개)
  baseline_*.jsonl             # 베이스라인 verdict/점수

data/real_data/New_Dataset/    # RAG 코퍼스: 법령 5,056 / 판례 28,711 / 매뉴얼 8,132 청크
legacy/                        # v2 재사용: law_index.json, alcv 분해기, MetaQA·SelfCheck 원본
data/evaluation/hallu_eval.jsonl  # 구 800건 (통제 실험 부록용으로만)
```

## 5. 실행 환경·주의사항

- 실행은 **사용자 로컬(Windows)**: `.venv\Scripts\python pipeline\<script>.py`.
  API 키는 `.env` (OPENAI_API_KEY). 샌드박스에서는 api.openai.com 차단.
- 모든 스크립트는 증분 저장 + 재개 지원 (출력 jsonl의 c_id/a_id 기준 skip). 중단 후 재실행 안전.
- 비용 감각: 지금까지 전부 합쳐 ~$10 미만. 전량 재실행해도 ~$15.
- NLI 모델: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (자동 다운로드).
- git: 브랜치 `methodology-v3`. 커밋은 사용자가 로컬에서 (과거 샌드박스 git 인덱스 호환 문제 있었음).
- 평가 시 test 분할만 보고: `int(hashlib.md5(q_id.encode()).hexdigest(),16)%10>=3`.

## 6. 알려진 한계 (논문 한계 절에 기재할 것)

1. 생성·라벨·판정 백본이 동일(GPT-4o-mini) — 정보 비대칭·3표결·독립 검수로 완화
2. silver 라벨 = 코퍼스 근거 기준 (객관적 참/거짓 아님) — 그룹핑 A·B 병행 보고
3. 질문 15~20% 다소 모호, L1 커버리지는 코퍼스 내 50개 법령 한정
4. 탐지기 정밀도의 주 병목 = BM25 검색이 gold 근거를 못 찾는 경우 (SUP→NOT 오판 413건, test)
   → 개선 여지: 하이브리드(dense) 검색, 인용 조문 주입 강화

## 7. 다음 할 일 (우선순위 순)

1. 베이스라인 4종 완료 확인 → `detect_final.jsonl` vs `baseline_*.jsonl` 비교표 (논문 Table 1)
2. selfcheck는 연속 점수 → PR-AUC 로 함께 비교 (verdict 계열은 F1 중심 + 점수화 옵션)
3. Track B (TriBench-Ko) 확보·적용
4. ablation: Stage1 유무 / NLI 게이트 / 프롬프트 v1 vs v2
5. 논문 초고 (방법·실험 섹션부터)
