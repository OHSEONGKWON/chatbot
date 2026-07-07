# LawsGuard — 법률 도메인 LLM 환각 탐지 (방법론 v3)

한국어 법률 QA에서 LLM 답변의 환각을 탐지하는 연구 프로젝트.
답변을 원자적 법률 클레임으로 분해한 뒤, **3단 하이브리드 검증**(법령 DB 결정적 검증 →
로컬 NLI entailment → 불확실 구간만 LLM 근거-대조)으로 클레임 단위 판정을 집계한다.
상세 설계는 [`환각탐지_방법론_제안_v3.md`](환각탐지_방법론_제안_v3.md) 참고.

- 브랜치: `methodology-v3`
- 평가셋: 규칙 주입(v2) 폐기 → **실제 LLM 환각 수집**(Track A) + 외부 벤치마크(Track B: TriBench-Ko·K-HALU)
- 성공 기준: Track A에서 MetaQA·vanilla LLM judge를 PR-AUC 기준 상회

## 저장소 구조

```text
chatbot/
├─ pipeline/                    # v3 파이프라인 (단계별 스크립트)
│  ├─ llm_utils.py              # 비동기 LLM 호출 공용 유틸 (재시도·동시성·비용 집계)
│  └─ generate_questions.py     # 1단계: 코퍼스 기반 질문 생성
├─ legacy/                      # v2 재사용 코드 (ALCV 분해기, MetaQA·SelfCheck 베이스라인, 법령 인덱스)
├─ data/
│  ├─ real_data/New_Dataset/    # RAG 코퍼스 (법령 5,056 / 판례 28,711 / 매뉴얼 8,132 청크)
│  ├─ collection/               # 새 평가셋 산출물 (questions.jsonl 등)
│  └─ evaluation/hallu_eval.jsonl  # 기존 800건 (통제 실험용으로 강등)
├─ 환각탐지_방법론_제안_v3.md    # 방법론 설계 문서
└─ .env                         # OPENAI_API_KEY 등 (Git 추적 제외)
```

## 실행 방법

각 단계는 로컬에서 실행 (병렬 API 호출·증분 저장·중단 후 재개 지원):

```bat
cd C:\GitHub\chatbot
.venv\Scripts\python pipeline\generate_questions.py --smoke    :: 15건 스모크
.venv\Scripts\python pipeline\generate_questions.py --n 400    :: 본 생성
.venv\Scripts\python pipeline\generate_questions.py --dry-run  :: API 없이 샘플링 검증
```

## 개발 일정 (목표: 8월 초 논문 초고 완성)

| 기간 | 작업 | 완료 기준 |
|---|---|---|
| 7/6–7 | 1단계 질문 생성 (스모크 → 본 400건) | questions.jsonl 400건, 표본 검수 통과 |
| 7/7–9 | 2단계 답변 생성(closed-book·약한 RAG) + 클레임 분해 | 클레임 2~3천 개 |
| 7/9–11 | 3단계 L1 결정적 + L2 silver 라벨링 | 전 클레임 라벨 완료 |
| 7/11–13 | L3 수동 스팟체크 150~200건 (직접 작업) | agreement ≥ 0.85 |
| 7/8–10 병행 | Track B 확보 (TriBench-Ko·K-HALU) | 로드·형식 통일 |
| 7/10–15 | 3단 하이브리드 탐지기 구현 (Chroma 재구축 포함) | 기존 800건 sanity check 통과 |
| 7/15 | end-to-end 리허설 | 전 구간 소량 통과 |
| 7/16–21 | 여행 — 작업 없음 | |
| 7/22–24 | 베이스라인 4종 (SelfCheck·MetaQA·vanilla judge·RAG judge) | 결과 테이블 |
| 7/24–26 | 본 평가 (Track A+B) | 성공 기준 판정 |
| 7/27–8/2 | 분석 (ablation·유형별 분해·오류 분석) + 성능 개선 버퍼 | 논문용 표·그림 |
| 7/29–8/6 | 논문 작성 (분석과 병행) | 초고 완성 |
| 8/7–8 | 퇴고·그림 정리 | 최종본 |

**리스크**: ① 7/15 탐지기 완성이 핵심 마일스톤(밀리면 전체 1주 지연), ② 성능 미달 시
7/27–8/2 버퍼로 개선 반복, ③ TriBench-Ko 미공개 시 K-HALU 중심으로 대체.
