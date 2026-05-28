"""LawsGuard 전체 시스템 아키텍처 이미지 생성 (오류 수정본)."""

import matplotlib
matplotlib.rcParams['font.family'] = ['Malgun Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── 색상 ──────────────────────────────────────────────────────────────────────
BG_L1       = '#EAF4FB'   # 레이어1 배경 (연파랑)
BG_L2       = '#E8F5E9'   # 레이어2 배경 (연초록)
BG_L3       = '#FFF8E1'   # 레이어3 배경 (연노랑)
BG_SERVICE  = '#F3E5F5'   # 서비스 패널 (연보라)

BOX_BLUE    = '#BBDEFB'   # 일반 박스
BOX_GREEN   = '#C8E6C9'   # 검색 박스
BOX_YELLOW  = '#FFF9C4'   # 강조 박스 (카카오)
BOX_ORANGE  = '#FFE0B2'   # 재질문/대체 박스 (조기종료)
BOX_PURPLE  = '#E1BEE7'   # 외부 API
BOX_DB      = '#B3E5FC'   # DB

LINE_BLUE   = '#1565C0'
LINE_GREEN  = '#2E7D32'
LINE_ORANGE = '#E65100'
LINE_PURPLE = '#6A1B9A'
LINE_GRAY   = '#546E7A'
LINE_YELLOW = '#F9A825'

TXT         = '#1A237E'
TXT_GRAY    = '#37474F'
TXT_ORANGE  = '#BF360C'

# ── 캔버스 ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def rect(x, y, w, h, fc, ec, lw=1.2, ls='-', alpha=1.0, radius=0.15):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.0,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw,
                         linestyle=ls, alpha=alpha)
    ax.add_patch(box)

def txt(x, y, s, size=8.5, color=TXT, ha='center', va='center',
        bold=False, wrap=False):
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight=weight, wrap=wrap,
            multialignment='center')

def arrow(x0, y0, x1, y1, color=LINE_GRAY, lw=1.4, arrowsize=8):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=arrowsize))

def dashed_line(x0, y0, x1, y1, color=LINE_GRAY, lw=1.2):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw,
            linestyle='--', alpha=0.7)

# ─────────────────────────────────────────────────────────────────────────────
# 제목
# ─────────────────────────────────────────────────────────────────────────────
txt(11, 13.6, 'LawsGuard 전체 시스템 파이프라인',
    size=14, bold=True, color='#0D1B2A')

# ─────────────────────────────────────────────────────────────────────────────
# 3개 레이어 배경
# ─────────────────────────────────────────────────────────────────────────────
LAYER_X, LAYER_W = 0.2, 14.1

# Layer 1
rect(LAYER_X, 9.3, LAYER_W, 3.9, BG_L1, LINE_BLUE, lw=1.2, alpha=0.6, radius=0.3)
txt(LAYER_X+0.25, 12.9, '1. 질문 분석 레이어', size=9.5, bold=True,
    color=LINE_BLUE, ha='left')

# Layer 2
rect(LAYER_X, 4.5, LAYER_W, 4.6, BG_L2, LINE_GREEN, lw=1.2, alpha=0.6, radius=0.3)
txt(LAYER_X+0.25, 8.8, '2. 검색 & 생성 레이어', size=9.5, bold=True,
    color=LINE_GREEN, ha='left')

# Layer 3
rect(LAYER_X, 0.3, LAYER_W, 4.0, BG_L3, '#F9A825', lw=1.2, alpha=0.6, radius=0.3)
txt(LAYER_X+0.25, 4.05, '3. 검증 레이어', size=9.5, bold=True,
    color='#E65100', ha='left')

# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 – 질문 분석 레이어
# ─────────────────────────────────────────────────────────────────────────────
# 박스 정의: (x_center, y_center, w, h, fc, ec, label_main, label_sub)
L1_Y = 11.35
L1_H = 1.55

# ① 카카오톡 질문
bx1 = 1.3
rect(bx1-0.9, L1_Y-L1_H/2, 1.8, L1_H, BOX_YELLOW, LINE_YELLOW, lw=1.5, radius=0.2)
txt(bx1, L1_Y+0.22, '카카오톡 질문', size=9, bold=True, color='#5D4037')
txt(bx1, L1_Y-0.22, '사용자 메시지 입력', size=7.5, color=TXT_GRAY)

# ② 명확화 모듈
bx2 = 4.1
rect(bx2-0.95, L1_Y-L1_H/2, 1.9, L1_H, BOX_BLUE, LINE_BLUE, lw=1.5, radius=0.2)
txt(bx2, L1_Y+0.22, '명확화 모듈', size=9, bold=True, color=LINE_BLUE)
txt(bx2, L1_Y-0.22, '법률 카테고리 분류', size=7.5, color=TXT_GRAY)

# ③ CaseFrame 생성
bx3 = 7.2
rect(bx3-1.1, L1_Y-L1_H/2, 2.2, L1_H, BOX_BLUE, LINE_BLUE, lw=1.5, radius=0.2)
txt(bx3, L1_Y+0.35, 'CaseFrame 생성', size=9, bold=True, color=LINE_BLUE)
txt(bx3, L1_Y+0.05, 'IssuePlan / AnswerContract', size=7.2, color=TXT_GRAY)
txt(bx3, L1_Y-0.28, '질문 구조 분석', size=7.5, color=TXT_GRAY)

# 화살표 Layer 1
arrow(bx1+0.9, L1_Y, bx2-0.95, L1_Y, color=LINE_BLUE)
arrow(bx2+0.95, L1_Y, bx3-1.1, L1_Y, color=LINE_BLUE)

# "정보 부족" 분기 → 재질문 반환
REQUERY1_X, REQUERY1_Y = 6.2, 9.9
rect(REQUERY1_X-1.05, REQUERY1_Y-0.55, 2.1, 1.1, BOX_ORANGE, LINE_ORANGE, lw=1.8, radius=0.2)
txt(REQUERY1_X, REQUERY1_Y+0.15, '재질문 반환', size=9, bold=True, color=TXT_ORANGE)
txt(REQUERY1_X, REQUERY1_Y-0.2, '사용자 응답 대기', size=7.5, color=TXT_ORANGE)

# CaseFrame → 재질문 (아래로)
ax.annotate('', xy=(REQUERY1_X, REQUERY1_Y+0.55), xytext=(bx3, L1_Y-L1_H/2),
            arrowprops=dict(arrowstyle='->', color=LINE_ORANGE, lw=1.4, mutation_scale=8))
txt(bx3+0.55, 10.3, '정보 부족', size=7.5, color=TXT_ORANGE, ha='left')

# CaseFrame → 다음 레이어 (Layer 2로)
arrow(bx3+1.1, L1_Y, 8.8, L1_Y, color=LINE_BLUE)
arrow(8.8, L1_Y, 8.8, 8.35, color=LINE_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 – 검색 & 생성 레이어
# ─────────────────────────────────────────────────────────────────────────────
L2_Y = 7.1
L2_H = 1.55

# ① RAG 검색 (수정: BM25 + ChromaDB + RRF)
bx_rag = 1.6
rect(bx_rag-1.2, L2_Y-L2_H/2, 2.4, L2_H, BOX_GREEN, LINE_GREEN, lw=1.5, radius=0.2)
txt(bx_rag, L2_Y+0.38, 'RAG 검색', size=9, bold=True, color=LINE_GREEN)
txt(bx_rag, L2_Y+0.08, 'BM25 + ChromaDB', size=7.2, color=TXT_GRAY)
txt(bx_rag, L2_Y-0.2, '벡터 검색 + RRF(k=60)', size=7.2, color=TXT_GRAY)

# ② 법률 문서 DB
bx_db2 = 4.6
# DB 원통 모양 대신 둥근 박스
rect(bx_db2-0.85, L2_Y-L2_H/2, 1.7, L2_H, BOX_DB, '#0277BD', lw=1.5, radius=0.2)
txt(bx_db2, L2_Y+0.22, '법률 문서 DB', size=9, bold=True, color='#01579B')
txt(bx_db2, L2_Y-0.22, '41,899개 문서', size=7.5, color=TXT_GRAY)
# DB 아이콘 표시
ax.text(bx_db2, L2_Y+0.6, '[DB]', fontsize=8, ha='center', va='center')

# ③ 일관성 검사
bx_cons = 7.55
rect(bx_cons-1.05, L2_Y-L2_H/2, 2.1, L2_H, BOX_BLUE, LINE_BLUE, lw=1.5, radius=0.2)
txt(bx_cons, L2_Y+0.22, '일관성 검사', size=9, bold=True, color=LINE_BLUE)
txt(bx_cons, L2_Y-0.22, '유사질문 × 10개 병렬', size=7.5, color=TXT_GRAY)

# ④ GPT-4o-mini (외부 점선 박스)
bx_gpt = 10.8
rect(bx_gpt-1.3, L2_Y-L2_H/2, 2.6, L2_H, BOX_PURPLE, LINE_PURPLE,
     lw=1.8, ls='--', radius=0.2)
txt(bx_gpt, L2_Y+0.35, 'GPT-4o-mini', size=9, bold=True, color=LINE_PURPLE)
txt(bx_gpt, L2_Y+0.05, '☁ 외부 Cloud API', size=7.5, color=LINE_PURPLE)
txt(bx_gpt, L2_Y-0.25, '답변 생성 및 일관성 검증', size=7.2, color=TXT_GRAY)

# 화살표 Layer 2
arrow(bx_rag+1.2, L2_Y, bx_db2-0.85, L2_Y, color=LINE_GREEN)
arrow(bx_db2+0.85, L2_Y, bx_cons-1.05, L2_Y, color=LINE_GREEN)
arrow(bx_cons+1.05, L2_Y, bx_gpt-1.3, L2_Y, color=LINE_BLUE)

# 입력 연결 (Layer1 → Layer2 RAG)
arrow(0.4, 7.1, bx_rag-1.2, 7.1, color=LINE_BLUE)
# 레이어간 점선 연결 표시
dashed_line(8.8, 9.35, 8.8, 8.35, color=LINE_BLUE)

# ★ 수정된 부분: 일관성 점수 미달 → 안전 템플릿 대체 (재질문 아님!)
SAFE_X, SAFE_Y = 9.5, 5.2
rect(SAFE_X-1.45, SAFE_Y-0.7, 2.9, 1.4, BOX_ORANGE, LINE_ORANGE, lw=1.8, radius=0.2)
txt(SAFE_X, SAFE_Y+0.32, '안전 템플릿 답변 대체', size=8.5, bold=True, color=TXT_ORANGE)
txt(SAFE_X, SAFE_Y-0.02, 'build_safe_contract_answer()', size=7.5,
    color=TXT_ORANGE)
txt(SAFE_X, SAFE_Y-0.35, '(할루시네이션 방지)', size=7.2, color=TXT_GRAY)

# GPT-4o-mini → 안전 템플릿 (일관성 미달 시)
ax.annotate('', xy=(SAFE_X+0.6, SAFE_Y+0.7), xytext=(bx_cons+0.35, L2_Y-L2_H/2),
            arrowprops=dict(arrowstyle='->', color=LINE_ORANGE, lw=1.4, mutation_scale=8))
txt(8.9, 6.08, '일관성 점수 미달', size=7.5, color=TXT_ORANGE)

# GPT-4o-mini → 검증 레이어 (통과 시)
arrow(bx_gpt+1.3, L2_Y, 13.0, L2_Y, color=LINE_PURPLE)
arrow(13.0, L2_Y, 13.0, 4.1, color=LINE_PURPLE)

# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 – 검증 레이어
# ─────────────────────────────────────────────────────────────────────────────
L3_Y = 2.55
L3_H = 1.5

# ① 법적 추론 검증
bx_lr = 1.7
rect(bx_lr-1.2, L3_Y-L3_H/2, 2.4, L3_H, BOX_BLUE, LINE_BLUE, lw=1.5, radius=0.2)
txt(bx_lr, L3_Y+0.22, '법적 추론 검증', size=9, bold=True, color=LINE_BLUE)
txt(bx_lr, L3_Y-0.22, '근거 없는 조문 감지', size=7.5, color=TXT_GRAY)

# ② NER 환각 탐지
bx_ner = 5.1
rect(bx_ner-1.2, L3_Y-L3_H/2, 2.4, L3_H, BOX_BLUE, LINE_BLUE, lw=1.5, radius=0.2)
txt(bx_ner, L3_Y+0.35, 'NER 환각 탐지', size=9, bold=True, color=LINE_BLUE)
txt(bx_ner, L3_Y+0.05, 'legal-ner-v3', size=7.5, color=TXT_GRAY)
txt(bx_ner, L3_Y-0.25, '언급처 규칙 포함', size=7.5, color=TXT_GRAY)

# ③ 품질 계약 검사
bx_qafs = 8.5
rect(bx_qafs-1.2, L3_Y-L3_H/2, 2.4, L3_H, BOX_BLUE, LINE_BLUE, lw=1.5, radius=0.2)
txt(bx_qafs, L3_Y+0.35, '품질 계약 검사', size=9, bold=True, color=LINE_BLUE)
txt(bx_qafs, L3_Y+0.05, 'QAFS', size=7.5, color=TXT_GRAY)
txt(bx_qafs, L3_Y-0.25, '출력 품질 기준 검증', size=7.5, color=TXT_GRAY)

# ④ 대화 이력 DB
bx_histdb = 11.7
rect(bx_histdb-1.0, L3_Y-L3_H/2, 2.0, L3_H, BOX_DB, '#0277BD', lw=1.5, radius=0.2)
txt(bx_histdb, L3_Y+0.35, '대화 이력 DB', size=9, bold=True, color='#01579B')
txt(bx_histdb, L3_Y+0.05, 'SQLite', size=7.5, color=TXT_GRAY)
txt(bx_histdb, L3_Y-0.25, '© Session 저장', size=7.5, color=TXT_GRAY)
ax.text(bx_histdb, L3_Y+0.7, '[DB]', fontsize=8, ha='center', va='center')

# 화살표 Layer 3
arrow(bx_lr+1.2, L3_Y, bx_ner-1.2, L3_Y, color=LINE_BLUE)
arrow(bx_ner+1.2, L3_Y, bx_qafs-1.2, L3_Y, color=LINE_BLUE)
arrow(bx_qafs+1.2, L3_Y, bx_histdb-1.0, L3_Y, color=LINE_BLUE)
# 입력 연결
arrow(0.35, L3_Y, bx_lr-1.2, L3_Y, color=LINE_BLUE)
# Layer2 → Layer3 연결
dashed_line(13.0, 4.5, 13.0, L3_Y+0.1, color=LINE_PURPLE)
ax.annotate('', xy=(0.35, L3_Y), xytext=(0.35, 4.5),
            arrowprops=dict(arrowstyle='->', color=LINE_BLUE, lw=1.4, mutation_scale=8))
dashed_line(0.35, 4.5, 13.0, 4.5, color=LINE_BLUE)

# ─────────────────────────────────────────────────────────────────────────────
# 우측 Service 패널
# ─────────────────────────────────────────────────────────────────────────────
SVC_X = 14.6
SVC_W = 7.0

rect(SVC_X, 0.3, SVC_W, 13.2, BG_SERVICE, '#7B1FA2', lw=1.5, alpha=0.45, radius=0.4)
txt(SVC_X + SVC_W/2, 13.2, 'Service', size=13, bold=True, color='#4A148C')

# ── 최종 포맷팅 (★ 수정: 5개 섹션)
rect(SVC_X+0.3, 10.4, SVC_W-0.6, 2.5, 'white', '#7B1FA2', lw=1.4, radius=0.2)
txt(SVC_X+SVC_W/2, 12.6, '최종 포맷팅', size=10, bold=True, color='#4A148C')
txt(SVC_X+SVC_W/2, 12.2,
    '① 상황정리  ② 법적판단',
    size=8.2, color=TXT_GRAY)
txt(SVC_X+SVC_W/2, 11.88,
    '③ 추가로 확인할 사항 및 법적 의미',
    size=8.2, color=TXT_GRAY)
txt(SVC_X+SVC_W/2, 11.55,
    '④ 지금할일  ⑤ 도움받을곳',
    size=8.2, color=TXT_GRAY)
txt(SVC_X+SVC_W/2, 11.1,
    '5개 섹션 구조화 · 포맷팅',
    size=8, color='#7B1FA2')

arrow(SVC_X+SVC_W/2, 10.4, SVC_X+SVC_W/2, 9.7, color='#4A148C')

# ── 카카오톡 말풍선 분할 (★ 수정: 950자)
rect(SVC_X+0.3, 8.5, SVC_W-0.6, 1.15, 'white', '#0277BD', lw=1.4, radius=0.2)
txt(SVC_X+SVC_W/2, 9.3, '카카오톡 말풍선 분할', size=10, bold=True, color='#01579B')
txt(SVC_X+SVC_W/2, 8.87, '최대 950자 × 최대 3개', size=8.5, color=TXT_GRAY)

arrow(SVC_X+SVC_W/2, 8.5, SVC_X+SVC_W/2, 7.9, color='#01579B')

# ── 카카오톡 최종 답변 전달
rect(SVC_X+0.3, 6.5, SVC_W-0.6, 1.35, BOX_YELLOW, LINE_YELLOW, lw=2.0, radius=0.2)
txt(SVC_X+SVC_W/2, 7.45, '카카오톡 최종 답변 전달', size=10, bold=True, color='#5D4037')
txt(SVC_X+SVC_W/2, 7.1, '최종 답변 전송', size=8.5, color=TXT_GRAY)
# 버튼 표시
rect(SVC_X+1.8, 6.56, 2.8, 0.42, '#1565C0', '#1565C0', radius=0.15)
txt(SVC_X+3.2, 6.78, '웹툰으로 요약 보기', size=8, bold=True, color='white')

arrow(SVC_X+SVC_W/2, 6.5, SVC_X+SVC_W/2, 5.85, color=LINE_YELLOW)
txt(SVC_X+SVC_W/2, 6.1, '자동 생성', size=7.5, color=TXT_ORANGE)

# ── 웹툰 자동 생성 (★ 수정: 버튼이 아닌 자동 생성)
rect(SVC_X+0.3, 2.8, SVC_W-0.6, 2.95, 'white', LINE_ORANGE, lw=1.4, radius=0.2)
txt(SVC_X+SVC_W/2, 5.45, '웹툰 자동 생성 기능', size=10, bold=True, color=TXT_ORANGE)
txt(SVC_X+SVC_W/2, 5.1, '(최종 답변 생성 후 자동 실행)', size=8, color=TXT_ORANGE)

# 웹툰 서브 박스
rect(SVC_X+0.5, 4.38, SVC_W-1.0, 0.55, '#FFF3E0', LINE_ORANGE, lw=1.0, radius=0.12)
txt(SVC_X+SVC_W/2, 4.65, 'DALL-E API → 4컷 웹툰 이미지 생성', size=8.2, color=TXT_GRAY)

arrow(SVC_X+SVC_W/2, 4.38, SVC_X+SVC_W/2, 4.0, color=LINE_ORANGE)

rect(SVC_X+0.5, 3.25, SVC_W-1.0, 0.65, '#FFF3E0', LINE_ORANGE, lw=1.0, radius=0.12)
txt(SVC_X+SVC_W/2, 3.58, '이미지 URL 생성', size=8.2, color=TXT_GRAY)

arrow(SVC_X+SVC_W/2, 3.25, SVC_X+SVC_W/2, 2.95, color=LINE_ORANGE)

# ─────────────────────────────────────────────────────────────────────────────
# 레이어 → Service 연결
# ─────────────────────────────────────────────────────────────────────────────
# Layer3 → Service (최종 포맷팅)
dashed_line(bx_histdb+1.0, L3_Y, SVC_X+SVC_W/2, L3_Y, color=LINE_GRAY)
ax.annotate('', xy=(SVC_X+0.3, 11.65), xytext=(SVC_X-0.5, 11.65),
            arrowprops=dict(arrowstyle='->', color=LINE_GRAY, lw=1.3, mutation_scale=8))
dashed_line(SVC_X-0.5, L3_Y, SVC_X-0.5, 11.65, color=LINE_GRAY)
dashed_line(bx_histdb+1.0, L3_Y, SVC_X-0.5, L3_Y, color=LINE_GRAY)

# ─────────────────────────────────────────────────────────────────────────────
# 범례
# ─────────────────────────────────────────────────────────────────────────────
LEG_X = 14.65
LEG_Y_START = 2.55
leg_items = [
    (BOX_BLUE, LINE_BLUE, '-', '처리 단계 (분석·검색·검증)'),
    (BOX_PURPLE, LINE_PURPLE, '--', '외부 API'),
    (BOX_ORANGE, LINE_ORANGE, '-', '조기 종료 / 안전 템플릿 대체'),
    (BOX_DB, '#0277BD', '-', '데이터베이스 (SQLite)'),
]
txt(LEG_X, LEG_Y_START+0.25, '범례', size=9, bold=True, color=TXT, ha='left')
for i, (fc, ec, ls, label) in enumerate(leg_items):
    iy = LEG_Y_START - 0.5 - i*0.48
    rect(LEG_X, iy-0.15, 0.55, 0.32, fc, ec, lw=1.2, ls=ls, radius=0.05)
    txt(LEG_X+0.75, iy+0.01, label, size=8, color=TXT_GRAY, ha='left')

# 화살표 범례
ax.annotate('', xy=(LEG_X+0.55, LEG_Y_START-3.0),
            xytext=(LEG_X, LEG_Y_START-3.0),
            arrowprops=dict(arrowstyle='->', color=LINE_GRAY, lw=1.3, mutation_scale=7))
txt(LEG_X+0.75, LEG_Y_START-3.0, '데이터 흐름 화살표', size=8, color=TXT_GRAY, ha='left')

ax.plot([LEG_X, LEG_X+0.55], [LEG_Y_START-3.48, LEG_Y_START-3.48],
        '--', color=LINE_GRAY, lw=1.2)
txt(LEG_X+0.75, LEG_Y_START-3.48, '레이어 간 흐름', size=8, color=TXT_GRAY, ha='left')

# ─────────────────────────────────────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────────────────────────────────────
out = r'c:\Users\CHOLONG\Documents\GitHub\chatbot\LawsGuard_Architecture.png'
plt.tight_layout(pad=0.5)
plt.savefig(out, dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f'이미지 저장 완료: {out}')
plt.close()
