"""4컷 웹툰 이미지 생성 모듈 (오리지널 모델 설정 유지 + 디버깅 강화)"""

from __future__ import annotations

import base64
import logging
import os
import re  # 정규표현식 모듈 추가
import traceback
import datetime
from pathlib import Path

import httpx

from ..config import config

print("[LAWSGUARD_DEBUG] >>> src/modules/webtoon.py 파일이 성공적으로 로드되었습니다! <<<", flush=True)

logger = logging.getLogger("lawsguard.webtoon")

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

WEBTOON_IMAGE_SIZE = os.getenv("LAWSGUARD_WEBTOON_IMAGE_SIZE", "1024x1024")
WEBTOON_IMAGE_QUALITY = os.getenv("LAWSGUARD_WEBTOON_IMAGE_QUALITY", "low")

_SANITIZE_MAP = {
    "성폭행": "부당한 피해",
    "폭행": "갈등 상황",
    "강간": "성적 피해",
    "구타": "신체 피해",
    "학대": "부당한 대우",
    "성희롱": "직장 내 부당 발언",
    "나체": "의상 입은",
    "누드": "의상 입은",
}


def _sanitize(text: str) -> str:
    for sensitive, safe in _SANITIZE_MAP.items():
        text = text.replace(sensitive, safe)
    return text


def _extract_specific_points(text: str) -> dict[int, str]:
    """답변 텍스트에서 숫자 리스트(1., 2) 등)로 시작하는 줄을 찾아 딕셔너리로 반환합니다."""
    points = {}
    pattern = re.compile(r'^\s*\[?(\d+)[\]\.\)]\s*(.*)')
    
    for line in text.split('\n'):
        match = pattern.match(line)
        if match:
            num = int(match.group(1))
            points[num] = match.group(2)
            
    return points


def _build_image_prompt(answer_text: str) -> str:
    safe_text = _sanitize(answer_text)
    
    points = _extract_specific_points(safe_text)
    p1 = points.get(1, "상황 발생 및 기초 사실관계 확인")
    p2 = points.get(2, "관련 증거 수집 및 법적 검토")
    p4 = points.get(4, "구체적인 기관 대응 및 절차 진행")
    p5 = points.get(5, "최종 사건 해결 및 권리 회복")
    
    character = "20대 한국인 주인공, 단정한 캐주얼 복장, 4컷 동일한 외모 유지"
    
    combined = (
        f"[패널 1] 묘사 내용: {p1} |  하단 박스: \"답변 1번 내용\"\n"
        f"[패널 2] 묘사 내용: {p2} |  하단 박스: \"답변 2번 내용\"\n"
        f"[패널 3] 묘사 내용: {p4} |  하단 박스: \"답변 4번 내용\"\n"
        f"[패널 4] 묘사 내용: {p5} |  하단 박스: \"답변 5번 내용\""
    )
    
    return (
        f"한국 웹툰 스타일 2×2 그리드 4컷 만화. 법률 교육 목적.\n\n"
        f"【주인공 외모 — 4컷 전체 동일 유지】 {character}\n\n"
        f"【전체 문맥 참고용 원본 답변】\n{safe_text}\n\n"
        f"【패널 구성 - 답변의 1, 2, 4, 5번 항목만 집중 묘사할 것】\n"
        f"좌상단(1컷) → 우상단(2컷) → 좌하단(3컷) → 우하단(4컷), 얇은 구분선\n"
        f"{combined}\n\n"   
        f"【아트 스타일】\n"
        f"- 전문 웹툰/만화 퀄리티, 굵은 검은 외곽선, 선명한 플랫 셀 채색\n"
        f"- 인물 표정을 풍부하고 생동감 있게 표현\n"
        f"- 배경은 단순하지만 장소가 명확하게 느껴지도록\n"
        f"- 각 패널 텍스트는 선명하고 읽기 쉽게, 배경 색과 대비 강하게\n"
        f"- 폭력, 신체 접촉, 성적 묘사 없음"
    )


async def generate_webtoon(answer_text: str) -> str | None:
    """법률 상담 내용으로 4컷 웹툰을 생성하고 저장된 파일명을 반환합니다. 실패 시 None."""
    print(f"\n[WEBTOON_MODULE] generate_webtoon 함수 가동! (텍스트 길이: {len(answer_text)})", flush=True)
    from openai import AsyncOpenAI

    api_key = config.llm.api_key
    if not api_key:
        print("[WEBTOON_MODULE] ❌ 오류: config.llm.api_key 주입 실패 (빈 값)", flush=True)
        return None

    print("[WEBTOON_MODULE] OpenAI 비동기 클라이언트 인스턴스 생성 완료", flush=True)
    client = AsyncOpenAI(api_key=api_key)

    try:
        image_prompt = _build_image_prompt(answer_text)
        print("[WEBTOON_MODULE] 프롬프트 빌드 완료. 오리지널 모델 커스텀 엔드포인트 호출 시작...", flush=True)
        
        # [기존 설정 전면 유지] 요청하신 대로 gpt-image-2 및 output_format 스펙 유지
        img_resp = await client.images.generate(
            model="gpt-image-2",
            prompt=image_prompt,
            size=WEBTOON_IMAGE_SIZE,
            quality=WEBTOON_IMAGE_QUALITY,
            output_format="jpeg",
            n=1,
        )
        print("[WEBTOON_MODULE] 이미지 생성 응답 수신 완료", flush=True)
        
        img = img_resp.data[0]
        if getattr(img, "b64_json", None):
            print("[WEBTOON_MODULE] b64_json 스트림 디코딩 가동", flush=True)
            image_bytes = base64.b64decode(img.b64_json)
        elif getattr(img, "url", None):
            print(f"[WEBTOON_MODULE] 이미지 URL 감지 -> {img.url} 다운로드 시작", flush=True)
            async with httpx.AsyncClient(timeout=30) as http:
                r = await http.get(img.url)
                image_bytes = r.content
        else:
            raise RuntimeError("이미지 응답 규격체에 url 및 b64_json 속성이 누락되었습니다.")
            
    except Exception as e:
        print(f"[WEBTOON_MODULE] ❌ 이미지 생성 내부 단계 실패 사유: {e}", flush=True)
        # 터미널에 에러가 발생한 위치와 상세 콜스택 추적을 강제로 뿌려줍니다.
        traceback.print_exc()
        return None

    import datetime
    filename = f"4cut_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpeg"
    try:
        target_path = OUTPUT_DIR / filename
        target_path.write_bytes(image_bytes)
        print(f"[WEBTOON_MODULE] ✅ 오리지널 파일 저장 완료: {target_path}", flush=True)
        return filename
    except Exception as save_err:
        print(f"[WEBTOON_MODULE] ❌ 디스크 저장(Write) 실패: {save_err}", flush=True)
        traceback.print_exc()
        return None