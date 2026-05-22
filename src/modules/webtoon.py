"""4컷 웹툰 이미지 생성 모듈."""

from __future__ import annotations

import base64
import logging
import os
import re  # 정규표현식 모듈 추가
from pathlib import Path

import httpx

from ..config import config


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
    # 매칭 패턴: 공백(선택) + 숫자 + 괄호나 점 + 내용 (예: " 1. 내용증명", "2) 고소장")
    pattern = re.compile(r'^\s*\[?(\d+)[\]\.\)]\s*(.*)')
    
    for line in text.split('\n'):
        match = pattern.match(line)
        if match:
            num = int(match.group(1))
            points[num] = match.group(2)
            
    return points


def _build_image_prompt(answer_text: str) -> str:
    safe_text = _sanitize(answer_text)
    
    # AI 답변에서 1, 2, 4, 5번 추출 (만약 해당 번호가 없을 경우를 대비해 기본값 설정)
    points = _extract_specific_points(safe_text)
    p1 = points.get(1, "상황 발생 및 기초 사실관계 확인")
    p2 = points.get(2, "관련 증거 수집 및 법적 검토")
    p4 = points.get(4, "구체적인 기관 대응 및 절차 진행")
    p5 = points.get(5, "최종 사건 해결 및 권리 회복")
    
    character = "20대 한국인 주인공, 단정한 캐주얼 복장, 4컷 동일한 외모 유지"
    
    # 추출한 답변 내용을 각 패널에 직접 주입합니다.
    combined = (
        f"[패널 1] 묘사 내용: {p1} | 상단 배너: \"1단계\" | 하단 박스: \"답변 1번 내용\"\n"
        f"[패널 2] 묘사 내용: {p2} | 상단 배너: \"2단계\" | 하단 박스: \"답변 2번 내용\"\n"
        f"[패널 3] 묘사 내용: {p4} | 상단 배너: \"4단계\" | 하단 박스: \"답변 4번 내용\"\n"
        f"[패널 4] 묘사 내용: {p5} | 상단 배너: \"5단계\" | 하단 박스: \"답변 5번 내용\""
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
    from openai import AsyncOpenAI

    api_key = config.llm.api_key
    if not api_key:
        logger.warning("OPENAI_API_KEY 없음 — 웹툰 생성 건너뜀")
        return None

    client = AsyncOpenAI(api_key=api_key)

    try:
        image_prompt = _build_image_prompt(answer_text)
        # 요청하신 대로 모델명(gpt-image-2) 등 기존 설정은 그대로 유지했습니다.
        img_resp = await client.images.generate(
            model="gpt-image-2",
            prompt=image_prompt,
            size=WEBTOON_IMAGE_SIZE,
            quality=WEBTOON_IMAGE_QUALITY,
            output_format="jpeg",
            n=1,
        )
        
        img = img_resp.data[0]
        if getattr(img, "b64_json", None):
            image_bytes = base64.b64decode(img.b64_json)
        elif getattr(img, "url", None):
            async with httpx.AsyncClient(timeout=30) as http:
                r = await http.get(img.url)
                image_bytes = r.content
        else:
            raise RuntimeError("이미지 응답에 url 또는 b64_json이 없습니다.")
    except Exception as e:
        logger.error(f"이미지 생성 실패: {e}")
        return None

    import datetime
    filename = f"4cut_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpeg"
    (OUTPUT_DIR / filename).write_bytes(image_bytes)
    logger.info(f"웹툰 저장 완료: {filename}")
    return filename