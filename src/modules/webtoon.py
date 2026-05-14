"""4컷 웹툰 이미지 생성 모듈."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

import httpx

from ..config import config


logger = logging.getLogger("lawsguard.webtoon")

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

WEBTOON_IMAGE_SIZE = os.getenv("LAWSGUARD_WEBTOON_IMAGE_SIZE", "512x512")
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


def _build_image_prompt(question: str) -> str:
    safe_question = _sanitize(question)
    character = "20대 한국인 주인공, 단정한 캐주얼 복장, 4컷 동일한 외모 유지"
    combined = (
        "[패널 1] 상담 상황 시작, 사용자가 고민을 털어놓는 장면 | 상단 배너: \"상황 발생\" | 하단 박스: \"사실관계 정리\"\n"
        "[패널 2] 주인공이 침착하게 증거/기록을 확인하는 장면 | 상단 배너: \"핵심 확인\" | 하단 박스: \"증거를 모아요\"\n"
        "[패널 3] 상담/신고 창구를 찾는 장면 | 상단 배너: \"대응 준비\" | 하단 박스: \"기관에 문의\"\n"
        "[패널 4] 절차를 진행하며 안도하는 장면 | 상단 배너: \"해결 단계\" | 하단 박스: \"권리 보호 시작\""
    )
    return (
        f"한국 웹툰 스타일 2×2 그리드 4컷 만화. 법률 교육 목적.\n\n"
        f"【주인공 외모 — 4컷 전체 동일 유지】 {character}\n\n"
        f"【사용자 상담 요약】 {safe_question}\n\n"
        f"【패널 배치】 좌상단(1컷) → 우상단(2컷) → 좌하단(3컷) → 우하단(4컷), 얇은 구분선\n\n"
        f"{combined}\n\n"
        f"【아트 스타일】\n"
        f"- 전문 웹툰/만화 퀄리티, 굵은 검은 외곽선, 선명한 플랫 셀 채색\n"
        f"- 인물 표정을 풍부하고 생동감 있게 표현\n"
        f"- 배경은 단순하지만 장소가 명확하게 느껴지도록\n"
        f"- 각 패널 텍스트는 선명하고 읽기 쉽게, 배경 색과 대비 강하게\n"
        f"- 폭력, 신체 접촉, 성적 묘사 없음"
    )


async def generate_webtoon(question: str) -> str | None:
    """법률 상담 내용으로 4컷 웹툰을 생성하고 저장된 파일명을 반환합니다. 실패 시 None."""
    from openai import AsyncOpenAI

    api_key = config.llm.api_key
    if not api_key:
        logger.warning("OPENAI_API_KEY 없음 — 웹툰 생성 건너뜀")
        return None

    client = AsyncOpenAI(api_key=api_key)

    try:
        image_prompt = _build_image_prompt(question)
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
