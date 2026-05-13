"""4컷 웹툰 이미지 생성 모듈."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import httpx

from ..config import config


logger = logging.getLogger("lawsguard.webtoon")

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

_SYSTEM_PROMPT = """당신은 한국 법률 전문 웹툰 스토리보드 작가입니다.
다음 상담 내용을 바탕으로 4컷 만화를 구성하고 JSON으로 답하세요.

[구성 지침]
1. 1컷 (상황 발생): 사건이 일어난 구체적인 장소와 인물을 묘사하세요.
2. 2컷 (감정/반응): 피해자 또는 주인공의 심리적 반응 장면.
3. 3컷 (행동): 증거 수집, 신고, 상담 등 구체적 행동 장면.
4. 4컷 (해결): 법적 절차 진행 또는 권리 회복 장면.

[인물 설정]
- 1~4컷 내내 주인공의 외모(헤어, 얼굴형, 피부톤, 옷차림, 체형)를 완벽하게 동일하게 유지하세요.
- 구체적인 외모를 먼저 "character" 필드에 정의하고, 모든 컷의 image_prompt에 동일하게 반복 명시하세요.
  예: "단발머리 20대 한국 여성, 흰색 후드티, 청바지, 작은 얼굴, 큰 눈"

[텍스트 삽입]
- 각 컷 이미지 상단에 소제목(top_text, 10자 이내)을 어두운 배너 안에 흰색 글씨로 표시하세요.
- 각 컷 이미지 하단에 핵심 조언(bottom_text, 20자 이내)을 밝은 박스 안에 검은 글씨로 표시하세요.
- 텍스트는 반드시 깔끔하고 선명하게, 배경과 대비되는 색상으로 작성하세요.

[안전 지침]
- 폭력, 신체 접촉, 성적 묘사는 절대 포함하지 마세요.
- 피해자의 표정, 증거물, 법적 기관 방문 장면으로 간접 묘사하세요.

[출력 형식 (JSON)]
{
  "character": "주인공 외모 상세 고정 설명",
  "panels": [
    {
      "top_text": "소제목 (10자 이내 한국어)",
      "bottom_text": "핵심 조언 (20자 이내 한국어)",
      "image_prompt": "이미지 생성용 프롬프트 (주인공 외모 반복 명시, 장면 묘사, 텍스트 배치 포함)"
    }
  ]
}"""

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


def _build_image_prompt(character: str, panels: list[dict]) -> str:
    panel_lines = []
    for i, p in enumerate(panels, 1):
        top = p.get("top_text", "")
        bot = p.get("bottom_text", "")
        scene = _sanitize(p.get("image_prompt", ""))
        panel_lines.append(
            f"[패널 {i}] 장면: {scene} "
            f"| 상단 배너(어두운 배경+흰 글씨): \"{top}\" "
            f"| 하단 박스(밝은 배경+검은 글씨): \"{bot}\""
        )

    combined = "\n".join(panel_lines)
    return (
        f"한국 웹툰 스타일 2×2 그리드 4컷 만화. 법률 교육 목적.\n\n"
        f"【주인공 외모 — 4컷 전체 동일 유지】 {character}\n\n"
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
        storyboard_resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"상담 내용: {question}"},
            ],
            max_tokens=1500,
        )
        storyboard = json.loads(storyboard_resp.choices[0].message.content.strip())
    except Exception as e:
        logger.error(f"스토리보드 생성 실패: {e}")
        return None

    panels = storyboard.get("panels", [])
    character = storyboard.get("character", "20대 한국인")
    if not panels or len(panels) != 4:
        logger.error("스토리보드 패널 수 이상")
        return None

    try:
        image_prompt = _build_image_prompt(character, panels)
        img_resp = await client.images.generate(
            model="gpt-image-2",
            prompt=image_prompt,
            size="1024x1024",
            quality="medium",
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
