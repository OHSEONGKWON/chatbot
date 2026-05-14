from __future__ import annotations

import re


MAX_SIMPLE_TEXT_CHARS = 900
MAX_OUTPUTS = 3


def build_simple_text(text: str, quick_replies: list[dict] | None = None) -> dict:
    chunks = split_for_kakao(text)
    outputs = [{"simpleText": {"text": chunk}} for chunk in chunks]
    template = {"outputs": outputs}
    if quick_replies:
        template["quickReplies"] = quick_replies[:10]
    return {"version": "2.0", "template": template}


def build_callback_response(waiting_message: str) -> dict:
    return {"version": "2.0", "useCallback": True, "data": {"text": waiting_message}}


def default_quick_replies(needs_requery: bool = False, category: str = "") -> list[dict]:
    if needs_requery:
        return [
            quick_reply("예시 보기", "예시를 보여줘"),
            quick_reply("처음부터", "처음부터 다시 상담할래요"),
        ]

    replies = [
        quick_reply("추가 질문", "추가 질문이 있어요"),
        quick_reply("증거 정리", "증거는 어떻게 정리해야 하나요?"),
    ]
    if category == "노동":
        replies.append(quick_reply("임금체불", "임금체불 신고 방법을 알려줘"))
    elif category == "성폭력":
        replies.append(quick_reply("피해지원", "성폭력 피해 지원기관을 알려줘"))
    return replies


def quick_reply(label: str, message_text: str) -> dict:
    return {"label": label, "action": "message", "messageText": message_text}


def build_image_response(image_url: str, alt_text: str = "4컷 법률 만화") -> dict:
    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleImage": {"imageUrl": image_url, "altText": alt_text}}]
        },
    }


def build_text_and_image_response(text: str, image_url: str, quick_replies: list[dict] | None = None) -> dict:
    # Kakao template outputs are limited; keep room for one image block.
    chunks = split_for_kakao(text)[: max(1, MAX_OUTPUTS - 1)]
    outputs = [{"simpleText": {"text": chunk}} for chunk in chunks]
    outputs.append({"simpleImage": {"imageUrl": image_url, "altText": "4컷 법률 만화"}})
    template = {"outputs": outputs}
    if quick_replies:
        template["quickReplies"] = quick_replies[:10]
    return {"version": "2.0", "template": template}


def split_for_kakao(text: str) -> list[str]:
    clean = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if not clean:
        return ["답변을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."]

    chunks: list[str] = []
    remaining = clean
    while remaining and len(chunks) < MAX_OUTPUTS:
        if len(remaining) <= MAX_SIMPLE_TEXT_CHARS:
            chunks.append(remaining)
            remaining = ""
            break

        cut = _best_cut(remaining, MAX_SIMPLE_TEXT_CHARS)
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    if remaining and chunks:
        suffix = "\n\n※ 답변이 길어 일부를 줄였습니다. 추가 질문으로 이어서 확인해 주세요."
        budget = MAX_SIMPLE_TEXT_CHARS - len(suffix)
        chunks[-1] = chunks[-1][:budget].rstrip() + suffix

    return chunks or [clean[:MAX_SIMPLE_TEXT_CHARS]]


def _best_cut(text: str, limit: int) -> int:
    paragraph_cut = text.rfind("\n\n", 0, limit)
    if paragraph_cut >= int(limit * 0.55):
        return paragraph_cut

    sentence_cuts = [text.rfind(mark, 0, limit) for mark in [". ", "? ", "! ", "다.\n", "요.\n", "\n"]]
    sentence_cut = max(sentence_cuts)
    if sentence_cut >= int(limit * 0.55):
        return sentence_cut + 1

    space_cut = text.rfind(" ", 0, limit)
    if space_cut >= int(limit * 0.55):
        return space_cut
    return limit
