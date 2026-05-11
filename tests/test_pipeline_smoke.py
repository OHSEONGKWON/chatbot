import pytest

from src.pipeline import pipeline


@pytest.mark.asyncio
async def test_pipeline_requeries_sparse_labor_question():
    result = await pipeline.process("pytest-sparse", "알바비를 못 받았어요")

    assert result.needs_requery is True
    assert "알려" in result.response_text
    assert result.legal_category == "노동"


@pytest.mark.asyncio
async def test_pipeline_completes_specific_labor_question():
    result = await pipeline.process(
        "pytest-specific",
        "저는 편의점 알바를 했는데 2026년 4월부터 5월까지 주 20시간 일했고 "
        "월급 80만원을 아직 못 받았습니다. 근로계약서는 없고 카톡 대화는 있습니다.",
    )

    assert result.needs_requery is False
    assert result.session_ended is True
    assert result.step_reached == 7
    assert "참고한 근거" in result.response_text
