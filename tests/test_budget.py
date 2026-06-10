"""Tests for BudgetedLLMClient.

Covers: spend tracking from actual usage, cap enforcement (the crossing call
completes, the next one raises), zero-cost handling for cache hits, custom
pricing, and that the default constants reflect Haiku 4.5 pricing.
"""

import pytest

from sentry.budget import (
    DEFAULT_INPUT_PRICE_PER_MTOK,
    DEFAULT_OUTPUT_PRICE_PER_MTOK,
    BudgetedLLMClient,
    BudgetExceededError,
)
from sentry.llm import LLMResponse, Message, MockLLMClient, Usage


def _response(
    input_tokens: int = 0, output_tokens: int = 0, text: str = "ok"
) -> LLMResponse:
    return LLMResponse(
        text=text,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_call_records_usage_and_spend() -> None:
    """A single call updates token counters and dollar spend from response.usage."""
    inner = MockLLMClient(
        [_response(input_tokens=1_000_000, output_tokens=200_000)]
    )
    client = BudgetedLLMClient(inner, cap_usd=10.0)

    client.complete([Message(role="user", content="hi")])

    assert client.total_input_tokens == 1_000_000
    assert client.total_output_tokens == 200_000
    # 1M input * $1.00 + 0.2M output * $5.00 = $1.00 + $1.00 = $2.00
    assert client.total_spend_usd == pytest.approx(2.0)


def test_multiple_calls_accumulate() -> None:
    """Spend accumulates across calls."""
    inner = MockLLMClient(
        [
            _response(input_tokens=500_000, output_tokens=100_000),  # $1.00
            _response(input_tokens=500_000, output_tokens=100_000),  # $1.00
        ]
    )
    client = BudgetedLLMClient(inner, cap_usd=10.0)

    client.complete([Message(role="user", content="a")])
    client.complete([Message(role="user", content="b")])

    assert client.total_spend_usd == pytest.approx(2.0)


def test_call_that_crosses_cap_completes() -> None:
    """The call that pushes spend over the cap is allowed to return normally."""
    inner = MockLLMClient(
        [_response(input_tokens=2_000_000, output_tokens=400_000)]  # $4.00
    )
    client = BudgetedLLMClient(inner, cap_usd=1.0)  # cap below the call's cost

    response = client.complete([Message(role="user", content="huge")])

    assert response.text == "ok"
    assert client.total_spend_usd == pytest.approx(4.0)


def test_subsequent_call_after_cap_raises_before_calling_inner() -> None:
    """After the cap is met, the next call raises and the inner is not consulted."""
    inner = MockLLMClient(
        [_response(input_tokens=2_000_000, output_tokens=400_000)]  # $4.00, only one scripted
    )
    client = BudgetedLLMClient(inner, cap_usd=3.0)

    client.complete([Message(role="user", content="big")])
    assert client.total_spend_usd >= 3.0

    # If the budget check didn't fire first, we'd hit the MockLLMClient's own
    # "exhausted" RuntimeError instead. Seeing BudgetExceededError proves the
    # budget check short-circuits before reaching the inner client.
    with pytest.raises(BudgetExceededError, match=r"\$3\.00"):
        client.complete([Message(role="user", content="too much")])


def test_zero_token_usage_costs_nothing() -> None:
    """A response with zero usage (cache hit) leaves spend unchanged."""
    inner = MockLLMClient(
        [_response(input_tokens=0, output_tokens=0, text="cached")]
    )
    client = BudgetedLLMClient(inner, cap_usd=0.01)  # very tight cap

    response = client.complete([Message(role="user", content="probe")])

    assert response.text == "cached"
    assert client.total_spend_usd == 0.0


def test_custom_pricing_overrides_defaults() -> None:
    """Pricing parameters are honored when overridden."""
    inner = MockLLMClient(
        [_response(input_tokens=1_000_000, output_tokens=1_000_000)]
    )
    client = BudgetedLLMClient(
        inner,
        cap_usd=100.0,
        input_price_per_mtok=2.0,
        output_price_per_mtok=10.0,
    )

    client.complete([Message(role="user", content="hi")])

    # 1M * $2 + 1M * $10 = $12
    assert client.total_spend_usd == pytest.approx(12.0)


def test_defaults_match_published_haiku_pricing() -> None:
    """Default constants reflect Claude Haiku 4.5 pricing."""
    assert DEFAULT_INPUT_PRICE_PER_MTOK == 1.0
    assert DEFAULT_OUTPUT_PRICE_PER_MTOK == 5.0