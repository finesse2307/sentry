"""Budget-aware wrapper around an LLMClient.

Tracks cumulative token usage and dollar spend across calls; refuses to make
further calls once the configured cap is met. Defense in depth on top of the
Anthropic platform-level prepaid balance.

Pricing constants default to Claude Haiku 4.5 rates as of June 2026 ($1.00 / $5.00
per million input/output tokens). Pass overrides at construction time if you
swap models or if rates change.
"""

from sentry.llm import LLMClient, LLMResponse, Message, ToolDef

# Claude Haiku 4.5 standard pricing, USD per million tokens.
# Verified against https://www.anthropic.com/claude/haiku (June 2026).
DEFAULT_INPUT_PRICE_PER_MTOK = 1.00
DEFAULT_OUTPUT_PRICE_PER_MTOK = 5.00


class BudgetExceededError(RuntimeError):
    """Raised when a call would proceed past the configured spend cap."""


class BudgetedLLMClient:
    """Wraps an LLMClient, tracks spend, refuses calls once the cap is met.

    The cap is a soft pre-call check: before each call we verify that
    cumulative spend is still below ``cap_usd``. The call that crosses the cap
    is allowed to complete (we don't have a free pre-call token count); all
    subsequent calls raise ``BudgetExceededError``. The Anthropic platform-level
    prepaid balance remains the true hard ceiling.

    Usage and spend are updated from the actual ``usage`` reported by the
    inner client, so cache hits (which report zero tokens) correctly cost zero.
    """

    def __init__(
        self,
        inner: LLMClient,
        *,
        cap_usd: float,
        input_price_per_mtok: float = DEFAULT_INPUT_PRICE_PER_MTOK,
        output_price_per_mtok: float = DEFAULT_OUTPUT_PRICE_PER_MTOK,
    ) -> None:
        self._inner = inner
        self.cap_usd = cap_usd
        self.input_price_per_mtok = input_price_per_mtok
        self.output_price_per_mtok = output_price_per_mtok

        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_spend_usd: float = 0.0

    def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        if self.total_spend_usd >= self.cap_usd:
            raise BudgetExceededError(
                f"Spend cap of ${self.cap_usd:.2f} reached "
                f"(current: ${self.total_spend_usd:.4f}); refusing call."
            )

        response = self._inner.complete(
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
        )

        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        call_cost = (
            response.usage.input_tokens / 1_000_000 * self.input_price_per_mtok
            + response.usage.output_tokens / 1_000_000 * self.output_price_per_mtok
        )
        self.total_spend_usd += call_cost

        return response