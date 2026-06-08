"""Tests for the LLM client abstraction.

Locks in the mock's contract (scripted, call-tracking, fail-loud-on-exhaustion)
and verifies the response schema's defaults and JSON behavior.
"""

import pytest

from sentry.llm import (
    LLMResponse,
    LLMToolCall,
    Message,
    MockLLMClient,
    ToolDef,
    Usage,
)


def test_mock_returns_scripted_responses_in_order() -> None:
    """Successive calls return the responses in the order they were provided."""
    client = MockLLMClient(
        [
            LLMResponse(text="first"),
            LLMResponse(text="second"),
            LLMResponse(text="third"),
        ]
    )

    assert client.complete([Message(role="user", content="a")]).text == "first"
    assert client.complete([Message(role="user", content="b")]).text == "second"
    assert client.complete([Message(role="user", content="c")]).text == "third"


def test_mock_records_each_call() -> None:
    """The mock remembers messages, system, tools, and max_tokens per call."""
    tool = ToolDef(
        name="ruff",
        description="Run the Ruff linter on the changed files.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
    )
    client = MockLLMClient([LLMResponse(text="ok"), LLMResponse(text="ok")])

    client.complete(
        messages=[Message(role="user", content="hi")],
        system="you are a reviewer",
        tools=[tool],
        max_tokens=2048,
    )
    client.complete(messages=[Message(role="user", content="bye")])

    assert len(client.calls) == 2

    msgs1, sys1, tools1, max1 = client.calls[0]
    assert [m.content for m in msgs1] == ["hi"]
    assert sys1 == "you are a reviewer"
    assert tools1 is not None and tools1[0].name == "ruff"
    assert max1 == 2048

    msgs2, sys2, tools2, max2 = client.calls[1]
    assert [m.content for m in msgs2] == ["bye"]
    assert sys2 is None
    assert tools2 is None
    assert max2 == 4096  # default


def test_mock_raises_on_exhaustion() -> None:
    """Calling more times than scripted is a loud failure."""
    client = MockLLMClient([LLMResponse(text="only one")])
    client.complete([Message(role="user", content="x")])

    with pytest.raises(RuntimeError, match="exhausted"):
        client.complete([Message(role="user", content="y")])


def test_llm_response_defaults() -> None:
    """A bare ``LLMResponse`` is an empty end_turn message with zero usage."""
    response = LLMResponse()

    assert response.text == ""
    assert response.tool_calls == []
    assert response.stop_reason == "end_turn"
    assert response.usage == Usage()


def test_tool_call_round_trips_through_json() -> None:
    """Tool calls with mixed-type argument dicts survive serialization."""
    call = LLMToolCall(
        id="call_001",
        name="ripgrep",
        arguments={"pattern": "RATE_PER_POUND", "max_count": 10},
    )

    payload = call.model_dump_json()
    restored = LLMToolCall.model_validate_json(payload)

    assert restored == call
    assert restored.arguments["pattern"] == "RATE_PER_POUND"
    assert restored.arguments["max_count"] == 10