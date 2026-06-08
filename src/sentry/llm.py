"""LLM client abstraction for the agent
"""

from collections.abc import Iterable
from typing import Literal, Protocol

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]
StopReason = Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]


class Message(BaseModel):
    """One message in the conversation. System prompts are passed separately."""

    role: Role
    content: str


class ToolDef(BaseModel):
    """Definition of a tool the model may call.

    The ``input_schema`` is a JSON Schema object describing the tool's arguments,
    in the same format the Anthropic API expects.
    """

    name: str
    description: str
    input_schema: dict[str, object]


class LLMToolCall(BaseModel):
    """A single tool invocation requested by the model."""

    id: str  # echoed back when we send the tool_result
    name: str
    arguments: dict[str, object]


class Usage(BaseModel):
    """Token usage for one LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    """The full result of one ``complete`` call.

    When the model wants to call tools, ``stop_reason`` is ``"tool_use"`` and
    ``tool_calls`` is non-empty; the caller is expected to execute the tools and
    continue the conversation. Otherwise ``text`` holds the final answer.
    """

    text: str = ""
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    stop_reason: StopReason = "end_turn"
    usage: Usage = Field(default_factory=Usage)


class LLMClient(Protocol):
    """Structural contract every LLM backend must implement."""

    def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...


class MockLLMClient:
    """A scripted LLM client for use in tests.

    Construct with an iterable of ``LLMResponse`` objects. Each call to
    ``complete`` returns the next response in the sequence. Raises if the
    agent makes more calls than were scripted, which keeps tests honest about
    expected interaction counts.

    Records every call in ``self.calls`` so tests can assert on what the agent
    actually sent (messages, system prompt, available tools).
    """

    def __init__(self, responses: Iterable[LLMResponse]) -> None:
        self._responses: list[LLMResponse] = list(responses)
        self._index: int = 0
        self.calls: list[
            tuple[list[Message], str | None, list[ToolDef] | None, int]
        ] = []

    def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.calls.append((list(messages), system, tools, max_tokens))
        if self._index >= len(self._responses):
            raise RuntimeError(
                f"MockLLMClient exhausted: agent made call #{self._index + 1} "
                f"but only {len(self._responses)} responses were scripted."
            )
        response = self._responses[self._index]
        self._index += 1
        return response