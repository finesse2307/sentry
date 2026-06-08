"""Tests for the run_tool node.

Covers: each tool fires in plan order with correct args, exceptions become
error results without breaking the run, missing-registry-entry raises, empty
plans return no results, and a missing plan short-circuits.
"""

import pytest

from sentry.nodes.run_tool import ToolFn, ToolRegistry, make_run_tool_node
from sentry.state import AgentState, Plan, PRMetadata, ToolCall, ToolName


def _make_state(plan: Plan | None) -> AgentState:
    return AgentState(
        pr=PRMetadata(
            repo="acme/x",
            pr_number=1,
            head_sha="a",
            base_sha="b",
            author="alice",
            title="t",
        ),
        raw_diff="(unused)",
        plan=plan,
    )


def test_happy_path_executes_each_call_in_order() -> None:
    """All planned calls execute; results appear in plan order with correct args."""
    log: list[tuple[ToolName, dict[str, str]]] = []

    def make_recorder(name: ToolName) -> ToolFn:
        def fn(args: dict[str, str]) -> str:
            log.append((name, args))
            return f"{name.value} ran"
        return fn

    registry: ToolRegistry = {
        ToolName.RUFF: make_recorder(ToolName.RUFF),
        ToolName.SEMGREP: make_recorder(ToolName.SEMGREP),
        ToolName.RIPGREP: make_recorder(ToolName.RIPGREP),
        ToolName.DOCS_LOOKUP: make_recorder(ToolName.DOCS_LOOKUP),
    }

    plan = Plan(
        reasoning="run two tools",
        calls=[
            ToolCall(
                tool=ToolName.SEMGREP,
                arguments={"config": "p/security"},
                rationale="security check",
            ),
            ToolCall(
                tool=ToolName.RIPGREP,
                arguments={"pattern": "get_user"},
                rationale="find callers",
            ),
        ],
    )

    results = make_run_tool_node(registry)(_make_state(plan))["tool_results"]

    assert log == [
        (ToolName.SEMGREP, {"config": "p/security"}),
        (ToolName.RIPGREP, {"pattern": "get_user"}),
    ]
    assert len(results) == 2
    assert results[0].tool is ToolName.SEMGREP
    assert results[0].output == "semgrep ran"
    assert results[0].error is None
    assert results[0].duration_ms >= 0
    assert results[1].tool is ToolName.RIPGREP
    assert results[1].output == "ripgrep ran"


def test_tool_exception_becomes_error_result() -> None:
    """A raising tool produces a ToolResult with error set and empty output."""

    def broken(args: dict[str, str]) -> str:
        raise RuntimeError("docker daemon offline")

    registry: ToolRegistry = {ToolName.RUFF: broken}
    plan = Plan(
        reasoning="lint it",
        calls=[
            ToolCall(tool=ToolName.RUFF, arguments={"path": "src"}, rationale="lint"),
        ],
    )

    [result] = make_run_tool_node(registry)(_make_state(plan))["tool_results"]

    assert result.output == ""
    assert result.error == "RuntimeError: docker daemon offline"
    assert result.tool is ToolName.RUFF


def test_other_tools_still_run_when_one_fails() -> None:
    """If one tool raises, subsequent tools still execute and report normally."""

    def broken(args: dict[str, str]) -> str:
        raise ValueError("first one broke")

    def ok(args: dict[str, str]) -> str:
        return "second one succeeded"

    registry: ToolRegistry = {ToolName.RUFF: broken, ToolName.SEMGREP: ok}
    plan = Plan(
        reasoning="two tools, first fails",
        calls=[
            ToolCall(tool=ToolName.RUFF, arguments={}, rationale="r1"),
            ToolCall(tool=ToolName.SEMGREP, arguments={}, rationale="r2"),
        ],
    )

    results = make_run_tool_node(registry)(_make_state(plan))["tool_results"]

    assert len(results) == 2
    assert results[0].error == "ValueError: first one broke"
    assert results[0].output == ""
    assert results[1].error is None
    assert results[1].output == "second one succeeded"


def test_missing_tool_in_registry_raises() -> None:
    """A planned call to an unregistered tool is a config bug; raise loudly."""
    registry: ToolRegistry = {
        ToolName.RUFF: lambda args: "ok",
        # SEMGREP intentionally omitted
    }
    plan = Plan(
        reasoning="needs semgrep",
        calls=[
            ToolCall(tool=ToolName.SEMGREP, arguments={}, rationale="check"),
        ],
    )

    with pytest.raises(KeyError, match="semgrep"):
        make_run_tool_node(registry)(_make_state(plan))


def test_empty_plan_calls_returns_empty_results() -> None:
    """A plan with no calls yields no tool results; the registry is never used."""

    def should_not_run(args: dict[str, str]) -> str:
        raise AssertionError("must not be called")

    registry: ToolRegistry = {ToolName.RUFF: should_not_run}
    plan = Plan(reasoning="nothing to do", calls=[])

    assert make_run_tool_node(registry)(_make_state(plan))["tool_results"] == []


def test_missing_plan_raises() -> None:
    """The node refuses to run when state.plan has not been populated."""
    registry: ToolRegistry = {}

    with pytest.raises(ValueError, match="plan_node"):
        make_run_tool_node(registry)(_make_state(plan=None))