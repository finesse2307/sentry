"""Integration test for the full agent graph.

End-to-end execution with a mocked LLM, stub tools, and the NoopPoster.
Confirms the six nodes wire correctly through LangGraph and the state machine
produces a fully populated final state.
"""

from sentry.graph import build_graph
from sentry.llm import LLMResponse, LLMToolCall, MockLLMClient
from sentry.nodes.run_tool import ToolRegistry
from sentry.posting import NoopPoster
from sentry.state import AgentState, PostStatus, PRMetadata, ToolName


def _stub_tools() -> ToolRegistry:
    return {
        ToolName.RUFF: lambda args: f"ruff stub for {args}",
        ToolName.SEMGREP: lambda args: f"semgrep stub for {args}",
        ToolName.RIPGREP: lambda args: f"ripgrep stub for {args}",
        ToolName.DOCS_LOOKUP: lambda args: f"docs stub for {args}",
    }


def _scripted_llm() -> MockLLMClient:
    """LLM scripted with two responses: a plan and a critique."""
    plan_response = LLMResponse(
        stop_reason="tool_use",
        tool_calls=[
            LLMToolCall(
                id="plan_call",
                name="submit_plan",
                arguments={
                    "reasoning": "SQL injection pattern present; run Semgrep.",
                    "calls": [
                        {
                            "tool": "semgrep",
                            "arguments": {"config": "p/security-audit"},
                            "rationale": "scan for known security patterns",
                        }
                    ],
                },
            )
        ],
    )
    critique_response = LLMResponse(
        stop_reason="tool_use",
        tool_calls=[
            LLMToolCall(
                id="critique_call",
                name="submit_findings",
                arguments={
                    "findings": [
                        {
                            "category": "security",
                            "severity": "high",
                            "file": "src/users.py",
                            "line": 12,
                            "message": "SQL injection via f-string; use parameterized queries.",
                        }
                    ]
                },
            )
        ],
    )
    return MockLLMClient([plan_response, critique_response])


def test_end_to_end_happy_path() -> None:
    """All six nodes execute; the final state has every relevant field populated."""
    diff_text = (
        "diff --git a/src/users.py b/src/users.py\n"
        "--- a/src/users.py\n"
        "+++ b/src/users.py\n"
        "@@ -10,2 +10,3 @@\n"
        " class UserRepo:\n"
        "     def __init__(self, db):\n"
        "+        self.db = db\n"
    )
    initial = AgentState(
        pr=PRMetadata(
            repo="acme/widgets",
            pr_number=42,
            head_sha="abc",
            base_sha="def",
            author="alice",
            title="Add user lookup",
        ),
        raw_diff=diff_text,
    )

    llm = _scripted_llm()
    poster = NoopPoster()
    graph = build_graph(llm=llm, tools=_stub_tools(), poster=poster)

    final = graph.invoke(initial)

    # parse_diff populated diff
    assert final["diff"] is not None
    assert final["diff"].files[0].path == "src/users.py"

    # plan populated plan
    assert final["plan"] is not None
    assert len(final["plan"].calls) == 1
    assert final["plan"].calls[0].tool is ToolName.SEMGREP

    # run_tool populated tool_results
    assert len(final["tool_results"]) == 1
    assert final["tool_results"][0].tool is ToolName.SEMGREP
    assert "semgrep stub" in final["tool_results"][0].output

    # critique populated findings
    assert len(final["findings"]) == 1
    assert final["findings"][0].file == "src/users.py"

    # format_review populated review_body
    assert final["review_body"] is not None
    assert "Sentry Code Review" in final["review_body"]
    assert "SQL injection" in final["review_body"]

    # post_comment posted with success
    assert final["post_status"] is PostStatus.SUCCESS
    assert final["post_url"] is not None

    # The mock LLM was called exactly twice (plan, critique)
    assert len(llm.calls) == 2

    # The poster was called exactly once with our PR
    assert len(poster.calls) == 1
    pr, body = poster.calls[0]
    assert pr.pr_number == 42
    assert "Sentry Code Review" in body