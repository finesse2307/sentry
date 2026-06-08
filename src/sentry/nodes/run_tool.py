"""run_tool node: executes the planned tool calls and records their results.

For each ToolCall in state.plan.calls, look up the function in the injected
registry, run it with timing, and capture either output or error in a
ToolResult. The list of results becomes the evidence the critique node reasons
over.
"""

import time
from collections.abc import Callable

from sentry.state import AgentState, ToolName, ToolResult

ToolFn = Callable[[dict[str, str]], str]
ToolRegistry = dict[ToolName, ToolFn]


def make_run_tool_node(
    registry: ToolRegistry,
) -> Callable[[AgentState], dict[str, list[ToolResult]]]:
    """Build a run_tool-node bound to a specific tool registry.

    The registry maps each ``ToolName`` to a callable that takes a string-keyed
    argument map and returns the tool's textual output. Tool functions that
    raise are caught; the exception message is recorded on the ToolResult and
    the rest of the plan still runs. A *missing* registry entry, by contrast,
    is a configuration bug and raises immediately.
    """

    def run_tool_node(state: AgentState) -> dict[str, list[ToolResult]]:
        if state.plan is None:
            raise ValueError(
                "run_tool_node requires state.plan to be populated; "
                "did plan_node run?"
            )

        results: list[ToolResult] = []
        for call in state.plan.calls:
            if call.tool not in registry:
                raise KeyError(
                    f"Tool {call.tool.value!r} is in the plan but missing from "
                    f"the registry. Registered tools: "
                    f"{sorted(t.value for t in registry)}"
                )

            tool_fn = registry[call.tool]
            output = ""
            error: str | None = None
            start = time.perf_counter()
            try:
                output = tool_fn(call.arguments)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            results.append(
                ToolResult(
                    tool=call.tool,
                    arguments=call.arguments,
                    output=output,
                    error=error,
                    duration_ms=elapsed_ms,
                )
            )

        return {"tool_results": results}

    return run_tool_node