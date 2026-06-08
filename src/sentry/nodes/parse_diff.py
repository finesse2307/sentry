"""parse_diff node: turns a raw unified-diff string into a ParsedDiff.

Pure CPU work; no LLM call, no I/O. Produces the structured view of the PR's
changes that downstream nodes (planner, critique) operate on.
"""

from unidiff import PatchSet  # type: ignore[import-untyped]

from sentry.state import AgentState, DiffFile, DiffHunk, ParsedDiff

# Minimal extension → language map. Extend as the eval set adds languages.
_LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".sh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".md": "markdown",
}


def _detect_language(path: str) -> str | None:
    """Map a file path to a language string based on its extension."""
    for ext, lang in _LANGUAGE_BY_EXTENSION.items():
        if path.endswith(ext):
            return lang
    return None


def parse_diff(state: AgentState) -> dict[str, ParsedDiff]:
    """Parse the raw unified diff in state into a ParsedDiff.

    Returns a partial state update; LangGraph (or a test) merges it into the
    running state. Empty or whitespace-only ``raw_diff`` yields a ParsedDiff
    with no files — a valid, if uninteresting, state, not an error.
    """
    if not state.raw_diff.strip():
        return {"diff": ParsedDiff(files=[])}

    patch_set = PatchSet(state.raw_diff)

    files: list[DiffFile] = []
    for patched_file in patch_set:
        hunks: list[DiffHunk] = []
        for hunk in patched_file:
            hunk_lines = str(hunk).splitlines()
            hunks.append(
                DiffHunk(
                    header=hunk_lines[0],
                    content="\n".join(hunk_lines[1:]),
                )
            )
        files.append(
            DiffFile(
                path=patched_file.path,
                language=_detect_language(patched_file.path),
                hunks=hunks,
            )
        )

    return {"diff": ParsedDiff(files=files)}