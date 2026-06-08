"""Tests for the parse_diff node.

Exercises: empty/whitespace input, single-file diffs, multi-file diffs,
multi-hunk files, and language detection.
"""

from sentry.nodes.parse_diff import parse_diff
from sentry.state import AgentState, PRMetadata


def _make_state(diff_text: str) -> AgentState:
    return AgentState(
        pr=PRMetadata(
            repo="acme/x",
            pr_number=1,
            head_sha="a",
            base_sha="b",
            author="alice",
            title="t",
        ),
        raw_diff=diff_text,
    )


def test_empty_diff_yields_empty_parsed_diff() -> None:
    """Empty or whitespace-only raw_diff produces a ParsedDiff with no files."""
    for diff_text in ("", "   \n\n  \t"):
        result = parse_diff(_make_state(diff_text))
        assert result["diff"].files == []


def test_single_file_single_hunk() -> None:
    """A one-file, one-hunk diff parses to one DiffFile with one DiffHunk."""
    diff_text = (
        "diff --git a/src/users.py b/src/users.py\n"
        "--- a/src/users.py\n"
        "+++ b/src/users.py\n"
        "@@ -1,2 +1,2 @@\n"
        " class UserRepo:\n"
        "-    pass\n"
        '+    def get(self, uid): return self.db.execute(f"SELECT ... id={uid}")\n'
    )
    parsed = parse_diff(_make_state(diff_text))["diff"]

    assert len(parsed.files) == 1
    file = parsed.files[0]
    assert file.path == "src/users.py"
    assert file.language == "python"
    assert len(file.hunks) == 1
    assert file.hunks[0].header == "@@ -1,2 +1,2 @@"
    assert "def get" in file.hunks[0].content


def test_multiple_files() -> None:
    """A diff touching multiple files yields one DiffFile per file."""
    diff_text = (
        "diff --git a/src/a.py b/src/a.py\n"
        "--- a/src/a.py\n"
        "+++ b/src/a.py\n"
        "@@ -1 +1 @@\n"
        "-old_a\n"
        "+new_a\n"
        "diff --git a/src/b.js b/src/b.js\n"
        "--- a/src/b.js\n"
        "+++ b/src/b.js\n"
        "@@ -1 +1 @@\n"
        "-old_b\n"
        "+new_b\n"
    )
    parsed = parse_diff(_make_state(diff_text))["diff"]

    assert [f.path for f in parsed.files] == ["src/a.py", "src/b.js"]
    assert [f.language for f in parsed.files] == ["python", "javascript"]


def test_multiple_hunks_in_one_file() -> None:
    """A single file with two non-contiguous changes yields two hunks."""
    diff_text = (
        "diff --git a/src/x.py b/src/x.py\n"
        "--- a/src/x.py\n"
        "+++ b/src/x.py\n"
        "@@ -1,2 +1,2 @@\n"
        " line_one\n"
        "-old_two\n"
        "+new_two\n"
        "@@ -10,2 +10,2 @@\n"
        " line_ten\n"
        "-old_eleven\n"
        "+new_eleven\n"
    )
    parsed = parse_diff(_make_state(diff_text))["diff"]

    assert len(parsed.files) == 1
    headers = [h.header for h in parsed.files[0].hunks]
    assert headers == ["@@ -1,2 +1,2 @@", "@@ -10,2 +10,2 @@"]


def test_unknown_extension_returns_none_language() -> None:
    """Files with extensions not in the map have language=None."""
    diff_text = (
        "diff --git a/data/raw.bin b/data/raw.bin\n"
        "--- a/data/raw.bin\n"
        "+++ b/data/raw.bin\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    parsed = parse_diff(_make_state(diff_text))["diff"]

    assert parsed.files[0].language is None