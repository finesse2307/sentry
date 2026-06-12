"""Validate that every case in evals/eval_set.json parses as a unified diff.

Reports each case that fails to parse, with the underlying error. Run this
after editing the eval set so structural mistakes are caught here rather than
silently corrupting eval runs.
"""

import json
import sys
from pathlib import Path

from unidiff import PatchSet  # type: ignore[import-untyped]
from unidiff.errors import UnidiffParseError  # type: ignore[import-untyped]


def synth(filename: str, diff: str) -> str:
    return (
        f"diff --git a/{filename} b/{filename}\n"
        f"--- a/{filename}\n"
        f"+++ b/{filename}\n"
        f"{diff}\n"
    )


def main() -> int:
    data = json.loads(Path("evals/eval_set.json").read_text())
    cases = data["cases"]
    bad: list[tuple[str, str]] = []

    for case in cases:
        try:
            PatchSet(synth(case["filename"], case["diff"]))
        except UnidiffParseError as exc:
            bad.append((case["id"], str(exc)))

    if not bad:
        print(f"OK: all {len(cases)} cases parse cleanly.")
        return 0

    print(f"FAIL: {len(bad)} of {len(cases)} cases failed to parse:")
    for cid, err in bad:
        print(f"  {cid}: {err}")
    return 1


if __name__ == "__main__":
    sys.exit(main())