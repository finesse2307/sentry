"""Recompute hunk headers in evals/eval_set.json based on actual body lines.

Reads each case's ``diff``, walks each hunk, counts context/removed/added
lines, and rewrites the header. Drops trailing artifacts: empty strings (from
trailing newlines) and bare lone "+" or "-" characters (off-by-one edits).

Idempotent on a clean file. Run after any manual edit of the eval set.
"""

import json
import re
import sys
from pathlib import Path

_HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)


def fix_diff(diff: str) -> str:
    """Return ``diff`` with all hunk headers recomputed from body line counts."""
    lines = diff.split("\n")
    out: list[str] = []

    i = 0
    while i < len(lines):
        header_match = _HUNK_HEADER.match(lines[i])
        if not header_match:
            out.append(lines[i])
            i += 1
            continue

        old_start = int(header_match.group(1))
        new_start = int(header_match.group(3))
        suffix = header_match.group(5)

        # Collect body until the next header or end.
        body: list[str] = []
        j = i + 1
        while j < len(lines) and not _HUNK_HEADER.match(lines[j]):
            body.append(lines[j])
            j += 1

        # Strip trailing artifacts: empty strings and lone "+"/"-".
        while body and (body[-1] == "" or body[-1] in ("+", "-")):
            body.pop()

        old_count = 0
        new_count = 0
        for b in body:
            if not b:
                continue
            prefix = b[0]
            if prefix == " ":
                old_count += 1
                new_count += 1
            elif prefix == "-":
                old_count += 1
            elif prefix == "+":
                new_count += 1
            # Any other prefix is malformed; not counted.

        out.append(
            f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{suffix}"
        )
        out.extend(body)
        i = j

    return "\n".join(out)


def main() -> int:
    path = Path("evals/eval_set.json")
    data = json.loads(path.read_text())
    for case in data["cases"]:
        case["diff"] = fix_diff(case["diff"])
    path.write_text(json.dumps(data, indent=4) + "\n")
    print(f"Rewrote hunk headers for {len(data['cases'])} cases in {path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())