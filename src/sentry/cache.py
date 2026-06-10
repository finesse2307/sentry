"""SQLite-backed response cache for LLMClient.

Wraps any ``LLMClient`` and memoizes responses keyed on the request inputs
(messages, system, tools, max_tokens, namespace). Cache hits do not call the
inner client and report zero token usage, so they pass through a budget
tracker as free.

This is OUR cache — request-level response memoization. It is unrelated to
Anthropic's server-side prompt caching, which deduplicates large prefixes
across separate API calls. The two are composable but only this one is used
for now.
"""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from sentry.llm import LLMClient, LLMResponse, Message, ToolDef, Usage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


class SQLiteCacheLLMClient:
    """LLMClient wrapper that caches responses in a SQLite file.

    Cache key is a SHA-256 of a canonical JSON encoding of the request fields.
    Cache hits return the stored response with usage zeroed, so downstream
    spend trackers correctly treat them as free.

    ``namespace`` lets you partition the cache (e.g. by model identifier or
    prompt version) so swapping the inner client doesn't surface stale entries.
    """

    def __init__(
        self,
        inner: LLMClient,
        *,
        db_path: Path,
        namespace: str = "",
    ) -> None:
        self._inner = inner
        self._db_path = db_path
        self._namespace = namespace
        self.hits: int = 0
        self.misses: int = 0
        self._init_db()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(_SCHEMA)

    def _key(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[ToolDef] | None,
        max_tokens: int,
    ) -> str:
        payload = {
            "namespace": self._namespace,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "system": system,
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in (tools or [])
            ],
            "max_tokens": max_tokens,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _lookup(self, key: str) -> LLMResponse | None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT response_json FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        response_json: str = row[0]
        return LLMResponse.model_validate_json(response_json)

    def _store(self, key: str, response: LLMResponse) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache "
                "(key, response_json, created_at) VALUES (?, ?, ?)",
                (
                    key,
                    response.model_dump_json(),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        key = self._key(messages, system, tools, max_tokens)

        cached = self._lookup(key)
        if cached is not None:
            self.hits += 1
            # Zero usage so budget trackers don't double-count cached calls.
            return cached.model_copy(update={"usage": Usage()})

        self.misses += 1
        response = self._inner.complete(
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
        )
        self._store(key, response)
        return response