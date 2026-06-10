"""Tests for SQLiteCacheLLMClient.

Covers: miss → call inner and store, hit → skip inner and return stored,
zero-usage on hit, key sensitivity (different requests → different entries),
namespace isolation, and persistence across separate cache instances.
"""

from pathlib import Path

from sentry.cache import SQLiteCacheLLMClient
from sentry.llm import LLMResponse, Message, MockLLMClient, ToolDef, Usage


def _resp(text: str, in_tok: int = 100, out_tok: int = 50) -> LLMResponse:
    return LLMResponse(
        text=text,
        usage=Usage(input_tokens=in_tok, output_tokens=out_tok),
    )


def test_miss_calls_inner_and_stores(tmp_path: Path) -> None:
    """A cache miss invokes the inner client and persists the response."""
    inner = MockLLMClient([_resp("first")])
    cache = SQLiteCacheLLMClient(inner, db_path=tmp_path / "cache.sqlite")

    response = cache.complete([Message(role="user", content="hello")])

    assert response.text == "first"
    assert cache.misses == 1
    assert cache.hits == 0
    assert len(inner.calls) == 1


def test_hit_does_not_call_inner_and_returns_stored(tmp_path: Path) -> None:
    """An identical second call is served from cache without touching inner."""
    # Only ONE scripted response — a second call to inner would explode.
    inner = MockLLMClient([_resp("once")])
    cache = SQLiteCacheLLMClient(inner, db_path=tmp_path / "cache.sqlite")
    messages = [Message(role="user", content="same prompt")]

    first = cache.complete(messages)
    second = cache.complete(messages)

    assert first.text == "once"
    assert second.text == "once"
    assert cache.hits == 1
    assert cache.misses == 1
    assert len(inner.calls) == 1


def test_hit_returns_zero_usage(tmp_path: Path) -> None:
    """A cached response is returned with usage zeroed for billing purposes."""
    inner = MockLLMClient([_resp("cached", in_tok=1000, out_tok=500)])
    cache = SQLiteCacheLLMClient(inner, db_path=tmp_path / "cache.sqlite")
    messages = [Message(role="user", content="probe")]

    first = cache.complete(messages)
    second = cache.complete(messages)

    # First call: real usage propagated.
    assert first.usage.input_tokens == 1000
    assert first.usage.output_tokens == 500
    # Second call (cache hit): usage zeroed.
    assert second.usage.input_tokens == 0
    assert second.usage.output_tokens == 0


def test_different_inputs_produce_different_keys(tmp_path: Path) -> None:
    """Changing any keyed input results in a fresh inner call."""
    inner = MockLLMClient([_resp("a"), _resp("b"), _resp("c"), _resp("d")])
    cache = SQLiteCacheLLMClient(inner, db_path=tmp_path / "cache.sqlite")

    cache.complete([Message(role="user", content="hi")])  # baseline
    cache.complete([Message(role="user", content="bye")])  # different content
    cache.complete([Message(role="user", content="hi")], system="be brief")  # add system
    cache.complete(
        [Message(role="user", content="hi")],
        tools=[
            ToolDef(name="t", description="d", input_schema={"type": "object"})
        ],
    )  # add tools

    assert cache.misses == 4
    assert cache.hits == 0


def test_namespace_isolation(tmp_path: Path) -> None:
    """Two caches at the same file but different namespaces don't share entries."""
    inner_a = MockLLMClient([_resp("from-a")])
    inner_b = MockLLMClient([_resp("from-b")])
    db = tmp_path / "cache.sqlite"
    cache_a = SQLiteCacheLLMClient(inner_a, db_path=db, namespace="model-a")
    cache_b = SQLiteCacheLLMClient(inner_b, db_path=db, namespace="model-b")

    msg = [Message(role="user", content="same")]
    resp_a = cache_a.complete(msg)
    resp_b = cache_b.complete(msg)

    assert resp_a.text == "from-a"
    assert resp_b.text == "from-b"
    assert cache_a.misses == 1
    assert cache_b.misses == 1


def test_persistence_across_instances(tmp_path: Path) -> None:
    """A second cache instance against the same file sees prior entries."""
    db = tmp_path / "cache.sqlite"

    inner1 = MockLLMClient([_resp("written")])
    cache1 = SQLiteCacheLLMClient(inner1, db_path=db)
    cache1.complete([Message(role="user", content="persist me")])

    # Fresh wrapper, fresh inner with NO scripted responses — would explode if called.
    inner2 = MockLLMClient([])
    cache2 = SQLiteCacheLLMClient(inner2, db_path=db)
    response = cache2.complete([Message(role="user", content="persist me")])

    assert response.text == "written"
    assert cache2.hits == 1
    assert len(inner2.calls) == 0