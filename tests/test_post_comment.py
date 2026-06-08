"""Tests for the post_comment node.

Covers: happy path (success status + URL stored), poster invocation (correct
PR and body passed), failure outcome (FAILED status without URL), and the
short-circuit when review_body is unset.
"""

import pytest

from sentry.nodes.post_comment import make_post_comment_node
from sentry.posting import NoopPoster
from sentry.state import AgentState, PostStatus, PRMetadata


def _make_state(*, with_review: bool = True) -> AgentState:
    return AgentState(
        pr=PRMetadata(
            repo="acme/x",
            pr_number=42,
            head_sha="a",
            base_sha="b",
            author="alice",
            title="Add user lookup",
        ),
        raw_diff="(unused)",
        review_body=(
            "## Sentry Code Review\n\nNo findings." if with_review else None
        ),
    )


def test_happy_path_records_success_status_and_url() -> None:
    """A successful post writes SUCCESS to post_status and a URL string."""
    poster = NoopPoster()
    result = make_post_comment_node(poster)(_make_state())

    assert result["post_status"] is PostStatus.SUCCESS
    url = result["post_url"]
    assert isinstance(url, str)
    assert "acme/x/pull/42" in url


def test_poster_receives_pr_and_review_body() -> None:
    """The node passes the PR metadata and review body through to the poster."""
    poster = NoopPoster()
    make_post_comment_node(poster)(_make_state())

    assert len(poster.calls) == 1
    pr, body = poster.calls[0]
    assert pr.repo == "acme/x"
    assert pr.pr_number == 42
    assert body.startswith("## Sentry Code Review")


def test_failed_poster_returns_failed_status_without_url() -> None:
    """When the poster returns FAILED, post_status reflects it and url is None."""
    poster = NoopPoster(force_error="403 Forbidden")
    result = make_post_comment_node(poster)(_make_state())

    assert result["post_status"] is PostStatus.FAILED
    assert result["post_url"] is None


def test_missing_review_body_raises_without_calling_poster() -> None:
    """Node refuses to run, and does not call the poster, when review_body is None."""
    poster = NoopPoster()
    node = make_post_comment_node(poster)

    with pytest.raises(ValueError, match="format_review"):
        node(_make_state(with_review=False))

    assert poster.calls == []