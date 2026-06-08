"""Abstraction for posting review comments back to the source-control host.

Phase 1 uses ``NoopPoster`` (records the attempt, doesn't network). Phase 4
will add ``GitHubPoster`` (real GitHub API). The ``CommentPoster`` protocol is
the seam between the agent and the outside world. Implementations are
expected to wrap their own errors into a FAILED ``CommentResult`` rather than
raising.
"""

from typing import Protocol

from pydantic import BaseModel

from sentry.state import PostStatus, PRMetadata


class CommentResult(BaseModel):
    """Outcome of one post attempt."""

    status: PostStatus
    url: str | None = None
    error: str | None = None


class CommentPoster(Protocol):
    """Structural contract for posting a markdown review comment.

    Implementations must not raise. Network/HTTP errors should be caught and
    returned as ``CommentResult(status=PostStatus.FAILED, error=...)``.
    """

    def post(self, *, pr: PRMetadata, body: str) -> CommentResult: ...


class NoopPoster:
    """A poster that records the attempt without making any network call.

    For Phase 1 and tests. Configure ``force_error`` to simulate failure paths.
    """

    def __init__(self, *, force_error: str | None = None) -> None:
        self.force_error = force_error
        self.calls: list[tuple[PRMetadata, str]] = []

    def post(self, *, pr: PRMetadata, body: str) -> CommentResult:
        self.calls.append((pr, body))
        if self.force_error is not None:
            return CommentResult(
                status=PostStatus.FAILED, error=self.force_error
            )
        return CommentResult(
            status=PostStatus.SUCCESS,
            url=f"https://github.com/{pr.repo}/pull/{pr.pr_number}#sentry-noop",
        )