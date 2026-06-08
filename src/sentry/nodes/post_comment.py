"""post_comment node: post the formatted review back to the source-control host.

Wraps a CommentPoster (NoopPoster in Phase 1, GitHubPoster in Phase 4). Returns
state updates for ``post_status`` and ``post_url``. Posters are contractually
required to return FAILED rather than raise; an unhandled exception here
indicates a broken poster, not a runtime condition.
"""

from collections.abc import Callable

from sentry.posting import CommentPoster
from sentry.state import AgentState, PostStatus


def make_post_comment_node(
    poster: CommentPoster,
) -> Callable[[AgentState], dict[str, PostStatus | str | None]]:
    """Build a post_comment-node bound to a specific poster."""

    def post_comment_node(
        state: AgentState,
    ) -> dict[str, PostStatus | str | None]:
        if state.review_body is None:
            raise ValueError(
                "post_comment_node requires state.review_body to be populated; "
                "did format_review run?"
            )

        result = poster.post(pr=state.pr, body=state.review_body)
        return {
            "post_status": result.status,
            "post_url": result.url,
        }

    return post_comment_node