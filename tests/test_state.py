"""Tests for the AgentState schema and its sub-models.

These tests lock in the contract: what's required, what defaults, what
serializes how, and what invalid input is rejected.
"""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from sentry.state import (
    AgentState,
    Category,
    Finding,
    PostStatus,
    PRMetadata,
    Severity,
)


def _sample_pr() -> PRMetadata:
    return PRMetadata(
        repo="acme/widgets",
        pr_number=42,
        head_sha="abc",
        base_sha="def",
        author="alice",
        title="Add user lookup",
    )


def test_minimal_state_only_requires_pr() -> None:
    """Constructing with only ``pr`` should succeed; other fields default."""
    state = AgentState(pr=_sample_pr())

    assert state.diff is None
    assert state.plan is None
    assert state.tool_results == []
    assert state.findings == []
    assert state.review_body is None
    assert state.post_status is PostStatus.PENDING
    assert state.post_url is None


def test_meta_auto_populates() -> None:
    """``meta`` should be created with a uuid run_id and a tz-aware timestamp."""
    state = AgentState(pr=_sample_pr())

    assert state.meta.run_id  # non-empty
    assert len(state.meta.run_id) == 36  # standard uuid4 string length
    assert isinstance(state.meta.started_at, datetime)
    assert state.meta.started_at.tzinfo is not None  # timezone-aware


def test_state_without_pr_raises() -> None:
    """``pr`` has no default; omitting it must raise ValidationError."""
    with pytest.raises(ValidationError):
        AgentState()  # type: ignore[call-arg]


def test_finding_rejects_unknown_category() -> None:
    """Categories are restricted to the Category enum."""
    with pytest.raises(ValidationError):
        Finding(
            category="cosmic-rays",  # type: ignore[arg-type]
            severity=Severity.LOW,
            file="src/foo.py",
            message="not a real category",
        )


def test_state_round_trips_through_json() -> None:
    """Serializing to JSON and back should preserve the state."""
    original = AgentState(
        pr=_sample_pr(),
        findings=[
            Finding(
                category=Category.SECURITY,
                severity=Severity.HIGH,
                file="src/users.py",
                line=12,
                message="SQL injection via f-string",
            )
        ],
    )

    payload = original.model_dump_json()
    restored = AgentState.model_validate_json(payload)

    assert restored == original
    decoded = json.loads(payload)
    assert decoded["findings"][0]["category"] == "security"
    assert decoded["findings"][0]["severity"] == "high"