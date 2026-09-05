"""Tests for the retrieval quality metrics."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.evaluation import (
    average_precision,
    mean_average_precision,
    precision_at_k,
    recall_at_k,
)


@pytest.mark.parametrize(
    ("retrieved", "relevant", "k", "expected"),
    [
        ([1, 2, 3], {1, 2, 3}, 3, 1.0),
        ([1, 2, 3], {1, 3}, 3, 2 / 3),
        ([1, 2, 3], set[int](), 3, 0.0),
        ([4, 5, 6], {1, 2, 3}, 3, 0.0),
        ([1, 2, 3], {1}, 1, 1.0),
        ([2, 1, 3], {1}, 1, 0.0),
        # Divided by what was returned, not by k, so three good hits out of a
        # possible ten score 1.0 rather than 0.3.
        ([1, 2], {1, 2}, 10, 1.0),
        ([], {1}, 5, 0.0),
        ([1], {1}, 0, 0.0),
    ],
)
def test_precision_at_k(
    retrieved: list[int], relevant: set[int], k: int, expected: float
) -> None:
    assert precision_at_k(retrieved, relevant, k) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("retrieved", "relevant", "k", "expected"),
    [
        ([1, 2, 3], {1, 2, 3}, 3, 1.0),
        ([1, 2, 3], {1, 2, 3, 4, 5, 6}, 3, 0.5),
        ([1, 2, 3], {1, 3}, 3, 1.0),
        ([1, 2, 3], {1, 3}, 1, 0.5),
        ([1, 2, 3], set[int](), 3, 0.0),
        ([], {1, 2}, 5, 0.0),
        ([1], {1}, 0, 0.0),
    ],
)
def test_recall_at_k(
    retrieved: list[int], relevant: set[int], k: int, expected: float
) -> None:
    assert recall_at_k(retrieved, relevant, k) == pytest.approx(expected)


def test_precision_and_recall_pull_against_each_other() -> None:
    """Returning everything maximises recall and ruins precision."""
    relevant = {1, 2}
    everything = list(range(1, 21))
    assert recall_at_k(everything, relevant, 20) == 1.0
    assert precision_at_k(everything, relevant, 20) == pytest.approx(0.1)
    assert precision_at_k([1], relevant, 1) == 1.0
    assert recall_at_k([1], relevant, 1) == 0.5


@pytest.mark.parametrize(
    ("retrieved", "relevant", "expected"),
    [
        # Both relevant, both at the top: perfect.
        ([1, 2, 3], {1, 2}, 1.0),
        # Hits at rank 1 and 3: (1/1 + 2/3) / 2
        ([1, 2, 3], {1, 3}, (1.0 + 2 / 3) / 2),
        # Same two documents, worse order: (1/2 + 2/3) / 2. Strictly lower.
        ([2, 1, 3], {1, 3}, (0.5 + 2 / 3) / 2),
        ([1, 2], set[int](), 0.0),
        ([], {1}, 0.0),
        # One of two relevant documents never retrieved, so the miss costs.
        ([1], {1, 99}, 0.5),
    ],
)
def test_average_precision(
    retrieved: list[int], relevant: set[int], expected: float
) -> None:
    assert average_precision(retrieved, relevant) == pytest.approx(expected)


def test_average_precision_is_the_metric_that_sees_order() -> None:
    """Precision@k cannot distinguish these two; average precision can."""
    relevant = {1, 3}
    better = [1, 3, 2]
    worse = [2, 1, 3]
    assert precision_at_k(better, relevant, 3) == precision_at_k(worse, relevant, 3)
    assert recall_at_k(better, relevant, 3) == recall_at_k(worse, relevant, 3)
    assert average_precision(better, relevant) > average_precision(worse, relevant)


def test_mean_average_precision() -> None:
    assert mean_average_precision([]) == 0.0
    assert mean_average_precision([([1], {1}), ([2], {9})]) == pytest.approx(0.5)


@given(
    st.lists(st.integers(0, 20), unique=True, max_size=12),
    st.sets(st.integers(0, 20), max_size=12),
    st.integers(0, 15),
)
def test_every_metric_stays_within_zero_and_one(
    retrieved: list[int], relevant: set[int], k: int
) -> None:
    assert 0.0 <= precision_at_k(retrieved, relevant, k) <= 1.0
    assert 0.0 <= recall_at_k(retrieved, relevant, k) <= 1.0
    assert 0.0 <= average_precision(retrieved, relevant) <= 1.0


@given(
    st.lists(st.integers(0, 20), unique=True, max_size=12),
    st.sets(st.integers(0, 20), min_size=1, max_size=12),
)
def test_recall_never_falls_as_k_grows(
    retrieved: list[int], relevant: set[int]
) -> None:
    """More results can only find more of what was relevant."""
    values = [recall_at_k(retrieved, relevant, k) for k in range(len(retrieved) + 1)]
    assert values == sorted(values)
