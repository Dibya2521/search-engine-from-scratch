"""Tests for the retrieval quality metrics."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.evaluation import (
    average_precision,
    discounted_cumulative_gain,
    mean_average_precision,
    mean_reciprocal_rank,
    normalized_discounted_cumulative_gain,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    sign_test,
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


@pytest.mark.parametrize(
    ("retrieved", "relevant", "expected"),
    [
        ([1, 2, 3], {1}, 1.0),
        ([1, 2, 3], {2}, 0.5),
        ([1, 2, 3], {3}, 1 / 3),
        ([1, 2, 3], {2, 3}, 0.5),
        ([1, 2, 3], {9}, 0.0),
        ([], {1}, 0.0),
        ([1], set[int](), 0.0),
    ],
)
def test_reciprocal_rank(
    retrieved: list[int], relevant: set[int], expected: float
) -> None:
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(expected)


def test_mean_reciprocal_rank_averages_across_queries() -> None:
    assert mean_reciprocal_rank([([1, 2], {1}), ([1, 2], {2})]) == pytest.approx(0.75)


def test_mean_reciprocal_rank_of_nothing_is_zero() -> None:
    assert mean_reciprocal_rank([]) == 0.0


def test_gain_is_discounted_by_the_logarithm_of_the_rank() -> None:
    """Rank 1 divides by log2(2) = 1, rank 3 by log2(4) = 2."""
    assert discounted_cumulative_gain([5, 9, 6], {5: 3, 6: 2}, 3) == pytest.approx(
        3 / 1.0 + 2 / 2.0
    )


def test_gain_counts_an_unjudged_document_as_worthless() -> None:
    assert discounted_cumulative_gain([9], {1: 3}, 5) == 0.0


def test_gain_stops_at_k() -> None:
    assert discounted_cumulative_gain([9, 1], {1: 3}, 1) == 0.0


def test_gain_of_no_results_is_zero() -> None:
    assert discounted_cumulative_gain([], {1: 3}, 5) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_gain_needs_a_positive_k(k: int) -> None:
    assert discounted_cumulative_gain([1], {1: 3}, k) == 0.0


def test_the_ideal_ranking_normalizes_to_one() -> None:
    grades = {1: 3, 2: 2, 3: 1}
    assert normalized_discounted_cumulative_gain([1, 2, 3], grades, 3) == pytest.approx(
        1.0
    )


def test_a_reversed_ranking_scores_below_the_ideal() -> None:
    grades = {1: 3, 2: 2, 3: 1}
    assert normalized_discounted_cumulative_gain([3, 2, 1], grades, 3) < 1.0


def test_normalizing_without_any_grades_is_zero() -> None:
    assert normalized_discounted_cumulative_gain([1, 2], {}, 3) == 0.0


def test_the_ideal_ranking_includes_documents_that_were_missed() -> None:
    """Otherwise failing to return a relevant document would cost nothing."""
    grades = {1: 3, 2: 3}
    assert normalized_discounted_cumulative_gain([1], grades, 5) < 1.0


def test_graded_relevance_sees_an_ordering_that_binary_metrics_cannot() -> None:
    """The whole reason to grade: both orderings retrieve the same set."""
    grades = {1: 3, 2: 1}
    better = normalized_discounted_cumulative_gain([1, 2], grades, 2)
    worse = normalized_discounted_cumulative_gain([2, 1], grades, 2)
    relevant = {1, 2}
    assert average_precision([1, 2], relevant) == average_precision([2, 1], relevant)
    assert better > worse


@given(
    st.lists(st.integers(0, 20), unique=True, max_size=12),
    st.dictionaries(st.integers(0, 20), st.integers(0, 3), max_size=12),
    st.integers(0, 15),
)
def test_normalized_gain_stays_within_zero_and_one(
    retrieved: list[int], grades: dict[int, int], k: int
) -> None:
    assert 0.0 <= normalized_discounted_cumulative_gain(retrieved, grades, k) <= 1.0


@pytest.mark.parametrize(
    ("differences", "expected"),
    [
        ([], 1.0),
        ([0.0, 0.0], 1.0),
        ([1.0], 1.0),
        ([1.0, 1.0], 0.5),
        ([1.0, -1.0], 1.0),
        ([1.0] * 10, 2 / 2**10),
        # Ties carry no information and are dropped before counting.
        ([1.0, 1.0, 0.0, 0.0, 0.0], 0.5),
    ],
)
def test_sign_test(differences: list[float], expected: float) -> None:
    assert sign_test(differences) == pytest.approx(expected)


def test_the_sign_test_ignores_how_large_a_difference_is() -> None:
    """Its defining property, and equally its weakness."""
    assert sign_test([0.001, 0.001, -50.0]) == sign_test([50.0, 50.0, -0.001])


@given(st.lists(st.floats(-10, 10), max_size=20))
def test_a_p_value_is_always_a_probability(differences: list[float]) -> None:
    assert 0.0 <= sign_test(differences) <= 1.0
