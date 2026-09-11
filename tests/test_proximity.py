"""Tests for proximity scoring.

The load-bearing test is `test_the_span_matches_an_exhaustive_search`. The sweep
visits each position once and the definition considers every combination of one
position per term, so only a differential test over generated input catches the
sweep taking a wrong turn.
"""

from __future__ import annotations

import itertools
import time

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.bm25 import BM25Ranker
from search_engine.index import InvertedIndex
from search_engine.proximity import (
    MINIMUM_TERMS,
    PROXIMITY_WEIGHT,
    minimum_span,
    proximity_boost,
)
from search_engine.query import search

LINEAR_TIME_BUDGET = 2.0


def exhaustive_span(position_lists: list[list[int]]) -> int | None:
    """The definition: the narrowest window over every choice of one per term."""
    if not position_lists or any(not positions for positions in position_lists):
        return None
    return min(
        max(chosen) - min(chosen) + 1 for chosen in itertools.product(*position_lists)
    )


@pytest.mark.parametrize(
    ("lists", "expected"),
    [
        ([[5], [6]], 2),
        ([[1, 5], [3]], 3),
        ([[3]], 1),
        ([[0, 10, 20], [11], [21]], 11),
        ([[1, 2, 3], [100]], 98),
        ([[1], [2], [3]], 3),
        ([[5, 6, 7], [1, 2, 3]], 3),
    ],
)
def test_the_span_on_known_cases(lists: list[list[int]], expected: int) -> None:
    assert minimum_span(lists) == expected


@given(
    st.lists(
        st.lists(st.integers(min_value=0, max_value=40), min_size=1, max_size=6).map(
            lambda positions: sorted(set(positions))
        ),
        min_size=1,
        max_size=4,
    )
)
def test_the_span_matches_an_exhaustive_search(position_lists: list[list[int]]) -> None:
    """The sweep against the definition it implements."""
    assert minimum_span(position_lists) == exhaustive_span(position_lists)


@given(
    st.lists(
        st.integers(min_value=0, max_value=200), min_size=2, max_size=8, unique=True
    )
)
def test_the_span_is_never_narrower_than_the_number_of_terms(
    positions: list[int],
) -> None:
    """Each term occupies its own position, so a window must fit them all.

    Positions are made distinct because two terms cannot share a position in one
    document: a position holds exactly one token.
    """
    lists = [[position] for position in sorted(positions)]
    span = minimum_span(lists)
    assert span is not None
    assert span >= len(lists)


def test_no_terms_have_no_window() -> None:
    assert minimum_span([]) is None


def test_a_term_with_no_positions_has_no_window() -> None:
    """A window containing every term cannot exist if one term is absent."""
    assert minimum_span([[1, 2], []]) is None


def test_all_of_one_term_before_all_of_another() -> None:
    """Adversarial: no interleaving at all, so the sweep never alternates."""
    assert minimum_span([[0, 1, 2], [50, 51, 52]]) == 49


def test_a_term_occurring_ten_thousand_times_stays_linear() -> None:
    """Adversarial: a quadratic sweep would not finish inside this budget."""
    crowded = list(range(0, 20_000, 2))
    start = time.perf_counter()
    assert minimum_span([crowded, [19_999]]) == 2
    assert time.perf_counter() - start < LINEAR_TIME_BUDGET


def test_adjacent_terms_earn_the_full_boost() -> None:
    """A span equal to the term count is the narrowest a window can be."""
    assert proximity_boost(MINIMUM_TERMS, MINIMUM_TERMS) == 1.0 + PROXIMITY_WEIGHT
    assert proximity_boost(3, 3) == 1.0 + PROXIMITY_WEIGHT


def test_distant_terms_earn_almost_nothing() -> None:
    boost = proximity_boost(1_000, 2)
    assert 1.0 < boost < 1.001


def test_a_single_term_is_never_boosted() -> None:
    """Proximity between a term and itself is not a question."""
    assert proximity_boost(1, 1) == 1.0
    assert proximity_boost(None, 1) == 1.0


def test_terms_that_never_meet_are_not_penalised() -> None:
    """The multiplier floors at 1.0, so proximity only ever rewards."""
    assert proximity_boost(None, 3) == 1.0


@given(
    st.integers(min_value=1, max_value=500),
    st.integers(min_value=MINIMUM_TERMS, max_value=6),
)
def test_the_boost_is_always_between_one_and_the_weight(span: int, terms: int) -> None:
    """It can never reduce a score, and can never exceed the stated weight."""
    boost = proximity_boost(max(span, terms), terms)
    assert 1.0 < boost <= 1.0 + PROXIMITY_WEIGHT


def test_a_narrower_window_never_scores_lower() -> None:
    boosts = [proximity_boost(span, 2) for span in range(2, 30)]
    assert boosts == sorted(boosts, reverse=True)


def test_the_boost_lifts_a_document_whose_terms_sit_together() -> None:
    """The whole point, end to end through the ranker."""
    index = InvertedIndex()
    index.add_document(1, "machine learning is the subject here")
    index.add_document(2, "machine " + "filler " * 40 + "learning")
    candidates = search(index, "machine learning")
    plain = BM25Ranker(index).rank("machine learning", candidates)
    boosted = BM25Ranker(index, proximity=True).rank("machine learning", candidates)
    assert dict(boosted)[1] > dict(plain)[1]
    assert dict(boosted)[2] == pytest.approx(dict(plain)[2], rel=0.02)


def test_ranking_without_the_boost_is_unchanged() -> None:
    """Off by default, so every existing caller must see what it saw before."""
    index = InvertedIndex()
    index.add_document(1, "machine learning")
    index.add_document(2, "machine and then later learning")
    candidates = search(index, "machine learning")
    assert BM25Ranker(index).rank("machine learning", candidates) == BM25Ranker(
        index, proximity=False
    ).rank("machine learning", candidates)


def test_a_single_term_query_ranks_identically_either_way() -> None:
    """There is no proximity to measure, so the two must not diverge."""
    index = InvertedIndex()
    index.add_document(1, "machine machine machine")
    index.add_document(2, "machine")
    candidates = search(index, "machine")
    assert BM25Ranker(index).rank("machine", candidates) == BM25Ranker(
        index, proximity=True
    ).rank("machine", candidates)
