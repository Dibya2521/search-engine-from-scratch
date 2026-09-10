"""Tests for BM25 scoring."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.bm25 import K1, B, BM25Ranker, bm25_inverse_document_frequency
from search_engine.index import InvertedIndex
from search_engine.ranking import Ranker, StaleRankerError
from tests.conftest import RANKING_CORPUS, build_index


def index_of(corpus: dict[int, str]) -> InvertedIndex:
    return build_index(corpus)


def test_an_unindexed_term_weighs_nothing(ranking_index: InvertedIndex) -> None:
    assert bm25_inverse_document_frequency(ranking_index, "absent") == 0.0


def test_a_rarer_term_weighs_more(ranking_index: InvertedIndex) -> None:
    common = bm25_inverse_document_frequency(ranking_index, "comput")
    rare = bm25_inverse_document_frequency(ranking_index, "garden")
    assert rare > common


def test_the_weight_stays_positive_for_a_term_in_every_document() -> None:
    """The classic formulation goes negative here, penalising a real match."""
    index = index_of({1: "alpha", 2: "alpha", 3: "alpha"})
    assert bm25_inverse_document_frequency(index, "alpha") > 0.0


@given(st.integers(1, 500), st.integers(1, 500))
def test_the_weight_is_never_negative(total: int, frequency: int) -> None:
    """The property the added constant exists to guarantee."""
    present = min(total, frequency)
    index = index_of(dict.fromkeys(range(present), "alpha"))
    for extra in range(present, total):
        index.add_document(extra, "beta")
    assert bm25_inverse_document_frequency(index, "alpha") >= 0.0


def test_term_frequency_saturates() -> None:
    """The property BM25 exists for: each repeat adds less than the last."""
    index = index_of(
        {
            1: "alpha",
            2: "alpha alpha",
            3: "alpha alpha alpha",
            4: "alpha alpha alpha alpha",
            5: "beta",
        }
    )
    ranker = BM25Ranker(index)
    weights = ranker.query_weights(["alpha"])
    scores = [ranker.score(weights, one) for one in (1, 2, 3, 4)]
    gains = [second - first for first, second in pairwise(scores)]
    assert all(gain > 0 for gain in gains)
    assert gains == sorted(gains, reverse=True)


def test_tf_idf_does_not_saturate() -> None:
    """The contrast that motivates BM25, asserted rather than assumed."""
    index = index_of(
        {
            1: "alpha",
            2: "alpha alpha",
            3: "alpha alpha alpha",
            4: "alpha alpha alpha alpha",
            5: "beta",
        }
    )
    ranker = Ranker(index)
    weights = ranker.query_weights(["alpha"])
    scores = [ranker.score(weights, one) for one in (1, 2, 3, 4)]
    assert len(set(scores)) == 1


def test_a_shorter_document_wins_at_equal_term_frequency() -> None:
    """What the length correction buys."""
    index = index_of(
        {
            1: "alpha beta",
            2: "alpha " + "gamma " * 40,
            3: "delta",
        }
    )
    ranker = BM25Ranker(index)
    weights = ranker.query_weights(["alpha"])
    assert ranker.score(weights, 1) > ranker.score(weights, 2)


def test_length_correction_can_be_switched_off() -> None:
    index = index_of(
        {
            1: "alpha beta",
            2: "alpha " + "gamma " * 40,
            3: "delta",
        }
    )
    ranker = BM25Ranker(index, b=0.0)
    weights = ranker.query_weights(["alpha"])
    assert ranker.score(weights, 1) == pytest.approx(ranker.score(weights, 2))


def test_zero_saturation_ignores_term_frequency() -> None:
    """With k1 at zero only the inverse document frequency survives."""
    index = index_of({1: "alpha", 2: "alpha alpha alpha alpha", 3: "beta"})
    ranker = BM25Ranker(index, k1=0.0)
    weights = ranker.query_weights(["alpha"])
    assert ranker.score(weights, 1) == pytest.approx(ranker.score(weights, 2))


def test_a_document_without_the_term_scores_zero() -> None:
    index = index_of({1: "alpha", 2: "beta"})
    ranker = BM25Ranker(index)
    assert ranker.score(ranker.query_weights(["alpha"]), 2) == 0.0


def test_an_empty_index_scores_nothing() -> None:
    ranker = BM25Ranker(InvertedIndex())
    assert ranker.rank("alpha", [1, 2]) == []
    assert ranker.score(ranker.query_weights(["alpha"]), 1) == 0.0


def test_a_corpus_of_empty_documents_has_no_average_length() -> None:
    """Documents holding no terms still count towards the corpus size."""
    ranker = BM25Ranker(index_of({1: "", 2: ""}))
    assert ranker.score(ranker.query_weights(["alpha"]), 1) == 0.0


def test_query_weights_ignore_repetition() -> None:
    ranker = BM25Ranker(index_of(RANKING_CORPUS))
    assert ranker.query_weights(["comput"]) == ranker.query_weights(
        [
            "comput",
            "comput",
        ]
    )


def test_adding_a_document_invalidates_the_ranker() -> None:
    index = index_of({1: "alpha", 2: "beta"})
    ranker = BM25Ranker(index)
    weights = ranker.query_weights(["alpha"])
    index.add_document(3, "gamma")
    with pytest.raises(StaleRankerError):
        ranker.score(weights, 1)


def test_ranking_returns_best_first(ranking_index: InvertedIndex) -> None:
    ranked = BM25Ranker(ranking_index).rank("computer science", [1, 2, 3, 4, 5])
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0][0] == 1


def test_ranking_an_empty_query_returns_nothing(ranking_index: InvertedIndex) -> None:
    assert BM25Ranker(ranking_index).rank("", [1, 2, 3]) == []


def test_a_repeated_word_still_outranks_an_unrelated_document() -> None:
    """The falsifying case: BM25 dampens keyword stuffing, it does not stop it.

    Document 4 is one query term repeated five times and contains nothing about
    the other. It ranks second under both scorers, because it does genuinely
    contain a query term many times. What BM25 changes is the margin: its score
    falls from 72 percent of the winner's to 52 percent.
    """
    index = index_of(RANKING_CORPUS)
    candidates = [1, 2, 3, 4, 5]
    tfidf = dict(Ranker(index).rank("computer science", candidates))
    okapi = dict(BM25Ranker(index).rank("computer science", candidates))
    assert tfidf[4] / tfidf[1] == pytest.approx(0.72, abs=0.01)
    assert okapi[4] / okapi[1] == pytest.approx(0.52, abs=0.01)


def test_the_defaults_are_the_conventional_ones() -> None:
    assert (K1, B) == (1.2, 0.75)


@given(
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=8),
    st.text(min_size=1, max_size=20),
)
def test_no_score_is_ever_negative(texts: list[str], query: str) -> None:
    """A negative score would mean a match made a document worse."""
    index = index_of(dict(enumerate(texts)))
    ranker = BM25Ranker(index)
    weights = ranker.query_weights([query])
    assert all(
        ranker.score(weights, document_id) >= 0.0 for document_id in range(len(texts))
    )


def test_the_formula_matches_a_hand_computation() -> None:
    """Independent arithmetic, so a refactor cannot quietly change the maths."""
    index = index_of({1: "alpha alpha beta", 2: "gamma"})
    ranker = BM25Ranker(index)
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    average = (3 + 1) / 2
    correction = 1 - B + B * 3 / average
    expected = idf * (2 * (K1 + 1)) / (2 + K1 * correction)
    assert ranker.score(ranker.query_weights(["alpha"]), 1) == pytest.approx(expected)


def assert_bounds_hold(corpus: dict[int, str]) -> None:
    """No document may score above the bound early termination prunes with."""
    index = index_of(corpus)
    ranker = BM25Ranker(index)
    for term in index.terms:
        postings = index.postings(term)
        highest = max(len(positions) for positions in postings.values())
        bound = ranker.upper_bound(term, highest)
        weights = ranker.query_weights([term])
        for document_id in postings:
            assert ranker.score(weights, document_id) <= bound


def test_the_upper_bound_holds_on_the_ranking_corpus() -> None:
    assert_bounds_hold(RANKING_CORPUS)


def test_the_upper_bound_holds_when_document_lengths_vary_widely() -> None:
    """The shortest document sets the bound, so a wide spread is the hard case."""
    assert_bounds_hold(
        {
            1: "alpha",
            2: " ".join(["alpha"] * 20),
            3: " ".join(["alpha", *["beta"] * 40]),
            4: "beta",
        }
    )


def test_the_upper_bound_rises_with_the_frequency_it_is_given() -> None:
    ranker = BM25Ranker(index_of(RANKING_CORPUS))
    assert ranker.upper_bound("comput", 5) > ranker.upper_bound("comput", 1)


def test_an_unindexed_term_bounds_nothing() -> None:
    ranker = BM25Ranker(index_of(RANKING_CORPUS))
    assert ranker.upper_bound("absent", 3) == 0.0
