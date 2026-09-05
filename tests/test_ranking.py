"""Tests for TF-IDF ranking.

The load-bearing test is `test_the_fast_score_equals_the_full_cosine`. `score`
is an optimisation that touches only query terms, while `cosine_similarity` over
two full vectors is the definition. Asserting they agree on generated input is a
differential test of the optimisation against its own specification, which is
the only thing that can catch the fast path drifting.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.index import InvertedIndex
from search_engine.query import search
from search_engine.ranking import (
    Ranker,
    StaleRankerError,
    cosine_similarity,
    inverse_document_frequency,
    normalize,
)
from tests.conftest import RANKING_CORPUS as CORPUS

TOLERANCE = 1e-9


def _corpus_index() -> InvertedIndex:
    index = InvertedIndex()
    for document_id, text in CORPUS.items():
        index.add_document(document_id, text)
    return index


@pytest.mark.parametrize(
    ("term", "frequency", "expected"),
    [
        ("comput", 3, 0.510826),
        ("scienc", 2, 0.916291),
        ("depart", 1, 1.609438),
        ("nonexistent", 0, 0.0),
    ],
)
def test_inverse_document_frequency(term: str, frequency: int, expected: float) -> None:
    """Rarer terms weigh more, and an unindexed term weighs nothing."""
    index = _corpus_index()
    assert index.document_frequency(term) == frequency
    assert inverse_document_frequency(index, term) == pytest.approx(expected, abs=1e-6)


def test_a_term_in_every_document_is_worth_nothing() -> None:
    """log(N/N) is zero, so such a term cannot move a ranking at all."""
    index = InvertedIndex()
    index.add_document(1, "alpha beta")
    index.add_document(2, "alpha gamma")
    assert inverse_document_frequency(index, "alpha") == 0.0
    assert inverse_document_frequency(index, "beta") > 0.0


@pytest.mark.parametrize("document_id", list(CORPUS))
def test_every_document_vector_has_unit_length(
    ranker: Ranker, document_id: int
) -> None:
    """This is what stops a long document outranking a relevant short one."""
    vector = ranker.document_vector(document_id)
    length = math.sqrt(sum(weight * weight for weight in vector.values()))
    assert length == pytest.approx(1.0, abs=TOLERANCE)


def test_a_document_vector_holds_only_the_terms_present(ranker: Ranker) -> None:
    """Document 4 is one word repeated, so its vector is a single component."""
    assert ranker.document_vector(4) == {"comput": pytest.approx(1.0)}


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("computer science", [(1, 0.6778), (4, 0.4869), (3, 0.2687), (2, 0.1042)]),
        ("computer", [(4, 1.0), (1, 0.3300), (2, 0.2139)]),
        ("science", [(1, 0.5920), (3, 0.3076)]),
        ("about", [(2, 0.2139), (5, 0.1802), (3, 0.1715)]),
        ("gardening", [(5, 0.5679)]),
    ],
)
def test_ranking(ranker: Ranker, query: str, expected: list[tuple[int, float]]) -> None:
    index = _corpus_index()
    actual = ranker.rank(query, search(index, query))
    assert [document_id for document_id, _ in actual] == [
        document_id for document_id, _ in expected
    ]
    for (_, got), (_, want) in zip(actual, expected, strict=True):
        assert got == pytest.approx(want, abs=1e-4)


def test_a_document_of_only_the_query_term_scores_exactly_one(
    ranker: Ranker,
) -> None:
    """Identical direction means a zero angle, and cosine of zero is one."""
    index = _corpus_index()
    assert ranker.rank("computer", search(index, "computer"))[0] == (
        4,
        pytest.approx(1.0, abs=TOLERANCE),
    )


@pytest.mark.parametrize(("query", "expected"), [("nonexistent", []), ("", [])])
def test_queries_that_can_match_nothing(
    ranker: Ranker, query: str, expected: list[tuple[int, float]]
) -> None:
    index = _corpus_index()
    assert ranker.rank(query, search(index, query)) == expected


def test_a_query_of_only_stopwords_ranks_nothing(ranker: Ranker) -> None:
    assert ranker.rank("the and of to", range(1, 6)) == []


@pytest.mark.parametrize("limit", [0, -1])
def test_a_nonpositive_limit_returns_nothing(ranker: Ranker, limit: int) -> None:
    assert ranker.rank("computer science", range(1, 6), limit=limit) == []


def test_the_limit_truncates_from_the_best_end(ranker: Ranker) -> None:
    index = _corpus_index()
    candidates = search(index, "computer science")
    full = ranker.rank("computer science", candidates, limit=10)
    assert ranker.rank("computer science", candidates, limit=2) == full[:2]


def test_zero_scoring_candidates_are_omitted(ranker: Ranker) -> None:
    """Document 5 shares no term with the query, so it is absent rather than 0.0."""
    ranked = ranker.rank("computer science", range(1, 6))
    assert 5 not in [document_id for document_id, _ in ranked]


def test_a_stale_ranker_refuses_to_score() -> None:
    """Its saved document count and vector lengths would both be wrong."""
    index = _corpus_index()
    ranker = Ranker(index)
    index.add_document(99, "a new document arrives")
    with pytest.raises(StaleRankerError, match="built for 5 documents"):
        ranker.rank("computer", [1])


def test_normalizing_an_all_zero_vector_is_a_no_op() -> None:
    """Happens when every query term appears in every document."""
    assert normalize({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}
    assert normalize({}) == {}


@given(st.lists(st.text(), min_size=1, max_size=6))
def test_the_fast_score_equals_the_full_cosine(texts: list[str]) -> None:
    """The optimisation checked against the definition it optimises.

    `score` walks only the query's terms; `cosine_similarity` walks two
    complete unit vectors. They must agree exactly, and this is the only test
    that would notice if the fast path started cutting a corner it should not.
    """
    index = InvertedIndex()
    for document_id, text in enumerate(texts):
        index.add_document(document_id, text)
    ranker = Ranker(index)
    terms = list(index.terms)[:3]
    if not terms:
        return
    weights = ranker.query_weights(terms)
    for document_id in range(len(texts)):
        definitional = cosine_similarity(weights, ranker.document_vector(document_id))
        assert ranker.score(weights, document_id) == pytest.approx(
            definitional, abs=TOLERANCE
        )


@given(st.lists(st.text(), min_size=1, max_size=6))
def test_every_score_lies_between_zero_and_one(texts: list[str]) -> None:
    """Both vectors are unit length and non-negative, so the cosine must."""
    index = InvertedIndex()
    for document_id, text in enumerate(texts):
        index.add_document(document_id, text)
    ranker = Ranker(index)
    for term in list(index.terms)[:3]:
        weights = ranker.query_weights([term])
        for document_id in range(len(texts)):
            score = ranker.score(weights, document_id)
            assert -TOLERANCE <= score <= 1.0 + TOLERANCE


@given(st.lists(st.text(), min_size=1, max_size=6))
def test_results_come_back_in_descending_score_order(texts: list[str]) -> None:
    index = InvertedIndex()
    for document_id, text in enumerate(texts):
        index.add_document(document_id, text)
    ranker = Ranker(index)
    terms = list(index.terms)[:2]
    if not terms:
        return
    scores = [score for _, score in ranker.rank(" ".join(terms), range(len(texts)))]
    assert scores == sorted(scores, reverse=True)


@given(st.lists(st.text(), min_size=1, max_size=6))
def test_a_document_vector_is_unit_length_or_empty(texts: list[str]) -> None:
    index = InvertedIndex()
    for document_id, text in enumerate(texts):
        index.add_document(document_id, text)
    ranker = Ranker(index)
    for document_id in range(len(texts)):
        vector = ranker.document_vector(document_id)
        length = math.sqrt(sum(weight * weight for weight in vector.values()))
        assert length == pytest.approx(0.0, abs=TOLERANCE) or length == pytest.approx(
            1.0, abs=TOLERANCE
        )
