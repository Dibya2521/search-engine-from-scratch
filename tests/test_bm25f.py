"""Tests for BM25F, BM25 over weighted fields.

The load-bearing test is `test_weighting_only_the_document_field_is_plain_bm25`.
BM25F is a generalisation, so it must reduce to what it generalises; a formula
that does not is a different formula wearing the name. The two orderings compute
the same algebra differently, so they agree to floating point rather than
bit for bit, and the test says which.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.bm25 import K1, B, BM25Ranker
from search_engine.bm25f import DEFAULT_FIELD_WEIGHTS, DOCUMENT_FIELD, BM25FRanker
from search_engine.index import TITLE_FIELD, InvertedIndex
from search_engine.query import search
from search_engine.ranking import StaleRankerError

TOLERANCE = 1e-12
WORDS = ("alpha", "beta", "gamma", "delta", "python")


def fielded(*documents: tuple[str, str]) -> InvertedIndex:
    """Index documents given as title and body, recording the title as a field."""
    index = InvertedIndex()
    for document_id, (title, body) in enumerate(documents):
        index.add_document(document_id, f"{title} {body}", {TITLE_FIELD: title})
    return index


def generated_index(seed: int) -> InvertedIndex:
    """Index sixty documents of varying title and body length.

    Built arithmetically rather than from a random source, so a failing example
    reproduces exactly from its seed alone.
    """
    index = InvertedIndex()
    for document_id in range(60):
        step = (document_id + seed) % len(WORDS)
        title = " ".join(WORDS[(step + at) % len(WORDS)] for at in range(1 + step % 4))
        body = " ".join(
            WORDS[(document_id * 7 + at * 3 + seed) % len(WORDS)]
            for at in range(3 + (document_id + seed) % 28)
        )
        index.add_document(document_id, f"{title} {body}", {TITLE_FIELD: title})
    return index


@given(st.integers(min_value=0, max_value=40))
def test_weighting_only_the_document_field_is_plain_bm25(seed: int) -> None:
    """A generalisation must reduce to what it generalises.

    The two compute the same algebra in a different order, so they agree to
    within floating point rather than bit for bit. The largest disagreement
    measured over generated indexes is one unit in the last place.
    """
    index = generated_index(seed)
    plain = BM25Ranker(index)
    document_only = BM25FRanker(index, {DOCUMENT_FIELD: 1.0})
    for query in ("python", "alpha", "beta gamma"):
        candidates = search(index, query)
        expected = dict(plain.rank(query, candidates, limit=100))
        actual = dict(document_only.rank(query, candidates, limit=100))
        assert actual.keys() == expected.keys()
        for document_id, score in expected.items():
            assert actual[document_id] == pytest.approx(score, abs=TOLERANCE)


def test_a_title_match_outscores_a_body_match_at_equal_frequency() -> None:
    """The entire reason for weighting fields."""
    index = fielded(
        ("python guide", "filler filler filler filler"),
        ("other heading", "filler python filler filler"),
    )
    candidates = search(index, "python")
    ranked = BM25FRanker(index).rank("python", candidates)
    assert [document_id for document_id, _ in ranked] == [0, 1]
    assert dict(ranked)[0] > dict(ranked)[1]


def test_a_field_weighted_zero_contributes_nothing() -> None:
    index = generated_index(1)
    candidates = search(index, "python")
    without = BM25FRanker(index, {DOCUMENT_FIELD: 1.0})
    zeroed = BM25FRanker(index, {DOCUMENT_FIELD: 1.0, TITLE_FIELD: 0.0})
    assert zeroed.rank("python", candidates, limit=100) == without.rank(
        "python", candidates, limit=100
    )


def test_a_field_the_index_does_not_hold_contributes_nothing() -> None:
    """Missing fields must not raise, so a weight list can outlive an index."""
    index = generated_index(2)
    candidates = search(index, "python")
    known = BM25FRanker(index, {DOCUMENT_FIELD: 1.0})
    unknown = BM25FRanker(index, {DOCUMENT_FIELD: 1.0, "nosuchfield": 5.0})
    assert unknown.rank("python", candidates, limit=100) == known.rank(
        "python", candidates, limit=100
    )


def test_an_empty_title_does_not_divide_by_zero() -> None:
    """Adversarial: a field length of zero, which the correction divides by."""
    index = fielded(
        ("", "python appears only in the body"), ("python", "a title match")
    )
    candidates = search(index, "python")
    ranked = BM25FRanker(index).rank("python", candidates)
    assert len(ranked) == 2
    assert all(score > 0 for _, score in ranked)


def test_a_corpus_where_no_document_has_a_title() -> None:
    """Adversarial: the whole field is empty, so its average length is zero."""
    index = InvertedIndex()
    index.add_document(0, "python and other words", {TITLE_FIELD: ""})
    index.add_document(1, "more python here", {TITLE_FIELD: ""})
    candidates = search(index, "python")
    ranked = BM25FRanker(index).rank("python", candidates)
    assert len(ranked) == 2
    assert all(score > 0 for _, score in ranked)


def saturate_then_combine(
    index: InvertedIndex, term: str, document_id: int, weights: dict[str, float]
) -> float:
    """The wrong ordering, for comparison: each field saturates on its own."""
    ranker = BM25FRanker(index, weights)
    idf = ranker.query_weights([term])[term]
    total = 0.0
    for field, weight in weights.items():
        key = term if field == DOCUMENT_FIELD else f"{field}:{term}"
        positions = index.postings(key).get(document_id)
        if positions is None:
            continue
        frequency = len(positions)
        total += weight * idf * frequency * (K1 + 1) / (frequency + K1 * B)
    return total


def test_the_two_orderings_differ_and_only_one_saturates_once() -> None:
    """Adversarial: a term 100 times in the body against 2 times in the title.

    Saturating each field and then adding runs up two separate curves, so the
    body's hundred occurrences carry far more than a single curve would allow.
    Combining first keeps one curve over the document, which is what saturation
    is for.
    """
    index = fielded(
        ("python python", "python " * 100 + "filler"),
        ("other", "a short unrelated body"),
    )
    combined = BM25FRanker(index).score(BM25FRanker(index).query_weights(["python"]), 0)
    separate = saturate_then_combine(index, "python", 0, DEFAULT_FIELD_WEIGHTS)
    assert combined != pytest.approx(separate)
    assert separate > combined


def test_repeating_a_term_saturates() -> None:
    """Each further occurrence adds less than the one before it.

    Document length is held constant across the four documents, because
    otherwise length normalisation moves with the frequency and the curve being
    measured is not the one under test.
    """
    index = InvertedIndex()
    for document_id, count in enumerate([1, 2, 3, 4]):
        body = " ".join(["python"] * count + ["filler"] * (100 - count))
        index.add_document(document_id, f"title {body}", {TITLE_FIELD: "title"})
    ranker = BM25FRanker(index)
    weights = ranker.query_weights(["python"])
    scores = [ranker.score(weights, document_id) for document_id in range(4)]
    assert scores == sorted(scores)
    gains = [scores[at + 1] - scores[at] for at in range(3)]
    assert gains == sorted(gains, reverse=True)


def test_the_default_weights_favour_the_title() -> None:
    """A convention rather than a measurement, pinned so it cannot drift."""
    assert DEFAULT_FIELD_WEIGHTS[TITLE_FIELD] > DEFAULT_FIELD_WEIGHTS[DOCUMENT_FIELD]


def test_a_stale_index_is_refused() -> None:
    """Its field averages would be computed from a corpus that has moved on."""
    index = fielded(("title", "python body"))
    ranker = BM25FRanker(index)
    index.add_document(99, "a new document", {TITLE_FIELD: "new"})
    with pytest.raises(StaleRankerError):
        ranker.rank("python", [0])


def test_an_empty_index_scores_nothing_without_dividing_by_zero() -> None:
    """Adversarial: no documents at all, so every field average is zero."""
    ranker = BM25FRanker(InvertedIndex())
    assert ranker.rank("python", []) == []
    assert ranker.score(ranker.query_weights(["python"]), 0) == 0.0
