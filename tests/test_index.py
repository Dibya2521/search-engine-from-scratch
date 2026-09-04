"""Tests for the inverted index."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.analysis import analyze, analyze_positioned
from search_engine.index import DuplicateDocumentError, InvertedIndex


def _index_of(texts: list[str]) -> InvertedIndex:
    index = InvertedIndex()
    for document_id, text in enumerate(texts):
        index.add_document(document_id, text)
    return index


def test_an_empty_index_reports_nothing() -> None:
    empty = InvertedIndex()
    assert empty.document_count == 0
    assert empty.vocabulary_size == 0
    assert empty.document_frequency("index") == 0
    assert empty.postings("index") == {}
    assert "index" not in empty


def test_counts(index: InvertedIndex) -> None:
    assert index.document_count == 3
    assert index.vocabulary_size == 11


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("index", {1: [6], 2: [1], 3: [5]}),
        ("web", {1: [0], 3: [2]}),
        ("search", {1: [1], 3: [0]}),
        ("build", {1: [3]}),
        ("posit", {2: [7]}),
    ],
)
def test_postings_record_document_and_position(
    index: InvertedIndex, term: str, expected: dict[int, list[int]]
) -> None:
    assert index.postings(term) == expected


@pytest.mark.parametrize("stopword", ["the", "an", "and", "to"])
def test_stopwords_are_never_indexed(index: InvertedIndex, stopword: str) -> None:
    assert stopword not in index
    assert index.postings(stopword) == {}


def test_dropping_a_stopword_leaves_a_gap_in_the_positions() -> None:
    """Document 1 has "an" at position 4, so no term claims position 4."""
    index = InvertedIndex()
    index.add_document(1, "Web search engines build an inverted index")
    claimed = {position for term in index.terms for position in index.postings(term)[1]}
    assert claimed == {0, 1, 2, 3, 5, 6}
    assert 4 not in claimed


def test_stemmed_variants_share_one_postings_list(index: InvertedIndex) -> None:
    """Document 1 says `search`, document 3 says `Searching`."""
    assert set(index.postings("search")) == {1, 3}
    assert index.postings("searching") == {}


def test_an_unknown_term_has_empty_postings_rather_than_raising(
    index: InvertedIndex,
) -> None:
    assert index.postings("nonexistent") == {}
    assert index.document_frequency("nonexistent") == 0


def test_a_repeated_term_records_every_position() -> None:
    """Positions 1, 3 and 4 held stopwords, so they are absent."""
    index = InvertedIndex()
    index.add_document(9, "index the index of an index")
    assert index.postings("index") == {9: [0, 2, 5]}
    assert index.document_frequency("index") == 1


def test_document_frequency_counts_documents_not_occurrences() -> None:
    index = InvertedIndex()
    index.add_document(1, "index index index index")
    index.add_document(2, "index")
    assert index.document_frequency("index") == 2


def test_a_document_with_no_terms_still_counts_as_a_document() -> None:
    """It has to, or the total used to weight rare terms would be wrong."""
    index = InvertedIndex()
    index.add_document(1, "...!!!")
    assert index.document_count == 1
    assert index.vocabulary_size == 0


def test_a_document_of_only_stopwords_still_counts_as_a_document() -> None:
    index = InvertedIndex()
    index.add_document(1, "the and of to a an")
    assert index.document_count == 1
    assert index.vocabulary_size == 0


def test_adding_the_same_document_twice_is_refused() -> None:
    """Silently appending a second set of positions would corrupt the first."""
    index = InvertedIndex()
    index.add_document(1, "web search")
    with pytest.raises(DuplicateDocumentError, match="already indexed"):
        index.add_document(1, "different text")
    assert index.document_count == 1


def test_postings_cannot_be_mutated_by_a_caller(index: InvertedIndex) -> None:
    postings = index.postings("index")
    with pytest.raises(TypeError):
        postings[4] = [0]  # type: ignore[index]


@given(st.text())
def test_a_position_always_points_at_the_term_that_claimed_it(text: str) -> None:
    """The invariant phrase queries rest on."""
    index = InvertedIndex()
    index.add_document(1, text)
    positioned = dict(analyze_positioned(text))
    for term in index.terms:
        for position in index.postings(term)[1]:
            assert positioned[position] == term


@given(st.text())
def test_every_term_occurrence_is_recorded_exactly_once(text: str) -> None:
    """Nothing lost and nothing duplicated between analysis and storage."""
    index = InvertedIndex()
    index.add_document(1, text)
    recorded = sum(len(index.postings(term)[1]) for term in index.terms)
    assert recorded == len(analyze(text))


@given(st.text())
def test_positions_are_stored_in_increasing_order(text: str) -> None:
    """Positional intersection assumes this, so it is asserted rather than hoped."""
    index = InvertedIndex()
    index.add_document(1, text)
    for term in index.terms:
        positions = index.postings(term)[1]
        assert list(positions) == sorted(positions)
        assert len(set(positions)) == len(positions)


@given(st.lists(st.text(), min_size=1, max_size=8))
def test_document_frequency_matches_a_direct_count(texts: list[str]) -> None:
    index = _index_of(texts)
    assert index.document_count == len(texts)
    for term in index.terms:
        expected = sum(1 for text in texts if term in analyze(text))
        assert index.document_frequency(term) == expected


@given(st.lists(st.text(), min_size=1, max_size=8))
def test_vocabulary_is_the_union_of_every_document(texts: list[str]) -> None:
    index = _index_of(texts)
    expected = {term for text in texts for term in analyze(text)}
    assert set(index.terms) == expected
    assert index.vocabulary_size == len(expected)
