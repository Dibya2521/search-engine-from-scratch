"""Tests for stopword handling."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from search_engine.index import InvertedIndex
from search_engine.stopwords import DEFAULT_STOPWORDS, derive
from search_engine.tokenizer import tokenize


def test_the_default_list_is_lowercase_single_tokens() -> None:
    """It is matched against tokenizer output, so anything else could never fire."""
    for word in DEFAULT_STOPWORDS:
        assert tokenize(word) == [word]


def test_the_default_list_is_immutable() -> None:
    assert isinstance(DEFAULT_STOPWORDS, frozenset)


def test_deriving_from_an_empty_index_does_not_divide_by_zero() -> None:
    assert derive(InvertedIndex()) == frozenset()


def test_derive_finds_terms_common_across_documents() -> None:
    index = InvertedIndex()
    index.add_document(1, "alpha beta gamma")
    index.add_document(2, "alpha beta delta")
    index.add_document(3, "alpha epsilon zeta")
    # alpha is in 3 of 3, beta in 2 of 3, the rest in 1 of 3.
    assert derive(index, threshold=1.0) == {"alpha"}
    assert derive(index, threshold=0.6) == {"alpha", "beta"}
    assert derive(index, threshold=0.3) == set(index.terms)


def test_derive_returns_stems_not_surface_words() -> None:
    """Because that is what an index actually stores."""
    index = InvertedIndex()
    index.add_document(1, "running running")
    assert derive(index, threshold=1.0) == {"run"}


@given(st.lists(st.text(), min_size=1, max_size=6), st.floats(0.0, 1.0))
def test_a_derived_term_always_meets_the_threshold(
    texts: list[str], threshold: float
) -> None:
    index = InvertedIndex()
    for document_id, text in enumerate(texts):
        index.add_document(document_id, text)
    for term in derive(index, threshold):
        assert index.document_frequency(term) >= threshold * index.document_count


@given(st.lists(st.text(), min_size=1, max_size=6))
def test_a_higher_threshold_never_derives_more_terms(texts: list[str]) -> None:
    index = InvertedIndex()
    for document_id, text in enumerate(texts):
        index.add_document(document_id, text)
    assert derive(index, 0.9) <= derive(index, 0.4)
