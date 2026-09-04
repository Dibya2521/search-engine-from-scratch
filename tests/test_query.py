"""Tests for query parsing and execution.

Property tests build their queries from the *raw* tokens of a document, never
from terms read back out of the index. That distinction is the contract:
analysis runs exactly once on each side, on the original text. Feeding a stored
term back in analyses it a second time, and stemming is not idempotent, so the
term can change. A property test found this the hard way, and
``test_a_term_read_back_from_the_index_is_not_a_valid_query`` pins it.
"""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.analysis import analyze, analyze_positioned
from search_engine.index import InvertedIndex
from search_engine.query import Query, parse, search
from search_engine.tokenizer import tokenize


def _index_of(texts: list[str]) -> InvertedIndex:
    index = InvertedIndex()
    for document_id, text in enumerate(texts):
        index.add_document(document_id, text)
    return index


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("computer", Query(terms=("comput",), offsets=(0,), is_phrase=False)),
        (
            "Computer SCIENCE",
            Query(terms=("comput", "scienc"), offsets=(0, 1), is_phrase=False),
        ),
        (
            '"computer science"',
            Query(terms=("comput", "scienc"), offsets=(0, 1), is_phrase=True),
        ),
        ('"computer"', Query(terms=("comput",), offsets=(0,), is_phrase=True)),
        ("", Query(terms=(), offsets=(), is_phrase=False)),
        ("   ", Query(terms=(), offsets=(), is_phrase=False)),
        ('""', Query(terms=(), offsets=(), is_phrase=True)),
        ('"', Query(terms=(), offsets=(), is_phrase=False)),
    ],
)
def test_parsing(text: str, expected: Query) -> None:
    assert parse(text) == expected


@pytest.mark.parametrize(
    ("text", "expected_offsets"),
    [
        ('"the computer science"', (1, 2)),
        ('"department of computer science"', (0, 2, 3)),
        ('"computer of science"', (0, 2)),
    ],
)
def test_a_dropped_stopword_leaves_a_gap_in_the_query_offsets(
    text: str, expected_offsets: tuple[int, ...]
) -> None:
    """These gaps are what make a phrase containing a stopword still matchable."""
    assert parse(text).offsets == expected_offsets


@pytest.mark.parametrize("text", ['computer "science"', '"computer" science'])
def test_a_partly_quoted_query_is_not_a_phrase(text: str) -> None:
    """Mixing a phrase with loose terms is unsupported, so quotes just separate."""
    assert parse(text) == Query(
        terms=("comput", "scienc"), offsets=(0, 1), is_phrase=False
    )


def test_query_terms_are_stemmed_like_documents(phrase_index: InvertedIndex) -> None:
    """`searching` finds a document that says `search`, which is the point."""
    assert search(phrase_index, "searching") == {4}
    assert search(phrase_index, "engines") == {2, 4}


def test_a_term_read_back_from_the_index_is_not_a_valid_query() -> None:
    """Query with language, never with a term taken out of the index.

    Stemming is not idempotent, so analysing a stored term a second time can
    change it. Here `A0SE` is stored as `a0s`, and querying `a0s` analyses to
    `a0`, which matches nothing. Found by a property test rather than reasoning.
    """
    index = InvertedIndex()
    index.add_document(1, "A0SE")
    assert set(index.terms) == {"a0s"}
    assert search(index, "A0SE") == {1}
    assert search(index, "a0s") == set[int]()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("computer", {1, 2, 3, 5}),
        ("science", {1, 2, 3, 5}),
        ("index", {4}),
        ("nonexistent", set[int]()),
        ("", set[int]()),
        ("the", set[int]()),
    ],
)
def test_one_word_queries(
    phrase_index: InvertedIndex, text: str, expected: set[int]
) -> None:
    assert search(phrase_index, text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("computer web", {1, 2, 3, 4, 5}),
        ("index courses", {1, 4}),
        ("nonexistent alsomissing", set[int]()),
        ("index nonexistent", {4}),
    ],
)
def test_free_text_queries_union_their_terms(
    phrase_index: InvertedIndex, text: str, expected: set[int]
) -> None:
    """Any term is enough, so more terms can only widen the result."""
    assert search(phrase_index, text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('"computer science"', {1, 3, 5}),
        ('"computer science department"', {1}),
        ('"department computer"', set[int]()),
        ('"computer"', {1, 2, 3, 5}),
        ('""', set[int]()),
        ('"computer nonexistent"', set[int]()),
    ],
)
def test_phrase_queries_require_adjacency_in_order(
    phrase_index: InvertedIndex, text: str, expected: set[int]
) -> None:
    assert search(phrase_index, text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Leading stopword: offsets (1,2), so it still matches.
        ('"the computer science"', {1, 3, 5}),
        # Document 3 reads "department of computer science", gap and all.
        ('"department of computer science"', {3}),
        # No document has computer, one word, science.
        ('"computer of science"', set[int]()),
    ],
)
def test_a_phrase_containing_a_stopword_still_matches_correctly(
    phrase_index: InvertedIndex, text: str, expected: set[int]
) -> None:
    """The reason query offsets exist rather than consecutive integers.

    Removing stopwords from both sides and then assuming the surviving terms
    were adjacent would make the first case match nothing and the third match
    document 3 wrongly.
    """
    assert search(phrase_index, text) == expected


def test_a_phrase_excludes_documents_holding_the_terms_apart(
    phrase_index: InvertedIndex,
) -> None:
    """Document 2 has both words, but reversed and with `and` between them."""
    assert 2 in search(phrase_index, "computer science")
    assert 2 not in search(phrase_index, '"computer science"')


def test_term_order_matters_to_a_phrase(phrase_index: InvertedIndex) -> None:
    """Document 5 repeats the pair, so it contains both orders."""
    assert search(phrase_index, '"science computer"') == {5}
    assert search(phrase_index, '"computer science"') == {1, 3, 5}


def test_a_single_term_phrase_behaves_like_a_one_word_query(
    phrase_index: InvertedIndex,
) -> None:
    assert search(phrase_index, '"computer"') == search(phrase_index, "computer")


def test_a_long_phrase_gives_up_as_soon_as_two_terms_are_not_adjacent() -> None:
    """Every term is present, and a later pair adjoins, but the first pair does not.

    This is the case the early exit exists for: once the overlap empties there
    is no point building the remaining position sets.
    """
    index = InvertedIndex()
    index.add_document(1, "alpha zzz beta gamma")
    assert search(index, "alpha beta gamma") == {1}
    assert search(index, '"beta gamma"') == {1}
    assert search(index, '"alpha beta gamma"') == set[int]()


@given(st.lists(st.text(), min_size=1, max_size=6))
def test_a_one_word_query_finds_exactly_the_documents_holding_that_term(
    texts: list[str],
) -> None:
    index = _index_of(texts)
    for token in {token for text in texts for token in tokenize(text)}:
        terms = analyze(token)
        if not terms:
            continue  # a stopword on its own is not a query
        expected = {
            document_id
            for document_id, text in enumerate(texts)
            if terms[0] in analyze(text)
        }
        assert search(index, token) == expected


@given(st.lists(st.text(), min_size=1, max_size=6))
def test_a_free_text_query_never_narrows_as_terms_are_added(
    texts: list[str],
) -> None:
    index = _index_of(texts)
    tokens = [token for text in texts for token in tokenize(text) if analyze(token)]
    if len(tokens) < 2:
        return
    assert search(index, tokens[0]) <= search(index, f"{tokens[0]} {tokens[1]}")


@given(st.lists(st.text(), min_size=1, max_size=6))
def test_a_phrase_result_is_always_a_subset_of_the_free_text_result(
    texts: list[str],
) -> None:
    """Adjacency is a strictly stronger condition than co-occurrence."""
    index = _index_of(texts)
    tokens = [token for text in texts for token in tokenize(text) if analyze(token)]
    if len(tokens) < 2:
        return
    pair = f"{tokens[0]} {tokens[1]}"
    assert search(index, f'"{pair}"') <= search(index, pair)


@given(st.text())
def test_any_span_of_a_document_is_findable_as_a_phrase(text: str) -> None:
    """The falsifying direction, and it exercises the stopword gaps hardest.

    Takes the literal text between two consecutive surviving terms, stopwords
    included, and insists the phrase query finds the document it came from.
    """
    index = InvertedIndex()
    index.add_document(7, text)
    tokens = tokenize(text)
    positioned = analyze_positioned(text)
    for (first, _), (second, _) in itertools.pairwise(positioned):
        span = " ".join(tokens[first : second + 1])
        assert 7 in search(index, f'"{span}"')
