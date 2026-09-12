"""Tests for the passage that shows why a document matched."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.analysis import analyze
from search_engine.snippet import BOUNDARY_TOLERANCE, SNIPPET_TOKENS, extract

SENTENCES = (
    "The first sentence is about gardening. "
    "The second sentence introduces the inverted index and its postings. "
    "The third sentence is about cookery."
)

# No sentence terminator anywhere, so neither edge can snap to one.
UNPUNCTUATED = " ".join(f"alpha{number}" for number in range(80))

CJK_TEXT = "A 検索エンジン B"


def marked(document: str, query: str, width: int = SNIPPET_TOKENS) -> list[str]:
    """Return the substrings a snippet highlights."""
    snippet = extract(document, query, width)
    return [snippet.text[start:end] for start, end in snippet.highlights]


def test_a_window_with_more_distinct_terms_beats_one_with_more_occurrences() -> None:
    """Showing more of the query explains the match better than showing it often."""
    document = (
        "index index index index index index index index. "
        "Here is an inverted index of postings."
    )
    snippet = extract(document, "inverted index postings", width=4)
    assert "inverted" in snippet.text
    assert "postings" in snippet.text
    assert marked(document, "inverted index postings", 4) == [
        "inverted",
        "index",
        "postings",
    ]


def test_highlights_land_on_the_words_even_where_stemming_changed_them() -> None:
    """`retrieval` is stored as `retriev`, and the range covers the word as written."""
    document = "Ranked retrieval returns documents ordered by how well they match."
    assert marked(document, "retrieval documents") == ["retrieval", "documents"]


def test_a_document_shorter_than_the_window_comes_back_whole() -> None:
    document = "  A short note about retrieval.  "
    assert extract(document, "retrieval").text == document


def test_a_query_matching_nothing_returns_the_opening() -> None:
    snippet = extract(UNPUNCTUATED, "nothing here matches", width=5)
    assert snippet.highlights == ()
    assert snippet.text.startswith("alpha0 alpha1 alpha2 alpha3 alpha4")
    assert "alpha40" not in snippet.text


def test_a_stopword_query_highlights_nothing_rather_than_nowhere() -> None:
    """`the` analyses away to nothing, so there is no term to mark."""
    snippet = extract(SENTENCES, "the of and")
    assert snippet.highlights == ()
    assert snippet.text.startswith("The first sentence")


def test_a_term_appearing_only_inside_a_longer_word_does_not_match() -> None:
    document = "The word concatenate contains it but is not it."
    assert marked(document, "cat") == []


def test_a_document_with_no_whitespace_at_all() -> None:
    document = "supercalifragilistic"
    snippet = extract(document, "supercalifragilistic")
    assert snippet.text == document
    assert snippet.highlights == ((0, len(document)),)


def test_the_passage_widens_to_the_sentence_it_sits_in() -> None:
    snippet = extract(SENTENCES, "inverted postings", width=4)
    assert snippet.text == (
        "The second sentence introduces the inverted index and its postings."
    )


def test_a_passage_with_no_sentence_edge_within_reach_keeps_the_window() -> None:
    """Both ends fall through, because there is no terminator to snap to."""
    snippet = extract(UNPUNCTUATED, "alpha40", width=4)
    assert not snippet.text.startswith("alpha0")
    assert not snippet.text.endswith("alpha79")
    assert "alpha40" in snippet.text
    assert len(snippet.text.split()) <= 4 + 2 * BOUNDARY_TOLERANCE


def test_a_term_visible_beside_the_window_is_still_marked() -> None:
    """Widening to a sentence brings terms in, and an unmarked one reads as a miss."""
    document = "Ranked retrieval returns documents ordered by how well they match."
    assert marked(document, "retrieval match", 2) == ["retrieval", "match"]


def test_overlapping_bigram_highlights_are_merged() -> None:
    """CJK bigrams overlap by a character, so their ranges have to be joined."""
    snippet = extract(CJK_TEXT, "検索エンジン")
    assert snippet.highlights == ((2, 8),)
    assert snippet.text[2:8] == "検索エンジン"


def test_a_document_holding_no_terms_comes_back_whole() -> None:
    """It cannot be a search result, having no postings, but it must not crash."""
    snippet = extract("... --- ...", "anything")
    assert snippet.text == "... --- ..."
    assert snippet.highlights == ()


def test_a_window_of_no_terms_is_refused() -> None:
    with pytest.raises(ValueError, match="width must be at least 1"):
        extract("some text", "text", 0)


def test_a_repeated_term_counts_once_towards_distinct() -> None:
    """Exercises a term entering a window it is already in, and leaving one twice."""
    document = "index alpha index beta gamma delta index epsilon"
    snippet = extract(document, "index", width=3)
    assert snippet.text == document
    assert len(snippet.highlights) == 3


@given(document=st.text(max_size=400), query=st.text(max_size=40))
def test_highlights_are_disjoint_ascending_and_name_query_terms(
    document: str, query: str
) -> None:
    snippet = extract(document, query)
    wanted = set(analyze(query))
    reached = 0
    for start, end in snippet.highlights:
        assert 0 <= start < end <= len(snippet.text)
        assert start >= reached
        reached = end
        assert set(analyze(snippet.text[start:end])) & wanted
