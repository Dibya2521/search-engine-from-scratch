"""Tests for the shared analysis pipeline."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.analysis import analyze, analyze_positioned
from search_engine.stemmer import stem
from search_engine.stopwords import DEFAULT_STOPWORDS
from search_engine.tokenizer import tokenize

NO_STOPWORDS: frozenset[str] = frozenset()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("!!!", []),
        (
            "Web search engines build an inverted index",
            ["web", "search", "engin", "build", "invert", "index"],
        ),
        (
            "The index maps terms to documents and positions",
            ["index", "map", "term", "document", "posit"],
        ),
    ],
)
def test_analysis_lowercases_splits_drops_stopwords_and_stems(
    text: str, expected: list[str]
) -> None:
    assert analyze(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # "an" at position 4 is dropped, so 4 is missing from the positions.
        (
            "Web search engines build an inverted index",
            [
                (0, "web"),
                (1, "search"),
                (2, "engin"),
                (3, "build"),
                (5, "invert"),
                (6, "index"),
            ],
        ),
        # "The", "to" and "and" go, leaving gaps at 0, 4 and 6.
        (
            "The index maps terms to documents and positions",
            [(1, "index"), (2, "map"), (3, "term"), (5, "document"), (7, "posit")],
        ),
    ],
)
def test_positions_are_indices_into_the_unfiltered_token_stream(
    text: str, expected: list[tuple[int, str]]
) -> None:
    """The gaps are the point: they keep phrase queries honest."""
    assert analyze_positioned(text) == expected


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Searching", "search"),
        ("SEARCHES", "searching"),
        ("inverted", "inverting"),
        ("positions", "position"),
    ],
)
def test_surface_variants_meet_on_one_term(first: str, second: str) -> None:
    """The whole reason the pipeline exists: different words, same key."""
    assert analyze(first) == analyze(second)


def test_a_text_of_only_stopwords_yields_nothing() -> None:
    assert analyze("the and of to a an is it") == []


def test_the_stopword_set_is_a_parameter() -> None:
    """So a corpus-derived list can replace the published default."""
    text = "the quick brown fox"
    assert analyze(text) == ["quick", "brown", "fox"]
    assert analyze(text, frozenset({"quick"})) == ["the", "brown", "fox"]


@given(st.text())
def test_with_no_stopwords_every_token_survives_at_its_own_position(
    text: str,
) -> None:
    """Filtering off reduces to one term per token at consecutive positions.

    This was the invariant before stopwords existed, and it still holds as the
    special case, which is what proves filtering is the only thing that changed.
    """
    positioned = analyze_positioned(text, NO_STOPWORDS)
    assert [position for position, _ in positioned] == list(range(len(tokenize(text))))
    assert len(positioned) == len(tokenize(text))


@given(st.text())
def test_every_surviving_position_points_at_its_own_token(text: str) -> None:
    """Ties each emitted term back to the exact token it came from."""
    tokens = tokenize(text)
    for position, term in analyze_positioned(text):
        assert term == stem(tokens[position])


@given(st.text())
def test_positions_strictly_increase_and_never_exceed_the_token_count(
    text: str,
) -> None:
    positions = [position for position, _ in analyze_positioned(text)]
    assert positions == sorted(set(positions))
    assert len(positions) <= len(tokenize(text))
    assert all(position < len(tokenize(text)) for position in positions)


@given(st.text())
def test_no_stopword_ever_survives(text: str) -> None:
    surviving = {position for position, _ in analyze_positioned(text)}
    tokens = tokenize(text)
    for position, token in enumerate(tokens):
        if token in DEFAULT_STOPWORDS:
            assert position not in surviving


@given(st.text())
def test_analyze_is_analyze_positioned_without_the_positions(text: str) -> None:
    assert analyze(text) == [term for _, term in analyze_positioned(text)]
