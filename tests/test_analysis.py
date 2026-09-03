"""Tests for the shared analysis pipeline."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.analysis import analyze
from search_engine.tokenizer import tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("!!!", []),
        (
            "Web search engines build an inverted index",
            ["web", "search", "engin", "build", "an", "invert", "index"],
        ),
        (
            "The index maps terms to documents and positions",
            ["the", "index", "map", "term", "to", "document", "and", "posit"],
        ),
    ],
)
def test_analysis_lowercases_splits_and_stems(text: str, expected: list[str]) -> None:
    assert analyze(text) == expected


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


@given(st.text())
def test_one_term_out_for_every_token_in(text: str) -> None:
    """Positions in the index are token positions, and this is what makes that true.

    Nothing in the pipeline currently drops or adds a token, so a term's index
    in this list is also its position in the original text. Adding a filtering
    stage would break this, and would silently change what a phrase query
    means: with the intervening words gone, "king of england" and "king
    england" become the same phrase. This test is here to fail loudly at that
    moment rather than let the semantics shift unnoticed.
    """
    assert len(analyze(text)) == len(tokenize(text))
