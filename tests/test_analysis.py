"""Tests for the shared analysis pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine import analysis
from search_engine.analysis import analyze, analyze_positioned, fingerprint
from search_engine.index import InvertedIndex
from search_engine.persistence import AnalyzerMismatchError, load, save
from search_engine.stemmer import stem
from search_engine.stopwords import DEFAULT_STOPWORDS
from search_engine.tokenizer import tokenize

if TYPE_CHECKING:
    from pathlib import Path

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
    """Ties each emitted term back to the exact token it came from.

    A term is its token stemmed when the token is ASCII and its token unchanged
    otherwise, and never anything else, whatever the script.
    """
    tokens = tokenize(text)
    for position, term in analyze_positioned(text):
        token = tokens[position]
        assert term == (stem(token) if token.isascii() else token)


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


def test_the_normalization_form_is_part_of_the_fingerprint() -> None:
    """Two forms produce different terms, so they must not share a digest."""
    assert fingerprint() != fingerprint(form="NFKC")


def test_the_stopword_list_is_part_of_the_fingerprint() -> None:
    assert fingerprint() != fingerprint(NO_STOPWORDS)


def test_adding_normalization_changed_the_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An index built before normalization is analysed differently from now.

    That is the failure the fingerprint exists to catch, and the pipeline
    version is what makes the two digests differ, since the token pattern, the
    stemmer and the stopword list are all unchanged.
    """
    current = fingerprint()
    monkeypatch.setattr(analysis, "_PIPELINE_VERSION", "1")
    assert fingerprint() != current


def test_an_index_built_before_normalization_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Written with the old digest, then read by this build, which refuses it.

    Without this an old index would answer queries analysed a new way and return
    nothing, with no error anywhere.
    """
    index = InvertedIndex()
    index.add_document(1, "web search engines")
    path = tmp_path / "old.index"
    monkeypatch.setattr(analysis, "_PIPELINE_VERSION", "1")
    save(index, path)
    monkeypatch.undo()
    with pytest.raises(AnalyzerMismatchError, match="rebuild the index"):
        load(path)


def test_analysis_carries_the_form_through_to_the_terms() -> None:
    """One case showing normalization and the stemming guard at once.

    Under NFC the ligature survives, so the token is not ASCII and is left
    unstemmed. Under NFKC it folds to ``fi``, the token becomes ASCII, and the
    English stemmer then applies.
    """
    text = f"{chr(0xFB01)}refighting"
    assert analyze(text) == [f"{chr(0xFB01)}refighting"]
    assert analyze(text, form="NFKC") == ["firefight"]


@pytest.mark.parametrize(
    ("script", "text"),
    [
        ("greek", "".join(chr(point) for point in (0x395, 0x3BB, 0x3BB, 0x3AC))),
        ("cyrillic", "".join(chr(point) for point in (0x41F, 0x440, 0x438))),
        ("arabic", "".join(chr(point) for point in (0x645, 0x631, 0x62D))),
    ],
)
def test_a_token_outside_ascii_is_never_stemmed(script: str, text: str) -> None:
    """The stemmer encodes English suffix rules and has no meaning elsewhere.

    Leaving a token whole costs recall on inflected forms. Stripping letters
    that spell nothing in that script costs correctness, which is worse.
    """
    assert analyze(text, NO_STOPWORDS) == [text.lower()], script


def test_ascii_tokens_are_still_stemmed() -> None:
    """The guard must not have switched stemming off for the language it fits."""
    assert analyze("running searches") == ["run", "search"]
