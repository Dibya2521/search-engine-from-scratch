"""Tests for the tokenizer.

Every expected value was read off a real run before being asserted.

Code points that matter are built from ``chr()`` or forced through
``unicodedata.normalize`` rather than pasted in as literals. These cases turn
entirely on which code points are present, and a literal leaves that to
whatever an editor or tool happened to write to disk. An earlier version used
literals, and the composed and decomposed columns below silently became the
same string.
"""

from __future__ import annotations

import re
import unicodedata

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.tokenizer import tokenize

TOKEN_SHAPE = re.compile(r"[a-z0-9]+")

NO_BREAK_SPACE = chr(0xA0)
SOFT_HYPHEN = chr(0xAD)
ZERO_WIDTH_JOINER = chr(0x200D)
KELVIN_SIGN = chr(0x212A)
TURKISH_CAPITAL_I = chr(0x130)
FULLWIDTH_AB = chr(0xFF21) + chr(0xFF22)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("   ", []),
        ("...!!!???", []),
        ("hello", ["hello"]),
        ("Hello, World!", ["hello", "world"]),
        ("web search engine", ["web", "search", "engine"]),
        ("  leading and trailing  ", ["leading", "and", "trailing"]),
        ("multiple\n\nnewlines\ttabs", ["multiple", "newlines", "tabs"]),
        ("a1b2c3", ["a1b2c3"]),
    ],
)
def test_the_basic_rule(text: str, expected: list[str]) -> None:
    assert tokenize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("don't", ["don", "t"]),
        ("state-of-the-art", ["state", "of", "the", "art"]),
        ("snake_case_name", ["snake", "case", "name"]),
        ("C++", ["c"]),
        ("Python3.14", ["python3", "14"]),
        ("192.168.0.1", ["192", "168", "0", "1"]),
        ("search \U0001f50d engine", ["search", "engine"]),
    ],
)
def test_every_non_alphanumeric_character_separates(
    text: str, expected: list[str]
) -> None:
    """Including ones a reader might expect to be kept: ``_ + . '``."""
    assert tokenize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (NO_BREAK_SPACE, []),
        (f"a{NO_BREAK_SPACE}b", ["a", "b"]),
        (f"a{ZERO_WIDTH_JOINER}b", ["a", "b"]),
        (f"co{SOFT_HYPHEN}operate", ["co", "operate"]),
    ],
)
def test_invisible_characters_separate_too(text: str, expected: list[str]) -> None:
    """A soft hyphen renders as nothing, yet it splits a word in two."""
    assert tokenize(text) == expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("café", ["caf"]),
        ("naïve", ["na", "ve"]),
        ("Müller", ["m", "ller"]),
        ("Straße", ["stra", "e"]),  # sharp s
        ("ﬁre", ["re"]),  # fi ligature, dropped whole
        ("x²", ["x"]),  # superscript two
        (FULLWIDTH_AB, []),
        ("Ⅳ", []),  # Roman numeral four
        ("日本語", []),
        ("Ελλάδα", []),
    ],
)
def test_the_ascii_only_rule_mangles_non_ascii_words(
    word: str, expected: list[str]
) -> None:
    """Whole scripts collapse to nothing, so they are unsearchable.

    Normalised to NFC first so the result does not depend on how this file was
    written to disk. NFC leaves ligatures, superscripts, Roman numerals and
    fullwidth forms intact, unlike NFKC.
    """
    assert tokenize(unicodedata.normalize("NFC", word)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (KELVIN_SIGN, ["k"]),
        (f"{TURKISH_CAPITAL_I}STANBUL", ["i", "stanbul"]),
    ],
)
def test_lowercasing_can_pull_characters_into_the_ascii_set(
    text: str, expected: list[str]
) -> None:
    """Which is why filtering cannot happen before lowercasing."""
    assert tokenize(text) == expected


def test_lowercasing_is_not_length_preserving() -> None:
    """U+0130 lowercases to two code points, so character offsets do not survive."""
    text = f"{TURKISH_CAPITAL_I}STANBUL"
    assert len(text.lower()) == len(text) + 1


@pytest.mark.parametrize(
    ("word", "composed_tokens", "decomposed_tokens"),
    [
        ("café", ["caf"], ["cafe"]),
        ("naïve", ["na", "ve"], ["nai", "ve"]),
        ("Müller", ["m", "ller"], ["mu", "ller"]),
    ],
)
def test_normalisation_form_changes_the_tokens_of_the_same_word(
    word: str,
    composed_tokens: list[str],
    decomposed_tokens: list[str],
) -> None:
    """Identical on screen, different tokens, so such words fail to match.

    Composed: the accent is one non-ASCII code point, so the whole letter goes.
    Decomposed: the base letter is ASCII, so only the combining mark goes.
    Pinned here so the eventual fix cannot happen silently.
    """
    composed = unicodedata.normalize("NFC", word)
    decomposed = unicodedata.normalize("NFD", word)
    assert composed != decomposed
    assert tokenize(composed) == composed_tokens
    assert tokenize(decomposed) == decomposed_tokens


@given(st.text())
def test_every_token_matches_the_specified_shape(text: str) -> None:
    for token in tokenize(text):
        assert TOKEN_SHAPE.fullmatch(token) is not None


@given(st.text())
def test_rejoining_tokens_and_retokenizing_is_stable(text: str) -> None:
    """Order, completeness and separator collapsing, in one invariant."""
    tokens = tokenize(text)
    assert tokenize(" ".join(tokens)) == tokens


@given(st.text(), st.text())
def test_tokenizing_distributes_over_a_separator(first: str, second: str) -> None:
    """What makes it safe to tokenize a title and body concatenated together."""
    assert tokenize(f"{first} {second}") == tokenize(first) + tokenize(second)


@given(st.text(alphabet=" \t\n\r!@#$%^&*()-_=+[]{};:'\",.<>/?\\|`~"))
def test_text_with_no_ascii_alphanumerics_yields_no_tokens(text: str) -> None:
    assert tokenize(text) == []
