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
import time
import unicodedata
from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.tokenizer import CJK_RANGES, DEFAULT_FORM, tokenize

TOKEN_SHAPE = re.compile(r"[^\W_]+")
ASCII_SHAPE = re.compile(r"[a-z0-9]+")
LINEAR_TIME_BUDGET = 2.0

NO_BREAK_SPACE = chr(0xA0)
SOFT_HYPHEN = chr(0xAD)
ZERO_WIDTH_JOINER = chr(0x200D)
KELVIN_SIGN = chr(0x212A)
TURKISH_CAPITAL_I = chr(0x130)
FULLWIDTH_AB = chr(0xFF21) + chr(0xFF22)
JAPANESE = "".join(chr(point) for point in (0x65E5, 0x672C, 0x8A9E))
TOKYO_UNIVERSITY = "".join(chr(point) for point in (0x6771, 0x4EAC, 0x5927, 0x5B66))
GREEK = "".join(chr(point) for point in (0x395, 0x3BB, 0x3BB, 0x3AC, 0x3B4, 0x3B1))
CYRILLIC = "".join(chr(point) for point in (0x41F, 0x440, 0x438, 0x432, 0x435, 0x442))
ARABIC = "".join(chr(point) for point in (0x645, 0x631, 0x62D, 0x628, 0x627))
HEBREW = "".join(chr(point) for point in (0x5E9, 0x5DC, 0x5D5, 0x5DD))
KOREAN = "".join(chr(point) for point in (0xD55C, 0xAD6D, 0xC5B4))
EMOJI = chr(0x1F50D) + chr(0x1F389)
COMBINING_ACUTE = chr(0x301)
HANGUL_SYLLABLE = chr(0xAC01)
HANGUL_JAMO = chr(0x1100) + chr(0x1161) + chr(0x11A8)
LIGATURE_FI = chr(0xFB01)
MATHEMATICAL_BOLD_A = chr(0x1D400)
ROMAN_NUMERAL_FOUR = chr(0x2163)


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
    ("script", "text", "expected_tokens"),
    [
        ("greek", GREEK, 1),
        ("cyrillic", CYRILLIC, 1),
        ("arabic", ARABIC, 1),
        ("hebrew", HEBREW, 1),
        ("japanese", JAPANESE, 2),
        ("korean", KOREAN, 2),
    ],
)
def test_six_scripts_that_used_to_be_unsearchable_now_tokenize(
    script: str, text: str, expected_tokens: int
) -> None:
    """Not degraded before, unsearchable: every one of these produced nothing.

    The alphabetic scripts give one token per word. The two written without
    spaces give overlapping bigrams instead, which is why their counts differ
    from their word counts.
    """
    tokens = tokenize(text)
    assert len(tokens) == expected_tokens, script
    assert all(TOKEN_SHAPE.fullmatch(token) for token in tokens)


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("café", ["café"]),
        ("naïve", ["naïve"]),
        ("Müller", ["müller"]),
        ("Straße", ["straße"]),
    ],
)
def test_an_accented_word_is_now_one_token(word: str, expected: list[str]) -> None:
    """It used to be split at every accent, so ``café`` indexed as ``caf``."""
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
    ("word", "expected"),
    [
        ("café", ["café"]),
        ("naïve", ["naïve"]),
        ("Müller", ["müller"]),
    ],
)
def test_both_spellings_of_a_word_produce_the_same_tokens(
    word: str, expected: list[str]
) -> None:
    """The defect this replaced: identical on screen, different tokens.

    Composed, the accent is one non-ASCII code point and the whole letter goes.
    Decomposed, the base letter is ASCII and only the combining mark goes. Before
    normalization those produced different terms, so a document stored in one
    form could never match a query typed in the other. Both are composed first
    now, so both give the composed answer.
    """
    composed = unicodedata.normalize("NFC", word)
    decomposed = unicodedata.normalize("NFD", word)
    assert composed != decomposed
    assert tokenize(composed) == tokenize(decomposed) == expected


@given(st.text())
def test_any_text_tokenizes_the_same_in_either_spelling(text: str) -> None:
    """The property the fix exists to provide, over generated input."""
    assert tokenize(unicodedata.normalize("NFC", text)) == tokenize(
        unicodedata.normalize("NFD", text)
    )


@given(st.text())
def test_normalizing_first_changes_nothing(text: str) -> None:
    """Tokenizing already-normalized text gives what tokenizing the raw text gives.

    Normalization is idempotent, so a caller that normalized its own text is not
    penalised and not treated differently.
    """
    assert tokenize(unicodedata.normalize(DEFAULT_FORM, text)) == tokenize(text)


@given(st.text(alphabet=st.characters(max_codepoint=127)))
def test_ascii_text_is_untouched_by_normalization(text: str) -> None:
    """The common case must be unaffected, since ASCII is already normalized."""
    assert unicodedata.is_normalized(DEFAULT_FORM, text)
    assert tokenize(text) == [
        match.group() for match in TOKEN_SHAPE.finditer(text.lower())
    ]


def test_a_syllable_composed_from_three_parts_tokenizes_either_way() -> None:
    """Hangul composes from jamo, which is where naive normalization breaks.

    Three code points become one, so any implementation assuming a fixed length
    or a one-to-one mapping fails here. One syllable is below the bigram length,
    so it stays a single token.
    """
    assert len(HANGUL_JAMO) == 3
    assert unicodedata.normalize("NFC", HANGUL_JAMO) == HANGUL_SYLLABLE
    assert tokenize(HANGUL_JAMO) == tokenize(HANGUL_SYLLABLE) == [HANGUL_SYLLABLE]
    assert tokenize(f"a{HANGUL_JAMO}b") == ["a", HANGUL_SYLLABLE, "b"]


def test_a_code_point_above_the_basic_plane_is_handled() -> None:
    """U+1D400 needs more than 16 bits, and is a word character in its own right."""
    assert tokenize(MATHEMATICAL_BOLD_A) == [MATHEMATICAL_BOLD_A]
    assert tokenize(MATHEMATICAL_BOLD_A, "NFKC") == ["a"]


@pytest.mark.parametrize(
    ("text", "under_nfc", "under_nfkc"),
    [
        (f"{LIGATURE_FI}re", [f"{LIGATURE_FI}re"], ["fire"]),
        ("x²", ["x²"], ["x2"]),
        (ROMAN_NUMERAL_FOUR, [chr(0x2173)], ["iv"]),
        (FULLWIDTH_AB, [chr(0xFF41) + chr(0xFF42)], ["ab"]),
    ],
)
def test_the_compatibility_form_recovers_tokens_and_loses_distinctions(
    text: str, under_nfc: list[str], under_nfkc: list[str]
) -> None:
    """Both halves of the NFKC trade, in one table.

    It recovers a ligature and a fullwidth word that NFC drops, and it makes a
    superscript two indistinguishable from an ordinary two, which is wrong
    wherever the difference carries meaning.
    """
    assert tokenize(text) == under_nfc
    assert tokenize(text, "NFKC") == under_nfkc


def test_the_two_forms_can_differ_in_length() -> None:
    """A ligature is one code point under NFC and two under NFKC."""
    assert len(unicodedata.normalize("NFC", LIGATURE_FI)) == 1
    assert len(unicodedata.normalize("NFKC", LIGATURE_FI)) == 2


@given(st.text())
def test_every_token_matches_the_specified_shape(text: str) -> None:
    for token in tokenize(text):
        assert TOKEN_SHAPE.fullmatch(token) is not None


@given(st.text(alphabet=st.characters(max_codepoint=127)))
def test_ascii_text_still_tokenizes_exactly_as_it_did(text: str) -> None:
    """Widening the pattern must not have moved the case that already worked."""
    assert tokenize(text) == ASCII_SHAPE.findall(text.lower())


@pytest.mark.parametrize(
    ("length", "expected"),
    [(1, 1), (2, 1), (3, 2), (4, 3)],
)
def test_a_run_without_spaces_becomes_overlapping_bigrams(
    length: int, expected: int
) -> None:
    """One character stays whole; every longer run gives length minus one."""
    run = TOKYO_UNIVERSITY[:length]
    tokens = tokenize(run)
    assert len(tokens) == expected
    assert all(len(token) <= 2 for token in tokens)
    assert "".join(token[0] for token in tokens) == run[: len(tokens)]


def test_a_run_without_spaces_overlaps_rather_than_partitions() -> None:
    """Partitioning would miss any word straddling a pair boundary."""
    tokens = tokenize(TOKYO_UNIVERSITY)
    assert tokens == [
        TOKYO_UNIVERSITY[0:2],
        TOKYO_UNIVERSITY[1:3],
        TOKYO_UNIVERSITY[2:4],
    ]


def test_scripts_are_cut_apart_within_a_single_match() -> None:
    """A run can hold both kinds, and each half needs its own rule."""
    assert tokenize(f"abc{JAPANESE}def") == [
        "abc",
        JAPANESE[0:2],
        JAPANESE[1:3],
        "def",
    ]


@pytest.mark.parametrize("text", [EMOJI, COMBINING_ACUTE, f"{EMOJI}{COMBINING_ACUTE}"])
def test_characters_that_are_neither_words_nor_syllables_yield_nothing(
    text: str,
) -> None:
    """Neither a word character nor a spaceless script, and neither raises."""
    assert tokenize(text) == []


def test_a_pathological_input_tokenizes_in_linear_time() -> None:
    """The pattern must not backtrack: query text is untrusted input.

    100,000 characters that almost match, run against a clock generous enough
    that only catastrophic backtracking could fail it.
    """
    start = time.perf_counter()
    assert tokenize("a" * 100_000 + "!") == ["a" * 100_000]
    assert tokenize("ab!" * 33_333) == ["ab"] * 33_333
    assert time.perf_counter() - start < LINEAR_TIME_BUDGET


def test_every_cjk_range_is_ordered_and_disjoint() -> None:
    """A range written backwards would silently classify nothing."""
    for low, high in CJK_RANGES:
        assert low < high
    for (_, first_high), (second_low, _) in pairwise(CJK_RANGES):
        assert first_high < second_low


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
