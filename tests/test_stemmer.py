"""Tests for the Porter stemmer.

The load-bearing test here is the differential one against an independent
implementation. Tests written from the same assumptions as the code can only
agree with it; a second implementation disagreeing is what actually falsifies
it. Every other expected value below was read off a real run first.
"""

# The measure and consonant helpers are internal, but their documented examples
# are part of the algorithm specification and are worth asserting directly.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import string

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.stemmer import (
    _consonant_flags,
    _measure,
    stem,
)
from tests._reference import reference_stem

words = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=20)


@pytest.mark.parametrize(
    ("word", "expected_measure"),
    [
        ("tr", 0),
        ("ee", 0),
        ("tree", 0),
        ("y", 0),
        ("by", 0),
        ("trouble", 1),
        ("oats", 1),
        ("trees", 1),
        ("ivy", 1),
        ("troubles", 2),
        ("private", 2),
        ("oaten", 2),
        ("orrery", 2),
    ],
)
def test_measure_counts_vowel_consonant_sequences(
    word: str, expected_measure: int
) -> None:
    """Every documented measure example. This gates every step condition."""
    assert _measure(word) == expected_measure


@pytest.mark.parametrize(
    ("word", "expected_flags"),
    [
        ("toy", [True, False, True]),
        ("syzygy", [True, False, True, False, True, False]),
        ("y", [True]),
        ("yes", [True, False, True]),
    ],
)
def test_y_is_a_consonant_only_at_the_start_or_after_a_vowel(
    word: str, expected_flags: list[bool]
) -> None:
    """The one letter whose class depends on its neighbour."""
    assert _consonant_flags(word) == expected_flags


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        # Plural and third-person forms.
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("caress", "caress"),
        ("cats", "cat"),
        # Past and progressive forms, and the repairs they need.
        ("agreed", "agre"),
        ("feed", "feed"),
        ("plastered", "plaster"),
        ("bled", "bled"),
        ("motoring", "motor"),
        ("sing", "sing"),
        ("conflated", "conflat"),
        ("troubled", "troubl"),
        ("hopping", "hop"),
        ("tanned", "tan"),
        ("falling", "fall"),
        ("hissing", "hiss"),
        ("fizzed", "fizz"),
        ("failing", "fail"),
        ("filing", "file"),
        # Terminal y becomes i so later rules can match it.
        ("happy", "happi"),
        ("sky", "sky"),
        # Derivational suffixes.
        ("relational", "relat"),
        ("rational", "ration"),
        ("digitizer", "digit"),
        ("vietnamization", "vietnam"),
        ("predication", "predic"),
        ("feudalism", "feudal"),
        ("decisiveness", "decis"),
        ("hopefulness", "hope"),
        ("callousness", "callous"),
        ("sensibiliti", "sensibl"),
        ("triplicate", "triplic"),
        ("formalize", "formal"),
        ("electrical", "electr"),
        ("goodness", "good"),
        ("revival", "reviv"),
        ("allowance", "allow"),
        ("airliner", "airlin"),
        ("adjustable", "adjust"),
        ("replacement", "replac"),
        ("adoption", "adopt"),
        ("communism", "commun"),
        ("activate", "activ"),
        ("homologous", "homolog"),
        ("effective", "effect"),
        ("bowdlerize", "bowdler"),
        # Final e and final double l.
        ("probate", "probat"),
        ("rate", "rate"),
        ("cease", "ceas"),
        ("controll", "control"),
        ("roll", "roll"),
    ],
)
def test_documented_examples(word: str, expected: str) -> None:
    assert stem(word) == expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("adoption", "adopt"),  # ion follows t, so it goes
        ("precision", "precis"),  # ion follows s, so it goes
        ("religion", "religion"),  # ion follows g, so it stays
        ("rebellion", "rebellion"),
        ("opinion", "opinion"),
        ("nation", "nation"),  # too little word left beneath the suffix
    ],
)
def test_ion_only_goes_when_it_followed_s_or_t(word: str, expected: str) -> None:
    assert stem(word) == expected


@pytest.mark.parametrize("word", ["a", "s", "is", "as", "by", "ss", "ie", "ed"])
def test_words_of_two_letters_or_fewer_are_untouched(word: str) -> None:
    """Without this guard the reference algorithm turns ``s`` into an empty term."""
    assert stem(word) == word


@pytest.mark.parametrize(
    ("word", "expected"),
    [("python3", "python3"), ("mp3s", "mp3"), ("x86", "x86"), ("123", "123")],
)
def test_digits_are_treated_as_consonants(word: str, expected: str) -> None:
    """The tokenizer emits alphanumerics, so digits must not crash or vanish."""
    assert stem(word) == expected


@pytest.mark.parametrize(
    "family",
    [
        ("connect", "connected", "connecting", "connection", "connections"),
        ("compute", "computer", "computing", "computed", "computation"),
        ("technology", "technologies", "technological", "technologically"),
        ("analogy", "analogous", "analogical"),
        ("relate", "related", "relating", "relational"),
    ],
)
def test_word_families_collapse_to_one_term(family: tuple[str, ...]) -> None:
    """The only thing that matters for retrieval: do variants meet on one term?"""
    assert len({stem(word) for word in family}) == 1


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # The stem left under -logi is too short for the rule to fire, so these
        # two forms of one word do not meet.
        ("biology", "biological"),
        # Over-stemming: -iviti strips more than the shared root.
        ("relativity", "relate"),
        # Under-stemming: -ment survives because the stem beneath it is short.
        ("argument", "argue"),
        ("apologetic", "apology"),
    ],
)
def test_known_families_that_fail_to_collapse(first: str, second: str) -> None:
    """Real limitations, pinned so a future change to them is visible."""
    assert stem(first) != stem(second)


@pytest.mark.parametrize(
    "word",
    [
        "caresses",
        "ponies",
        "agreed",
        "hopping",
        "filing",
        "happy",
        "relational",
        "vietnamization",
        "sensibiliti",
        "triplicate",
        "electrical",
        "adoption",
        "controll",
        "technology",
        "possibli",
        "apologies",
        "analogy",
        "biology",
        "s",
        "is",
        "python3",
    ],
)
def test_matches_the_reference_implementation_on_known_words(word: str) -> None:
    assert stem(word) == reference_stem(word)


@given(words)
def test_matches_the_reference_implementation_on_arbitrary_words(word: str) -> None:
    """The differential oracle, and the only test that can falsify the rules.

    Exempts words of two letters or fewer, where this implementation
    deliberately departs by leaving them alone.
    """
    if len(word) <= 2:
        return
    assert stem(word) == reference_stem(word)


@given(words)
def test_never_returns_an_empty_term(word: str) -> None:
    """An empty term in an index is unusable, so this is a hard requirement."""
    assert stem(word) != ""


@given(words)
def test_never_returns_more_than_it_was_given(word: str) -> None:
    """Every rule either deletes a suffix or swaps one for a shorter one."""
    assert len(stem(word)) <= len(word)


@given(words)
def test_always_preserves_the_first_letter(word: str) -> None:
    """No rule touches the front of a word, which keeps stems browsable."""
    assert stem(word)[0] == word[0]


@given(words)
def test_output_stays_alphabetic_for_alphabetic_input(word: str) -> None:
    assert stem(word).isalpha()


def test_stemming_is_not_idempotent_and_that_is_the_algorithm() -> None:
    """Stripping one suffix can expose another, so a second pass keeps going.

    Documented rather than asserted away: the reference implementation behaves
    the same, so this is the algorithm's nature and not a defect here. Callers
    must stem exactly once, on both documents and queries.
    """
    assert stem("atye") == "aty"
    assert stem("aty") == "ati"
    assert stem(stem("atye")) != stem("atye")
    assert reference_stem(reference_stem("atye")) != reference_stem("atye")
