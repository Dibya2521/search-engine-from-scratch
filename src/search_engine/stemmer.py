"""Porter stemming algorithm, reducing inflected words to a common form.

Why the index needs it: `connect`, `connected`, `connecting` and `connection`
describe one concept, so a query for any of them should match documents using
the others. Without stemming they are four unrelated terms.

Five ordered steps of suffix rewriting, each gated on how many
vowel-consonant sequences the remaining stem has. Not a dictionary lookup, so
the output is often not a word: `relational` becomes `relat`. Documents and
queries pass through the same function and meet on the same non-word.

Within a step, the rule with the *longest* matching suffix fires. If its
condition fails the step makes no change; a shorter rule does not get a turn.

Follows the algorithm author's revised rules, which generalise `abli -> able`
to `bli -> ble` and add `logi -> log`. The latter is what lets `technology`
and `technological` meet.
"""

from __future__ import annotations

from typing import Final

# Names the variant implemented here, which is the algorithm author's revised
# rules rather than the 1980 paper. Part of the analyzer fingerprint, so
# changing the rules invalidates indexes built with the old ones.
ALGORITHM_VERSION: Final = "porter-revised"


_VOWELS: Final = frozenset("aeiou")

# Shortest word worth stemming. Below this the algorithm can consume the word
# entirely, and an empty term is unusable in an index.
_MIN_STEMMABLE_LENGTH: Final = 3
_CVC_LENGTH: Final = 3
_MIN_DOUBLE_LENGTH: Final = 2

# Suffix, replacement. Order is irrelevant: the longest match is selected.
_STEP2_RULES: Final = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("bli", "ble"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
    ("logi", "log"),
)

_STEP3_RULES: Final = (
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
)

# Every step 4 rule deletes its suffix outright.
_STEP4_SUFFIXES: Final = (
    "al",
    "ance",
    "ence",
    "er",
    "ic",
    "able",
    "ible",
    "ant",
    "ement",
    "ment",
    "ent",
    "ion",
    "ou",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
)


def _consonant_flags(word: str) -> list[bool]:
    """Mark each position as consonant or vowel.

    ``y`` is the awkward case: a consonant at the start of a word or after a
    vowel, a vowel after a consonant. So ``toy`` ends in a consonant while the
    middle letters of ``syzygy`` are vowels. Computed left to right rather than
    recursively, since a long run of ``y`` would otherwise recurse per letter.
    """
    flags: list[bool] = []
    for index, letter in enumerate(word):
        if letter in _VOWELS:
            flags.append(False)
        elif letter == "y":
            flags.append(index == 0 or not flags[index - 1])
        else:
            flags.append(True)
    return flags


def _measure(base: str) -> int:
    """Count vowel-consonant sequences, the algorithm's notion of word length.

    Every word has the shape ``[C](VC)^m[V]``, and this returns ``m``. Step
    conditions test it so a suffix only goes when enough word survives beneath
    it: ``ate`` goes from ``activate`` but not from ``rate``.
    """
    groups: list[bool] = []
    for flag in _consonant_flags(base):
        if not groups or groups[-1] != flag:
            groups.append(flag)
    return sum(
        1 for index in range(1, len(groups)) if groups[index] and not groups[index - 1]
    )


def _contains_vowel(base: str) -> bool:
    return any(not flag for flag in _consonant_flags(base))


def _ends_with_double_consonant(word: str) -> bool:
    if len(word) < _MIN_DOUBLE_LENGTH or word[-1] != word[-2]:
        return False
    return _consonant_flags(word)[-1]


def _ends_cvc(word: str) -> bool:
    """Consonant, vowel, consonant, where the last is not ``w``, ``x`` or ``y``.

    Marks a short word whose final consonant was doubled or needs an ``e``
    restored, which is how ``filing`` keeps its ``e`` and ``failing`` does not.
    """
    if len(word) < _CVC_LENGTH or word[-1] in "wxy":
        return False
    flags = _consonant_flags(word)
    return flags[-3] and not flags[-2] and flags[-1]


def _longest_suffix(word: str, suffixes: tuple[str, ...]) -> str | None:
    matches = [suffix for suffix in suffixes if word.endswith(suffix)]
    return max(matches, key=len) if matches else None


def _rewrite_longest(word: str, rules: tuple[tuple[str, str], ...]) -> str:
    suffix = _longest_suffix(word, tuple(rule[0] for rule in rules))
    if suffix is None:
        return word
    base = word[: -len(suffix)]
    if _measure(base) == 0:
        return word
    replacement = next(rule[1] for rule in rules if rule[0] == suffix)
    return base + replacement


def _step1a(word: str) -> str:
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _restore_after_removal(base: str) -> str:
    """Repair a stem left malformed by dropping ``ed`` or ``ing``.

    Removing the suffix can leave something unpronounceable or ambiguous, so
    the vowel goes back (``conflat`` to ``conflate``), a doubled consonant
    collapses (``hopp`` to ``hop``), or a short stem regains its ``e``.
    """
    if base.endswith(("at", "bl", "iz")):
        return base + "e"
    if _ends_with_double_consonant(base) and base[-1] not in "lsz":
        return base[:-1]
    if _measure(base) == 1 and _ends_cvc(base):
        return base + "e"
    return base


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        return word[:-1] if _measure(word[:-3]) > 0 else word
    for suffix in ("ing", "ed"):
        if word.endswith(suffix):
            base = word[: -len(suffix)]
            return _restore_after_removal(base) if _contains_vowel(base) else word
    return word


def _step1c(word: str) -> str:
    if word.endswith("y") and _contains_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


def _step4(word: str) -> str:
    suffix = _longest_suffix(word, _STEP4_SUFFIXES)
    if suffix is None:
        return word
    base = word[: -len(suffix)]
    # A measure above one already guarantees several letters, so base is never
    # empty here.
    if _measure(base) <= 1:
        return word
    # ion only goes when it followed s or t, so religion keeps its i.
    if suffix == "ion" and base[-1] not in "st":
        return word
    return base


def _step5a(word: str) -> str:
    if not word.endswith("e"):
        return word
    base = word[:-1]
    measure = _measure(base)
    if measure > 1 or (measure == 1 and not _ends_cvc(base)):
        return base
    return word


def _step5b(word: str) -> str:
    if word.endswith("l") and _ends_with_double_consonant(word) and _measure(word) > 1:
        return word[:-1]
    return word


def stem(word: str) -> str:
    """Reduce a word to its stem.

    Very short words are returned untouched, since the rules can consume them
    entirely and leave an empty term.
    """
    if len(word) < _MIN_STEMMABLE_LENGTH:
        return word
    stemmed = _step1c(_step1b(_step1a(word)))
    stemmed = _rewrite_longest(stemmed, _STEP2_RULES)
    stemmed = _rewrite_longest(stemmed, _STEP3_RULES)
    return _step5b(_step5a(_step4(stemmed)))
