"""Split text into the tokens the index stores.

Normalize, lowercase, keep maximal runs of word characters, then cut any run of
CJK characters into overlapping bigrams.

Normalizing comes first because one word can be written as more than one
sequence of code points. ``cafe`` with a composed acute accent is a single
non-ASCII code point; the same word with a combining accent is an ASCII ``e``
followed by a mark. The two render identically, so without normalizing first a
document stored in one form could never match a query typed in the other.

**The pattern matches word characters in every script, not only ASCII.** An
ASCII-only pattern produces no tokens at all for Chinese, Japanese, Korean,
Greek, Cyrillic, Hebrew or Arabic, which leaves those documents unsearchable
rather than merely ranked badly.

Chinese and Japanese are written without spaces, so a run of them arrives as one
enormous token. Splitting it properly needs a dictionary and a segmentation
model, which is a different project. Overlapping character bigrams are the
standard cheap answer and what Lucene's CJK analyser does. They cost roughly
twice the postings, and they match across word boundaries, so precision is
traded for recall that an ASCII-only pattern does not provide at all.

Documents and queries must both go through this function. If they diverge,
queries produce terms the index never stored.
"""

from __future__ import annotations

import re
import unicodedata
from itertools import groupby
from typing import Final, Literal

type NormalForm = Literal["NFC", "NFKC"]

# Word characters except the underscore, which stays a separator as every other
# punctuation mark does. Applied after lowercasing, so no A-Z is needed. Public
# because the analyzer fingerprint is built from it: changing this changes every
# term. It has no alternation and no nested quantifier, so matching is linear in
# the input; query text is untrusted and a backtracking pattern would be a way
# to hang the engine. Keep any replacement linear too.
TOKEN_PATTERN: Final = re.compile(r"[^\W_]+")

DEFAULT_FORM: Final[NormalForm] = "NFC"

# Scripts written without spaces between words, so a run of them is one match.
CJK_RANGES: Final = (
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
)

_BIGRAM = 2


def tokenize(text: str, form: NormalForm = DEFAULT_FORM) -> list[str]:
    """Return tokens in order of appearance.

    Normalizing must come before lowercasing, and lowercasing before matching:
    some code points lowercase *into* ASCII, such as U+212A becoming a plain
    ``k``.

    ``NFC`` merges spellings that render identically and changes nothing else.
    ``NFKC`` also folds compatibility forms, which recovers tokens that would
    otherwise be dropped whole, and in doing so destroys distinctions: a
    superscript two and an ordinary two become the same token. Anything indexed
    with a form must be queried with it, and `analysis.fingerprint` takes the
    form for exactly that reason.
    """
    normalized = unicodedata.normalize(form, text).lower()
    return [
        token
        for match in TOKEN_PATTERN.finditer(normalized)
        for token, _ in _segment(match.group())
    ]


def tokenize_spans(
    text: str, form: NormalForm = DEFAULT_FORM
) -> list[tuple[str, int, int]]:
    """Return each token with the range of text it came from.

    Ranges index into ``unicodedata.normalize(form, text)`` and not into
    ``text``, because normalization composes two code points into one and
    nothing maps an offset back across that. A caller slicing by these ranges
    normalizes the text the same way first.

    Produces the same tokens as `tokenize`, from the same code, which is the
    only way to be sure a highlighted word is a word that matched.
    """
    normalized = unicodedata.normalize(form, text)
    lowered = normalized.lower()
    origins = _origins(normalized, lowered)
    spans: list[tuple[str, int, int]] = []
    for match in TOKEN_PATTERN.finditer(lowered):
        for token, offset in _segment(match.group()):
            start = match.start() + offset
            spans.append((token, *_locate(origins, start, start + len(token))))
    return spans


def _origins(normalized: str, lowered: str) -> list[int] | None:
    """Map each lowered position back to the character it came from.

    None when lowering changed no length, which is every text not containing
    U+0130, the one code point in Unicode whose lowercase is two characters.
    Building the map is skipped in that case because it costs a list entry per
    character of the document.
    """
    if len(lowered) == len(normalized):
        return None
    return [at for at, character in enumerate(normalized) for _ in character.lower()]


def _locate(origins: list[int] | None, start: int, end: int) -> tuple[int, int]:
    return (start, end) if origins is None else (origins[start], origins[end - 1] + 1)


def _segment(run: str) -> list[tuple[str, int]]:
    """Split one match into CJK bigrams and whole words, with their offsets.

    A match can hold both, as a product name with Latin and Japanese in it
    does, so the run is cut at every change of script rather than classified as
    a whole.
    """
    pieces: list[tuple[str, int]] = []
    at = 0
    for cjk, characters in groupby(run, _is_cjk):
        piece = "".join(characters)
        if cjk:
            pieces.extend((bigram, at + start) for start, bigram in _bigrams(piece))
        else:
            pieces.append((piece, at))
        at += len(piece)
    return pieces


def _bigrams(run: str) -> list[tuple[int, str]]:
    """Return a run as overlapping pairs, or whole when it is a single character.

    Overlapping rather than partitioning, because a partition would miss any
    word straddling a pair boundary.
    """
    if len(run) < _BIGRAM:
        return [(0, run)]
    return [(start, run[start : start + _BIGRAM]) for start in range(len(run) - 1)]


def _is_cjk(character: str) -> bool:
    """Return whether a character belongs to a script written without spaces."""
    point = ord(character)
    return any(low <= point <= high for low, high in CJK_RANGES)
