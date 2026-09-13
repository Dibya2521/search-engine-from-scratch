"""The passage of a document that shows why it matched.

A list of identifiers and scores tells a person nothing about whether to open a
result. The passage under each one is what does.

**The passage is chosen by sliding a window over the document's terms** and
keeping the window that shows the most of the query. Windows are scored on the
number of distinct query terms first and the total number of occurrences
second: a window holding all three query terms once explains the match better
than one holding a single term five times. Ties go to the earlier window, so
the answer is deterministic and the opening of a document wins when nothing
distinguishes two passages.

**The window is then widened to the sentence it sits in**, within a bounded
number of terms, because a passage that begins mid-sentence reads as damage
rather than as an extract.

**Highlights come back as character ranges, not as marked-up text.** A terminal
wants ANSI codes, an HTTP response wants ranges in JSON, and a test wants
neither. Marking in the wrong layer is expensive to undo later.

**The terms are found with the same function the index was built with**, which
is the only way to be sure a highlighted word is a word that matched. Stemming
means the highlighted characters are usually not the term itself: `retrieval`
in the text is `retriev` in the index, and the range covers the word as
written.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from search_engine.analysis import analyze, analyze_spans
from search_engine.tokenizer import DEFAULT_FORM

if TYPE_CHECKING:
    from collections.abc import Sequence
    from collections.abc import Set as AbstractSet

    from search_engine.tokenizer import NormalForm

SNIPPET_TOKENS: Final = 40
BOUNDARY_TOLERANCE: Final = 10
SENTENCE_END: Final = frozenset(".!?")

type Span = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class Snippet:
    """A passage of a document, and where the query terms fall inside it."""

    text: str
    highlights: tuple[tuple[int, int], ...]


def extract(
    text: str,
    query: str,
    width: int = SNIPPET_TOKENS,
    form: NormalForm = DEFAULT_FORM,
) -> Snippet:
    """Return the passage that best shows why a document matched a query.

    ``query`` is the query as typed, analysed here rather than taken already
    analysed. Stemming is not idempotent, so a sequence of strings would not
    say which it held, and `BaseRanker.rank` takes a query the same way.

    A document holding no terms at all comes back whole. It has no passage to
    choose between, and no such document can be a search result, because a
    document with no terms has no postings to match.

    Raises:
        ValueError: If the window would hold no terms.
    """
    if width < 1:
        message = f"width must be at least 1, got {width}"
        raise ValueError(message)
    normalized = unicodedata.normalize(form, text)
    spans = analyze_spans(normalized, form=form)
    if not spans:
        return Snippet(text=normalized, highlights=())

    wanted = frozenset(analyze(query, form=form))
    first, last = _best_window(spans, wanted, width)
    start = _open_at(normalized, spans, first)
    end = _close_at(normalized, spans, last)
    return Snippet(
        text=normalized[start:end],
        highlights=_highlights(_visible(spans, start, end, first, last), wanted, start),
    )


def _visible(
    spans: Sequence[Span], start: int, end: int, first: int, last: int
) -> Sequence[Span]:
    """Return every term inside the passage, not only those inside the window.

    Widening to a sentence brings terms in on either side of the window, and a
    query term the reader can see but that is not marked reads as a miss.
    """
    while first > 0 and spans[first - 1][1] >= start:
        first -= 1
    while last + 1 < len(spans) and spans[last + 1][2] <= end:
        last += 1
    return spans[first : last + 1]


def _best_window(
    spans: Sequence[Span], wanted: AbstractSet[str], width: int
) -> tuple[int, int]:
    """Return the first and last term index of the window showing the most query.

    One pass with a running count rather than one pass per window, so the cost
    is the document's length and not the document's length times the width.

    The comparison is strict, so the earliest window with a given score keeps
    it. That is what makes a query matching nothing return the opening of the
    document rather than an arbitrary part of it.
    """
    counts: dict[str, int] = {}
    distinct = 0
    total = 0
    best = (-1, -1)
    best_first = 0
    for last, (term, _, _) in enumerate(spans):
        if term in wanted:
            counts[term] = counts.get(term, 0) + 1
            total += 1
            distinct += 1 if counts[term] == 1 else 0
        if last >= width:
            gone = spans[last - width][0]
            if gone in wanted:
                counts[gone] -= 1
                total -= 1
                distinct -= 1 if counts[gone] == 0 else 0
        if (distinct, total) > best:
            best = (distinct, total)
            best_first = max(last - width + 1, 0)
    return best_first, min(best_first + width - 1, len(spans) - 1)


def _open_at(text: str, spans: Sequence[Span], first: int) -> int:
    """Return where the passage starts: a sentence edge, or the document's.

    Walking back from the window rather than forward from the document, so the
    cost is bounded by the tolerance and not by where the window landed.
    """
    for index in range(first, max(first - BOUNDARY_TOLERANCE, 0) - 1, -1):
        if index == 0:
            return 0
        gap = text[spans[index - 1][2] : spans[index][1]]
        if any(character in SENTENCE_END for character in gap):
            return spans[index - 1][2] + _past_terminator(gap)
    return spans[first][1]


def _close_at(text: str, spans: Sequence[Span], last: int) -> int:
    """Return where the passage ends, including the full stop that ends it."""
    for index in range(last, min(last + BOUNDARY_TOLERANCE, len(spans) - 1) + 1):
        if index == len(spans) - 1:
            return len(text)
        gap = text[spans[index][2] : spans[index + 1][1]]
        if any(character in SENTENCE_END for character in gap):
            return spans[index][2] + _first_terminator(gap) + 1
    return spans[last][2]


def _past_terminator(gap: str) -> int:
    """Return where the next sentence's own text begins within a gap.

    Past the terminator and past the space after it, so a passage opens on the
    sentence's first word rather than on its first word that is not a stopword.
    """
    last = max(at for at, character in enumerate(gap) if character in SENTENCE_END)
    tail = gap[last + 1 :]
    return last + 1 + len(tail) - len(tail.lstrip())


def _first_terminator(gap: str) -> int:
    return next(at for at, character in enumerate(gap) if character in SENTENCE_END)


def _highlights(
    window: Sequence[Span], wanted: AbstractSet[str], start: int
) -> tuple[tuple[int, int], ...]:
    """Return the query terms' ranges within the passage, merged and ascending.

    CJK bigrams overlap by a character, so two neighbouring matches can name
    overlapping ranges. Merging them is both what a reader wants to see and
    what keeps every range disjoint from the next.
    """
    merged: list[list[int]] = []
    for term, at, to in window:
        if term not in wanted:
            continue
        left, right = at - start, to - start
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return tuple((left, right) for left, right in merged)
