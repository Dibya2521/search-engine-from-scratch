"""Scoring a document higher when its query terms appear close together.

Matching and ranking are independent everywhere else in this engine. A phrase
query decides *which* documents match, and the scorer then treats every one of
them as a bag of words, so word order changes what is returned and never how
well it scores. A document with `machine learning` in its title is more about
the subject than one mentioning `machine` in the first paragraph and `learning`
in the fortieth.

**This needs no format change.** Postings already map a document to the
positions of a term inside it, stored for phrase queries, and the positions are
exactly what a proximity measure needs.

**The measure is the minimum window span**: the width of the smallest stretch of
text holding at least one occurrence of every query term. Adjacent terms give a
span equal to the number of terms, which is the smallest a window can be, and
terms scattered across a long document give a wide one.

The sweep keeps one cursor per term and repeatedly advances whichever sits
earliest, recording the window between the earliest and latest cursors each
time. Every position is visited once, so the cost is proportional to the
positions rather than to their product.

**The 0.3 weight is a convention, not a measurement.** It is the value commonly
quoted for this kind of boost. Tuning it means a test collection whose queries
are hard enough for a ranking change to show, and the collection here is not:
see `docs/06-ranking.md`. The weight is a module constant so a later measurement
can move it without touching the algorithm.
"""

from __future__ import annotations

from heapq import heapify, heappop, heappush
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

PROXIMITY_WEIGHT: Final = 0.3

# Proximity between one term and itself is not a question, so a single-term
# query is left alone rather than given a boost every document shares.
MINIMUM_TERMS: Final = 2


def minimum_span(position_lists: Sequence[Sequence[int]]) -> int | None:
    """Return the width of the smallest window containing one of every term.

    Each list must be in ascending order, which is how postings store them.
    Returns None when there are no terms, or when any term has no positions,
    because no window containing all of them exists.
    """
    if not position_lists or any(not positions for positions in position_lists):
        return None
    cursors = [(positions[0], at, 0) for at, positions in enumerate(position_lists)]
    heapify(cursors)
    latest = max(positions[0] for positions in position_lists)
    best = latest - cursors[0][0] + 1
    while True:
        _, at, offset = heappop(cursors)
        following = offset + 1
        if following == len(position_lists[at]):
            return best
        position = position_lists[at][following]
        latest = max(latest, position)
        heappush(cursors, (position, at, following))
        best = min(best, latest - cursors[0][0] + 1)


def proximity_boost(span: int | None, term_count: int) -> float:
    """Return a score multiplier for a minimum window span.

    A span equal to the term count means the terms are adjacent and earns the
    full weight. The multiplier decays as the window widens, and is exactly 1.0
    when the terms never occur together, so a document is never penalised for
    proximity, only rewarded for it.
    """
    if span is None or term_count < MINIMUM_TERMS:
        return 1.0
    return 1.0 + PROXIMITY_WEIGHT * (term_count / span)
