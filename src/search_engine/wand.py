"""Early termination: return the best documents without scoring the rest.

`BaseRanker.rank` scores every candidate and then keeps ten of them. Scoring a
document that cannot reach the results is waste, and on a common term it is
almost all of the work.

WAND removes it. Each query term is walked by its own cursor over that term's
postings in document order, and each cursor carries an upper bound on what its
term can contribute to any document. Adding those bounds in cursor order gives
the largest score any document at or before a given position could reach. When
that total cannot beat the lowest result already held, no document there is
worth scoring and the cursors jump past it.

Two properties make it exact rather than approximate.

Cursors are kept sorted by document identifier, which is what makes "no cursor
before the pivot has reached it yet" true, and is why postings have to be
document-ordered.

The bound is compared strictly greater than the threshold, which skips a
document whose best possible score only equals the lowest result held. That is
safe because documents are walked in ascending identifier order and ties are
resolved on the lower identifier, so every document that could win such a tie
has already been scored. Comparing greater or equal is equally exact and only
scores more documents.

**BM25 only.** The bound comes from `BM25Ranker.upper_bound`, which holds
because the score rises with term frequency and falls with document length.
TF-IDF has no equivalent: its cosine normalisation divides by a per-document
vector length that is not known until the document is chosen, so there is
nothing to bound the contribution with in advance.

A single segment is walked through its own skip pointers, so a cursor jumps
without decoding the postings it passes over. Every other index is walked over
the postings mapping it returns. A directory index is deliberately left on that
path: it filters deleted documents inside `postings`, and reading the segment
bytes underneath it would return documents a tombstone has removed.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import TYPE_CHECKING

from search_engine.ranking import TopK
from search_engine.segment import SegmentReader, advance_to

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from search_engine.bm25 import BM25Ranker
    from search_engine.codecs import Bytelike
    from search_engine.index import ReadableIndex
    from search_engine.segment import TermEntry


@dataclass(frozen=True, slots=True)
class WandResult:
    """The documents a search returned, and how many it scored to return them.

    The count is what a benchmark compares against the candidate set. It does
    not move with machine load, so it measures the pruning rather than the
    machine.
    """

    documents: list[tuple[int, float]]
    scored: int


class _Cursor:
    """One query term's postings, walked in ascending document order."""

    __slots__ = ("bound", "document")

    def __init__(self, bound: float) -> None:
        self.bound = bound
        self.document: int | None = None

    def advance(self, target: int) -> None:
        """Move to the first document at or after target, or past the end."""
        raise NotImplementedError


class _MappingCursor(_Cursor):
    """A cursor over postings an index has already materialised."""

    __slots__ = ("_at", "_documents")

    def __init__(self, bound: float, documents: list[int]) -> None:
        super().__init__(bound)
        self._documents = documents
        self._at = 0
        self.document = documents[0]

    def advance(self, target: int) -> None:
        """Move to the first document at or after target, or past the end."""
        self._at = bisect_left(self._documents, target, self._at)
        if self._at == len(self._documents):
            self.document = None
        else:
            self.document = self._documents[self._at]


class _SegmentCursor(_Cursor):
    """A cursor over a segment's postings, read straight from the mapped file.

    Nothing is decoded but the document identifiers walked past, and the skip
    list carries the walk over most of those.
    """

    __slots__ = ("_data", "_entry")

    def __init__(self, bound: float, data: Bytelike, entry: TermEntry) -> None:
        super().__init__(bound)
        self._data = data
        self._entry = entry
        self.document = advance_to(data, entry, 0)

    def advance(self, target: int) -> None:
        """Move to the first document at or after target, or past the end."""
        self.document = advance_to(self._data, self._entry, target)


def rank_wand(
    index: ReadableIndex,
    ranker: BM25Ranker,
    terms: Sequence[str],
    limit: int,
) -> list[tuple[int, float]]:
    """Return the best scoring documents without scoring all of them.

    Equivalent to ranker.rank over every candidate, and typically much cheaper,
    because a document is scored only when the remaining terms could still lift
    it into the results.

    Raises:
        ValueError: If limit is not positive.
    """
    return search_wand(index, ranker, terms, limit).documents


def search_wand(
    index: ReadableIndex,
    ranker: BM25Ranker,
    terms: Sequence[str],
    limit: int,
) -> WandResult:
    """Return what `rank_wand` returns, and how many documents were scored.

    Raises:
        ValueError: If limit is not positive.
    """
    results = TopK(limit)
    weights = ranker.query_weights(terms)
    cursors = _open_cursors(index, ranker, weights)
    scored = 0
    while True:
        live = _in_document_order(cursors)
        pivot = _pivot(live, results.threshold)
        if pivot is None:
            break
        if _step(live, pivot, ranker, weights, results):
            scored += 1
    return WandResult(results.best(), scored)


def _step(
    live: Sequence[tuple[int, _Cursor]],
    pivot: int,
    ranker: BM25Ranker,
    weights: Mapping[str, float],
    results: TopK,
) -> bool:
    """Score the pivot document, or move a cursor closer to it.

    Returns whether a document was scored. Only the cursors already positioned
    on the pivot document move past it: the rest have not reached it yet, and
    moving them would skip documents between here and there.
    """
    target = live[pivot][0]
    if live[0][0] != target:
        live[0][1].advance(target)
        return False
    results.push(target, ranker.score(weights, target))
    for document, cursor in live:
        if document == target:
            cursor.advance(target + 1)
    return True


def _open_cursors(
    index: ReadableIndex, ranker: BM25Ranker, weights: Mapping[str, float]
) -> list[_Cursor]:
    """Return one cursor per query term that some document holds."""
    cursors: list[_Cursor] = []
    for term in weights:
        cursor = _open_cursor(index, ranker, term)
        if cursor is not None:
            cursors.append(cursor)
    return cursors


def _open_cursor(index: ReadableIndex, ranker: BM25Ranker, term: str) -> _Cursor | None:
    """Return a cursor over one term's postings, or None if no document holds it."""
    if isinstance(index, SegmentReader):
        entry = index.dictionary.lookup(term)
        if entry is None:
            return None
        bound = ranker.upper_bound(term, entry.max_term_frequency)
        return _SegmentCursor(bound, index.raw, entry)
    postings = index.postings(term)
    if not postings:
        return None
    highest = max(len(positions) for positions in postings.values())
    return _MappingCursor(ranker.upper_bound(term, highest), sorted(postings))


def _in_document_order(cursors: Iterable[_Cursor]) -> list[tuple[int, _Cursor]]:
    """Return the cursors still holding a document, lowest identifier first.

    The identifier is carried alongside so that the caller reads it without
    having to prove again that it is not None.
    """
    live = [
        (cursor.document, cursor) for cursor in cursors if cursor.document is not None
    ]
    live.sort(key=lambda entry: entry[0])
    return live


def _pivot(live: Sequence[tuple[int, _Cursor]], threshold: float) -> int | None:
    """Return the first cursor whose document could still beat the threshold.

    None means no remaining document can, which is where the search stops: the
    cursors are in document order, so every document still to come is at or
    after this one and no better placed than it.
    """
    running = 0.0
    for at, (_, cursor) in enumerate(live):
        running += cursor.bound
        if running > threshold:
            return at
    return None
