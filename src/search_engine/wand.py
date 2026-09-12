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

One bound for a whole term is loose. A term occurring fifty times in one
document and once everywhere else carries a bound set by that single outlier, so
almost nothing is pruned. A segment therefore stores a bound per block of
postings, and the search sums the bounds of the blocks the cursors currently sit
in. When that total cannot beat the threshold either, no document from here to
the earliest block end can, and every cursor jumps past its block without the
postings inside being read. The stop is held back to the next cursor's document,
since a term that no cursor here bounds could contribute beyond that point.

A single segment is walked through its block table, so a cursor jumps without
decoding the postings it passes over. Every other index is walked over the
postings mapping it returns and reports the term's own bound as its block bound,
which the pivot rule guarantees is above the threshold, so the block test never
fires there and nothing changes for it. A directory index is deliberately left
on that path: it filters deleted documents inside `postings`, and reading the
segment bytes underneath it would return documents a tombstone has removed.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import TYPE_CHECKING

from search_engine.access import allow_all
from search_engine.ranking import DOCUMENTS_SCORED, TopK
from search_engine.segment import (
    SegmentReader,
    block_identifiers,
    read_blocks,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from search_engine.access import Permit
    from search_engine.bm25 import BM25Ranker
    from search_engine.codecs import Bytelike
    from search_engine.index import ReadableIndex
    from search_engine.segment import PostingsBlock


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

    @property
    def block_bound(self) -> float:
        """Return the bound over the postings this cursor currently sits among.

        A cursor with no blocks reports the whole term's bound. The pivot is
        chosen so that the bounds up to it already exceed the threshold, so a
        sum including this one can never fall to or below it, and such a cursor
        is never asked for a block end.
        """
        return self.bound

    @property
    def block_end(self) -> int:
        """Return the last document identifier of the current block."""
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

    Only the block table and the identifiers of the block currently under the
    cursor are decoded. Positions are never touched, and a block the search
    jumps over is never read at all.
    """

    __slots__ = ("_at", "_blocks", "_bounds", "_data", "_identifiers", "_position")

    def __init__(
        self, data: Bytelike, blocks: list[PostingsBlock], bounds: list[float]
    ) -> None:
        # A term's highest frequency is the highest any of its blocks holds, so
        # the largest block bound is the bound for the term as a whole.
        super().__init__(max(bounds))
        self._data = data
        self._blocks = blocks
        self._bounds = bounds
        self._at = 0
        self._identifiers: list[int] = []
        self._position = 0
        self._open(0)

    @property
    def block_bound(self) -> float:
        """Return the bound over the block this cursor currently sits in."""
        return self._bounds[self._at]

    @property
    def block_end(self) -> int:
        """Return the last document identifier of the current block."""
        return self._blocks[self._at].last_document

    def advance(self, target: int) -> None:
        """Move to the first document at or after target, or past the end."""
        at = self._at
        while at < len(self._blocks) and self._blocks[at].last_document < target:
            at += 1
        if at == len(self._blocks):
            self.document = None
            return
        if at != self._at:
            self._open(at)
        self._position = bisect_left(self._identifiers, target, self._position)
        self.document = self._identifiers[self._position]

    def _open(self, at: int) -> None:
        """Decode one block's identifiers and sit on its first document."""
        self._at = at
        self._identifiers = block_identifiers(self._data, self._blocks[at])
        self._position = 0
        self.document = self._identifiers[0]


def rank_wand(
    index: ReadableIndex,
    ranker: BM25Ranker,
    terms: Sequence[str],
    limit: int,
    permit: Permit | None = None,
) -> list[tuple[int, float]]:
    """Return the best scoring documents without scoring all of them.

    Equivalent to ranker.rank over every candidate, and typically much cheaper,
    because a document is scored only when the remaining terms could still lift
    it into the results.

    Raises:
        ValueError: If limit is not positive, or if the ranker applies a
            proximity boost.
    """
    return search_wand(index, ranker, terms, limit, permit).documents


def search_wand(
    index: ReadableIndex,
    ranker: BM25Ranker,
    terms: Sequence[str],
    limit: int,
    permit: Permit | None = None,
) -> WandResult:
    """Return what `rank_wand` returns, and how many documents were scored.

    ``permit`` is consulted before a document is scored and before it can enter
    the results. That ordering is the whole of it: a forbidden document that
    reached the heap would set a threshold, and pruning against a threshold set
    by documents the caller cannot see drops documents they can.

    Raises:
        ValueError: If limit is not positive, or if the ranker applies a
            proximity boost.
    """
    _refuse_proximity(ranker)
    results = TopK(limit)
    weights = ranker.query_weights(terms)
    scoring = _Scoring(ranker, weights, permit or allow_all)
    cursors = _open_cursors(index, ranker, weights)
    scored = 0
    while True:
        live = _in_document_order(cursors)
        pivot = _pivot(live, results.threshold)
        if pivot is None:
            break
        if _step(live, pivot, scoring, results):
            scored += 1
    # Counted once rather than per document: the point of this path is that
    # the number is smaller than the candidate count, and a lock per skipped
    # document would cost more than the skipping saves.
    DOCUMENTS_SCORED.increment(scored)
    return WandResult(results.best(), scored)


@dataclass(frozen=True, slots=True)
class _Scoring:
    """What scoring one document takes: the scorer, the query, and who may see it."""

    ranker: BM25Ranker
    weights: Mapping[str, float]
    permit: Permit


def _step(
    live: Sequence[tuple[int, _Cursor]],
    pivot: int,
    scoring: _Scoring,
    results: TopK,
) -> bool:
    """Score the pivot document, jump over its blocks, or close the gap to it.

    Returns whether a document was scored. Only the cursors already positioned
    on the pivot document move past it: the rest have not reached it yet, and
    moving them would skip documents between here and there.
    """
    target = live[pivot][0]
    if live[0][0] != target:
        live[0][1].advance(target)
        return False
    if not scoring.permit(target):
        # Before the score and before the threshold, so a document this caller
        # cannot see never prunes one they can.
        _past(live, target)
        return False
    holders = live[: _last_holder(live) + 1]
    if sum(cursor.block_bound for _, cursor in holders) <= results.threshold:
        _skip_blocks(live, holders)
        return False
    results.push(target, scoring.ranker.score(scoring.weights, target))
    _past(live, target)
    return True


def _past(live: Sequence[tuple[int, _Cursor]], target: int) -> None:
    """Move every cursor sitting on a document to just past it."""
    for document, cursor in live:
        if document == target:
            cursor.advance(target + 1)


def _last_holder(live: Sequence[tuple[int, _Cursor]]) -> int:
    """Return the index of the last cursor sitting on the first one's document.

    Every cursor up to the pivot holds it, and a cursor past the pivot may hold
    it too. Leaving one of those out would leave its term unbounded in a total
    that decides whether the document can be skipped.
    """
    target = live[0][0]
    at = 0
    while at + 1 < len(live) and live[at + 1][0] == target:
        at += 1
    return at


def _skip_blocks(
    live: Sequence[tuple[int, _Cursor]], holders: Sequence[tuple[int, _Cursor]]
) -> None:
    """Move every cursor on the pivot document past the blocks it sits in.

    The stop is the first document after the earliest of those blocks ends, held
    back to the next cursor's document, because a term none of these cursors
    bounds could contribute at or beyond it.
    """
    stops = [cursor.block_end + 1 for _, cursor in holders]
    stops += [document for document, _ in live[len(holders) :]]
    stop = min(stops)
    for _, cursor in holders:
        cursor.advance(stop)


def _refuse_proximity(ranker: BM25Ranker) -> None:
    """Refuse a ranker whose scores this cannot reproduce.

    A proximity boost is applied by `rank` after scoring, and depends on where
    every query term sits in the document. The bounds here are per term and know
    nothing about position, so early termination would prune against one scale
    and report another. Returning quietly wrong results is the one outcome this
    module exists to avoid.

    Raises:
        ValueError: If the ranker applies a proximity boost.
    """
    if ranker.proximity:
        message = (
            "early termination cannot reproduce a proximity boost: "
            "rank the candidates directly, or build the ranker without it"
        )
        raise ValueError(message)


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
        blocks = read_blocks(index.raw, entry)
        bounds = [
            ranker.upper_bound(term, block.max_term_frequency) for block in blocks
        ]
        return _SegmentCursor(index.raw, blocks, bounds)
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
