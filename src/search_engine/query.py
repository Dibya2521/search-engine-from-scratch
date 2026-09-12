"""Reading a query and finding the documents that satisfy it.

Three query forms, distinguished by how the text is written:

- ``search`` matches documents containing that term.
- ``web search engine`` matches documents containing *any* of those terms.
- ``"web search engine"`` matches only documents where those terms appear
  adjacent, in that order.

Results are unordered. Every matching document is equally a match until
something scores them, so a set is the honest return type.

A free-text query can be expanded with synonyms before it reaches the postings.
That widens what matches, so scores shift: an expanded query matches on more
terms than the one that was typed. See `search_engine.synonyms`.

Query text goes through the same analysis as documents. It has to: a query
analyzed differently would ask for terms the index never stored.

Parsing is pure, so `CachedParser` can keep a parse for the life of the process
without any way for it to go stale. It is a class rather than a cache inside
`parse` so that nothing here holds mutable state at module level: two callers
can measure their own hit rates, and a test never inherits what an earlier test
parsed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from search_engine.access import refuse_expensive
from search_engine.analysis import analyze_positioned
from search_engine.cache import MISSING, QUERY_CACHE_SIZE, LruCache
from search_engine.metrics import REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from search_engine.index import ReadableIndex
    from search_engine.synonyms import SynonymTable

QUERIES: Final = REGISTRY.counter("queries_total", "Searches run.")
QUERY_DURATION: Final = REGISTRY.histogram(
    "query_duration_seconds", "How long a search took, from text to identifiers."
)
QUERY_ERRORS: Final = REGISTRY.counter(
    "query_errors_total", "Searches that raised instead of answering."
)
QUERY_ZERO_RESULTS: Final = REGISTRY.counter(
    "query_zero_results_total", "Searches that matched no document at all."
)
QUERY_CACHE_HITS: Final = REGISTRY.counter(
    "query_cache_hits_total", "Query parses answered from a cache."
)
QUERY_CACHE_MISSES: Final = REGISTRY.counter(
    "query_cache_misses_total", "Query parses that had to be made."
)

LOGGER: Final = logging.getLogger("search_engine.query")


@dataclass(frozen=True, slots=True)
class Query:
    """A parsed query.

    ``offsets[i]`` is where ``terms[i]`` sat in the original token stream, so a
    dropped stopword leaves a gap. Phrase matching subtracts these rather than
    consecutive integers, which is what lets a phrase containing a stopword
    still match.
    """

    terms: tuple[str, ...]
    offsets: tuple[int, ...]
    is_phrase: bool


def parse(text: str) -> Query:
    """Parse query text into terms, their offsets, and a phrase flag.

    A query counts as a phrase only when the whole of it is wrapped in double
    quotes. Mixing a phrase with loose terms is not supported, and in that case
    the quotes are simply separators like any other punctuation.
    """
    stripped = text.strip()
    # The last clause stops a lone quote character satisfying both ends.
    is_phrase = stripped.startswith('"') and stripped.endswith('"') and stripped != '"'
    positioned = analyze_positioned(stripped)
    return Query(
        terms=tuple(term for _, term in positioned),
        offsets=tuple(offset for offset, _ in positioned),
        is_phrase=is_phrase,
    )


def _matching_any_term(index: ReadableIndex, terms: Sequence[str]) -> set[int]:
    """Union the postings, which is what a one-word or free-text query wants."""
    matches: set[int] = set()
    for term in terms:
        matches.update(index.postings(term))
    return matches


def _contains_phrase(
    postings: Sequence[Mapping[int, Sequence[int]]],
    offsets: Sequence[int],
    document_id: int,
) -> bool:
    """Test whether one document holds the terms at the queried spacing.

    Subtract each term's query offset from its document positions. A phrase
    occurring at document position ``p`` puts the term queried at offset ``i``
    at ``p + i``, so subtracting ``i`` maps every term of a real occurrence
    onto the same ``p``. Adjacency, and any gap a dropped stopword left,
    becomes a plain set intersection.

    Sets are built one at a time and the loop stops the moment the overlap
    empties, so a phrase failing early never pays to build the rest. Sorting
    them by size first measured slower: sorting must build every set before it
    can compare lengths, which is the expensive part.
    """
    common = {position - offsets[0] for position in postings[0][document_id]}
    for index in range(1, len(postings)):
        if not common:
            return False
        common &= {
            position - offsets[index] for position in postings[index][document_id]
        }
    return bool(common)


def _matching_phrase(
    index: ReadableIndex, terms: Sequence[str], offsets: Sequence[int]
) -> set[int]:
    """Find documents holding every term at the spacing the query asked for."""
    if not terms:
        return set()
    postings = [index.postings(term) for term in terms]
    if any(not entry for entry in postings):
        return set()
    # Start from the rarest term, so the candidate set begins as small as it
    # can. Skipping that entry avoids intersecting it with itself.
    rarest = min(range(len(postings)), key=lambda offset: len(postings[offset]))
    candidates = set(postings[rarest])
    for offset, entry in enumerate(postings):
        if offset == rarest:
            continue
        candidates.intersection_update(entry)
        if not candidates:
            return set()
    return {
        document_id
        for document_id in candidates
        if _contains_phrase(postings, offsets, document_id)
    }


def execute(
    index: ReadableIndex, query: Query, synonyms: SynonymTable | None = None
) -> set[int]:
    """Run a parsed query and return the identifiers of matching documents.

    Synonyms expand a free-text query and never a phrase query. A phrase is
    matched on the spacing between its terms, and a term standing in for another
    occupies no position in the document, so there is nothing to space it
    against.

    The cost is checked against the terms actually about to be read, expansion
    included, and before any posting is touched.

    Raises:
        QueryTooExpensiveError: If the query would cost more than the limits
            allow.
    """
    terms = query.terms if query.is_phrase else _expanded(query.terms, synonyms)
    refuse_expensive(index, terms)
    if query.is_phrase:
        return _matching_phrase(index, query.terms, query.offsets)
    return _matching_any_term(index, terms)


def _expanded(terms: Sequence[str], synonyms: SynonymTable | None) -> list[str]:
    """Return the query terms plus every term equivalent to one of them."""
    if synonyms is None:
        return list(terms)
    return sorted({member for term in terms for member in synonyms.expand(term)})


def search(
    index: ReadableIndex, text: str, synonyms: SynonymTable | None = None
) -> set[int]:
    """Parse and run query text in one step."""

    def run() -> tuple[Query, set[int]]:
        parsed = parse(text)
        return parsed, execute(index, parsed, synonyms)

    return _recorded(text, run)


def _recorded(text: str, run: Callable[[], tuple[Query, set[int]]]) -> set[int]:
    """Run one search, recording that it happened, what it cost and what it found.

    Both entry points funnel through here rather than one being counted and the
    other not. A metric covering some of the traffic is worse than no metric,
    because it still looks like a number.

    Finding nothing is counted on its own because it is the cheapest proxy
    there is for relevance health. Nothing else in a running system notices
    that the engine has started answering nothing.

    Parsing is inside the timing, because a caller asked for a search and does
    not care which part of one was slow.
    """
    started = time.perf_counter()
    QUERIES.increment()
    try:
        parsed, found = run()
    except Exception:
        QUERY_ERRORS.increment()
        raise
    finally:
        elapsed = time.perf_counter() - started
        QUERY_DURATION.observe(elapsed)
    if not found:
        QUERY_ZERO_RESULTS.increment()
    LOGGER.info(
        "query",
        extra={
            "query": text,
            "terms": len(parsed.terms),
            "phrase": parsed.is_phrase,
            "candidates": len(found),
            "duration_ms": round(elapsed * 1000, 3),
        },
    )
    return found


class CachedParser:
    """Parses query text, keeping the parses it has already made.

    A parsed query is immutable, so handing the same one to two callers is
    safe, and parsing is pure, so a kept parse can never be wrong. Together
    those are the whole argument for this cache: there is no state to
    invalidate and no way for a hit to be stale.
    """

    __slots__ = ("_cache",)

    def __init__(self, capacity: int = QUERY_CACHE_SIZE) -> None:
        """Keep at most `capacity` parses.

        Raises:
            ValueError: If capacity is not positive.
        """
        self._cache: LruCache[str, Query] = LruCache(capacity)

    def parse(self, text: str) -> Query:
        """Return the parse of this text, from the cache when it is held."""
        cached = self._cache.get(text)
        if cached is not MISSING:
            QUERY_CACHE_HITS.increment()
            return cached
        QUERY_CACHE_MISSES.increment()
        parsed = parse(text)
        self._cache.put(text, parsed)
        return parsed

    def search(self, index: ReadableIndex, text: str) -> set[int]:
        """Run query text, reusing an earlier parse of the same text."""

        def run() -> tuple[Query, set[int]]:
            parsed = self.parse(text)
            return parsed, execute(index, parsed)

        return _recorded(text, run)

    @property
    def cache(self) -> LruCache[str, Query]:
        """Return the cache, so a caller can read its hit rate."""
        return self._cache
