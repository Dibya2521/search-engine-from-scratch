"""Bounded caches, and the reason none of them needs invalidating.

Query traffic repeats heavily, so three things are worth remembering: the parse
of a query's text, a term's decoded postings, and a ranked result. They are kept
as three caches rather than one because they are keyed differently and would
evict each other if they shared a budget.

| Layer | Key | Value | Goes stale |
| --- | --- | --- | --- |
| Parsed query | the query text | a parsed query | never, parsing is pure |
| Postings | segment name, term | decoded postings | never, segments never change |
| Results | generation, text, limit, scorer | the ranked list | never, see below |

Cache invalidation is the hard half of caching, and here it does not exist. A
segment file never changes after it is written, so an entry keyed by the segment
it came from cannot describe something that has since moved. The manifest
generation changes on every publish, so a result keyed by the generation it was
computed under is never returned for a different one. **Neither cache needs an
invalidation call, and neither can serve a stale answer.** That falls out of
deciding that segments would be immutable, which was decided for crash recovery
rather than for caching, and it is the clearest case in this project of a design
decision paying off a long way from where it was made.

`functools.lru_cache` is not used for two reasons. It reports one global hit
rate rather than one per layer, and the per-layer rate is the number that says
whether a cache is earning its memory. On a method it also holds a strong
reference to the instance, which would keep a `SegmentReader` alive after its
last use and, on Windows, keep its file mapped so that nothing could replace it.
"""

from __future__ import annotations

from collections import OrderedDict
from enum import Enum, auto
from typing import Final

QUERY_CACHE_SIZE: Final = 1_000
POSTINGS_CACHE_SIZE: Final = 10_000
RESULT_CACHE_SIZE: Final = 1_000


class Absent(Enum):
    """The type of the marker returned for a key a cache does not hold.

    An enum with a single member is the sentinel shape a type checker can
    narrow: testing `is not MISSING` against it leaves the stored type behind.
    """

    MARKER = auto()


MISSING: Final = Absent.MARKER
"""Returned by `LruCache.get` when a key is not held.

None is a value a caller may legitimately store, so returning it for a miss
would make a stored None indistinguishable from an absent key and the caller
would recompute it on every lookup. Testing `is not MISSING` also narrows the
result to the stored type.
"""


class LruCache[K, V]:
    """A bounded cache that evicts the least recently used entry.

    Hits and misses are counted because the hit rate is the only thing that says
    whether a cache is earning the memory it holds. A cache nobody measures is
    memory spent on faith.
    """

    __slots__ = ("_capacity", "_entries", "_hits", "_misses")

    def __init__(self, capacity: int) -> None:
        """Hold at most `capacity` entries.

        Raises:
            ValueError: If capacity is not positive. A cache of nothing reports
                a miss for every lookup, which reads as a cache working badly
                rather than as a cache that is switched off.
        """
        if capacity <= 0:
            message = f"capacity must be positive, got {capacity}"
            raise ValueError(message)
        self._capacity = capacity
        self._entries: OrderedDict[K, V] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: K) -> V | Absent:
        """Return the value held for a key, or MISSING, and count the outcome.

        A hit counts as a use, so the entry it found becomes the last one that
        would be evicted.
        """
        if key not in self._entries:
            self._misses += 1
            return MISSING
        self._hits += 1
        self._entries.move_to_end(key)
        return self._entries[key]

    def put(self, key: K, value: V) -> None:
        """Store a value, evicting the least recently used entry when full.

        Storing a key already held replaces its value and counts as a use, so
        rewriting an entry does not bring it closer to eviction.
        """
        if key in self._entries:
            self._entries.move_to_end(key)
        elif len(self._entries) >= self._capacity:
            self._entries.popitem(last=False)
        self._entries[key] = value

    def clear(self) -> None:
        """Drop every entry and reset the counters."""
        self._entries.clear()
        self._hits = 0
        self._misses = 0

    def __len__(self) -> int:
        """Return how many entries are held."""
        return len(self._entries)

    @property
    def capacity(self) -> int:
        """Return the most entries this cache will hold."""
        return self._capacity

    @property
    def hits(self) -> int:
        """Return how many lookups found a value."""
        return self._hits

    @property
    def misses(self) -> int:
        """Return how many lookups did not find one."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Return hits as a share of lookups, or 0.0 before any lookup."""
        lookups = self._hits + self._misses
        if lookups == 0:
            return 0.0
        return self._hits / lookups


def result_key(
    generation: int, text: str, limit: int, scorer: str
) -> tuple[int, str, int, str]:
    """Return the key a ranked result is cached under.

    The generation comes first because it is what makes the key safe rather
    than merely unique: publishing a new manifest moves it, so a result
    computed against an older set of segments can never be returned for a newer
    one. The scorer is part of the key because two scorers rank the same query
    differently and neither is more correct than the other.
    """
    return (generation, text, limit, scorer)
