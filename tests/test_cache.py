"""Tests for the bounded caches.

The load-bearing test is `test_a_cache_never_exceeds_capacity_or_invents_values`.
A cache that grows past its bound is a memory leak with a hit rate, and a cache
that returns a value stored under a different key is worse than no cache at all,
so both are checked against generated sequences of operations rather than
against the handful of orders a unit test would think to try.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.cache import (
    MISSING,
    QUERY_CACHE_SIZE,
    LruCache,
    result_key,
)
from search_engine.index import InvertedIndex
from search_engine.query import CachedParser, parse, search


def filled(capacity: int, count: int) -> LruCache[int, str]:
    """Return a cache of `capacity` holding `count` entries, oldest first."""
    cache: LruCache[int, str] = LruCache(capacity)
    for number in range(count):
        cache.put(number, f"v{number}")
    return cache


@pytest.mark.parametrize("capacity", [0, -1])
def test_a_capacity_that_is_not_positive_is_rejected(capacity: int) -> None:
    """A cache of nothing reads as a cache working badly, not one switched off."""
    with pytest.raises(ValueError, match="capacity must be positive"):
        LruCache[int, str](capacity)


def test_the_least_recently_used_entry_is_the_one_evicted() -> None:
    cache = filled(3, 3)
    cache.put(3, "v3")
    assert cache.get(0) is MISSING
    assert cache.get(1) == "v1"


def test_reading_an_entry_counts_as_using_it() -> None:
    """Without this a hot entry read but never rewritten would be evicted."""
    cache = filled(3, 3)
    assert cache.get(0) == "v0"
    cache.put(3, "v3")
    assert cache.get(0) == "v0"
    assert cache.get(1) is MISSING


def test_rewriting_an_entry_counts_as_using_it() -> None:
    cache = filled(3, 3)
    cache.put(0, "again")
    cache.put(3, "v3")
    assert cache.get(0) == "again"
    assert cache.get(1) is MISSING
    assert len(cache) == 3


def test_a_capacity_of_one_holds_only_the_last_entry() -> None:
    cache = filled(1, 3)
    assert len(cache) == 1
    assert cache.get(2) == "v2"
    assert cache.get(1) is MISSING


def test_hits_and_misses_are_counted() -> None:
    cache = filled(2, 2)
    assert cache.get(0) == "v0"
    assert cache.get(9) is MISSING
    assert (cache.hits, cache.misses) == (1, 1)
    assert cache.hit_rate == 0.5


def test_the_hit_rate_of_an_untouched_cache_is_zero() -> None:
    """Nothing has been asked for, so the rate must not divide by zero."""
    cache: LruCache[int, str] = LruCache(4)
    assert cache.hit_rate == 0.0
    assert (cache.hits, cache.misses) == (0, 0)


def test_clearing_drops_the_entries_and_the_counters() -> None:
    cache = filled(4, 3)
    cache.get(0)
    cache.clear()
    assert len(cache) == 0
    assert (cache.hits, cache.misses) == (0, 0)
    assert cache.hit_rate == 0.0
    assert cache.get(0) is MISSING


def test_none_is_storable_and_is_not_a_miss() -> None:
    """The reason `get` returns a marker rather than None.

    A cache that could not tell a stored None from an absent key would
    recompute that value on every single lookup, quietly.
    """
    cache: LruCache[str, str | None] = LruCache(2)
    cache.put("here", None)
    assert cache.get("here") is None
    assert cache.get("absent") is MISSING
    assert (cache.hits, cache.misses) == (1, 1)


def test_reading_with_an_unhashable_key_is_refused_at_once() -> None:
    """Not at eviction time, when the caller is long gone from the stack."""
    cache: LruCache[object, str] = LruCache(2)
    with pytest.raises(TypeError, match="unhashable"):
        cache.get(["a", "list"])


def test_writing_with_an_unhashable_key_is_refused_at_once() -> None:
    cache: LruCache[object, str] = LruCache(2)
    with pytest.raises(TypeError, match="unhashable"):
        cache.put(["a", "list"], "value")


@given(
    capacity=st.integers(min_value=1, max_value=8),
    operations=st.lists(
        st.tuples(st.booleans(), st.integers(min_value=0, max_value=12)),
        max_size=60,
    ),
)
def test_a_cache_never_exceeds_capacity_or_invents_values(
    capacity: int, operations: list[tuple[bool, int]]
) -> None:
    """Two invariants no order of operations may break.

    The cache holds no more than it was told to, and anything it returns was
    stored under the key it was asked for. A value appearing under the wrong
    key would corrupt every answer built on it.
    """
    cache: LruCache[int, str] = LruCache(capacity)
    for writing, key in operations:
        if writing:
            cache.put(key, f"v{key}")
        else:
            held = cache.get(key)
            assert held is MISSING or held == f"v{key}"
        assert len(cache) <= capacity


@given(
    capacity=st.integers(min_value=1, max_value=6),
    keys=st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=40),
)
def test_the_most_recently_used_entries_are_the_ones_kept(
    capacity: int, keys: list[int]
) -> None:
    """Whatever the sequence, the last `capacity` distinct keys used are held."""
    cache: LruCache[int, str] = LruCache(capacity)
    used: list[int] = []
    for key in keys:
        cache.put(key, f"v{key}")
        if key in used:
            used.remove(key)
        used.append(key)
    for key in used[-capacity:]:
        assert cache.get(key) == f"v{key}"


def test_a_result_key_separates_generations() -> None:
    """A publish moves the generation, so an older result cannot be returned."""
    cache: LruCache[tuple[int, str, int, str], list[int]] = LruCache(4)
    cache.put(result_key(1, "alpha", 10, "bm25"), [1, 2, 3])
    assert cache.get(result_key(1, "alpha", 10, "bm25")) == [1, 2, 3]
    assert cache.get(result_key(2, "alpha", 10, "bm25")) is MISSING


@pytest.mark.parametrize(
    ("generation", "text", "limit", "scorer"),
    [(1, "beta", 10, "bm25"), (1, "alpha", 5, "bm25"), (1, "alpha", 10, "tfidf")],
)
def test_every_part_of_a_result_key_matters(
    generation: int, text: str, limit: int, scorer: str
) -> None:
    """Two scorers rank the same query differently, and neither is wrong."""
    cache: LruCache[tuple[int, str, int, str], list[int]] = LruCache(8)
    cache.put(result_key(1, "alpha", 10, "bm25"), [1])
    assert cache.get(result_key(generation, text, limit, scorer)) is MISSING


def test_a_cached_parser_returns_the_same_parse_twice() -> None:
    parser = CachedParser()
    first = parser.parse("web search engine")
    assert parser.parse("web search engine") is first
    assert parser.cache.capacity == QUERY_CACHE_SIZE
    assert (parser.cache.hits, parser.cache.misses) == (1, 1)


def test_a_cached_parse_equals_an_uncached_one() -> None:
    """The cache must not change what parsing means, only what it costs."""
    for text in ("alpha beta", '"alpha beta"', "", '"'):
        assert CachedParser().parse(text) == parse(text)


def test_a_cached_parser_searches_the_same_as_the_plain_function() -> None:
    index = InvertedIndex()
    index.add_document(1, "web search engines rank documents")
    index.add_document(2, "a lazy dog sleeps")
    parser = CachedParser()
    for text in ("search", "web engines", '"search engines"', "absent"):
        assert parser.search(index, text) == search(index, text)
