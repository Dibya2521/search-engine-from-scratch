# Caching

Three things worth remembering between queries, and the reason none of them can
go stale.

## What it is

A bounded cache that evicts the least recently used entry, counts its hits and
misses, and is used at three separate layers.

Implemented at [`src/search_engine/cache.py`](../src/search_engine/cache.py).

| Layer | Key | Value | Where it lives |
| --- | --- | --- | --- |
| Parsed query | the query text | a parsed query | `CachedParser` in [query.py](../src/search_engine/query.py) |
| Postings | segment and term | decoded postings | `SegmentReader` in [segment.py](../src/search_engine/segment.py) |
| Results | generation, text, limit, scorer | the ranked list | `result_key`, not yet wired to a serving path |

Three caches rather than one, because they are keyed differently and would evict
each other out of a shared budget. A burst of distinct queries would flush the
postings that every one of them needs.

## Cache invalidation, and why there is none

Cache invalidation is the hard half of caching. Here it does not exist, and that
is not carefulness. It falls out of a decision taken for an entirely different
reason.

**A segment file never changes after it is written.** That was decided so that a
crash could not corrupt one, and so that a half-written file left behind by a
crash could simply be ignored. See [segments](13-segments.md). The consequence
here is that an entry keyed by the segment it came from describes a file that
cannot have moved underneath it. It stays true for as long as the file exists.

**The manifest generation changes on every publish.** That was decided so that
publishing could be atomic. The consequence here is that a result keyed by the
generation it was computed under is never returned for a different one. A
publish retires every result computed before it, at no cost and with no code.

**Neither cache needs an invalidation call, and neither can serve a stale
answer.** This is the clearest case in this project of a design decision paying
off a long way from where it was taken.

Parsing is pure, so the query cache has nothing to be stale about at all.

## The postings cache is load-bearing

This is not an optimisation that can be switched off.

`BM25Ranker.score` asks the index for a term's postings **once per document
scored**. With the cache that is a dictionary lookup. Without it, a
`SegmentReader` decodes the term's entire postings list on every one of those
calls. Scoring is therefore linear in the candidates with the cache and
quadratic without it.

Measured by
[`benchmarks/early_termination.py`](../benchmarks/early_termination.py), the
same query against the same segment:

| Documents | Candidates | Cache on s | Cache off s | Off / on | Hit rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 500 | 0.0170 | 3.7363 | 220x | 99.8% |
| 1,000 | 1,000 | 0.0159 | 13.1044 | 825x | 99.9% |
| 2,000 | 1,997 | 0.0647 | 106.8129 | 1,652x | 99.9% |
| 4,000 | 3,988 | 0.1214 | 436.7176 | 3,598x | 100.0% |

**Read the growth of the ratio, not its value.** A constant ratio would mean a
constant-factor optimisation. This one roughly doubles as the corpus doubles,
which is the signature of a different complexity class. The individual cells are
noisy: an earlier run of the same code gave 427x, 1,135x, 2,004x and 3,321x. The
growth appears in both.

That is why the comparison is measured at 500 to 4,000 documents and not at the
50,000 used everywhere else in that benchmark. At 49,658 candidates the uncached
side needs on the order of a billion posting decodes. An attempt to measure it
there ran for 73.7 minutes at 94 percent of a core with 24 GB of memory free and
produced nothing before it was killed. **That failure is the measurement**, not
an obstacle to it.

A merge is the one place caching is switched off deliberately, with
`SegmentReader(..., cache=False)`. A merge reads every term exactly once, so a
cache would retain everything it had already passed and buy nothing. It is also
the one place scoring never happens, which is why the quadratic cost above does
not apply to it.

**The scorer could stop depending on this.** If `score` took the postings it
needed rather than fetching them per document, the ranker would be cheap without
a cache underneath it. That changes the scorer's interface, so it has not been
done here.

## A miss is a marker, not None

`LruCache.get` returns `MISSING`, a single-member enum, rather than `None`.

`None` is a value a caller may legitimately store. A cache that returned it for
a miss could not tell a stored `None` from an absent key, and would recompute
that value on every lookup, silently and forever. The enum is also the sentinel
shape a type checker narrows, so `if cached is not MISSING` leaves the stored
type behind without a cast.

A capacity of zero is refused rather than treated as "off". A cache holding
nothing reports a miss for every lookup, so its hit rate is 0.0, which reads
exactly like a cache working badly. A reader that wants none passes
`cache=False` and gets none, which reads as what it is.

## The hit rate is published because it has to be

Every layer counts its hits and misses and exposes `hit_rate`.

`functools.lru_cache` was not used, for two reasons. It reports one figure for
the whole process rather than one per layer, and the per-layer rate is the
number that says whether a cache earns its memory. On a method it also holds a
strong reference to the instance, which would keep a `SegmentReader` alive past
its last use and, on Windows, keep its file mapped so that nothing could replace
it. See [segments](13-segments.md) for why that matters.

**The 99.8 to 100.0 percent above characterises this benchmark and nothing
else.** It repeats one term against one segment, which is close to the best case
a cache can have. Real hit rates depend on real query distributions, and no
synthetic mix supplies one. The number is published because a cache whose hit
rate nobody has measured is memory spent on faith, not because it predicts
anything about production traffic.

## Sizes

```python
QUERY_CACHE_SIZE = 1_000
POSTINGS_CACHE_SIZE = 10_000
RESULT_CACHE_SIZE = 1_000
```

**These are guesses.** Nothing has measured the memory a full postings cache
holds against the hit rate it buys, and until something does they should be read
as starting points rather than as answers.

## Connections

- **[Segments](13-segments.md)** supply the immutability and the generation
  counter that between them remove the need to invalidate anything.
- **[Early termination](14-early-termination.md)** reduces how many documents
  are scored; this reduces what each of those scores costs. The two are
  independent and both are needed.
- **[Ranking](06-ranking.md)** is the caller whose per-document postings lookup
  makes the postings cache load-bearing rather than optional.
- **[Querying](04-querying.md)** supplies the parse that `CachedParser` keeps.
