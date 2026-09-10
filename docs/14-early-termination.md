# Early termination

Returning the best ten documents without scoring the rest.

## What it is

Ranking scores every candidate and then throws almost all of them away. Early
termination keeps an upper bound on what each query term can contribute, and
uses it to prove a document cannot reach the results before spending anything on
it.

Implemented at [`src/search_engine/wand.py`](../src/search_engine/wand.py).

```python
from search_engine.wand import rank_wand

best = rank_wand(index, ranker, ["web", "search"], limit=10)
```

The algorithm is **WAND**, and with per-block bounds **Block-Max WAND**. Both
are exact: they return what scoring every candidate returns, identifier for
identifier and score for score. That is the property the whole idea rests on,
and it is checked against generated indexes and queries rather than assumed.

## Why it exists

Scoring one candidate costs **1.7 microseconds**, measured in
[`benchmarks/index_build.py`](../benchmarks/index_build.py) and recorded in
[ranking](06-ranking.md). That is fast, and it is the wrong thing to measure,
because the count is what grows. A term appearing in a million documents costs
1.7 seconds per query, and 999,990 of those scores are computed only to be
discarded.

The waste is not in the scoring. It is in scoring documents that could never
have finished in the top ten.

## The upper bound

BM25's contribution from one term rises with the term frequency and falls with
the document length. The most a term can contribute to any document in a segment
is therefore fixed by the highest frequency it reaches anywhere and the shortest
document in the corpus:

```text
ub(t) = idf(t) * (max_tf * (k1 + 1))
                 / (max_tf + k1 * (1 - b + b * min_length / average_length))
```

**No postings are read to compute it.** `max_tf` is stored per term in the
segment's dictionary and the document lengths are stored in the segment's
document section, both put there for this before anything used them. That is the
entire reason the bound is affordable.

The shortest document in the whole corpus is used rather than the shortest
document containing the term. The second is tighter and would prune more, and it
costs a pass over the term's postings, which is the cost this exists to avoid.
The looser value is never an underestimate, so it is safe.

The bound lives on the scorer, at
[`src/search_engine/bm25.py`](../src/search_engine/bm25.py), rather than beside
the search. A bound derived from different `k1` and `b` than the score it bounds
is not a bound.

## The loop, worked through

Each query term gets a **cursor** over its postings in document order. The
cursors are kept sorted by the document each is currently on. Adding their
bounds in that order gives the best score any document at or before a given
position could reach, and the first cursor at which that running total beats the
threshold is the **pivot**.

Ten documents and three terms, with the postings below. This is a real run, not
an illustration:

| Term | Documents | Frequency | Upper bound |
| --- | --- | ---: | ---: |
| `alpha` | 0 to 9 | 1 | 0.0599 |
| `beta` | 2, 5, 8 | 2 | 1.8599 |
| `gamma` | 5, 9 | 3 | 2.6364 |

Asking for the best **two**:

```text
step  cursors sit on        threshold   pivot   action
  1   0, 2, 5               0.0000      0       score document 0
  2   1, 2, 5               0.0000      1       score document 1
  3   2, 2, 5               0.0599      2       score document 2
  4   3, 5, 5               0.0599      5       move alpha from 3 to 5
  5   5, 5, 5               0.0599      5       score document 5
  6   6, 8, 9               1.4690      8       move alpha from 6 to 8
  7   8, 8, 9               1.4690      8       score document 8
  8   9, 9                  1.4690      9       score document 9

result   document 5 at 2.7863, document 9 at 2.0158
scored   6 of 10
```

**Documents 3, 4, 6 and 7 are never scored.** Each holds only `alpha`, whose
bound of 0.0599 cannot beat a threshold that has risen past it, so the loop
moves `alpha`'s cursor straight to the pivot and skips them. A full scan returns
the same two documents with the same two scores.

Two properties make this exact rather than approximate.

**Cursors are sorted by document identifier.** That is what makes "no cursor
before the pivot has reached this document yet" a true statement, and it is why
postings have to be document-ordered. The consequence is large enough to have
its own record: see [ADR 0006](adr/0006-document-ordered-postings.md).

**The bound is compared strictly greater than the threshold.** At step 4 above,
`alpha`'s bound is 0.0599 and the threshold is exactly 0.0599, so the loop
refuses to score document 3, whose score is also exactly 0.0599. That is
correct, and the reason is worth stating precisely: documents are walked in
ascending identifier order and ties are resolved on the lower identifier, so
every document that could win such a tie has already been scored. Document 0
holds that slot and document 3 could never have taken it. Comparing greater or
equal is equally exact and merely scores more documents: over 400 generated
searches it changed no result and scored 8 more documents in total.

## Block maxima

One bound for a whole term is set by that term's most extreme document. A term
occurring 500 times in one document and once in 4,999 others carries a bound so
far above what an ordinary document scores that it never falls below the
threshold, and nothing is pruned at all.

A term's postings are therefore cut into blocks of 128 documents, and each block
records the highest term frequency inside it. The search sums the bounds of the
blocks its cursors currently sit in, and when that total cannot beat the
threshold either, every cursor jumps past its block without the postings inside
being decoded. See [segments](13-segments.md) for the format.

The check is made only where every cursor up to the pivot already sits on the
pivot document. A cursor whose block ended earlier bounds nothing there, so the
sum would be an underestimate anywhere else in the walk. Every cursor on that
document is counted, not only those up to the pivot, because one past the pivot
can sit on the same document and leaving it out would leave its term unbounded.
The jump stops at the earliest block end, held back to the next cursor's
document, since beyond that a term none of these cursors bounds could
contribute.

An index that is not a single segment has no blocks and reports the term's own
bound in their place. The pivot rule guarantees that total already exceeds the
threshold, so the block test cannot fire and those indexes behave exactly as
they did before blocks existed.

## What it costs, and when it pays

Measured by [`benchmarks/early_termination.py`](../benchmarks/early_termination.py)
over 50,000 documents in one segment, comparing a full scan against early
termination with one bound per term and with a bound per block.

**Documents scored is exact.** It is a counter, it does not move with machine
load, and it was identical across every run. The timings are not exact: the same
row has moved by 34 percent between runs of identical code. Read the counts and
treat the clock as direction only.

| Query, limit 10 | Candidates | Scan s | Term bounds s | Scored |
| --- | ---: | ---: | ---: | ---: |
| the most common term | 49,658 | 0.0683 | 0.1652 | 49,658 |
| two common terms | 49,939 | 0.0945 | 0.2978 | 47,825 |
| one common, one rare | 49,658 | 0.0906 | **0.0129** | **1,510** |
| three of mixed frequency | 49,659 | 0.1018 | **0.0220** | **2,118** |
| two rare terms | 1,084 | 0.0021 | 0.0053 | 1,084 |

**It pays when one query term is rare enough to drive the pivot**, scoring 3.0
percent of the candidates and running 7.0x faster at limit 10 and 14.1x faster
at limit 1.

**It costs when every query term is common.** The most common term alone scores
100 percent of the candidates and runs 2.4x slower than the scan, because
nothing can be pruned and the cursor machinery is added on top of the same
scoring work. Two common terms prune 4.2 percent and lose by 3.2x.

**It also costs on a small candidate set.** A thousand candidates do not repay
the fixed cost of opening cursors.

That is a genuine negative result and the code stays, with the condition stated,
exactly as [`save_text`](07-corpus-and-persistence.md) and the naive phrase
matcher stay.

### Where per-block bounds matter, and where they do nothing

On the corpus above, per-block bounds matched per-term bounds exactly on four of
the five query mixes and beat them only on the three-term query, 2,118 against
1,715 at limit 10.

The reason is the corpus, not the technique. That generator draws every
document's length independently, so a document with a high term frequency is as
likely to land in one block as any other and every block ends up with a maximum
close to the term's own. There is nothing left to tighten. **Per-block bounds
need high frequencies concentrated in few blocks, and postings are ordered by
document identifier, so they need identifier order to correlate with term
frequency.**

On a corpus where one document of 50,000 holds a term 500 times and the rest
hold it once:

| Bound | Documents scored | Seconds |
| --- | ---: | ---: |
| one per term | 50,000 | 0.1299 |
| one per block | **138** | **0.0084** |

362x fewer documents scored, 10.7x faster than the full scan. Whether real text
has that correlation is a question a synthetic corpus cannot answer.

## What does not work

**TF-IDF cannot use this.** Its cosine similarity divides by a per-document
vector length that is not known until the document has been chosen, so there is
nothing to bound a term's contribution with in advance. `rank_wand` takes a
`BM25Ranker` in its signature rather than the base class, so the restriction is
in the type rather than in a comment. Accepting either scorer and being quietly
wrong for one is the failure this avoids.

## Connections

- **[Ranking](06-ranking.md)** supplies the 1.7 microseconds per candidate this
  exists to stop paying, and the top-k selection rule the threshold comes from.
- **[BM25](09-bm25.md)** supplies the bound, and the monotonicity that makes it
  sound.
- **[The on-disk index](12-on-disk-index.md)** stores the term's highest
  frequency and each document's length, which is what makes the bound free.
- **[Segments](13-segments.md)** hold the block table the per-block bounds are
  read from, and a directory of segments is deliberately walked through its
  postings mapping instead, because that is where deleted documents are filtered
  out.
- **[Caching](15-caching.md)** is what keeps scoring linear in the candidates.
  Early termination reduces how many documents are scored; the postings cache is
  what stops each of those scores costing a decode.
- **[ADR 0006](adr/0006-document-ordered-postings.md)** records the trade this
  forced on the shape of a postings list.
