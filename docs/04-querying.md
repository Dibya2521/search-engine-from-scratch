# Querying

Reading what somebody typed and turning it into a set of documents. This is
where the index finally does the job it was built for.

Implemented at [`src/search_engine/query.py`](../src/search_engine/query.py).

## The three query forms

Distinguished by how the query is written, not by a flag or a syntax tree.

| Written as | Means | Answer |
| --- | --- | --- |
| `search` | documents containing that term | one postings lookup |
| `web search engine` | documents containing **any** of those terms | union of postings |
| `"web search engine"` | those terms **adjacent, in that order** | intersection, then position arithmetic |

A one-word query is not actually a separate case. It is a union of one postings
list, so the same code serves both, and only the phrase form needs different
machinery.

### Why free text is OR and not AND

Requiring every term (AND) would be the more useful default for a modern search
box, and it is what most engines do. Union is used here because it is what the
design specifies, and because it composes with ranking: a union produces a wide
candidate set, and scoring then sorts the ones matching more terms to the top.
Under AND semantics a single unusual word in the query silently empties the
result.

That reasoning depends on ranking existing, and it does:
[TF-IDF scoring](06-ranking.md) turns the wide candidate set into an ordered
list, so breadth costs nothing at the top of the results.

## The phrase query trick

The interesting algorithm in the whole project, and it is very short.

Finding documents containing every term is just a set intersection over the
postings keys. The hard part is *adjacency*: knowing the terms sat next to each
other rather than scattered through the document.

The trick is to **subtract each term's offset in the query from its positions in
the document**, then intersect.

Take the phrase `computer science department` and a document holding:

```text
computer at positions  2, 5
science  at position       3
department at positions    4, 6
```

The query offsets here are 0, 1 and 2. Subtract each term's offset from its
positions:

```text
computer   {2, 5} - 0  ->  {2, 5}
science    {3}    - 1  ->  {2}
department {4, 6} - 2  ->  {2, 4}

intersection ->  {2}
```

The result is non-empty, so the phrase occurs, starting at position 2.

Why it works: if the phrase begins at document position `p`, then the term
queried at offset `i` sits at `p + i`, so subtracting `i` maps every term of a
genuine occurrence onto the same number `p`. A shared value in the intersection
*is* a starting position. Adjacency, which sounds like it needs ordered
traversal, collapses into one set intersection.

**The offsets come from the query, not from a counter.** That distinction looks
pedantic until stopwords exist. Once they do, both the document and the query
have gaps where common words were removed, and only the query's real offsets
line the two up:

```text
document "computer of science"  ->  comput@0, scienc@2

query "computer science"      offsets (0,1)  ->  {0-0} & {2-1} = {}   no match
query "computer of science"    offsets (0,2)  ->  {0-0} & {2-2} = {0}  match
query "the computer science"    offsets (1,2)  ->  only differences matter
```

With no filtering the offsets are `0, 1, 2, ...` and this reduces exactly to the
version above, so gaps are the general case of the same arithmetic rather than a
special case bolted on. See [Stopwords](05-stopwords.md) for why the two obvious
alternatives are both wrong.

## Parsing rules

- A query is a phrase only when the **whole** of it is wrapped in double
  quotes. Mixing a phrase with loose terms, as in `web "search engine"`, is not
  supported: the quotes are then just separators like any other punctuation.
- A lone `"` is not a phrase, even though it starts and ends with a quote.
- `""` parses as a phrase with no terms, and matches nothing.
- Empty and whitespace-only queries match nothing.

## The contract: analyse once, on raw text

Query text goes through the same analysis as documents. That is not a
convenience, it is the thing that makes matching possible at all: a query
analysed differently would ask for terms the index never stored, and the engine
would return nothing while appearing to work perfectly.

There is a sharper consequence, and it was found by a property test rather than
by reasoning. **A term read back out of the index is not a valid query.**

```text
document "A0SE"  ->  indexed as the term  "a0s"
query    "A0SE"  ->  finds the document
query    "a0s"   ->  finds nothing
```

Because stemming is not idempotent, analysing a stored term a second time can
change it: `a0s` stems to `a0`. Both sides applied the pipeline exactly once to
their own raw input, which is correct. But it means anything programmatic that
takes terms out of the index and feeds them back as queries is wrong. Pinned by
a test at [`tests/test_query.py`](../tests/test_query.py).

## Complexity

Let `k` be the number of query terms, `d` the number of documents containing
all of them, and `p` the average positions per term per document.

| Query | Work |
| --- | --- |
| One word | one hash lookup |
| Free text | O(sum of postings lengths), one pass per term |
| Phrase | O(k) document-set intersections, then O(k x p) per surviving document |

The phrase cost is the only one that scales with document *content* rather than
just document count, which is what makes positions expensive to use as well as
to store.

## Measured: an optimisation that had to be earned twice

The standard advice for phrase intersection is to handle the smallest position
set first and stop as soon as the overlap empties. The first implementation
followed that advice literally, sorting the shifted position sets by size.

Measured against a deliberately naive version, on 2,000 documents each holding
200 occurrences of one very frequent term
([`benchmarks/phrase_query.py`](../benchmarks/phrase_query.py)):

| Case | Sorted version | Naive | Verdict |
| --- | --- | --- | --- |
| 2 terms, frequent first | 32.15 ms | 19.44 ms | naive 1.65x faster |
| 2 terms, both rare | 6.15 ms | 3.14 ms | naive 1.96x faster |
| 4 rare terms | 8.96 ms | 7.43 ms | naive 1.20x faster |
| 4 terms, frequent first | 27.09 ms | 30.14 ms | sorted 1.11x faster |
| 4 terms, fails on the last | 7.99 ms | 4.26 ms | naive 1.89x faster |
| fails on the first pair | 4.04 ms | 1.37 ms | naive 2.94x faster |

**The optimisation lost five cases out of six.** Textbook advice, implemented
faithfully, and slower than doing nothing clever.

The cause is specific and worth understanding: `sorted()` has to build *every*
position set before it can compare their lengths. Building the sets is the
expensive part, so sorting them guarantees paying the full cost up front, and
then adds a sort on top. The early exit had nothing left to skip.

The real version of the advice never materialises what it might not need. Build
the shifted sets **one at a time**, and stop the moment the overlap empties:

```python
common = set(postings[0][document_id])
for offset in range(1, len(postings)):
    if not common:
        return False
    common &= {position - offset for position in postings[offset][document_id]}
return bool(common)
```

Re-measured:

| Case | Lazy version | Naive | Verdict |
| --- | --- | --- | --- |
| 2 terms, frequent first | 13.96 ms | 25.37 ms | **1.82x faster** |
| 2 terms, both rare | 2.58 ms | 2.56 ms | even |
| 4 rare terms | 4.23 ms | 3.06 ms | naive 1.38x faster |
| 4 terms, frequent first | 7.17 ms | 12.28 ms | **1.71x faster** |
| 4 terms, fails on the last | 2.19 ms | 2.16 ms | even |
| fails on the first pair | 1.07 ms | 1.23 ms | 1.15x faster |

Still not a clean sweep, and the honest reading is not "the optimisation wins".
It is that the optimisation wins **where the work is**. It saves 11.4 ms and
5.1 ms on the two slowest cases and costs 1.2 ms on a fast one. When every term
is rare all the position sets are tiny, so there is nothing to avoid building
and the bulk list comprehension in the naive version is marginally quicker.

Two lessons, and the second is the transferable one:

1. Choosing the rarest term to seed the candidate set is nearly free, because
   the length of a postings mapping is known without touching it.
2. **An optimisation that decides what to skip must not have already built the
   thing it is skipping.** That is why the first attempt failed while the second
   worked, and it is not visible from reading either version.

## Alternatives

| Approach | How it works | Wins when | Loses when |
| --- | --- | --- | --- |
| **Positional index (this)** | store every position, intersect with offsets | exact phrases of any length | positions dominate index size |
| **Biword index** | index adjacent pairs as single terms | two-word phrases, very fast | longer phrases need verification, and vocabulary explodes |
| **Non-positional index** | document identifiers only | smallest index, fastest boolean queries | phrase queries are impossible |
| **Post-verification** | boolean match, then re-read the documents | tiny index | needs the original text, and is slow per candidate |
| **Suffix automaton** | index every substring | wildcard and infix search | far larger than the text |

Real systems often combine the biword and positional approaches: common phrases
get their own terms, everything else falls back to position arithmetic.

## How it connects to everything else

- **Analysis** must be the same function used at index time. Everything here
  depends on that, and nothing enforces it except sharing one function.
- **Positions in the index** exist solely for the phrase case. A
  non-positional index would make this file about a third of its size and the
  index several times smaller.
- **Stopword filtering**, if it ever removes terms before positions are
  assigned, changes what a phrase *means* rather than just what it costs:
  `"king of england"` would match text reading `king england`.
- **[Ranking](06-ranking.md)** consumes exactly what these functions return. A candidate set is
  the input to scoring, which is why they return sets rather than ordered
  results: imposing an order here would be inventing one.
- **The document frequency** already exposed by the index is what lets the
  phrase path pick the rarest term to start from, and it is the same number
  ranking will use to weight rare terms more heavily.
