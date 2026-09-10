# The inverted index

The data structure the whole engine exists to build. Everything before it
prepares terms; everything after it reads this.

## What it is

A map from each term to the places it occurs.

```python
{
    "index": {1: [6], 2: [1], 3: [5]},
    "web": {1: [0], 3: [2]},
    "search": {1: [1], 3: [0]},
}
```

Read the first line as: the term `index` appears in document 1 at position 6,
in document 2 at position 1, and in document 3 at position 5.

Implemented at [`src/search_engine/index.py`](../src/search_engine/index.py).

## Why it exists

The obvious structure is the opposite one, a **forward index**, mapping each
document to the terms it contains. That is how the data naturally arrives, and
it answers "what is in document 5".

Searching never asks that. It asks "which documents contain this term", and a
forward index can only answer it by examining every document. For a corpus of
N documents that is O(N) work per query, and it grows without limit as the
corpus grows.

Inverting the map makes the answer a single dictionary lookup, independent of
corpus size. That is the entire trick, and it is why the structure is named
after the inversion rather than after what it stores.

| | Forward index | Inverted index |
| --- | --- | --- |
| Shape | document to terms | term to documents |
| "What is in document 5?" | one lookup | scan everything |
| "Who contains `search`?" | scan everything | one lookup |

Both directions are useful, and real systems keep both. Only one of them makes
search possible.

## What a posting stores, and what that costs

A **posting** is one term's presence in one document. The minimum it can hold
is the document identifier. This implementation also stores every **position**
the term occupies in that document.

Positions are not free, and they are not optional. Without them, a search for
the phrase `"computer science"` can only find documents containing both words
somewhere, which includes a document about computers that mentions science in
an unrelated sentence. Positions are the only way to know the two words were
adjacent.

The measured cost is 131 bytes per posting, and total index memory of **9.4
times the source text**. That is the price of positions, and it is the single
strongest argument for the [on-disk format](07-corpus-and-persistence.md): a 300 MB corpus
projects to roughly 2.8 GB of memory, which is more than a build should need.

## Design decisions

### Postings keyed by document, not held as a sorted list

The classic presentation is a sorted list of postings, traversed with a merge
to intersect two terms. This implementation uses a mapping from document
identifier to positions instead.

The reason is that every query operation needed here is a **set operation over
document identifiers**: a free-text query is a union, a phrase query starts
with an intersection. Dictionary keys give those directly, whereas a sorted
list requires a merge walk to get the same answer. The sorted list wins when
postings must be streamed from disk in order, and that is what happened: the
[on-disk format](07-corpus-and-persistence.md) writes each term line with its
document identifiers sorted, precisely so it can be read in order. The two
shapes coexist because they answer to different constraints.

**Partly superseded for the on-disk form.** That reasoning still holds for this
in-memory structure, which keeps its dictionary. It does not hold for a segment
on disk, where [early termination](14-early-termination.md) needs to skip
forward through a term's postings and to attach an upper bound to a range of
them, and a dictionary can do neither. A segment now stores each term's postings
as an ordered sequence of blocks. See
[ADR 0006](adr/0006-document-ordered-postings.md) for the decision and what it
cost.

### Positions are token positions, and one pipeline produces both sides

The index calls the shared analysis function and numbers the terms it gets
back. It does not tokenize or stem itself.

This matters more than it looks. The same function must analyze documents at
index time and queries at search time. If they ever diverge, the query produces
terms the index never stored, and the engine returns nothing at all while
appearing to work perfectly. Sharing one function is what makes that
impossible.

Positions are indices into the **unfiltered** token stream, so a term dropped
by stopword filtering leaves a gap rather than closing the sequence up. The
index does not know or care that filtering happened; it stores whatever
positions analysis reports.

That gap is load-bearing. Without it, removing `of` would make `king of
england` and `king england` store identical positions, and a phrase query could
not tell them apart. The full argument is in
[Stopwords](05-stopwords.md); the short version is that the query keeps its own
offsets too, and phrase matching subtracts those rather than assuming the
surviving terms were adjacent.

There is a property test at
[`tests/test_analysis.py`](../tests/test_analysis.py) asserting that with
filtering switched off, positions are exactly `0, 1, 2, ...` and one term comes
out per token. That is the original invariant surviving as the special case,
which is what demonstrates filtering was the only thing that changed.

### Adding the same document twice is refused

Not defensive programming. Positions are appended, so indexing a document twice
would leave one postings list holding two interleaved position sequences for
what the index believes is a single document. Every phrase query touching that
document would then be wrong, and nothing would report an error. Refusing the
second add converts a silent corruption into an immediate exception.

### An unknown term returns empty postings rather than raising

A term nobody indexed is a term with no matches, which is a perfectly ordinary
answer to an ordinary question. Raising would push a `try`/`except` into every
call site of every query type for no gain.

### A document with no terms still counts as a document

A document of pure punctuation contributes nothing to the vocabulary but must
still increment the document count, because that count is the `N` that
determines how rare a term is. Dropping such documents would inflate the
apparent rarity of every term in the corpus.

## Measured

From [`benchmarks/index_build.py`](../benchmarks/index_build.py), over 10,000
synthetic documents totalling one million tokens and 10.2 MB of text.

```text
distinct terms   : 8,000
postings entries : 733,596

analysis         :  6.28 s   (69.4% of build)
insertion        :  2.78 s
total build      :  9.06 s

throughput       : 110,396 tokens/s, 1.1 MB/s
peak memory      : 95.8 MB, 131 bytes per posting, 9.4x the source text
```

**Analysis dominates, at roughly 70 percent of build time.** Storing postings
is the cheap part. So if indexing is ever too slow, the tokenizer and stemmer
are where to look, not the data structure. That is consistent with the stemmer
measuring about 27 times more expensive per token than the tokenizer.

Projected to a 300 MB corpus at 1.1 MB/s: roughly four and a half minutes to
build, and around 2.8 GB of peak memory.

### A note on how that number was obtained

The first run of this benchmark reported the opposite conclusion, that
insertion took 67 percent of build time. It was wrong. `tracemalloc` was active
during the build pass but not during the analysis pass, and instrumenting every
allocation roughly doubled the time of the pass it was watching.

The benchmark now times the two passes with instrumentation off and measures
memory in a separate pass, where the overhead does not distort anything. The
lesson is worth more than the number: a measurement is only evidence if both
sides of a comparison were measured the same way.

## Alternatives

| Approach | How it works | Wins when | Loses when |
| --- | --- | --- | --- |
| **Forward index** | document to terms | reconstructing or displaying a document | answering any search query |
| **Inverted index (this)** | term to documents and positions | keyword search at any corpus size | substring and wildcard queries it cannot express |
| **Signature files** | a bloom-filter-like bitmask per document | very compact | false positives need a verification pass |
| **Suffix array or suffix tree** | every suffix of the text, sorted | arbitrary substring and wildcard search | several times the size of the text, and expensive to update |
| **Vector index (HNSW, IVF)** | nearest neighbours over embeddings | semantic similarity, paraphrases, multilingual | no exact-match guarantee, needs a model, and cannot do phrase queries |

The last row is where modern retrieval mostly lives, and it does not replace
this structure so much as complement it. An inverted index matches strings
exactly and can prove a document contains a phrase. A vector index matches
meanings approximately and cannot prove anything. Production systems commonly
run both and fuse the results, precisely because the failure modes do not
overlap: lexical search misses paraphrases, semantic search misses exact
identifiers.

## How it connects to everything else

- **Tokenization and stemming** decide the keys. Their output *is* the
  vocabulary, so every choice made there shows up here as index size.
- **Stopword filtering**, when added, will shrink the index most where postings
  are longest, since stopwords are by definition the highest-frequency terms.
  It will also change position numbering, which is the open decision described
  above.
- **One-word queries** are a single lookup on this structure.
- **Free-text queries** are a union over the document keys of several postings
  mappings.
- **Phrase queries** intersect the document keys first, then intersect
  positions within each surviving document. They are the only reason positions
  are stored at all.
- **[Ranking](06-ranking.md)** needs three things, and this provides all of
  them: a term document frequency, the total document count, and each term
  frequency within a document, which is simply the length of its position list.
  The positional postings stored for phrase queries turned out to carry the term
  frequencies for free, so ranking needed no new stored data at all.
- **[Persistence](07-corpus-and-persistence.md)** serialises this structure, and
  the 9.4x memory measurement above is why it matters. Document identifiers are
  stored separately there for the same reason they are tracked separately here:
  a document holding no terms appears in no postings list.
- **[The on-disk index](12-on-disk-index.md)** holds the same information in a
  file that is searched without being loaded, which is what makes the memory
  figure above survivable. It stores each term's highest frequency and each
  document's length alongside the postings, neither of which this structure
  needs but both of which a bound does.
- **[Segments](13-segments.md)** are many such files read as one index, with a
  manifest deciding which exist. That is the form an index that grows actually
  takes; this one is built in a single shot and never changes afterwards.
- **[Early termination](14-early-termination.md)** is the reason the on-disk
  postings are ordered and blocked rather than shaped like this mapping.
