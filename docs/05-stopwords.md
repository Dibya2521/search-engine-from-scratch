# Stopwords

Removing the words too common to help. A small technique with one genuinely
subtle consequence, which the usual presentation of it gets wrong.

Implemented at [`src/search_engine/stopwords.py`](../src/search_engine/stopwords.py)
and applied in [`src/search_engine/analysis.py`](../src/search_engine/analysis.py).

## What it is

A stopword is a term so common that knowing a document contains it tells you
almost nothing. `the` appears in essentially every English document, so its
postings list names every document and separates none of them.

```text
"The index maps terms to documents and positions"
  ->  index, map, term, document, posit          (the, to, and dropped)
```

## Why it exists

Not primarily to save space in the dictionary. The measured reason is more
interesting than that.

| | Terms | Postings | Stored occurrences | Build time |
| --- | --- | --- | --- | --- |
| No filtering | 4,022 | 192,115 | 300,000 | 3.49 s |
| 25 stopwords removed | 3,998 | 132,883 | **174,247** | 2.96 s |
| Change | **-0.6%** | -30.8% | **-41.9%** | -15% |

Measured over 3,000 synthetic documents of 100 words each, with a
Zipf-distributed vocabulary and stopwords at a realistic 42 percent of all word
instances.

**Twenty-five words out of four thousand remove 42 percent of everything
stored.** The vocabulary barely changes, because 25 entries out of 4,000 is
nothing. What collapses is the number of *occurrences*, because a handful of
words account for an enormous share of all word instances.

That is Zipf's law made concrete: term frequency falls off so steeply that the
top few words dominate the corpus. It is also why the saving lands exactly where
the index is fattest, on the longest postings lists.

The cost is precision. `to be or not to be` is entirely stopwords, so it becomes
unsearchable. Any query whose meaning lives in its common words is damaged.

## Where the list comes from

Two sources, deliberately.

**A published list of 25 terms**, from Manning, Raghavan and Schutze,
*Introduction to Information Retrieval*, table 2.2. Chosen because it is
published rather than invented, and small: a longer list removes more but starts
discarding words that carry meaning in some queries. A hardcoded list is
arbitrary data, and the least arbitrary hardcoded list is a citable one.

**A function deriving one from a real index.** `derive(index, threshold)`
returns terms appearing in at least that fraction of documents. This is the
honest way to decide what is common *in this corpus* rather than in English
generally, and it is possible only because the index already exposes document
frequency.

The two are not interchangeable, and the distinction matters: the published list
holds **surface words** and is applied before stemming; `derive` returns
**stems**, because that is what an index stores.

## Order in the pipeline, and why

```text
tokenize  ->  drop stopwords  ->  stem
```

Filtering happens on raw tokens, before stemming. The alternative would work,
but the list would then have to hold stems: `was` stems to `wa`, so a
stem-stage list would need the entry `wa`, which nobody can read or maintain.

## The consequence everybody gets wrong

This is the part worth understanding, and it is where this implementation
departs from the source design.

### The problem

Removing a word from the middle of a document changes what "adjacent" means. If
positions are assigned *after* filtering, then:

```text
"king of england"   ->  king@0, england@1     ("of" gone, positions closed up)
"king england"      ->  king@0, england@1     identical
```

Those two documents are now indistinguishable, so the phrase query
`"king of england"` matches both. That is silently wrong.

### The obvious fix, which is also wrong

Keep the original positions, so the gap survives:

```text
"king of england"   ->  king@0, england@2     (gap at 1 where "of" was)
"king england"      ->  king@0, england@1
```

The documents are now distinguishable. But the **query** goes through the same
pipeline, so `"king of england"` also loses its `of` and becomes two terms. If
the phrase matcher assumes those two terms were adjacent, it looks for
`england` at `king + 1`, finds it at `king + 2`, and matches nothing. A phrase
containing a stopword becomes permanently unmatchable.

### What actually works

Filter both sides, keep the original positions on both sides, and have the
phrase matcher subtract **the query's own offsets** rather than consecutive
integers.

```text
document "computer of science"  ->  comput@0, scienc@2

query "computer science"       offsets (0,1)  ->  {0-0} & {2-1} = {}   no match
query "computer of science"    offsets (0,2)  ->  {0-0} & {2-2} = {0}  match
query "the computer science"   offsets (1,2)  ->  normalises the same way
```

Every case comes out right. The first must not match, because the document does
not contain those two words adjacent. The second must match. The third shows
that a leading stopword is harmless, because only the *differences* between
offsets matter.

**Why it works:** with no filtering, the query offsets are `0, 1, 2, ...` and
this reduces exactly to the original algorithm. Stopword gaps are then just the
general case of the same arithmetic. Nothing special-cases them.

This is strictly better than removing stopwords before positions, and it costs
one extra tuple on the `Query` object.

Verified at [`tests/test_query.py`](../tests/test_query.py), including a
property test that takes the literal text between two consecutive surviving
terms of a generated document, stopwords included, and insists the phrase query
finds the document it came from.

## Alternatives

| Approach | Wins when | Loses when |
| --- | --- | --- |
| **No stopword removal** | phrases matter more than size; modern engines mostly do this now | 42 percent more stored occurrences |
| **Small published list (this)** | predictable, citable, reproducible | not tuned to the corpus |
| **Derived from document frequency** | tuned to the actual corpus | needs a built index first, and the threshold is another arbitrary number |
| **Keep them but weight them down** | best of both: phrases stay exact and common terms score near zero | needs ranking to exist first, so it cannot be the only mechanism |

The last row is worth noting, because it is what most current systems do.
Inverse document frequency already drives the weight of a term appearing in
every document towards zero, so ranking makes stopwords nearly harmless without
deleting them. Removing them is a space optimisation from an era when space was
the binding constraint. It is kept here because the space measurement is real,
and because the position problem is worth having solved.

## How it connects to everything else

- **Tokenization** feeds this, so the list is defined against a specific
  tokenizer. A tokenizer change can silently stop entries matching.
- **Stemming** runs after, which is why the list holds surface words.
- **The inverted index** stores the surviving terms, and positions now contain
  gaps. Nothing in the index needed changing: it stores whatever positions
  analysis reports.
- **Phrase queries** are the only part that had to change, and the change was to
  stop assuming query terms are consecutive.
- **TF-IDF** will make the deletion partly redundant, since `log(N/df)` already
  drives a term in every document to a weight of zero. The two mechanisms
  overlap, and measuring how much is worth doing once ranking exists.
