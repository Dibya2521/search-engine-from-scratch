# Ranking with TF-IDF

Matching decides *whether* a document qualifies. Ranking decides *how well*,
and that is the difference between a result list that is correct and one that
is useful.

Implemented at [`src/search_engine/ranking.py`](../src/search_engine/ranking.py).

## What it is

Every document and every query becomes a vector over the vocabulary. Relevance
is the cosine of the angle between them: a small angle means the two point the
same way, so the document is about what the query asked for.

Each component of a vector is one term's weight, and the weight is a product of
two forces pulling in different directions.

### Term frequency: how much this document is about the term

A document mentioning `computer` ten times is more about computers than one
mentioning it once. Raw counts, straight off the positional postings, because
**the number of positions a term holds in a document is its frequency there.**
That is worth pausing on: storing positions for phrase queries turned out to
give term frequencies for free, so ranking needed no new stored data at all.

### Inverse document frequency: how much the term tells you anything

A term appearing in every document separates nothing. `the` in a corpus of
English tells you precisely zero about which document you want.

```text
idf(term) = log(N / df(term))
```

`N` is the document count, `df` the number of documents holding the term. When
a term is everywhere, `df = N`, so `log(1) = 0` and the term contributes
nothing to any score. Rarer terms produce larger values.

Measured on the five-document corpus in the tests:

| Term | Documents holding it | idf |
| --- | --- | --- |
| `depart` | 1 of 5 | 1.609 |
| `scienc` | 2 of 5 | 0.916 |
| `comput` | 3 of 5 | 0.511 |
| a term in all 5 | 5 of 5 | **0.000** |
| unknown term | 0 | 0.000 |

**Why the logarithm.** Without it, `N/df` punishes common terms far too
harshly: a term in 1 document out of 1,000 would weigh 1,000 times more than
one in every document, which swamps everything else. The log compresses that
into a usable range while keeping the ordering.

Note the two zeros at the bottom of the table meet from opposite directions. A
term in every document is uninformative; a term in no document cannot inform
either. Returning 0.0 for the unknown case is not just a guard against dividing
by zero, it is the right answer.

## Length normalisation, and why it matters

Multiply term frequency by inverse document frequency and you have raw weights.
Used directly, long documents win everything, because they mention every term
more often.

The fix is to divide each vector by its own length, making every document a
unit vector:

```text
||D|| = sqrt(sum of every weight squared)
```

Measured: every document vector in the test corpus has length exactly
1.000000, including a five-word document and a much longer one.

That has a second benefit which is more than a convenience. The cosine of the
angle between two vectors is normally their dot product divided by both
lengths. If both are already length 1, **the cosine is just the dot product**,
so the expensive part of the formula disappears.

## The whole thing on one example

```text
document 4: "computer computer computer computer computer"
query: "computer"

document vector: {comput: 1.0}      one term, so after normalising, exactly 1.0
query vector:    {comput: 1.0}
cosine = 1.0 * 1.0 = 1.0            a perfect score, and it should be
```

And a real ranking, measured:

```text
query "computer science"

1.  0.6778  doc 1  "the computer science department teaches computer science"
2.  0.4869  doc 4  "computer computer computer computer computer"
3.  0.2687  doc 3  "science journal about biology and chemistry"
4.  0.1042  doc 2  "a short note about computers"
```

Document 1 wins because it is about both terms. Document 4 is about `computer`
intensely but knows nothing of science. Document 3 has only the rarer of the
two terms. Document 2 mentions `computer` once among other things. That order
is defensible line by line, which is the test of a scoring function.

## The performance failure, and the fix

This is the part worth reading, because the first implementation was correct
and unusable.

Scoring a document needs its vector, and building its vector needs every term
it holds. The obvious way to find those is to ask every term in the vocabulary
whether this document appears in its postings. That is a pass over the whole
vocabulary **per document scored**.

Measured on a 3,000 document, 4,000 term index:

```text
document_vector for ONE document : 1.3 ms
rank() over 2,998 candidates     : 4.91 s
per candidate                    : 1.64 ms
projected at a 600,000 term vocabulary : 737 s
```

Twelve minutes for one query. The design was finished and worthless.

### Why it was slow, precisely

Cost per candidate was O(vocabulary), and the vocabulary is the one thing that
grows without limit as a corpus grows. Nothing about the query's size entered
into it, which is the tell: a two-word query was paying for 4,000 terms.

### What fixes it

Two observations, both about avoiding work rather than doing it faster.

**A term absent from the query contributes nothing to the dot product.** So
scoring only needs the query's terms, not the vocabulary. That makes the cost
O(query length), which is small and fixed.

**But the document's vector length still depends on all its terms.** That
cannot be avoided per query, so it is computed **once**, for every document, in
a single pass over the postings. One pass over the index total, rather than one
pass over the vocabulary per document scored.

That is what `Ranker` is: a snapshot of every term's inverse document frequency
and every document's vector length, computed on construction.

```text
Ranker construction (once)   : 0.05 s
rank() over 2,998 candidates : 5.2 ms
per candidate                : 1.7 microseconds
speedup                      : 939x
```

The top results were identical before and after, which is what makes it an
optimisation rather than a change.

### The test that keeps it honest

An optimisation that skips work can skip the wrong work. So `score`, the fast
path over query terms, is asserted equal to `cosine_similarity` of the two
complete vectors, which is the definition, over generated corpora. If the fast
path ever starts cutting a corner it should not, that test fails and no other
one would.

This is the same technique used on the stemmer: check the thing against an
independent statement of what it should do, rather than against your own
assumptions.

## Why a stale ranker refuses to work

`Ranker` snapshots the document count and every vector length. Add a document
to the index afterwards and both are wrong: `N` changed, so every inverse
document frequency changed, so every weight and every length changed.

Nothing about that failure is visible in the output. The scores would still be
numbers between 0 and 1, still sorted, still plausible, and quietly incorrect.
So the ranker records how many documents existed when it was built and raises
if that changes. Build it after indexing is finished.

## Top-K with a heap

Returning the best 10 of 3,000 candidates does not need all 3,000 sorted.
`heapq.nlargest` maintains a heap of size k, giving O(n log k) rather than
O(n log n). At k=10 and n=3,000 that is roughly a third of the comparisons.

Ties break on the lower document identifier, so two runs over the same data
produce the same order. Without that, tests could not assert an exact list and
two runs would not be comparable.

## What this discards

**Word order.** `dog bites man` and `man bites dog` produce identical vectors.
That is the bag-of-words assumption, and it is a deliberate trade: phrase
queries preserve order exactly, ranking does not. The two mechanisms cover each
other.

## Alternatives

| Approach | How it works | Wins when | Loses when |
| --- | --- | --- | --- |
| **TF-IDF cosine (this)** | unit vectors, dot product | general text, no tuning, fully explainable | term frequency grows unboundedly, so repetition can be gamed |
| **BM25** | saturating term frequency plus a document-length prior | almost always better in practice; the modern lexical default | two tuning parameters, and less obvious why a score is what it is |
| **Pure boolean** | matched or not | you only need to know whether | no ordering at all |
| **Learning to rank** | a model over many features | large labelled click data available | needs that data, and stops being explainable |
| **Dense retrieval** | cosine over learned embeddings | paraphrases, synonyms, other languages | needs a model, cannot prove an exact match, no phrase support |

**BM25 is the obvious next step**, and the reason is visible in the numbers
above: document 4, five repetitions of one word, scores 0.4869 on
`computer science` while knowing nothing about science. TF-IDF rewards
repetition linearly and forever. BM25 saturates it, so the tenth mention adds
almost nothing, which matches how relevance actually behaves. It would be a
change worth measuring against the evaluation metrics rather than assumed.

## How it connects to everything else

- **Positional postings** supply term frequencies at no extra storage cost,
  since a frequency is a position-list length.
- **Document frequency**, already exposed by the index, is the whole of the
  inverse document frequency calculation.
- **The document count** must include documents holding no terms, or `N` is
  wrong and every score with it. That is why the index tracks identifiers
  separately from postings.
- **Stopword removal** overlaps with this: `log(N/df)` already drives a term in
  every document to zero, so ranking makes stopwords nearly harmless without
  deleting them. Deleting them is a space optimisation, not a scoring one.
- **Querying** produces the candidate set this scores. Ranking never widens
  that set, only orders it.
- **Evaluation** is the only thing that can say whether this scoring is good,
  as opposed to correct.
