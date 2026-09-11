# Query understanding

Most relevance gains in real systems come from understanding the query rather
than from a better scorer. This chapter covers five attempts at it, what each
one does, and what each one could be shown to be worth.

**Read the last section first if you read only one.** Three of the five could
not be shown to help, and the reason is the same every time and is not the
features.

## Spelling correction

A query returning nothing is the worst answer a search engine can give, and a
misspelling is the commonest reason for it. The engine already knows every term
it holds, so it can offer the near ones.

Implemented at [`src/search_engine/spelling.py`](../src/search_engine/spelling.py).

### The BK-tree, and the inequality it rests on

Comparing a query against every term in the vocabulary is correct and costs one
edit distance per term. A **BK-tree** is a metric tree that skips most of them.

It works because Levenshtein distance obeys the **triangle inequality**: for any
three strings, `d(a, c) <= d(a, b) + d(b, c)`. So if the query sits at distance
`d` from some node, a term reachable through that node at distance `k` from it
can be no closer to the query than `d - k` and no further than `d + k`.

**Only children whose distance from the node lies in `[d - max, d + max]` can
hold a match.** Every other subtree is skipped without one distance being
computed inside it. That is the whole of the idea.

```text
                    receive
                   /   |    \
                  1    2     3      children keyed by distance from "receive"
                  |    |     |
              relieve recipe  ...
```

Searching for `recieve` at a maximum distance of 2: it sits 2 from `receive`, so
only children keyed 0 to 4 are worth descending into.

### The distance takes no limit, deliberately

The obvious optimisation is to stop computing once the distance exceeds the
threshold and return "too far". **A BK-tree cannot use it.** At every node it
needs the *true* distance, because the range of children to descend into is
derived from it. A distance capped at the threshold sends the search into the
wrong subtrees and silently returns fewer suggestions.

The two-row formulation is kept, with the shorter string on the inner axis, so
memory is proportional to the shorter string rather than to the product.

### Suggestions are stemmed, and that is visible

The index holds analysed terms, and analysis stems them. So a suggestion is an
index term and may not look like a word.

```text
serach    ->  search
documnet  ->  document
engnie    ->  engin
```

The first two read perfectly. The third does not, and `engin` is nonetheless
**the term that would actually have matched**, which is the useful thing to
report. Returning a readable surface form instead would mean storing surface
forms the index does not keep.

When the command line finds nothing it prints suggestions to stderr and keeps
exit code 1. The code already means "no results" and a suggestion does not
change what happened; sending prose to stderr keeps stdout usable in a pipeline.

**Measurability.** Not measured against relevance, and it is not a ranking
change: it turns an empty result into a second chance. Whether that helps is a
question about users rather than about documents, and answering it needs query
logs and click data that a project without users does not have.

## Synonyms

A document that only says `notebook` should be found by a search for `laptop`.

Implemented at [`src/search_engine/synonyms.py`](../src/search_engine/synonyms.py).

**Expansion happens at query time and never at index time.** That decision has
its own record: see [ADR 0007](adr/0007-query-time-expansion.md).

Groups are equivalences rather than directed rules, because a directed rule
needs a direction decided for every pair and there is rarely a principled
answer. Groups sharing a term merge into one, since equivalence is transitive and
keeping them apart would make the answer depend on the order lines appear in.

```text
# Groups are bidirectional: any member matches any other.
laptop, notebook
car, automobile, vehicle
```

**A member that analyses to more than one term is refused, naming the line.**
Matching a multi-word synonym means matching a phrase, and expansion produces
terms. Accepting one and contributing its words separately would make `portable
computer` match any document containing `computer`: wrong at query time for
someone who never sees the file, where refusing it is wrong at load time for the
person editing it.

**A phrase query is never expanded.** A phrase matches on the spacing between
its terms, and a term standing in for another occupies no position in the
document, so there is nothing to space it against.

**Scores shift, and there is no down-weighting.** An expanded query matches on
more terms, so its scores are not comparable with those of the query that was
typed. The conventional remedy is to weight an expanded term below the original,
commonly at 0.7, which is a convention and not a measurement. It needs the
ranker to accept weighted query terms and it does not, so the weighting is
absent rather than faked.

**Measurability.** Not measured. The judgement set has no query whose relevant
documents use a different word for the same thing, so there is nothing here for
expansion to find. Settling it needs judged queries built around vocabulary
mismatch, which is a property of the collection rather than of the code.

## Query segmentation

Recognising that `new york times` is one entity, not three words.

```text
"new york times review"  ->  ['"new york times"', 'review']
"new york bagel"         ->  ['"new york"', 'bagel']
```

Longest match wins, so the three-word phrase is preferred over the two-word one
that starts it.

**The missing half is learning the phrase set from data.** The set is supplied
by the caller here. The statistic that would build it is **pointwise mutual
information over adjacent term pairs**: pairs that occur together far more often
than their individual frequencies predict are the ones worth binding. It needs a
corpus large enough for the counts to mean something, which this project does
not have.

**Measurability.** Not measured, and not wired into the command line, because a
phrase set nobody has built cannot be evaluated.

## Proximity

A document with `machine learning` in its title is more about the subject than
one mentioning `machine` in the first paragraph and `learning` in the fortieth.

Implemented at [`src/search_engine/proximity.py`](../src/search_engine/proximity.py).

The measure is the **minimum window span**: the width of the narrowest stretch
of text holding at least one occurrence of every query term.

### A worked span

Three terms in one document, at these positions:

```text
alpha  0, 10, 20
beta   11
gamma  21
```

The sweep keeps one cursor per term and repeatedly advances whichever sits
earliest, recording the window between earliest and latest each time:

```text
cursors at   0, 11, 21   window 0..21   span 22
cursors at  10, 11, 21   window 10..21  span 12
cursors at  20, 11, 21   window 11..21  span 11   <- narrowest
```

`alpha` then runs out of positions and the sweep stops. **Span 11.** Every
position is visited once, so the cost is proportional to the positions rather
than to their product; ten thousand occurrences of one term resolve in about
five milliseconds.

The multiplier is `1 + 0.3 * (terms / span)`. Adjacent terms give a span equal
to the term count and the full weight; terms that never occur together give
exactly 1.0, so a document is never penalised for proximity, only rewarded.

**Early termination refuses a ranker built with this boost.** The boost is
applied after scoring and depends on where every term sits; the per-term bounds
that early termination prunes with know nothing about position. It would have
pruned against one scale and reported another, disagreeing with a full scan
silently. See [early termination](14-early-termination.md).

**Measurability.** Measured and **not distinguishable from chance**: mean
average precision +0.1 percent, nDCG **-0.1 percent**, sign test p **1.000**,
permutation test 0.9378, three queries better and two worse out of fifty-four
with fifty unchanged. The collection lacks the power to distinguish it: mean
reciprocal rank is already 0.9818, so the first result is relevant for nearly
every query and a reordering has nowhere to show. A collection with harder
queries, judged before the feature existed, would settle it. **Off by default.**

## BM25F, weighting fields

With per-field terms in the index, the scorer can weight them.

Implemented at [`src/search_engine/bm25f.py`](../src/search_engine/bm25f.py).

### The ordering is the whole idea

BM25F does **not** score each field separately and add the results. It combines
the per-field frequencies first and saturates once over the total:

```text
combined = sum over fields of  weight * frequency / length_correction
score    = idf * combined * (k1 + 1) / (k1 + combined)
```

Here is why that matters, with `k1 = 1.2`, `b = 0.75`, a title weight of 3.0, an
average document of 20 tokens and an average title of 4. The same four
occurrences of one term, arranged three ways:

| Document | combined frequency | Combine first | Saturate first |
| --- | ---: | ---: | ---: |
| A: 2 in title, 2 in body | 4.0 + 6.0 = 10.0 | **1.9643** | 5.8173 |
| B: 4 in body, none in title | 4.0 | **1.6923** | 1.6923 |
| C: 4 in title, none in body | 4.0 + 12.0 = 16.0 | **2.0465** | 6.7692 |

**Look at A against B under the wrong ordering.** The same four occurrences,
merely split across two fields instead of one, score 5.82 against 1.69: a factor
of 3.4 for nothing but the arrangement. Each field ran up its own saturation
curve from zero, so splitting occurrences between fields buys the steep early
part of the curve twice.

Combining first keeps one curve over the document as a whole, and A scores 1.96
against B's 1.69, which is the modest advantage a title mention actually
deserves. **Saturation exists to make the tenth occurrence worth less than the
first, and saturating per field throws that away.**

### The fields here overlap

Textbook BM25F partitions a document into disjoint fields. This index stores the
whole document under unqualified terms, so that every query written before
fields existed still works, and stores the title a second time under qualified
ones.

**So a title occurrence is counted by the document field and by the title
field**, and the default title weight of 3.0 gives a title term an effective 4.0
against a body term's 1.0. That is a boost rather than a partition. It is what
this index supports without a format change, and it is written down rather than
left for a reader to deduce from a surprising number.

**Weighting only the document field reproduces plain BM25**, to within one unit
in the last place over generated indexes. A generalisation must reduce to what
it generalises, and that is the strongest correctness check available for this
formula.

**Measurability.** Measured and **not distinguishable from chance**: mean
average precision +0.3 percent, nDCG +0.6, reciprocal rank +0.9, precision at
five unchanged, eight queries better and five worse out of fifty-four, sign test
p **0.581**, permutation test 0.6060. Every metric moved the right way, which
makes this the most encouraging of the three null results, and encouraging is
not evidence. The collection lacks the power to distinguish it, for the same
reason as above. **The default scorer is unchanged.**

## The thing all three null results have in common

Three features in this release were built, measured, and could not be shown to
help. BM25 was the first, one release earlier. The cause is the same every time
and it is not the features:

**Mean reciprocal rank on this collection is 0.9818 before any of them.** The
first result is already relevant for essentially every query, so there is no
room above rank one for a reordering to occupy. A better scorer has nothing left
to win.

**A bigger corpus does not fix it.** Relevance measurement needs *judgements*,
and judgements are the expensive half. Ten thousand unlabelled articles supply
none of them.

**What a decisive test needs:** queries whose relevant documents are not already
at rank one, enough of them that a few reversals carry a p value, and judgements
written before the feature existed. Judgements made after seeing results follow
the engine instead of judging it.

**Every one of the three is kept and switched off.** That is the same rule BM25
was held to, and holding a fourth, fifth and sixth feature to it is what makes
it a rule rather than a story told once. **A default changes on evidence, never
on expectation.**

## Connections

- **[Unicode](16-unicode.md)** is what happens to the text before any of this.
- **[Ranking](06-ranking.md)** and **[BM25](09-bm25.md)** are the scorers these
  extend, and hold the evaluation numbers these are compared against.
- **[Measuring retrieval quality](08-evaluation.md)** is where the significance
  tests quoted here come from, and why a difference between two means is not a
  result.
- **[Early termination](14-early-termination.md)** is the one part of the engine
  that refuses to work with proximity, and says why.
- **[ADR 0007](adr/0007-query-time-expansion.md)** records the expansion timing
  and the field storage decisions.
