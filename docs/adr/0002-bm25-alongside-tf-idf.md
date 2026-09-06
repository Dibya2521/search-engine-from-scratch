# ADR 0002: Ship BM25 alongside TF-IDF, and keep TF-IDF as the default

- **Status:** accepted
- **Date:** 2026-09-06
- **Supersedes:** nothing
- **Superseded by:** nothing

## Context

TF-IDF has a documented weakness. Term frequency enters the score linearly and
without limit, so a document repeating a term twenty times scores twenty times a
document mentioning it once. The concrete case recorded when the ranker was
built: in a five-document corpus, a document consisting of the single word
`computer` repeated five times scores 0.4869 against the query `computer
science`, ranking second of four while containing nothing about science.

BM25 is the standard correction. It saturates term frequency and adds an explicit
document length prior. It has been the strongest simple baseline in information
retrieval for three decades.

Everything BM25 requires was already stored: term frequency is a position list
length, and document length is the sum of a document's position list lengths.

The question was therefore not whether BM25 could be implemented, but whether it
should replace TF-IDF as the default. That is a claim about relevance, and the
project's rule is that a claim about relevance is settled by measurement.

## Decision

**Implement BM25, ship it as a selectable scorer, and keep TF-IDF as the
default.**

Three parts, each decided separately.

### 1. Both scorers stay in the codebase

`BaseRanker` holds what does not depend on the formula: inverse document
frequencies, the staleness check, top-k selection. `Ranker` and `BM25Ranker`
supply only how a query becomes weighted terms and how a document scores against
them.

Keeping both is what makes the comparison reproducible by anyone, rather than a
number in a document that has to be believed. This follows the precedent set by
`benchmarks/phrase_query.py`, which keeps a deliberately naive phrase matcher for
the same reason.

### 2. The inverse document frequency uses the variant with the added constant

```text
log( 1 + (N - df + 0.5) / (df + 0.5) )     instead of     log( (N - df + 0.5) / (df + 0.5) )
```

The classic form goes negative for any term present in more than half the
documents, which means a document is penalised for containing a query term. A
document containing every query term can then rank below one containing fewer.

Measured rather than assumed: of the 1,061 terms in the 70-document evaluation
corpus, three take a negative weight under the classic form. The worst is
`document`, present in 47 of 70, at **-0.704**.

### 3. TF-IDF stays the default

The comparison, on a judgement set committed before BM25 existed:

| Metric | TF-IDF | BM25 | Change |
| --- | --- | --- | --- |
| Mean average precision | 0.7485 | 0.7586 | +1.3% |
| Mean nDCG@10 | 0.8660 | 0.8772 | +1.3% |
| Mean reciprocal rank | 0.9788 | 0.9818 | +0.3% |
| Mean precision@5 | 0.5921 | 0.5848 | -1.2% |

Per query: BM25 wins 14, loses 12, ties 29. Sign test p = 0.845. Paired
permutation test over 100,000 resamplings p = 0.3275.

**The difference is not distinguishable from chance**, and BM25 is worse on one
of the four metrics. The rule for this decision was written down before the
result was known: if the measurement does not show BM25 is better, record that
and keep TF-IDF, because adopting it anyway would be an appeal to authority. The
measurement did not show it.

Reproduce with `uv run python benchmarks/ranking_quality.py`.

## Options considered

**Replace TF-IDF with BM25 outright.** Rejected. It is what the literature would
predict and what most projects do, but this project's own measurement does not
support it, and there would be no way to re-run the comparison afterwards.

**Do not implement BM25 until a better collection exists.** Rejected. The
implementation is what made the measurement possible, and it also produced the
finding that the collection is too easy, which is worth more than the scorer.

**Make BM25 the default anyway, on the strength of the literature.** Rejected.
The whole project is built on preferring a measurement to an argument from
authority. Making an exception the first time the measurement is inconvenient
would mean the rule never applied at all.

**Tune `k1` and `b` on the judgement set to make BM25 win.** Rejected as
dishonest. Fitting parameters on the same data used to evaluate them is
overfitting, and 55 queries cannot be split into a training and a held-out half
large enough for either to mean anything.

## Consequences

**Good.**

- The relevance claim in this repository is now backed by a reproducible
  measurement with a significance test, not by an appeal to what is standard.
- A second scorer exists, so future ranking work has a baseline to beat and a
  harness to beat it in.
- The negative inverse document frequency defect is fixed before it could ever
  produce a wrong ranking.
- BM25 measurably improves the specific case that motivated it: the repeated-word
  document falls from 72 percent to 52 percent of the winning score.

**Bad.**

- Two scorers is more surface than one: two code paths, two sets of tests, and a
  command line flag that a user has to understand.
- The default is now defended by a null result, which is a harder thing to
  explain than a win.

**Follow-up this creates.**

- A larger and harder test collection. Mean reciprocal rank of 0.979 means the
  first result is already relevant for nearly every query, so the collection has
  almost no room to distinguish two scorers. Until that exists, no ranking change
  can be shown to help.
- This decision should be revisited, and this record superseded rather than
  edited, once a collection with enough power exists.
