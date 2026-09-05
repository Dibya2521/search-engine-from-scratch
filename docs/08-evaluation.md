# Measuring retrieval quality

Every other test in this project asks whether the engine does what it claims.
These metrics ask a different question: **given what a person considers
relevant, how well does the ranking agree?**

That is the only question a search engine is finally judged on, and no number
of passing unit tests answers it. A search engine can be provably correct and
return useless results.

Implemented at [`src/search_engine/evaluation.py`](../src/search_engine/evaluation.py).

## What these need that nothing else does

**Human judgements.** A set of document identifiers a person has marked
relevant for a given query. They cannot be derived from the corpus, computed
from the index, or inferred from the scores. Somebody has to decide.

That is why retrieval evaluation is genuinely hard in practice, and why
standard test collections exist: the judgements are the expensive part, not the
arithmetic.

## The three metrics

### Precision@k: is what I returned any good?

Of the top k results, what fraction are relevant?

```text
retrieved [1, 2, 3], relevant {1, 3}, k=3  ->  2/3
```

Punishes returning rubbish. Blind to what was missed.

**One decision worth stating:** it divides by how many results were actually
returned, not by k. An engine returning three good results when ten were
allowed scores 1.0, not 0.3. Dividing by k would punish an engine for the
corpus being small, which is not a property of the engine.

### Recall@k: did I miss anything?

Of everything relevant, what fraction reached the top k?

```text
retrieved [1, 2, 3], relevant {1, 2, 3, 4, 5, 6}, k=3  ->  3/6 = 0.5
```

Punishes missing things. Blind to how much rubbish came with them.

### They pull against each other, and that is the point

Measured:

| Strategy | Precision | Recall |
| --- | --- | --- |
| Return all 20 documents | 0.10 | **1.00** |
| Return only the single best | **1.00** | 0.50 |

Perfect recall is trivially achievable by returning everything, and perfect
precision by returning one certain hit. Neither is a search engine. Any claim
about one number alone is meaningless, which is why they are always quoted
together.

### Average precision: was the order any good?

Precision recomputed at every rank where a relevant document appears, then
averaged.

```text
retrieved [1, 2, 3], relevant {1, 3}
  rank 1: relevant, precision 1/1 = 1.000
  rank 3: relevant, precision 2/3 = 0.667
  average = 0.833
```

**This is the metric that sees ordering, and the other two cannot.** Measured on
two result lists containing the identical documents:

| Result | Precision@3 | Recall@3 | Average precision |
| --- | --- | --- | --- |
| `[1, 3, 2]` | 0.667 | 1.0 | **0.833** |
| `[2, 1, 3]` | 0.667 | 1.0 | **0.583** |

Same documents, same set, same precision, same recall. The first puts the
relevant ones first and the second buries one behind an irrelevant result.
Average precision is the only one of the three that notices, which is exactly
why ranking work has to be judged by it.

It divides by the total number of relevant documents, not by how many were
found, so failing to retrieve a relevant document costs the same as ranking it
last. Retrieving one of two relevant documents perfectly scores 0.5, not 1.0.

### Mean average precision

The average of average precision across several queries. The conventional
single number for comparing two retrieval systems, because it accounts for both
what came back and the order it came back in.

## The conventions, stated rather than hidden

Two cases are mathematically undefined and get a documented answer instead:

- **Nothing relevant.** Recall and average precision are undefined, since the
  denominator is zero. Both return 0.0, which keeps averages over many queries
  finite.
- **Nothing retrieved.** Precision is undefined. It returns 0.0.

These are conventions, not truths. They are chosen so that a batch of queries
produces a usable number rather than an error, and they are written down here
because a reader comparing against another implementation needs to know which
convention was picked.

## What to do with these

The immediate use is deciding whether a change helped. Ranking has an obvious
next step, BM25 instead of TF-IDF, motivated by a specific weakness visible in
the [ranking measurements](06-ranking.md): a document of one word repeated five
times scores 0.4869 on a two-word query while knowing nothing about half of it.

Without these metrics, adopting BM25 would be an appeal to authority. With
them, and a hand-labelled judgement set, it becomes a measurement: build both,
score the same queries, compare mean average precision. That is the difference
between engineering and preference.

The judgement set does not exist yet, and building an honest one is real work:
the queries have to be chosen before the results are seen, or the labelling
follows the engine rather than judging it.

## Alternatives and what they add

| Metric | Adds | Needs |
| --- | --- | --- |
| **Precision, recall (these)** | the basic trade-off | binary relevant or not |
| **F1** | one number blending both | the same, plus a choice about their relative weight |
| **Average precision (this)** | sensitivity to rank order | the same |
| **nDCG** | graded relevance, and discounting by position | judgements on a scale, not just yes or no |
| **MRR** | how soon the first correct answer appears | only useful when one answer is wanted |
| **Click-through, dwell time** | what users actually did | live traffic, and it measures the interface as much as the ranking |

**nDCG is the standard in modern retrieval**, because real relevance is graded
rather than binary: a document can be perfect, useful, or tangential. It needs a
richer judgement set, which is why these three come first.

## How it connects to everything else

- **Ranking** is the thing being measured. These metrics are the only way to
  tell a scoring improvement from a scoring change.
- **Querying** supplies the candidate set, and recall is bounded by it: a
  document the matcher never returned cannot be ranked into the top k. So a
  recall problem may be a matching problem rather than a scoring one, and these
  numbers alone do not distinguish the two.
- **Stopword removal** trades recall for index size, and this is where that
  trade would be quantified rather than assumed.
- **Tokenization** sets a hard ceiling on recall. Under the ASCII-only rule,
  non-Latin documents produce no tokens at all, so no metric here can ever
  score them above zero. A ceiling set three stages upstream cannot be fixed by
  better ranking.
