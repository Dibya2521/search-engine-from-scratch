# Changelog

Everything notable that has changed in this project, newest first.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**While the major version is `0`, a minor bump may change a public interface.**
Anything that would be a breaking change after `1.0.0` is called out under
**Changed** or **Removed** with what to do instead.

Each entry says what moved and *why it moved*, because a list of features is a
worse record than a list of reasons. Where a change was made on the strength of
a measurement, the measurement is quoted and the command that produced it is
named, so the claim can be re-checked rather than believed.

## [Unreleased]

Nothing yet.

## [0.2.0] - 2026-09-06

Relevance, measured rather than assumed. The headline is a null result: the new
ranking function did not beat the old one on this project's own test
collection, so the default did not change.

### Added

- This changelog. The engine reached a working state through thirteen commits
  with no running record of what changed between them, so the starting point
  below was reconstructed from the source and the commit history. Keeping it
  from here means the next change can always be compared against the one before
  it.
- A short contributing guide, stating the five rules the project runs on rather
  than a submission process, since there is one maintainer.
- A test collection: seventy documents in `tests/fixtures/evaluation_corpus.xml`
  and fifty-five queries with graded relevance judgements in
  `tests/fixtures/judgements.jsonl`. The existing seven-document fixture is
  large enough to assert postings by hand and far too small to tell two ranking
  functions apart, which is what this collection exists to do.
- `search_engine.judgements`, which reads a judgement file and refuses a
  malformed one. A file silently missing half its labels produces a plausible
  number that is wrong, which is worse than producing no number.
- Graded evaluation metrics: discounted cumulative gain and its normalized form,
  plus reciprocal rank and its mean. nDCG is the first metric here that uses the
  grade rather than a yes or no, which lets it separate two rankings that
  retrieved the same documents in a different order. Average precision scores
  both of those 1.0; nDCG scores them 1.0 and 0.80.
- `search_engine.bm25`, a BM25 scorer, and `--scorer` on the `search` command to
  choose between it and TF-IDF. BM25 saturates term frequency so repeated words
  stop helping, and corrects document length explicitly through a tunable
  parameter rather than implicitly through the cosine.
- `BaseRanker`, holding what does not depend on the scoring formula, so two
  scorers see the same candidates and the same selection rule and any difference
  in results is caused by the formula alone.
- `evaluation.sign_test`, an exact two-sided sign test, so a difference between
  two systems can be reported with whether it is distinguishable from chance.
- `benchmarks/ranking_quality.py`, which compares the two scorers over the
  judgement set and reports both a sign test and a paired permutation test.

### Changed

- Nothing about the default ranking. See below.

### Measured

- **BM25 did not beat TF-IDF on this collection, and the default did not
  change.** Mean average precision 0.7485 to 0.7586 (+1.3%), nDCG@10 0.8660 to
  0.8772 (+1.3%), mean reciprocal rank 0.9788 to 0.9818 (+0.3%), but
  precision@5 0.5921 to 0.5848 (-1.2%). Per query BM25 wins 14, loses 12, ties
  29. Sign test p = 0.845, paired permutation test p = 0.3275. The rule set down
  before the measurement was that BM25 would be adopted only if it proved
  better, so TF-IDF remains the default and BM25 ships as an option. Reproduce
  with `uv run python benchmarks/ranking_quality.py`.
- **The collection, not the scorer, is the limiting factor.** TF-IDF already
  achieves a mean reciprocal rank of 0.979, meaning the first result is relevant
  for nearly every query, so there is almost no headroom for a better scorer to
  occupy.
- **BM25 does fix the case that motivated it.** On the recorded example where a
  document is one query term repeated five times, its score falls from 72 percent
  of the winning document's to 52 percent.
- **The classic BM25 inverse document frequency would have produced negative
  weights on this corpus.** Three of 1,061 terms, worst case `document` at
  -0.704, present in 47 of 70 documents. A negative weight penalises a document
  for containing a query term, so the variant with an added constant is used
  instead.

## [0.1.0] - 2026-09-05

The first working engine: text goes in, ranked results come out, and every
mechanism in between was built rather than installed.

### Added

- **Tokenizer.** Lowercase, then keep maximal runs of `[a-z0-9]`. ASCII-only by
  design, which is a recall ceiling for non-Latin text and is documented as one.
  Measured at 21.1 MB/s.
- **Porter stemmer**, implemented from the published algorithm rather than
  imported, so that `connect` and `connecting` meet on one term. Verified
  against an independent implementation as a differential oracle: **zero
  disagreements across 34,814 words**. Measured at 128,688 words/s, 2.2 times
  the reference implementation.
- **Stopword filtering** over a published 25-word list, removing 41.9 percent of
  stored occurrences. Positions survive filtering on both sides, so phrase
  queries containing a stopword still match.
- **A single analysis pipeline** shared by indexing and querying. If those two
  paths ever diverge, queries ask for terms the index never stored and the
  engine silently returns nothing, which is the easiest way to break a search
  engine.
- **Inverted index with positional postings.** Each term maps to the documents
  containing it and the positions within them. Positions cost memory and buy
  phrase queries. Measured at 110,396 tokens/s, 1.1 MB/s, and 131 bytes per
  posting.
- **One-word, free-text and phrase queries.** A quoted query requires its terms
  adjacent and in order; an unquoted one matches any of them.
- **TF-IDF ranking with cosine similarity** over the vector space model, with
  top-k selection by heap. Measured at 1.7 microseconds per candidate, after a
  939-fold correction: scoring a document by building its full vector costs a
  pass over the whole vocabulary, which projected to 737 seconds per query
  before document vector lengths were precomputed once.
- **Index persistence** in a plain-text format, one line per term. Readable with
  `head`, streamable, and unable to execute code on load, which `pickle` is not.
- **Corpus reader** streaming `<page>` records a page at a time, written as a
  tolerant scanner rather than an XML parser so that an unwrapped dump still
  parses and entity-expansion attacks cannot apply.
- **Retrieval quality metrics**: precision@k, recall@k, average precision and
  mean average precision. These ask whether the results are *good*, which no
  amount of passing unit tests can answer.
- **Command line interface** with `index` and `search`, and exit codes that
  follow the shell convention so it composes in a pipeline.
- **Quality gates**: ruff format, ruff check, pylint, pyright strict and pytest
  with branch coverage, enforced by pre-commit and run in CI on Python 3.12,
  3.13 and 3.14.

### Notes on what this release deliberately does not do

Recorded here because a limitation you chose is worth more than a limitation you
did not notice.

- **No runtime dependencies.** `project.dependencies` is empty and stays that
  way. `numpy` was predicted as the likely first exception and measured to be
  unnecessary, because the document vectors are sparse and a dict of non-zero
  components beats a dense array.
- **Text that is not ASCII produces no tokens at all**, so such documents are
  unsearchable rather than merely degraded.
- **The same word in composed and decomposed Unicode form tokenizes
  differently**, so a document stored in one form and a query typed in the other
  will never match. A test pins this behaviour rather than hiding it.
- **Adding a document to a finished index is not possible.** The only way to add
  one is a full rebuild.

[Unreleased]: https://github.com/Dibya2521/search-engine-from-scratch/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Dibya2521/search-engine-from-scratch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Dibya2521/search-engine-from-scratch/releases/tag/v0.1.0
