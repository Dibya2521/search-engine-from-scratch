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

### Added

- This changelog. The engine reached a working state through thirteen commits
  with no running record of what changed between them, so the starting point
  below was reconstructed from the source and the commit history. Keeping it
  from here means the next change can always be compared against the one before
  it.
- A short contributing guide, stating the five rules the project runs on rather
  than a submission process, since there is one maintainer.

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

[Unreleased]: https://github.com/Dibya2521/search-engine-from-scratch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Dibya2521/search-engine-from-scratch/releases/tag/v0.1.0
