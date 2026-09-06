# search-engine-from-scratch

An inverted-index search engine written from scratch in Python, on the standard
library alone.

The point is not to have a search engine. The point is to understand one. A
mechanism that can be imported can also be used without being understood, so
every retrieval mechanism here is built rather than installed: the tokenizer,
the stopword handling, the Porter stemmer, the inverted index and its on-disk
format, the postings intersection algorithms, TF-IDF, the vector space model,
cosine similarity and top-K selection.

## Status

Complete and working end to end.

| Component | State |
| --- | --- |
| Tokenizer | done |
| Stopword filtering | done |
| Porter stemmer | done |
| Corpus parser | done |
| Inverted index with positions | done |
| Index persistence | done |
| One-word and free-text queries | done |
| Phrase queries | done |
| TF-IDF ranking and cosine similarity | done |
| BM25 ranking | done |
| Retrieval quality evaluation | done |

450 tests, 100 percent branch coverage, verified on Python 3.12, 3.13 and 3.14.

## Requirements

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/) for dependencies and
environments.

```bash
# Linux and macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

`uv` provisions the right Python itself, so no separate Python install is
needed.

## Getting started

```bash
git clone https://github.com/Dibya2521/search-engine-from-scratch
cd search-engine-from-scratch
uv sync --all-groups
```

`uv sync` creates the virtual environment, installs the pinned development
tools from `uv.lock`, and installs this project in editable mode. `uv run`
executes inside that environment without needing to activate it.

## Using it

Build an index once, then query it as often as you like. A small corpus is
included so this works immediately.

```bash
uv run search-engine index tests/fixtures/sample_corpus.xml sample.index
```

```text
indexed 7 documents, 66 distinct terms -> sample.index
```

A one-word query, ranked by relevance:

```bash
uv run search-engine search sample.index "computer"
```

```text
  1. 0.7397  document 2
```

Free text matches any of the terms, and `--limit` truncates from the best end:

```bash
uv run search-engine search sample.index "term frequency document" --limit 3
```

```text
  1. 0.7604  document 5
  2. 0.1030  document 1
  3. 0.0347  document 3
```

A quoted query is a phrase, and requires the terms adjacent and in order:

```bash
uv run search-engine search sample.index '"computer science department"'
```

```text
  1. 0.8541  document 2
```

Two ranking functions are available. TF-IDF is the default; BM25 is selectable
and scores the same results differently:

```bash
uv run search-engine search sample.index "computer science" --scorer bm25
```

Queries are stemmed exactly as documents are, so `connecting` finds a document
that says `connect`:

```bash
uv run search-engine search sample.index "connecting"
```

```text
  1. 0.4637  document 3
```

Exit codes follow the shell convention, so this composes in a pipeline: `0`
found results, `1` found none, `2` the input was wrong.

### Corpus format

A sequence of page records. Content outside them is ignored, so a dump wrapped
in a root element also works.

```xml
<page><id>1</id><title>Inverted index</title><text>An inverted index maps each term to...</text></page>
<page><id>2</id><title>Computer science</title><text>Computer science is the study of...</text></page>
```

## Design constraints

1. **No runtime dependencies.** `project.dependencies` is empty and stays that
   way. Anything added needs an architecture decision record explaining where
   the standard library fell short.
2. **Development dependencies are unrestricted**, and are used as verification
   instruments rather than implementation shortcuts. The clearest case: the
   from-scratch Porter stemmer is verified against an independent
   implementation as a differential oracle, with zero disagreements across
   34,814 words.
3. **Every claim carries a measurement.** Index sizes, build times and query
   latencies in the documentation come from a reproducible command in
   `benchmarks/`, never from an estimate.
4. **Documents and queries are processed by the same code path.** If the two
   ever diverge, queries produce terms the index never stored and the engine
   silently returns nothing.

## Quality gates

Every commit must pass all five. CI runs the same commands on Python 3.12, 3.13
and 3.14.

```bash
uv run ruff format --check .
uv run ruff check .
uv run pylint src tests
uv run pyright
uv run pytest --cov --cov-report=term-missing
```

To have them run automatically before each commit:

```bash
uv run pre-commit install
```

## Layout

```text
src/search_engine/   the package, standard library only
tests/               unit and property-based tests
tests/fixtures/      a small committed corpus
docs/                how each mechanism works, and what it costs
docs/adr/            architecture decision records
benchmarks/          measurement scripts
```

## Documentation

Each of these explains one mechanism from first principles: what it is, why it
exists, how it works internally, what it costs as measured, and what the
alternatives are.

- [Tokenization](docs/01-tokenization.md)
- [Stemming](docs/02-stemming.md)
- [The inverted index](docs/03-inverted-index.md)
- [Querying](docs/04-querying.md)
- [Stopwords](docs/05-stopwords.md)
- [Ranking with TF-IDF](docs/06-ranking.md)
- [BM25](docs/09-bm25.md)
- [Reading a corpus, and saving the index](docs/07-corpus-and-persistence.md)
- [Measuring retrieval quality](docs/08-evaluation.md)
- [ADR 0001: Toolchain and quality gates](docs/adr/0001-toolchain.md)
- [ADR 0002: BM25 alongside TF-IDF](docs/adr/0002-bm25-alongside-tf-idf.md)

What changed between versions, and why, is in the
[changelog](CHANGELOG.md).

## Measured

Everything below came from a script in `benchmarks/`, not from an estimate.

| | |
| --- | --- |
| Tokenizer | 21.1 MB/s |
| Stemmer | 128,688 words/s, 2.2x an independent implementation |
| Index build | 110,396 tokens/s, 1.1 MB/s |
| Index memory | 131 bytes per posting, 9.4x the source text |
| Ranking | 1.7 microseconds per candidate |

Two findings worth the space:

**The textbook phrase-query optimisation was slower than not doing it**, losing
five cases out of six, because sorting the position sets by size has to build
every set before it can compare their lengths. Rebuilt to construct them lazily,
it wins 1.8x on the cases that are actually slow.

**BM25 did not beat TF-IDF on this project's own test collection.** It wins on
three of four metrics by about one percent, loses on the fourth, and wins 14
queries against 12. A sign test gives p = 0.845 and a paired permutation test
p = 0.33, so the difference cannot be told from chance. TF-IDF therefore remains
the default and BM25 ships as an option, which is what the rule written before
the measurement said to do. The finding underneath it is that the collection is
too easy to decide the question: the first result is already relevant for
essentially every query.

**Ranking was 939x too slow before it was measured.** Scoring a document by
building its full vector costs a pass over the whole vocabulary, projecting to
737 seconds per query on a realistic index. Precomputing document vector lengths
once brings it to 5.2 ms.

## Acknowledgements

The design follows the classic inverted index and vector space model treatment
in Arden Dertat's series on implementing a search engine.

## Contributing

One maintainer, so pull requests are welcome but not expected.
[CONTRIBUTING.md](CONTRIBUTING.md) is short and states the five rules the
project runs on, of which the first is that nothing gets installed that could be
built.

## License

[MIT](LICENSE)
