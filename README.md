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

Under active development.

| Component | State |
| --- | --- |
| Tokenizer | done |
| Stopword filtering | not started |
| Porter stemmer | done |
| Corpus parser | not started |
| Inverted index with positions | done |
| Index persistence | not started |
| One-word and free-text queries | not started |
| Phrase queries | not started |
| TF-IDF ranking and cosine similarity | not started |
| Retrieval quality evaluation | not started |

## Design constraints

1. **No runtime dependencies.** `project.dependencies` is empty and stays that
   way. Anything added needs an architecture decision record explaining where
   the standard library fell short.
2. **Development dependencies are unrestricted**, and are used as verification
   instruments rather than implementation shortcuts. The clearest case: the
   from-scratch Porter stemmer is verified against an independent
   implementation as a differential oracle, so a disagreement falsifies this
   code rather than confirming it.
3. **Every claim carries a measurement.** Index sizes, build times and query
   latencies in the documentation come from a reproducible command, never from
   an estimate.
4. **Documents and queries are processed by the same code path.** If the two
   ever diverge, queries produce keys the index does not contain and the engine
   silently returns nothing.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for dependency and environment management

## Getting started

```bash
uv sync --all-groups
uv run search-engine --version
```

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
docs/                how each mechanism works, and what it costs
docs/adr/            architecture decision records
benchmarks/          measurement scripts
scripts/             corpus preparation utilities
```

## Documentation

- [Tokenization](docs/01-tokenization.md)
- [Stemming](docs/02-stemming.md)
- [The inverted index](docs/03-inverted-index.md)
- [ADR 0001: Toolchain and quality gates](docs/adr/0001-toolchain.md)

## Acknowledgements

The design follows the classic inverted index and vector space model treatment
in Arden Dertat's series on implementing a search engine.

## License

[MIT](LICENSE)
