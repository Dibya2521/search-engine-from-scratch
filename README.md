# search-engine-from-scratch

An inverted-index search engine written from scratch in Python, on the standard
library alone.

The point is not to have a search engine. The point is to understand one. If a
mechanism can be imported, it can also be used without understanding it, so
every retrieval mechanism here is built rather than installed: the tokenizer,
the stopword handling, the Porter stemmer, the inverted index and its on-disk
format, the postings intersection algorithms, TF-IDF, the vector space model,
cosine similarity and top-K selection.

## Status

Phase 0 complete: packaging, typing, linting, tests and CI are in place. The
engine itself is not built yet. Progress is tracked phase by phase, and each
phase is documented under [`docs/`](docs/).

| Phase | Scope | State |
| --- | --- | --- |
| 0 | Project foundation and quality gates | done |
| 1 | Tokenizer, stemmer, corpus parser, inverted index | not started |
| 2 | One-word, free-text and phrase queries | not started |
| 3 | TF-IDF ranking, vector space model, evaluation | not started |

## Design constraints

1. **No runtime dependencies.** `project.dependencies` is empty and stays that
   way. Anything added needs an architecture decision record explaining why the
   standard library was not enough.
2. **Development dependencies are unrestricted**, and are used as verification
   instruments rather than as implementation shortcuts. The clearest example:
   the from-scratch Porter stemmer is tested against an independent
   implementation as a differential oracle, so a disagreement falsifies our own
   code rather than confirming it.
3. **Every claim carries a measurement.** Index sizes, build times and query
   latencies in the documentation come from a reproducible command, not from an
   estimate.

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
docs/                design notes and how each mechanism works
docs/adr/            architecture decision records
benchmarks/          measurement scripts
scripts/             corpus preparation utilities
```

## Source material

The design follows Arden Dertat's three-part series, which specifies the corpus
format, the index structure, the phrase query algorithm and the TF-IDF
formulation:

1. [Create the index](https://www.ardendertat.com/2011/05/30/how-to-implement-a-search-engine-part-1-create-index/)
2. [Query the index](https://www.ardendertat.com/2011/05/31/how-to-implement-a-search-engine-part-2-query-index/)
3. [Ranking with TF-IDF](https://www.ardendertat.com/2011/07/17/how-to-implement-a-search-engine-part-3-ranking-tf-idf/)

## License

[MIT](LICENSE)
