# Contributing

A personal project with one maintainer. Anyone may use it; pull requests are
welcome but not expected.

## Setup

```bash
uv sync --all-groups
uv run pre-commit install
```

## The rules the project runs on

1. **No runtime dependencies.** `project.dependencies` is empty and stays empty.
   Anything added needs an [ADR](docs/adr/) saying which standard library
   approach was tried and where it fell short. Development dependencies are
   unrestricted, but they are verification instruments rather than
   implementation shortcuts: `nltk` appears nowhere in `src/` and exists only as
   the stemmer's differential oracle.
2. **All five gates pass on every commit.** `pre-commit` runs them, and CI runs
   the same commands on Python 3.12, 3.13 and 3.14. Read a finding rather than
   silencing it.
3. **Never write a number you did not produce with a command.** Every figure in
   the documentation came from a script in [`benchmarks/`](benchmarks/). Change
   something with a published number and you re-measure it, both sides with the
   same method.
4. **Comment the why, never the what.** Reasoning about a decision belongs in
   [`docs/`](docs/) or an ADR, not scattered above the code.
5. **Committed prose stands alone.** A reader has only this repository.

Each mechanism has a document in [`docs/`](docs/) explaining what it is, why it
exists, how it works, what it costs as measured, and what the alternatives are.
A change to a mechanism updates its document in the same commit, and adds a
[changelog](CHANGELOG.md) entry.
