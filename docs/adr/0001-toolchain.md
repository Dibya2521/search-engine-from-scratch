# ADR 0001: Toolchain and quality gates

- Status: accepted
- Date: 2026-09-03

## Context

This project has two goals that pull against each other. It must be a genuine
from-scratch implementation, which wants as little machinery as possible between
the reader and the algorithm. It must also be maintainable and verifiable, which
wants packaging, typing, linting, tests and CI.

The resolution is to put the strictness in the toolchain and the simplicity in
the runtime. Tooling is where rigour is cheap: it costs configuration once.
Dependencies are where rigour is expensive: every import is a mechanism nobody
here understands from the inside.

Each decision below records what it is, what it is for, how it is wired, what
else was available, and the fact that settled it.

## Decision 1: no runtime dependencies

`project.dependencies` is empty, and stays empty.

- **Purpose.** A from-scratch implementation is the premise. An imported stemmer
  or an imported TF-IDF vectoriser defeats it.
- **Alternatives.** `nltk` for tokenizing and stemming, `numpy` for the ranking
  vector arithmetic, `scikit-learn` for TF-IDF outright. All three would produce
  a working search engine sooner, and all three would leave the mechanism
  unexplained.
- **Why this one.** Three payoffs beyond the obvious: the code is auditable end
  to end, there is no dependency-resolution risk on new Python releases, and any
  eventual service container stays small because the runtime layer is just
  Python.
- **Escape hatch.** A runtime dependency is permitted, but only with its own ADR
  stating what standard library approach was tried and where it fell short.
  `numpy` for the ranking arithmetic is the likely candidate, and it must be
  justified by a measurement rather than an assumption about speed.

Development and test dependencies are deliberately unrestricted, because their
job is verification rather than implementation. `nltk` will be used in tests
only, as a differential oracle against the from-scratch stemmer.

## Decision 2: uv for environments and locking

- **Purpose.** Reproducible environments and a committed lockfile.
- **How.** `uv sync --all-groups` locally; CI uses `uv sync --locked`, so a stale
  `uv.lock` fails the build instead of drifting silently.
- **Alternatives.** `pip` with `requirements.txt`, which has no real lock and no
  dependency groups. Poetry, which is slower and whose `[tool.poetry]` metadata
  predates PEP 621. PDM, standards-compliant but with a smaller ecosystem. Plain
  venv and pip, with no locking at all.
- **Why this one.** It reads standard PEP 621 metadata and PEP 735 dependency
  groups, so nothing in this repository is locked to the tool, and it resolves
  fast enough that CI can afford a full sync on three Python versions.

## Decision 3: `requires-python = ">=3.12"`, CI matrix 3.12 to 3.14

- **Purpose.** Define the supported range and prove it rather than claim it.
- **Alternatives.** Pin to a single local interpreter and test one version. Or
  also support 3.11, which is still widely deployed.
- **Why this one.** 3.12 is the floor because nothing in the design needs newer
  syntax. 3.11 was dropped because supporting it buys nothing for a project
  nobody installs from a package index, while adding a fourth CI job and a
  fourth chance of a version-specific type error. The upper end matters more
  than the lower: 3.14 is recent, and testing it is how a version-specific break
  surfaces here rather than at the point where it blocks work.
- **Not yet verified.** Whether every development dependency has working wheels
  on 3.14 in CI. No CI run has happened yet, so the first run is the check. If a
  tool lags, the fix is to drop that one job with a comment, not to lower the
  floor.

## Decision 4: ruff as the primary linter and the only formatter

- **Purpose.** One fast tool covering formatting, import sorting and the bulk of
  linting.
- **Alternatives.** Black plus isort plus flake8 plus a stack of plugins, which
  is the arrangement ruff was built to replace, and which needs four configs and
  four processes.
- **Why `select = ["ALL"]` rather than a curated set.** It inverts the burden of
  proof. A rule is off only when there is a written reason, rather than on only
  when somebody remembered it. The ignore list is therefore short, and each
  entry is justified at the line:
  - `COM812` and `ISC001` are off because the formatter owns trailing commas and
    implicit string concatenation, and leaving them on makes linter and
    formatter fight each other.
  - `CPY001` is off because a per-file copyright header adds noise that `LICENSE`
    already covers.
  - Conflicting docstring rules are resolved by `convention = "google"` rather
    than by hand, so the mutually exclusive pairs are switched off by the tool
    that knows which pairs conflict.
- **Per-file relaxations.** Tests may use bare `assert` (`S101`), bare numbers in
  assertions (`PLR2004`), and no per-test docstring (`D103`). The last is
  deliberate: a docstring that restates the test's own name is noise, so test
  names carry the documentation and a docstring appears only where there is a
  non-obvious reason. `scripts/` and `benchmarks/` may print and are not
  importable packages.
- **Accepted cost.** The first run on new code usually fails. That is the gate
  working, and the response is to read the finding rather than add an ignore.

## Decision 5: pylint kept in addition to ruff

This is the one decision that looks redundant, so it carries the most reasoning.

- **Purpose.** Catch design-level problems ruff does not implement.
- **What it uniquely adds.** `duplicate-code` (R0801), a whole-program
  similarity analysis with no ruff equivalent, and the limits under
  `[tool.pylint.design]`.
- **How the overlap is resolved.** Every message pylint raises that ruff already
  owns is disabled by name, with the equivalent ruff code in a comment on each
  line. So each class of problem is reported by exactly one tool, and the
  reasoning is auditable rather than assumed.
- **Alternatives.** Drop pylint and accept the loss of duplication detection,
  which is a real loss on a codebase that will grow several structurally similar
  postings-intersection routines. Or drop ruff and keep pylint alone, which is
  far slower and has no formatter.
- **Why both.** Once the overlap is cut they are complementary: ruff is a fast
  per-file linter, pylint is a slower whole-program analyser. The cost is one
  extra process in the gate; the benefit is the class of finding that only
  appears across files.
- **Side benefit.** `max-nested-blocks = 3` enforces a two-levels-of-nesting
  guideline that no ruff rule expresses. Without pylint that guideline would rest
  on review discipline alone.

## Decision 6: pyright in strict mode

- **Purpose.** An unchecked type is an unverified assumption, and the postings
  structures here are nested enough that a wrong shape is easy to write and hard
  to see by eye.
- **Alternatives.** mypy, the older standard with broader plugin support. No
  type checking at all.
- **Why pyright.** Strict mode is a single switch rather than a dozen
  `disallow_*` flags, and it is the checker behind the editor integration in use,
  so the terminal and the editor agree. mypy would be an equally defensible
  choice; this is not a close-run correctness question, so it was settled on
  ergonomics.
- **Consequence accepted.** Tests are type-checked too. That is deliberate: a
  test with a wrong type may not be asserting what it appears to.

## Decision 7: pytest, with branch coverage floored at 90 percent

- **How.** `--strict-markers` and `--strict-config` turn typos in markers and
  config into errors. `xfail_strict = true` makes a test marked
  expected-to-fail that passes into a failure. `filterwarnings = ["error"]`
  treats a warning as a defect that has not been diagnosed yet.
- **Alternatives.** `unittest` from the standard library, which would be
  consistent with the no-runtime-dependency rule but has no fixtures, no
  parametrisation and no coverage plugin. A lower coverage floor, or none.
- **Why this one.** The no-dependency rule governs the runtime, not the test
  harness, and parametrisation plus property-based testing is what makes the
  stemmer and intersection suites tractable at all.
- **What the floor is and is not.** 90 percent branch coverage stops coverage
  silently rotting. It is explicitly not the definition of tested: a component is
  done when it has a test built from input the design did not anticipate, and
  coverage cannot measure that.
- **One documented exclusion.** The `if __name__ == "__main__":` guard cannot
  execute in-process, so in-process coverage was structurally incapable of
  measuring it and reported a misleading 90.00 percent. It is excluded from
  coverage and verified instead by a test that runs the module in a real
  subprocess, which is a stronger check than coverage could give.

## Decision 8: pre-commit runs all five gates, not a fast subset

- **Purpose.** Make committing through a red gate impossible by accident.
- **How.** Third-party hooks for file hygiene and ruff; local `system` hooks
  calling `uv run` for pylint, pyright and pytest, so the versions match
  `uv.lock` and therefore match CI exactly.
- **Alternatives.** The common arrangement is fast hooks on commit and slow ones
  on push or in CI only, which is better once a test suite is slow.
- **Why this one, for now.** The suite currently runs in about a second, so the
  honest choice is to run everything and measure rather than pre-optimise for a
  slowness that does not exist. The decision is explicitly conditional: if the
  hook run exceeds roughly ten seconds, pytest and pylint move to a pre-push
  stage and this ADR gets a superseding note. The trigger is a measurement, not
  a feeling.
- **Note.** `check-added-large-files` is the load-bearing hook. Corpus data that
  reaches git history cannot be removed cleanly afterwards.

## Consequences

- Five gate commands rather than one. Accepted, because CI runs the identical
  list and there is no divergence to reconcile.
- `select = ["ALL"]` will occasionally surface a rule that needs a judgement
  call. That is the intended cost, and each resolution becomes a documented
  decision.
- Hook and action versions are never hand-written. Three were guessed during
  setup and all three were wrong, so they are resolved by `pre-commit autoupdate`
  or looked up at the source.
