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
- **The predicted first exception did not materialise.** `numpy` was named here
  as the likely candidate, for the ranking vector arithmetic. Ranking is now
  complete without it, at 1.7 microseconds per candidate, because the vectors
  are sparse: a document holds a few dozen of several thousand possible terms,
  so a dict of the non-zero components beats a dense array. `numpy` would have
  been the wrong tool, and assuming otherwise would have cost the constraint for
  nothing. The measurement is in docs/06-ranking.md.

Development and test dependencies are deliberately unrestricted, because their
job is verification rather than implementation. `nltk` is used in tests
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
- **Now verified.** CI passes on all three versions, so every development
  dependency does have working wheels on 3.14. This was recorded as unverified
  until the first run, and the first run settled it.
- **What the first CI run actually caught** was unrelated and worth recording:
  `astral-sh/setup-uv@v10` does not resolve, because that action stops
  publishing moving major tags at `v7` even though releases reach `v10.0.1`.
  Verifying that a version exists is not the same as verifying that the ref you
  wrote resolves, and only a real run distinguishes them.

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
- **The mainstream alternative, and when to prefer it.** Most large projects
  select rule *families* instead of everything, for example
  `select = ["E", "F", "I", "N", "UP", "B", "SIM", "RUF"]`. Eight codes,
  readable at a glance, and no long ignore list. That is the better choice when
  a team has to agree on the configuration, because a family list states intent
  while `ALL` plus exceptions states history.
- **Why `ALL` still wins here.** 829 stable rules are enabled and exactly
  **four** required a global suppression. A suppression rate under one percent
  is evidence that the maximal setting is not fighting this codebase, which is
  the usual reason projects retreat to a family list.
- **Nobody reads 829 rules, and the config does not require it.** The ignore
  list is not designed up front, it accumulates: a rule fires, it is either
  fixed or given an ignore with a reason, one at a time. Rules that have paid
  for themselves this way include `RUF001` (visually ambiguous Unicode in a
  string), `RUF100` (a suppression that suppresses nothing), `PLR2004` (magic
  values), `PLR0915` (an over-long function), and `D403`. None of them would
  have been chosen deliberately from a list.

## Decision 5: pylint kept, but restricted to four checks

Most projects that once ran both have dropped pylint, and they are right to.
This is the decision that looks redundant, so it is the one settled by
measurement rather than argument.

### Why the usual arrangement is indefensible

Configured with its default rule set, pylint cost **18.64s of a 31.9s gate**,
58 percent of the total, and almost every message it produced was one ruff had
already reported. Paying more than half the gate for duplicate findings cannot
be justified. That is the real reason the ecosystem moved on, and dropping
pylint outright is a reasonable conclusion from it.

### What only pylint can do

Four checks, and nothing else:

| Check | Situation in ruff |
| --- | --- |
| `redefined-outer-name` | **No equivalent at any setting.** Verified by running ruff with every rule plus preview against a parameter shadowing a module-level function: it reports nothing. |
| `duplicate-code` | No equivalent. Nothing in ruff compares across files. |
| `too-many-nested-blocks` | `PLR1702`, preview only. |
| `too-many-locals` | `PLR0914`, preview only. |

Enabling ruff's preview mode to recover the last two was measured and rejected:
it adds 42 findings on this codebase, 24 of them demanding a `Returns:` section
in every docstring, which contradicts the rule that a docstring carries the
non-obvious fact and nothing more.

`redefined-outer-name` has already earned its keep. It found eight real defects
here that ruff missed: seven helper parameters in the stemmer named `stem`,
shadowing the module-level `stem` function, and one in a fixture. That is a
readability defect rather than a style preference, and nothing else would have
caught it.

`duplicate-code` has not fired yet. It is kept on the expectation that several
structurally similar postings-intersection routines are coming, which is exactly
what a cross-file similarity analysis is for.

### How it is configured

Inverted from the usual arrangement: `disable = ["all"]`, then those four
enabled by name. The design limits ruff *can* enforce (`PLR0913` arguments,
`PLR0915` statements, `PLR0912` branches) moved into `[tool.ruff.lint.pylint]`,
so each limit is applied by exactly one tool.

The result, measured when the suite held 231 tests, was **9.70s instead of
18.64s**, and a total gate of 20.8s instead of 31.9s, and a role that states itself: pylint runs the four checks ruff cannot.

### Alternatives

- **Drop pylint entirely.** Gate falls to 13.1s, one less tool to explain, and
  it matches common practice. Rejected because shadowing detection would be
  lost permanently rather than temporarily: ruff cannot do it at any setting, so
  the eight defects found here would have shipped unnoticed.
- **Keep the default rule set.** Rejected on the 58 percent measurement.
- **Move pylint to a pre-push stage.** Rejected under Decision 8: a commit that
  has not passed the gates is no longer a safe point to reset to.

### The transferable point

"Two linters" and "two overlapping linters" are different decisions. The second
is waste. The first is only worth defending when the second tool's contribution
can be named exactly, which here is four checks and one proven class of defect.

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
- **Why this one.** A commit that has not passed the gates is not a safe point
  to reset to, and the whole value of committing freely is that every commit is
  such a point.
- **Revising an earlier condition in this document.** An earlier version set a
  ten second budget for the hook run, chosen before anything had been measured.
  The gate took 20.8s at that point, and takes 28.8s now, so by that rule the
  slow checks should have moved to pre-push twice over. The rule was wrong
  rather than the gate: it treated hook duration as
  the thing to minimise, when the thing to protect is the guarantee that any
  commit is green. At a slice-per-commit cadence that is a fair price, and the
  earlier figure was a guess dressed as a threshold.
- **The condition that replaces it.** Move the slow checks to pre-push when the
  gate starts changing behaviour, meaning commits get batched or skipped to
  avoid the wait. That is observable, unlike a number picked in advance.
- **Measured composition**, re-taken once the suite reached 365 tests: ruff
  0.6s, pylint 9.2s, pyright 3.6s, pytest 15.4s, for 28.8s total. pytest is now
  the largest share, having overtaken pylint, and most of it is the
  property-based tests generating hundreds of cases. That is the gate earning
  its cost rather than wasting it, so the replacement condition above still says
  leave it alone.
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
