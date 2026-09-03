# ADR 0001: Toolchain and quality gates

- Status: accepted
- Date: 2026-09-03

## Context

This project has two goals that pull in different directions. It must teach me
information retrieval by making me build every mechanism myself, and it must
also read like a repository a large organisation would ship. The first goal
wants as little machinery as possible between me and the algorithm. The second
wants packaging, typing, linting, tests and CI.

The resolution is to put the strictness in the toolchain and the simplicity in
the runtime. Tooling is where rigour is cheap, since it costs configuration
once. Dependencies are where rigour is expensive, since every import is a
mechanism I did not build.

Every choice below records what it is, what it is for, how it is wired, what
else was available, and the fact that decided it.

## Decision 1: no runtime dependencies

`project.dependencies` in `pyproject.toml:31` is empty, and stays empty.

- **Purpose.** "From scratch" is the entire premise. An imported stemmer or an
  imported `Counter`-based index defeats the project.
- **Alternatives.** `nltk` for tokenizing and stemming, `numpy` for the vector
  arithmetic in Phase 3, `scikit-learn` for TF-IDF outright. All three would
  produce a working search engine faster, and all three would leave me unable
  to explain it.
- **Why this one.** Three payoffs beyond the learning: the code is auditable end
  to end, there is no dependency-resolution risk on new Python versions, and the
  Phase 5 container image stays small because the runtime layer is just Python.
- **Escape hatch.** A runtime dependency is allowed, but only with its own ADR
  stating what standard library approach was tried and where it fell short.
  `numpy` in Phase 3 is the likely candidate, and it must be justified with a
  measurement rather than an assumption about speed.

Development and test dependencies are deliberately unrestricted, because their
job is verification rather than implementation. `nltk` will be used in tests
only, as a differential oracle for the from-scratch Porter stemmer.

## Decision 2: uv for environments and locking

- **Purpose.** Reproducible environments and a committed lockfile.
- **How.** `uv sync --all-groups` locally; CI uses `uv sync --locked` so a stale
  `uv.lock` fails the build instead of drifting silently
  (`.github/workflows/ci.yml:37-39`).
- **Alternatives.** `pip` with `requirements.txt` (no real lock, no dependency
  groups), Poetry (slower, and its non-standard `[tool.poetry]` metadata predates
  PEP 621), PDM (standards-compliant but smaller ecosystem), plain venv plus pip
  (no locking at all).
- **Why this one.** uv is installed and verified at version 0.12.5, it reads
  standard PEP 621 metadata and PEP 735 dependency groups so nothing here is
  locked to the tool, and it resolves fast enough that CI can afford a full sync
  on three Python versions.

## Decision 3: `requires-python = ">=3.12"`, CI matrix 3.12 to 3.14

- **Purpose.** Define the supported range and prove it rather than claim it.
- **How.** `pyproject.toml:9`, and the matrix at
  `.github/workflows/ci.yml:22-23`.
- **Alternatives.** Pin to the local interpreter only, which is 3.14.5, and test
  one version. Or support 3.11, which is still widely deployed.
- **Why this one.** 3.12 is the floor because nothing in the design needs newer
  syntax, and it is the version most projects treat as their minimum in 2026.
  3.11 was dropped because supporting it buys nothing for a project nobody
  installs from an index, while adding a fourth job and a fourth chance of a
  version-specific type error. The upper end matters more than the lower end:
  3.14 is new, and testing it is how a version-specific break gets caught here
  rather than at the point where it blocks work.
- **Derived, not verified.** Whether every development dependency has working
  wheels on 3.14 has not been confirmed on CI yet, because no CI run has
  happened. The first run is the check. If a tool lags, the fix is to drop that
  one job with a comment, not to lower the floor.

## Decision 4: ruff as the primary linter and the only formatter

- **Purpose.** One fast tool covering formatting, import sorting and the bulk of
  linting.
- **How.** `[tool.ruff]` in `pyproject.toml`, with `select = ["ALL"]`.
- **Alternatives.** Black plus isort plus flake8 plus a stack of plugins, which
  is the arrangement ruff was built to replace, and which needs four configs and
  four processes.
- **Why `ALL` rather than a curated set.** On a learning project, seeing every
  rule the ecosystem has agreed on is itself educational, and it inverts the
  burden of proof: a rule is off only when there is a written reason, rather than
  on only when someone remembered it. The ignore list is therefore short and each
  entry is justified at the line:
  - `COM812` and `ISC001` are disabled because the formatter owns trailing
    commas and implicit string concatenation, and leaving them on makes the
    linter and formatter fight.
  - `CPY001` is disabled because a per-file copyright header adds noise that
    `LICENSE` already covers.
  - `D` rule conflicts are resolved by `convention = "google"` rather than by
    hand, so the mutually exclusive docstring rules are switched off by the tool
    that knows which pairs conflict.
- **Per-file relaxations.** Tests may use bare `assert` (`S101`) and bare numbers
  in assertions (`PLR2004`), since a named constant in a test usually hides the
  value being asserted. `scripts/` and `benchmarks/` may print and are not
  importable packages.

## Decision 5: pylint kept in addition to ruff

This is the one decision that looks redundant, so it carries the most reasoning.

- **Purpose.** Catch design-level problems that ruff does not implement.
- **What it uniquely adds.** `duplicate-code` (R0801), which is a whole-program
  similarity analysis ruff has no equivalent for, and the design limits
  configured at `pyproject.toml` under `[tool.pylint.design]`.
- **How the overlap is resolved.** Rather than accepting duplicate reports, every
  message pylint raises that ruff already owns is disabled by name under
  `[tool.pylint.messages_control]`, with the equivalent ruff code named in a
  comment on each line. So each class of problem is reported by exactly one tool,
  and the reason is auditable.
- **Alternatives.** Drop pylint entirely and accept the loss of duplication
  detection, which is a real loss on a project that will grow several similar
  postings-intersection routines. Or drop ruff and keep pylint alone, which is
  far slower and has no formatter.
- **Why both.** They are complementary once the overlap is cut: ruff is a fast
  per-file linter, pylint is a slower whole-program analyser. The cost is one
  extra process in the gate. The benefit is the class of finding that only shows
  up across files.
- **Notable side benefit.** `max-nested-blocks = 3` encodes the two-levels-of-
  nesting guideline from my coding rules directly into tooling, which no ruff
  rule does. Without pylint that guideline would rest on review discipline alone.

## Decision 6: pyright in strict mode

- **Purpose.** An unchecked type is an unverified assumption, and the postings
  structures here are nested enough that a wrong shape is easy to write and hard
  to see.
- **How.** `[tool.pyright]` with `typeCheckingMode = "strict"` over `src`,
  `tests`, `scripts` and `benchmarks`.
- **Alternatives.** mypy, which is the older standard and has broader plugin
  support. No type checking at all.
- **Why pyright.** Strict mode is a single switch rather than a dozen
  `disallow_*` flags, it is the checker behind the editor integration already in
  use so the terminal and the editor agree, and it is faster on a small codebase.
  mypy would be an equally defensible choice; this is not a close-run
  correctness question, so it was settled on ergonomics.
- **Consequence accepted.** Tests are type-checked too. That is deliberate,
  since a test with a wrong type is a test that may not be asserting what it
  appears to.

## Decision 7: pytest, with branch coverage floored at 90 percent

- **How.** `[tool.pytest.ini_options]` and `[tool.coverage.*]` in
  `pyproject.toml`. `--strict-markers` and `--strict-config` turn typos in
  markers and config into errors. `xfail_strict = true` means a test marked as
  expected-to-fail that passes is itself a failure. `filterwarnings = ["error"]`
  treats a warning as a defect that has not been diagnosed yet.
- **Alternatives.** `unittest` from the standard library, which would be
  consistent with the no-runtime-dependency rule but has no fixtures, no
  parametrisation and no plugin for coverage. A lower coverage floor, or none.
- **Why this one.** The no-dependency rule applies to the runtime, not to the
  test harness, and pytest's parametrisation is what makes the stemmer and
  intersection test suites tractable.
- **What the floor is and is not.** 90 percent branch coverage is a floor that
  stops coverage silently rotting. It is explicitly not the definition of tested.
  A slice is done when it has a test built from inputs the design did not
  anticipate, and coverage cannot measure that.

## Decision 8: pre-commit runs all five gates, not a fast subset

- **Purpose.** Make a red gate impossible to commit through by accident.
- **How.** `.pre-commit-config.yaml`. Third-party hooks for file hygiene and
  ruff; local `system` hooks calling `uv run` for pylint, pyright and pytest so
  the versions match `uv.lock` and therefore match CI exactly.
- **Alternatives.** The common arrangement is fast hooks on commit and slow ones
  on push or in CI only. That arrangement is better when the test suite is slow.
- **Why this one, for now.** The suite is currently trivial, so the honest choice
  is to run everything and measure, rather than to pre-optimise for a slowness
  that does not exist. The decision is explicitly conditional: if the hook run
  exceeds roughly ten seconds, pytest and pylint move to a pre-push stage and
  this ADR gets a superseding note. The trigger is a measurement recorded in
  `MEASUREMENTS`, not a feeling.
- **Note.** `check-added-large-files` is the load-bearing hook here. The corpus
  is hundreds of megabytes, and data that reaches git history cannot be removed
  cleanly afterwards.

## Consequences

- Five gate commands rather than one. Accepted, since CI runs the identical list
  and there is no divergence to reconcile.
- `select = ["ALL"]` will occasionally surface a rule that needs a judgement
  call. That is the intended cost, and each resolution becomes a documented
  decision.
- No badges are in the README yet, because there is no remote and a badge
  pointing at a repository that does not exist is a broken claim. They go in when
  the remote does.
