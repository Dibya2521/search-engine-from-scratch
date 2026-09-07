"""Measure what an index costs on disk, and how long it takes to read back.

Run with ``uv run python benchmarks/index_format.py``.

Written before the format was changed, so that the comparison afterwards is
against a recorded number rather than a remembered one. Every format this
project ships is measured here with the same corpus and the same method, and
the corpus generator is shared with ``index_build.py`` so the two benchmarks
describe the same index.

Load time is reported as the best of several runs rather than the mean. The
interesting quantity is how long reading the file takes, and anything slower
than the fastest observed run was measuring something else competing for the
machine.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from search_engine.index import InvertedIndex
from search_engine.persistence import load, save, save_text

DOCUMENT_COUNT = 10_000
TOKENS_PER_DOCUMENT = 100
VOCABULARY_SIZE = 8_000
SEED = 20260903
REPEATS = 7

STEMS = (
    "connect",
    "compute",
    "retrieve",
    "document",
    "position",
    "invert",
    "rank",
    "score",
    "weight",
    "frequent",
    "index",
    "search",
    "query",
    "term",
    "vector",
    "normal",
    "similar",
    "relate",
    "general",
    "national",
)
SUFFIXES = ("", "s", "ed", "ing", "ion", "ions", "al", "ally", "ive", "ness")

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class Measurement:
    """What one index format cost for one corpus."""

    name: str
    file_bytes: int
    save_seconds: float
    load_seconds: float


def make_vocabulary(size: int) -> list[str]:
    """Build distinct inflected words, so stemming has real work to do."""
    words: list[str] = []
    for number in range(size):
        base = STEMS[number % len(STEMS)]
        suffix = SUFFIXES[(number // len(STEMS)) % len(SUFFIXES)]
        words.append(f"{base}{suffix}{number // (len(STEMS) * len(SUFFIXES))}")
    return words


def make_corpus(rng: random.Random, vocabulary: list[str]) -> list[str]:
    """Generate documents whose term frequencies follow a Zipf-like curve."""
    weights = [1.0 / (rank + 1) for rank in range(len(vocabulary))]
    return [
        " ".join(rng.choices(vocabulary, weights=weights, k=TOKENS_PER_DOCUMENT))
        for _ in range(DOCUMENT_COUNT)
    ]


def build(corpus: list[str]) -> InvertedIndex:
    """Index every document."""
    index = InvertedIndex()
    for document_id, text in enumerate(corpus):
        index.add_document(document_id, text)
    return index


def time_load(path: Path) -> float:
    """Return the seconds one load of an index file took."""
    start = time.perf_counter()
    load(path)
    return time.perf_counter() - start


def measure(
    name: str,
    writer: Callable[[InvertedIndex, Path], None],
    index: InvertedIndex,
    path: Path,
) -> Measurement:
    """Time writing and reading one format, best of several runs each.

    Both directions are timed the same way. An earlier version timed the save
    once and the load best-of-five, and the save figure then swung between
    plus four and plus sixty-four percent across runs of the same code: it was
    reporting how busy the machine was, not how fast the format is.
    """
    save_seconds = min(_time_save(writer, index, path) for _ in range(REPEATS))
    return Measurement(
        name=name,
        file_bytes=path.stat().st_size,
        save_seconds=save_seconds,
        load_seconds=min(time_load(path) for _ in range(REPEATS)),
    )


def _time_save(
    writer: Callable[[InvertedIndex, Path], None], index: InvertedIndex, path: Path
) -> float:
    start = time.perf_counter()
    writer(index, path)
    return time.perf_counter() - start


def describe(index: InvertedIndex, source_bytes: int) -> tuple[int, int]:
    """Print what the index holds, and return its posting and occurrence counts."""
    postings = sum(index.document_frequency(term) for term in index.terms)
    occurrences = sum(
        len(entry) for term in index.terms for entry in index.postings(term).values()
    )
    print(f"documents          : {index.document_count:,}")
    print(f"distinct terms     : {index.vocabulary_size:,}")
    print(f"postings entries   : {postings:,}")
    print(f"stored occurrences : {occurrences:,}")
    print(f"source text        : {source_bytes / 1_000_000:.1f} MB")
    return postings, occurrences


def compare(
    baseline: Measurement, current: Measurement, source_bytes: int, occurrences: int
) -> None:
    """Print both formats side by side, with the change between them."""
    header = f"{'':<22}{baseline.name:>14}{current.name:>14}{'change':>12}"
    print()
    print(header)
    print("-" * len(header))
    rows = (
        ("file size, MB", baseline.file_bytes / 1e6, current.file_bytes / 1e6),
        (
            "vs source text",
            baseline.file_bytes / source_bytes,
            current.file_bytes / source_bytes,
        ),
        (
            "bytes / occurrence",
            baseline.file_bytes / occurrences,
            current.file_bytes / occurrences,
        ),
        ("save, s", baseline.save_seconds, current.save_seconds),
        ("load, s", baseline.load_seconds, current.load_seconds),
    )
    for label, left, right in rows:
        print(f"{label:<22}{left:>14.3f}{right:>14.3f}{(right - left) / left:>11.1%}")


def main() -> None:
    """Run the benchmark and print a report."""
    rng = random.Random(SEED)
    corpus = make_corpus(rng, make_vocabulary(VOCABULARY_SIZE))
    index = build(corpus)
    source_bytes = sum(len(text) for text in corpus)
    _, occurrences = describe(index, source_bytes)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        baseline = measure("v1 text", save_text, index, root / "v1.index")
        current = measure("v2 binary", save, index, root / "v2.index")
    compare(baseline, current, source_bytes, occurrences)


if __name__ == "__main__":
    main()
