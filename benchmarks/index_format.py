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

from search_engine.index import InvertedIndex
from search_engine.persistence import load, save

DOCUMENT_COUNT = 10_000
TOKENS_PER_DOCUMENT = 100
VOCABULARY_SIZE = 8_000
SEED = 20260903
REPEATS = 5

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


def measure(index: InvertedIndex, directory: Path) -> Measurement:
    """Time writing and reading the plain-text format."""
    path = directory / "v1.index"

    start = time.perf_counter()
    save(index, path)
    save_seconds = time.perf_counter() - start

    return Measurement(
        name="v1 plain text",
        file_bytes=path.stat().st_size,
        save_seconds=save_seconds,
        load_seconds=min(time_load(path) for _ in range(REPEATS)),
    )


def report(index: InvertedIndex, source_bytes: int, result: Measurement) -> None:
    """Print the observed figures for one format."""
    postings = sum(index.document_frequency(term) for term in index.terms)
    occurrences = sum(
        len(entry) for term in index.terms for entry in index.postings(term).values()
    )
    print(f"documents          : {index.document_count:,}")
    print(f"distinct terms     : {index.vocabulary_size:,}")
    print(f"postings entries   : {postings:,}")
    print(f"stored occurrences : {occurrences:,}")
    print(f"source text        : {source_bytes / 1_000_000:.1f} MB")
    print()
    print(f"format             : {result.name}")
    print(f"file size          : {result.file_bytes / 1_000_000:.2f} MB")
    print(f"file / source      : {result.file_bytes / source_bytes:.2f}x")
    print(f"bytes per posting  : {result.file_bytes / postings:.1f}")
    print(f"bytes per occurrence: {result.file_bytes / occurrences:.1f}")
    print()
    print(f"save               : {result.save_seconds:7.3f} s")
    print(f"load, best of {REPEATS}   : {result.load_seconds:7.3f} s")
    print(
        f"load throughput    : {result.file_bytes / result.load_seconds / 1e6:.1f} MB/s"
    )


def main() -> None:
    """Run the benchmark and print a report."""
    rng = random.Random(SEED)
    corpus = make_corpus(rng, make_vocabulary(VOCABULARY_SIZE))
    index = build(corpus)
    source_bytes = sum(len(text) for text in corpus)
    with TemporaryDirectory() as directory:
        report(index, source_bytes, measure(index, Path(directory)))


if __name__ == "__main__":
    main()
