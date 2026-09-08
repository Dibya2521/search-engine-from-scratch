"""Measure what reading an index in place costs against loading it whole.

Run with ``uv run python benchmarks/on_disk.py``.

The claim an on-disk segment has to earn is that opening it is cheap and that it
holds almost nothing in memory. The claim it has to survive is that a term
lookup followed by a decode is slower than reading a value out of a dict that is
already in memory. Both are measured here, on the same corpus the index format
benchmark uses, so the numbers line up with what is already published for that
format.

A cold read uses a reader that has never seen the terms, because a reader caches
what it decodes and a second read of the same term would measure the cache
rather than the format. The warm column is that cache, reported separately
rather than allowed to flatter the cold one.

Timing and memory are measured in separate passes, because ``tracemalloc`` costs
time and a benchmark in this project once inverted its own conclusion by having
it on for one side and off for the other.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from corpus import fixed_length_corpus, make_vocabulary

from search_engine.index import InvertedIndex
from search_engine.persistence import load, save
from search_engine.segment import SegmentReader, write_segment

if TYPE_CHECKING:
    from collections.abc import Callable

DOCUMENT_COUNT = 10_000
TOKENS_PER_DOCUMENT = 100
VOCABULARY_SIZE = 8_000
QUERY_COUNT = 300
REPEATS = 5
BYTES_PER_MB = 1_000_000


@dataclass(frozen=True, slots=True)
class Result:
    """What one way of reading an index cost."""

    name: str
    open_seconds: float
    open_bytes: int
    cold_seconds: float
    warm_seconds: float
    read_bytes: int


def build(corpus: list[str]) -> InvertedIndex:
    """Index every document."""
    index = InvertedIndex()
    for document_id, text in enumerate(corpus):
        index.add_document(document_id, text)
    return index


def query_terms(index: InvertedIndex) -> list[str]:
    """Return terms spanning the common, the middling and the rare.

    Only common terms would measure the longest postings lists and only rare
    ones the shortest, and the two behave differently enough that either alone
    would mislead.
    """
    ordered = sorted(index.terms, key=index.document_frequency, reverse=True)
    third = QUERY_COUNT // 3
    middle = len(ordered) // 2
    return ordered[:third] + ordered[middle : middle + third] + ordered[-third:]


def measure_whole_file(path: Path, terms: list[str]) -> Result:
    """Load the entire index, then answer every term from memory.

    Cold and warm are the same thing here, because the whole index is decoded
    before the first term is asked for. Both are reported so a column means the
    same thing on every row.
    """
    open_seconds = min(_time(lambda: load(path)) for _ in range(REPEATS))
    index = load(path)
    cold = min(_time(lambda: _read(index, terms)) for _ in range(REPEATS))
    warm = min(_time(lambda: _read(index, terms)) for _ in range(REPEATS))

    tracemalloc.start()
    opened = load(path)
    _, open_bytes = tracemalloc.get_traced_memory()
    _read(opened, terms)
    _, read_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return Result("whole file", open_seconds, open_bytes, cold, warm, read_bytes)


def measure_segment(path: Path, terms: list[str], *, verify: bool) -> Result:
    """Map the segment and decode only the postings each term needs."""
    open_seconds = min(_time_open(path, verify=verify) for _ in range(REPEATS))
    cold = min(_time_cold(path, terms, verify=verify) for _ in range(REPEATS))
    with SegmentReader(path, verify=verify) as reader:
        _read(reader, terms)
        warm = min(_time(lambda: _read(reader, terms)) for _ in range(REPEATS))

    tracemalloc.start()
    with SegmentReader(path, verify=verify) as opened:
        _, open_bytes = tracemalloc.get_traced_memory()
        _read(opened, terms)
        _, read_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    name = "segment, verified" if verify else "segment, unverified"
    return Result(name, open_seconds, open_bytes, cold, warm, read_bytes)


def _time_open(path: Path, *, verify: bool) -> float:
    started = time.perf_counter()
    with SegmentReader(path, verify=verify):
        pass
    return time.perf_counter() - started


def _time_cold(path: Path, terms: list[str], *, verify: bool) -> float:
    """Time reading terms through a reader that has cached none of them."""
    with SegmentReader(path, verify=verify) as reader:
        started = time.perf_counter()
        _read(reader, terms)
        return time.perf_counter() - started


def _read(source: InvertedIndex | SegmentReader, terms: list[str]) -> int:
    return sum(len(source.postings(term)) for term in terms)


def _time(action: Callable[[], object]) -> float:
    started = time.perf_counter()
    action()
    return time.perf_counter() - started


def report(results: list[Result], file_bytes: dict[str, int]) -> None:
    """Print the comparison as a markdown table."""
    print()
    print(
        f"| Reader | File MB | Open s | Memory after open MB "
        f"| Cold read of {QUERY_COUNT} terms s | Warm read s "
        f"| Memory after reads MB |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            f"| {result.name} "
            f"| {file_bytes[result.name] / BYTES_PER_MB:.2f} "
            f"| {result.open_seconds:.4f} "
            f"| {result.open_bytes / BYTES_PER_MB:.1f} "
            f"| {result.cold_seconds:.4f} "
            f"| {result.warm_seconds:.4f} "
            f"| {result.read_bytes / BYTES_PER_MB:.1f} |"
        )


def main() -> None:
    """Run the benchmark and print the report."""
    corpus = fixed_length_corpus(
        DOCUMENT_COUNT, TOKENS_PER_DOCUMENT, make_vocabulary(VOCABULARY_SIZE)
    )
    index = build(corpus)
    terms = query_terms(index)
    print(f"documents      : {index.document_count:,}")
    print(f"distinct terms : {index.vocabulary_size:,}")

    with TemporaryDirectory() as directory:
        root = Path(directory)
        whole = root / "whole.index"
        segment = root / "one.seg"
        save(index, whole)
        write_segment(index, segment)
        sizes = {
            "whole file": whole.stat().st_size,
            "segment, verified": segment.stat().st_size,
            "segment, unverified": segment.stat().st_size,
        }
        results = [
            measure_whole_file(whole, terms),
            measure_segment(segment, terms, verify=True),
            measure_segment(segment, terms, verify=False),
        ]
        report(results, sizes)


if __name__ == "__main__":
    main()
