"""Measure what a query costs as a directory accumulates segments.

Run with ``uv run python benchmarks/segments.py``.

Merging exists on the argument that a query over many segments is slower than
the same query over one, because every segment is opened and every term is
looked up in each of them. That argument is the justification for a background
process that rewrites data, so it should be a number rather than a claim.

The same documents are written at several buffer sizes, which is the only thing
that decides how many segments they land in. Nothing else differs, so the
comparison is controlled.
"""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory

from corpus import make_vocabulary, varied_length_corpus

from search_engine.directory import DirectoryIndex
from search_engine.writer import IndexWriter

DOCUMENT_COUNT = 4_000
VOCABULARY_SIZE = 4_000
QUERY_COUNT = 200
REPEATS = 5
SPLITS = (1, 2, 4, 8, 16)


def write(directory: Path, texts: list[str], segments: int) -> None:
    """Write the documents so they land in exactly this many segments."""
    per_segment = -(-len(texts) // segments)
    with IndexWriter(directory, buffer_documents=per_segment, merge=False) as writer:
        for document_id, text in enumerate(texts):
            writer.add(document_id, text)


def query_terms(directory: Path) -> list[str]:
    """Return terms spanning the common, the middling and the rare."""
    with DirectoryIndex(directory, verify=False) as index:
        ordered = sorted(index.terms, key=index.document_frequency, reverse=True)
    third = QUERY_COUNT // 3
    middle = len(ordered) // 2
    return ordered[:third] + ordered[middle : middle + third] + ordered[-third:]


def measure(directory: Path, terms: list[str]) -> tuple[float, float, int]:
    """Return seconds to open, seconds to read every term, and the segment count."""
    opening = min(_time_open(directory) for _ in range(REPEATS))
    with DirectoryIndex(directory, verify=False) as index:
        reading = min(
            _time(lambda: sum(len(index.postings(term)) for term in terms))
            for _ in range(REPEATS)
        )
        count = index.segment_count
    return opening, reading, count


def _time_open(directory: Path) -> float:
    started = time.perf_counter()
    with DirectoryIndex(directory, verify=False):
        pass
    return time.perf_counter() - started


def _time(action: object) -> float:
    started = time.perf_counter()
    if callable(action):
        action()
    return time.perf_counter() - started


def main() -> None:
    """Run the benchmark and print the report."""
    texts = list(varied_length_corpus(DOCUMENT_COUNT, make_vocabulary(VOCABULARY_SIZE)))
    print(f"documents  : {len(texts):,}")
    print(f"terms read : {QUERY_COUNT}")
    print()
    print(
        f"| Segments | Open s | Read {QUERY_COUNT} terms s "
        f"| Per term ms | Against one segment |"
    )
    print("| ---: | ---: | ---: | ---: | ---: |")

    baseline: float | None = None
    for split in SPLITS:
        with TemporaryDirectory() as name:
            directory = Path(name)
            write(directory, texts, split)
            terms = query_terms(directory)
            opening, reading, count = measure(directory, terms)
        if baseline is None:
            baseline = reading
        print(
            f"| {count} | {opening:.4f} | {reading:.4f} "
            f"| {reading / QUERY_COUNT * 1000:.3f} | {reading / baseline:.2f}x |"
        )


if __name__ == "__main__":
    main()
