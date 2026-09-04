"""Measure where index build time and memory actually go.

Run with ``uv run python benchmarks/index_build.py``.

Document text is synthetic, drawn from a fixed vocabulary with a Zipf-shaped
frequency so postings lists have a realistic length spread. Only the timings
and memory figures are claimed; anything about a real corpus needs one.

Timing and memory are measured in separate passes. ``tracemalloc`` instruments
every allocation, so leaving it on during a timed pass inflates that pass and
makes it incomparable with one measured without it.
"""

from __future__ import annotations

import random
import time
import tracemalloc
from dataclasses import dataclass

from search_engine.analysis import analyze
from search_engine.index import InvertedIndex

DOCUMENT_COUNT = 10_000
TOKENS_PER_DOCUMENT = 100
VOCABULARY_SIZE = 8_000
SEED = 20260903

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
    """Everything one benchmark run observed."""

    characters: int
    distinct_terms: int
    postings: int
    analysis_seconds: float
    build_seconds: float
    peak_bytes: int


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


def measure(corpus: list[str]) -> Measurement:
    """Time analysis and building separately, then measure memory on its own."""
    start = time.perf_counter()
    for text in corpus:
        analyze(text)
    analysis_seconds = time.perf_counter() - start

    start = time.perf_counter()
    index = build(corpus)
    build_seconds = time.perf_counter() - start

    tracemalloc.start()
    measured = build(corpus)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del measured

    return Measurement(
        characters=sum(len(text) for text in corpus),
        distinct_terms=index.vocabulary_size,
        postings=sum(index.document_frequency(term) for term in index.terms),
        analysis_seconds=analysis_seconds,
        build_seconds=build_seconds,
        peak_bytes=peak_bytes,
    )


def report(result: Measurement) -> None:
    """Print the observed figures."""
    tokens = DOCUMENT_COUNT * TOKENS_PER_DOCUMENT
    insertion = result.build_seconds - result.analysis_seconds
    share = result.analysis_seconds / result.build_seconds
    megabytes_per_second = result.characters / result.build_seconds / 1_000_000
    print(f"documents        : {DOCUMENT_COUNT:,}")
    print(f"tokens           : {tokens:,}")
    print(f"source text      : {result.characters / 1_000_000:.1f} MB")
    print(f"distinct terms   : {result.distinct_terms:,}")
    print(f"postings entries : {result.postings:,}")
    print()
    print(f"analysis         : {result.analysis_seconds:7.2f} s")
    print(f"insertion        : {insertion:7.2f} s")
    print(f"total build      : {result.build_seconds:7.2f} s")
    print(f"analysis share   : {share:7.1%}")
    print()
    print(f"throughput       : {DOCUMENT_COUNT / result.build_seconds:,.0f} docs/s")
    print(f"                   {tokens / result.build_seconds:,.0f} tokens/s")
    print(f"                   {megabytes_per_second:.1f} MB/s")
    print()
    print(f"peak memory      : {result.peak_bytes / 1_000_000:.1f} MB")
    print(f"bytes per posting: {result.peak_bytes / result.postings:.0f}")
    print(f"memory / source  : {result.peak_bytes / result.characters:.2f}x")


def main() -> None:
    """Run the benchmark and print a report."""
    rng = random.Random(SEED)
    corpus = make_corpus(rng, make_vocabulary(VOCABULARY_SIZE))
    report(measure(corpus))


if __name__ == "__main__":
    main()
