"""Measure where index build time and memory actually go.

Run with ``uv run python benchmarks/index_build.py``.

The document text is synthetic, drawn from a fixed vocabulary with a
Zipf-shaped frequency so the postings lists have realistic length spread. Only
the timings and memory figures are claimed here; anything about a real corpus
needs a real corpus.
"""

from __future__ import annotations

import random
import time
import tracemalloc

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


def make_vocabulary(size: int) -> list[str]:
    """Build distinct inflected words, so stemming has real work to do."""
    suffixes = ("", "s", "ed", "ing", "ion", "ions", "al", "ally", "ive", "ness")
    words: list[str] = []
    for number in range(size):
        base = STEMS[number % len(STEMS)]
        suffix = suffixes[(number // len(STEMS)) % len(suffixes)]
        words.append(f"{base}{suffix}{number // (len(STEMS) * len(suffixes))}")
    return words


def make_corpus(rng: random.Random, vocabulary: list[str]) -> list[str]:
    """Generate documents whose term frequencies follow a Zipf-like curve."""
    weights = [1.0 / (rank + 1) for rank in range(len(vocabulary))]
    return [
        " ".join(rng.choices(vocabulary, weights=weights, k=TOKENS_PER_DOCUMENT))
        for _ in range(DOCUMENT_COUNT)
    ]


def main() -> None:
    """Run the benchmark and print a report."""
    rng = random.Random(SEED)
    vocabulary = make_vocabulary(VOCABULARY_SIZE)
    corpus = make_corpus(rng, vocabulary)
    total_tokens = DOCUMENT_COUNT * TOKENS_PER_DOCUMENT
    characters = sum(len(text) for text in corpus)

    # Analysis alone, to separate text processing from index insertion.
    start = time.perf_counter()
    for text in corpus:
        analyze(text)
    analysis_seconds = time.perf_counter() - start

    # Timed without tracemalloc: it instruments every allocation and would
    # inflate this pass but not the analysis pass above, making them
    # incomparable.
    start = time.perf_counter()
    index = InvertedIndex()
    for document_id, text in enumerate(corpus):
        index.add_document(document_id, text)
    build_seconds = time.perf_counter() - start

    # Memory in a separate pass, where the instrumentation cost does not matter.
    tracemalloc.start()
    measured = InvertedIndex()
    for document_id, text in enumerate(corpus):
        measured.add_document(document_id, text)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del measured

    postings = sum(index.document_frequency(term) for term in index.terms)
    insertion_seconds = build_seconds - analysis_seconds

    print(f"documents        : {DOCUMENT_COUNT:,}")
    print(f"tokens           : {total_tokens:,}")
    print(f"source text      : {characters / 1_000_000:.1f} MB")
    print(f"distinct terms   : {index.vocabulary_size:,}")
    print(f"postings entries : {postings:,}")
    print()
    print(f"analysis         : {analysis_seconds:7.2f} s")
    print(f"insertion        : {insertion_seconds:7.2f} s")
    print(f"total build      : {build_seconds:7.2f} s")
    print(f"analysis share   : {analysis_seconds / build_seconds:7.1%}")
    print()
    print(f"throughput       : {DOCUMENT_COUNT / build_seconds:,.0f} documents/s")
    print(f"                   {total_tokens / build_seconds:,.0f} tokens/s")
    print(f"                   {characters / build_seconds / 1_000_000:.1f} MB/s")
    print()
    print(f"peak memory      : {peak_bytes / 1_000_000:.1f} MB")
    print(f"bytes per posting: {peak_bytes / postings:.0f}")
    print(f"memory / source  : {peak_bytes / characters:.2f}x")


if __name__ == "__main__":
    main()
