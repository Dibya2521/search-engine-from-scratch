"""Measure how the engine grows, rather than projecting it from one point.

Run with ``uv run python benchmarks/at_scale.py``. It takes several minutes.

Every scale claim this project has made so far was extrapolated from a single
10,000 document run, and an extrapolation is a belief rather than a
measurement. This builds indexes at six sizes and reports how build time,
memory, vocabulary and file size actually move with the corpus.

Two properties are reported that a single point cannot show. The first is
whether memory per source byte is constant: if it is, the earlier projections
hold, and if it rises they were optimistic. The second is Heaps' law, the
observation that vocabulary grows as ``K * N ** beta`` in the token count, with
published English values of ``K`` between 10 and 100 and ``beta`` between 0.4
and 0.6. Whether the synthetic generator reproduces that decides whether any
vocabulary-dependent projection made from it is worth anything.

Each size is built with ``tracemalloc`` off to time it and again with it on to
measure memory, because tracing costs time and mixing the two inverted the
conclusion of an earlier benchmark in this project.

Build time is the best of three runs rather than a single one. A first version
timed it once, and its two smallest sizes were measured while another process
happened to be busy: they reported 90,000 and 88,000 tokens per second against
109,000 to 116,000 for every larger size, which reads as a startup cost that is
not there. The fastest observed run is the one where the machine was doing
least of anything else.

The range stops at 20,000 documents. 50,000 was measured first and dropped: at
the throughput and footprint recorded below it needs about 2.8 GB of resident
memory and around 29 minutes for the two passes, which is a benchmark nobody
would run twice. Where the trend is clear over a 40x span, extending it to 100x
buys confirmation rather than information.
"""

from __future__ import annotations

import math
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from corpus import make_vocabulary, varied_length_corpus

from search_engine.index import InvertedIndex
from search_engine.persistence import load, save

if TYPE_CHECKING:
    from collections.abc import Sequence

SIZES = (500, 1_000, 2_000, 5_000, 10_000, 20_000)

# Large enough that the Zipf draw keeps finding unseen terms as the corpus
# grows. A vocabulary the corpus exhausts would flatten Heaps' law by
# construction and measure the generator's ceiling instead of the engine.
VOCABULARY_SIZE = 100_000

BUILD_REPEATS = 3
LOAD_REPEATS = 3
BYTES_PER_MB = 1_000_000


@dataclass(frozen=True, slots=True)
class Measurement:
    """What one corpus size cost to index, store and read back."""

    documents: int
    source_bytes: int
    tokens: int
    vocabulary: int
    build_seconds: float
    peak_bytes: int
    index_bytes: int
    load_seconds: float


def build(documents: Sequence[str]) -> InvertedIndex:
    """Index every document."""
    index = InvertedIndex()
    for document_id, text in enumerate(documents):
        index.add_document(document_id, text)
    return index


def count_tokens(index: InvertedIndex) -> int:
    """Return the number of stored occurrences, which is the analysed token count."""
    return sum(
        len(positions)
        for term in index.terms
        for positions in index.postings(term).values()
    )


def peak_memory(documents: Sequence[str]) -> int:
    """Return the peak bytes allocated while building an index of these documents.

    The documents already exist when tracing starts, so the figure covers the
    index and the transient cost of analysis and nothing else.
    """
    tracemalloc.start()
    build(documents)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def measure(documents: Sequence[str], path: Path) -> Measurement:
    """Build, store and read back one corpus, and report what each step cost.

    The discarded builds come first and the kept one last, so only one index is
    alive at a time. Holding all three would triple the footprint of the very
    thing being measured.
    """
    discarded = min(_time_build(documents) for _ in range(BUILD_REPEATS - 1))
    started = time.perf_counter()
    index = build(documents)
    build_seconds = min(discarded, time.perf_counter() - started)

    save(index, path)
    return Measurement(
        documents=len(documents),
        source_bytes=sum(len(text) for text in documents),
        tokens=count_tokens(index),
        vocabulary=index.vocabulary_size,
        build_seconds=build_seconds,
        peak_bytes=peak_memory(documents),
        index_bytes=path.stat().st_size,
        load_seconds=min(_time_load(path) for _ in range(LOAD_REPEATS)),
    )


def _time_build(documents: Sequence[str]) -> float:
    started = time.perf_counter()
    build(documents)
    return time.perf_counter() - started


def _time_load(path: Path) -> float:
    started = time.perf_counter()
    load(path)
    return time.perf_counter() - started


def heaps_law(measurements: Sequence[Measurement]) -> tuple[float, float]:
    """Fit vocabulary against token count as K * N ** beta, and return K and beta.

    Least squares on the logarithms, which is the standard fit for a power law
    and needs no dependency beyond ``math``.
    """
    xs = [math.log(item.tokens) for item in measurements]
    ys = [math.log(item.vocabulary) for item in measurements]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    variance = sum((x - mean_x) ** 2 for x in xs)
    beta = covariance / variance
    return math.exp(mean_y - beta * mean_x), beta


def print_costs(measurements: Sequence[Measurement]) -> None:
    """Print what each corpus size cost, as a markdown table."""
    print()
    print(
        "| Documents | Source MB | Tokens | Vocabulary | Build s | Tokens/s "
        "| Peak MB | Index MB | Load s |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in measurements:
        print(
            f"| {item.documents:,} "
            f"| {item.source_bytes / BYTES_PER_MB:.1f} "
            f"| {item.tokens:,} "
            f"| {item.vocabulary:,} "
            f"| {item.build_seconds:.2f} "
            f"| {item.tokens / item.build_seconds:,.0f} "
            f"| {item.peak_bytes / BYTES_PER_MB:.0f} "
            f"| {item.index_bytes / BYTES_PER_MB:.2f} "
            f"| {item.load_seconds:.3f} |"
        )


def print_ratios(measurements: Sequence[Measurement]) -> None:
    """Print the per-unit costs, which are what reveal whether growth is linear."""
    print()
    print(
        "| Documents | Peak memory / source | Index bytes / occurrence "
        "| Build s / source MB | Load s / index MB |"
    )
    print("| ---: | ---: | ---: | ---: | ---: |")
    for item in measurements:
        source_mb = item.source_bytes / BYTES_PER_MB
        print(
            f"| {item.documents:,} "
            f"| {item.peak_bytes / item.source_bytes:.1f}x "
            f"| {item.index_bytes / item.tokens:.2f} "
            f"| {item.build_seconds / source_mb:.1f} "
            f"| {item.load_seconds / (item.index_bytes / BYTES_PER_MB):.3f} |"
        )


def print_heaps(measurements: Sequence[Measurement]) -> None:
    """Print the fitted vocabulary growth against the published English range."""
    constant, exponent = heaps_law(measurements)
    print()
    print(f"Heaps' law fit     : V = {constant:.1f} * N ** {exponent:.3f}")
    print("Published, English : K between 10 and 100, beta between 0.4 and 0.6")
    print()
    print("| Documents | Vocabulary | Heaps predicts | Error |")
    print("| ---: | ---: | ---: | ---: |")
    for item in measurements:
        predicted = constant * item.tokens**exponent
        error = (predicted - item.vocabulary) / item.vocabulary
        print(
            f"| {item.documents:,} | {item.vocabulary:,} "
            f"| {predicted:,.0f} | {error:+.1%} |"
        )


def main() -> None:
    """Run the benchmark and print the report."""
    vocabulary = make_vocabulary(VOCABULARY_SIZE)
    measurements: list[Measurement] = []
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for size in SIZES:
            documents = list(varied_length_corpus(size, vocabulary))
            measurements.append(measure(documents, root / f"{size}.index"))
            print(f"measured {size:,} documents", flush=True)
    print_costs(measurements)
    print_ratios(measurements)
    print_heaps(measurements)


if __name__ == "__main__":
    main()
