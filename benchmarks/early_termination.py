"""Measure what early termination saves, and where it costs instead.

Run with ``uv run python benchmarks/early_termination.py``.

Early termination is justified by making queries cheaper. This measures by how
much, and on which query shapes it does not.

Three strategies run over the same segment, with the same scorer and the same
queries:

- a full scan, which is what ranking did before either commit;
- early termination with one upper bound per term;
- early termination with a bound per block of postings.

The second and the third differ only in where the bound comes from. `rank_wand`
gives a single segment per-block bounds and gives any other index one bound per
term, so `TermBounds` below is that other index wrapped around the same reader.
The data, the scorer and the postings are identical on both sides.

**Documents scored is exact.** It is a counter, not a timing, so it does not
move with machine load and it is the figure to compare. The timings are best of
seven and are not exact: this project has already had three benchmarks whose
timings swung far enough to invert a conclusion.

The full scan is timed generously. Gathering its candidates costs a decode of
every query term's postings, and that decode happens outside the timer, so the
figure reported for it is lower than the work a real query would do.
"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from corpus import make_vocabulary, varied_length_corpus

from search_engine.bm25 import BM25Ranker
from search_engine.index import InvertedIndex
from search_engine.segment import SegmentReader, write_segment
from search_engine.wand import search_wand

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

DOCUMENT_COUNT = 50_000
VOCABULARY_SIZE = 4_000
REPEATS = 7
LIMIT = 10
OUTLIER_FREQUENCY = 500

# Scoring without the postings cache is quadratic, so the cache comparison runs
# on corpora small enough to finish. 50,000 documents would take days.
CACHE_SIZES = (500, 1_000, 2_000, 4_000)
CACHE_VOCABULARY = 1_000


class TermBounds:
    """The same segment, read through the path that has no per-block bounds.

    Delegates every question to the reader. It exists only so that
    `search_wand` takes its mapping path, which is what the engine did before
    per-block bounds, and so the two bound granularities can be compared on one
    set of files.
    """

    def __init__(self, reader: SegmentReader) -> None:
        self._reader = reader

    def postings(self, term: str) -> Mapping[int, Sequence[int]]:
        """Return the documents holding a term, and where in each it occurs."""
        return self._reader.postings(term)

    def document_frequency(self, term: str) -> int:
        """Return how many documents hold a term."""
        return self._reader.document_frequency(term)

    def document_length(self, document_id: int) -> int:
        """Return a document's length in analysed tokens."""
        return self._reader.document_length(document_id)

    def __contains__(self, term: str) -> bool:
        """Return whether the segment holds a term."""
        return term in self._reader

    @property
    def document_count(self) -> int:
        """Return how many documents the segment holds."""
        return self._reader.document_count

    @property
    def vocabulary_size(self) -> int:
        """Return how many distinct terms the segment holds."""
        return self._reader.vocabulary_size

    @property
    def terms(self) -> Iterable[str]:
        """Return every term the segment holds."""
        return self._reader.terms

    @property
    def document_ids(self) -> Iterable[int]:
        """Return every document identifier the segment holds."""
        return self._reader.document_ids


def build() -> InvertedIndex:
    """Index a corpus whose document lengths span two orders of magnitude."""
    index = InvertedIndex()
    vocabulary = make_vocabulary(VOCABULARY_SIZE)
    for document_id, text in enumerate(
        varied_length_corpus(DOCUMENT_COUNT, vocabulary)
    ):
        index.add_document(document_id, text)
    return index


def outlier_index() -> InvertedIndex:
    """One term everywhere, once per document, except one document holding it 500 times.

    The generator above draws each document's length independently, so a
    document with a high term frequency is as likely to land in one block as
    any other and every block ends up with much the same maximum. Per-block
    bounds need the opposite: high frequencies concentrated in few blocks.
    Postings are ordered by document identifier, so that happens when identifier
    order correlates with term frequency, which is what this builds.
    """
    spike = DOCUMENT_COUNT // 2
    postings = {
        "alpha": {
            document_id: list(range(OUTLIER_FREQUENCY if document_id == spike else 1))
            for document_id in range(DOCUMENT_COUNT)
        }
    }
    return InvertedIndex.from_postings(list(range(DOCUMENT_COUNT)), postings)


def query_mixes(reader: SegmentReader) -> list[tuple[str, list[str]]]:
    """Return query shapes that behave differently, named for what they are.

    Terms are chosen by their measured document frequency rather than written
    down, so the mixes describe this corpus rather than an assumption about it.
    """
    ordered = sorted(reader.terms, key=reader.document_frequency, reverse=True)
    common, second = ordered[0], ordered[1]
    middle = ordered[len(ordered) // 2]
    rare = ordered[-1]
    return [
        ("the most common term", [common]),
        ("two common terms", [common, second]),
        ("one common, one rare", [common, rare]),
        ("three of mixed frequency", [common, middle, rare]),
        ("two rare terms", [ordered[-2], rare]),
    ]


def measure(
    reader: SegmentReader, ranker: BM25Ranker, terms: list[str], limit: int
) -> tuple[float, float, float, int, int, int]:
    """Return the three timings and the three documents-scored counts."""
    candidates = {
        document_id for term in terms for document_id in reader.postings(term)
    }
    query = " ".join(terms)
    bounded = TermBounds(reader)
    scan = _time(lambda: ranker.rank(query, candidates, limit))
    per_term = search_wand(bounded, ranker, terms, limit)
    per_block = search_wand(reader, ranker, terms, limit)
    if per_term.documents != per_block.documents:
        message = "the two bound granularities disagreed, which is a bug"
        raise AssertionError(message)
    return (
        scan,
        _time(lambda: search_wand(bounded, ranker, terms, limit)),
        _time(lambda: search_wand(reader, ranker, terms, limit)),
        len(candidates),
        per_term.scored,
        per_block.scored,
    )


def report(
    reader: SegmentReader,
    ranker: BM25Ranker,
    limit: int,
    mixes: list[tuple[str, list[str]]],
) -> None:
    """Print one table of every query mix at one result limit."""
    say()
    say(f"### Limit {limit}")
    say()
    say(
        "| Query | Terms | Candidates | Scan s | Term bounds s | Block bounds s "
        "| Scored, term bounds | Scored, block bounds | Block bounds, share |"
    )
    say("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, terms in mixes:
        scan, by_term, by_block, candidates, term_scored, block_scored = measure(
            reader, ranker, terms, limit
        )
        share = block_scored / candidates if candidates else 0.0
        say(
            f"| {name} | {len(terms)} | {candidates:,} | {scan:.4f} | {by_term:.4f} "
            f"| {by_block:.4f} | {term_scored:,} | {block_scored:,} | {share:.1%} |"
        )


def say(line: str = "") -> None:
    """Print a line and flush it.

    This benchmark builds a 50,000 document corpus and runs for minutes. Python
    block-buffers stdout when it is not a terminal, so without the flush a run
    redirected to a file shows nothing at all until it exits, and there is no
    way to tell progress from a hang.
    """
    print(line, flush=True)


def _time(work: Callable[[], object]) -> float:
    """Return the best of REPEATS runs, which is the least contaminated one."""
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        work()
        best = min(best, time.perf_counter() - start)
    return best


def _time_once(work: Callable[[], object]) -> float:
    """Return one run, for a side too slow to run seven times."""
    start = time.perf_counter()
    work()
    return time.perf_counter() - start


def _search_common(reader: SegmentReader, ranker: BM25Ranker, term: str) -> None:
    """Run the query the cache comparison repeats."""
    search_wand(reader, ranker, [term], LIMIT)


def measure_caching(directory: Path) -> None:
    """Time the same query with the postings cache on and off, as size grows.

    **The postings cache is not an optimisation, it is load-bearing.**
    `BM25Ranker.score` asks the index for a term's postings once per document
    scored, and a reader with no cache decodes the whole list on every one of
    those calls. Scoring is therefore linear in the candidates with the cache
    and quadratic without it, which the growth of the ratio below shows.

    That is why this runs on corpora far smaller than the tables above. At
    50,000 candidates the uncached side needs billions of posting decodes and
    does not finish in any useful time, and that failure is the result rather
    than an obstacle to it.

    Timings here are single runs, not best of seven, because the difference is
    orders of magnitude rather than percent and seven runs of the uncached side
    would take many minutes.
    """
    say()
    say("### The postings cache, on and off, as the corpus grows")
    say()
    say("| Documents | Candidates | Cache on s | Cache off s | Off / on | Hit rate |")
    say("| ---: | ---: | ---: | ---: | ---: | ---: |")
    vocabulary = make_vocabulary(CACHE_VOCABULARY)
    for count in CACHE_SIZES:
        index = InvertedIndex()
        for document_id, document in enumerate(varied_length_corpus(count, vocabulary)):
            index.add_document(document_id, document)
        path = directory / f"cache{count}.seg"
        write_segment(index, path)
        say(_cache_row(path, count))


def _cache_row(path: Path, count: int) -> str:
    """Return one row of the cache table, both sides over the same segment."""
    timings: dict[bool, float] = {}
    candidates = 0
    rate = "n/a"
    for caching in (True, False):
        with SegmentReader(path, verify=False, cache=caching) as reader:
            ranker = BM25Ranker(reader)
            term = max(reader.terms, key=reader.document_frequency)
            candidates = reader.document_frequency(term)
            timings[caching] = _time_once(partial(_search_common, reader, ranker, term))
            cache = reader.postings_cache
            if cache is not None:
                rate = f"{cache.hit_rate:.1%}"
    ratio = timings[False] / timings[True] if timings[True] else float("inf")
    return (
        f"| {count:,} | {candidates:,} | {timings[True]:.4f} | {timings[False]:.4f} "
        f"| {ratio:,.0f}x | {rate} |"
    )


def measure_corpus(index: InvertedIndex, directory: Path, name: str) -> None:
    """Write one index as a segment and report every mix at two limits."""
    path = directory / f"{name}.seg"
    write_segment(index, path)
    say(f"segment MB     : {path.stat().st_size / 1_000_000:.2f}")
    with SegmentReader(path, verify=False) as reader:
        ranker = BM25Ranker(reader)
        highest = max(reader.terms, key=reader.document_frequency)
        say(
            f"most common    : {highest} in "
            f"{reader.document_frequency(highest):,} of "
            f"{reader.document_count:,} documents"
        )
        mixes = (
            query_mixes(reader) if name == "generated" else [("one term", ["alpha"])]
        )
        report(reader, ranker, LIMIT, mixes)
        report(reader, ranker, 1, mixes)


def main() -> None:
    """Measure the generated corpus, then the shape block bounds need."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        say("## A generated corpus, lengths spread over two orders of magnitude")
        index = build()
        say(f"documents      : {index.document_count:,}")
        say(f"distinct terms : {index.vocabulary_size:,}")
        measure_corpus(index, root, "generated")
        del index
        say()
        say("## One document holding the term 500 times, the rest holding it once")
        measure_corpus(outlier_index(), root, "outlier")
        say()
        say("## What the postings cache is worth")
        measure_caching(root)


if __name__ == "__main__":
    main()
