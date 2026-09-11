"""Command line entry point.

Two subcommands, matching the two phases of using a search engine: build an
index once, then query it many times.

Exit codes follow the shell convention that zero means success, so this is
usable in a pipeline: 0 found results, 1 found none, 2 the input was wrong.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final

from search_engine import __version__
from search_engine.analysis import analyze
from search_engine.bm25 import BM25Ranker
from search_engine.bm25f import BM25FRanker
from search_engine.corpus import CorpusFormatError, read
from search_engine.index import InvertedIndex, ReadableIndex
from search_engine.persistence import IndexFormatError, load, save
from search_engine.query import search
from search_engine.ranking import BaseRanker, Ranker
from search_engine.segment import (
    SEGMENT_HEADER,
    SegmentFormatError,
    SegmentReader,
    write_segment,
)
from search_engine.spelling import suggest

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

EXIT_OK = 0
EXIT_NO_RESULTS = 1
EXIT_BAD_INPUT = 2

# TF-IDF remains the default because the comparison between the two did not
# find a difference distinguishable from chance on the collection available.
# See docs/09-bm25.md.
SCORERS: Final[dict[str, Callable[[ReadableIndex], BaseRanker]]] = {
    "tfidf": Ranker,
    "bm25": BM25Ranker,
    "bm25f": BM25FRanker,
}
DEFAULT_SCORER = "tfidf"


def build_parser() -> argparse.ArgumentParser:
    """Build the top level argument parser."""
    parser = argparse.ArgumentParser(
        prog="search-engine",
        description="An inverted-index search engine built from scratch.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command")

    build = subcommands.add_parser("index", help="build an index from a corpus")
    build.add_argument("corpus", type=Path, help="corpus file of <page> records")
    build.add_argument("output", type=Path, help="index file to write")
    build.add_argument(
        "--on-disk",
        action="store_true",
        help="write a segment that is searched in place rather than loaded",
    )

    find = subcommands.add_parser("search", help="query an index")
    find.add_argument("index", type=Path, help="index file to read")
    find.add_argument("query", help='terms, or "a quoted phrase"')
    find.add_argument("--limit", type=int, default=10, help="results to show")
    find.add_argument(
        "--scorer",
        choices=sorted(SCORERS),
        default=DEFAULT_SCORER,
        help="ranking function to score results with",
    )

    return parser


def build_index(corpus: Path, output: Path, *, on_disk: bool = False) -> int:
    """Index every document in a corpus file and write the result.

    The two formats differ in how they are read rather than in what they hold:
    one is decoded whole when it is opened, the other is mapped and decoded a
    term at a time. See docs/12-on-disk-index.md for which to prefer.
    """
    index = InvertedIndex()
    try:
        for document in read(corpus):
            index.add_document(document.identifier, document.indexable_text)
    except (OSError, CorpusFormatError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT
    if on_disk:
        write_segment(index, output)
    else:
        save(index, output)
    print(
        f"indexed {index.document_count} documents, "
        f"{index.vocabulary_size} distinct terms -> {output}"
    )
    return EXIT_OK


def is_segment(path: Path) -> bool:
    """Return whether a file is a segment, from its first bytes alone.

    Reads the header rather than the file, so deciding the format costs
    nothing on an index that is not going to be loaded.
    """
    with path.open("rb") as handle:
        return handle.read(len(SEGMENT_HEADER)) == SEGMENT_HEADER


@contextmanager
def open_index(path: Path) -> Generator[ReadableIndex]:
    """Open an index in whichever format it was written in.

    A segment holds the file mapped for as long as it is read, which is why
    this is a context manager and not a function returning an index.
    """
    if is_segment(path):
        with SegmentReader(path) as reader:
            yield reader
    else:
        yield load(path)


def run_query(index_path: Path, query: str, limit: int, scorer: str) -> int:
    """Search an index in either format and print ranked results."""
    try:
        with open_index(index_path) as index:
            return _ranked(index, query, limit, scorer)
    except (OSError, IndexFormatError, SegmentFormatError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT


def _ranked(index: ReadableIndex, query: str, limit: int, scorer: str) -> int:
    candidates = search(index, query)
    ranked = (
        SCORERS[scorer](index).rank(query, candidates, limit=limit)
        if candidates
        else []
    )
    if not ranked:
        print("no matching documents")
        _suggest(index, query)
        return EXIT_NO_RESULTS
    for rank, (document_id, score) in enumerate(ranked, start=1):
        print(f"{rank:>3}. {score:.4f}  document {document_id}")
    return EXIT_OK


def _suggest(index: ReadableIndex, query: str) -> None:
    """Print near spellings for each query term the index does not hold.

    Written to stderr, so a caller piping results is not handed prose. The exit
    code stays at "no results", which is still what happened.

    Suggestions are index terms, and index terms are stemmed, so a suggestion
    may not look like a word. It is nonetheless the term that would have
    matched, which is the useful thing to report.
    """
    for term in dict.fromkeys(analyze(query)):
        if term in index:
            continue
        near = suggest(index, term)
        if near:
            print(f"did you mean: {', '.join(near)}?", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and return a process exit code.

    ``argv`` defaults to ``sys.argv[1:]``, which the console script relies on.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "index":
        return build_index(
            arguments.corpus, arguments.output, on_disk=arguments.on_disk
        )
    if arguments.command == "search":
        return run_query(
            arguments.index, arguments.query, arguments.limit, arguments.scorer
        )
    parser.print_help()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
