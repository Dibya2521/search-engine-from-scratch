"""Command line entry point.

Two subcommands, matching the two phases of using a search engine: build an
index once, then query it many times.

Exit codes follow the shell convention that zero means success, so this is
usable in a pipeline: 0 found results, 1 found none, 2 the input was wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

from search_engine import __version__
from search_engine.bm25 import BM25Ranker
from search_engine.corpus import CorpusFormatError, read
from search_engine.index import InvertedIndex
from search_engine.persistence import IndexFormatError, load, save
from search_engine.query import search
from search_engine.ranking import BaseRanker, Ranker

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

EXIT_OK = 0
EXIT_NO_RESULTS = 1
EXIT_BAD_INPUT = 2

# TF-IDF remains the default because the comparison between the two did not
# find a difference distinguishable from chance on the collection available.
# See docs/09-bm25.md.
SCORERS: Final[dict[str, Callable[[InvertedIndex], BaseRanker]]] = {
    "tfidf": Ranker,
    "bm25": BM25Ranker,
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


def build_index(corpus: Path, output: Path) -> int:
    """Index every document in a corpus file and write the result."""
    index = InvertedIndex()
    try:
        for document in read(corpus):
            index.add_document(document.identifier, document.indexable_text)
    except (OSError, CorpusFormatError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT
    save(index, output)
    print(
        f"indexed {index.document_count} documents, "
        f"{index.vocabulary_size} distinct terms -> {output}"
    )
    return EXIT_OK


def run_query(index_path: Path, query: str, limit: int, scorer: str) -> int:
    """Search an index and print ranked results."""
    try:
        index = load(index_path)
    except (OSError, IndexFormatError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT
    candidates = search(index, query)
    if not candidates:
        print("no matching documents")
        return EXIT_NO_RESULTS
    ranked = SCORERS[scorer](index).rank(query, candidates, limit=limit)
    if not ranked:
        print("no matching documents")
        return EXIT_NO_RESULTS
    for rank, (document_id, score) in enumerate(ranked, start=1):
        print(f"{rank:>3}. {score:.4f}  document {document_id}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and return a process exit code.

    ``argv`` defaults to ``sys.argv[1:]``, which the console script relies on.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "index":
        return build_index(arguments.corpus, arguments.output)
    if arguments.command == "search":
        return run_query(
            arguments.index, arguments.query, arguments.limit, arguments.scorer
        )
    parser.print_help()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
