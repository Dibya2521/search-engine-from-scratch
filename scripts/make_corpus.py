"""Convert a Wikipedia dump into the corpus format this engine reads.

Run with:

    uv run python scripts/make_corpus.py DUMP OUTPUT [--limit N]

The dump may be plain XML, or compressed as `.gz` or `.bz2`, and is read
without expanding it first. Everything streams, so a dump larger than memory is
fine.

Dumps come from https://dumps.wikimedia.org. The one usually wanted is
`enwiki-latest-pages-articles.xml.bz2`, which is around 20 GB expanded. Start
with `simplewiki-latest-pages-articles.xml.bz2`, around 1 GB expanded, which is
enough to measure against and quick to download.

**Never commit the output.** `data/` is in `.gitignore` and pre-commit refuses
large files, which are the two guards.
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

from search_engine.wikipedia import read_dump, write_corpus

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from search_engine.wikipedia import Article


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="make_corpus",
        description="Convert a Wikipedia dump into the corpus format.",
    )
    parser.add_argument("dump", type=Path, help="dump file, optionally compressed")
    parser.add_argument("output", type=Path, help="corpus file to write")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after this many articles, for a smaller sample",
    )
    return parser


def report_progress(
    articles: Iterator[Article], every: int = 10_000
) -> Iterator[Article]:
    """Pass articles through, printing a count as they go.

    A full dump takes long enough that silence is indistinguishable from a
    hang.
    """
    started = time.perf_counter()
    for count, article in enumerate(articles, start=1):
        if count % every == 0:
            elapsed = time.perf_counter() - started
            print(f"  {count:,} articles, {count / elapsed:,.0f}/s", file=sys.stderr)
        yield article


def main(argv: Sequence[str] | None = None) -> int:
    """Convert a dump and report what was written."""
    arguments = build_parser().parse_args(argv)
    if not arguments.dump.exists():
        print(f"error: {arguments.dump} does not exist", file=sys.stderr)
        return 2
    articles = report_progress(read_dump(arguments.dump))
    if arguments.limit is not None:
        articles = islice(articles, arguments.limit)
    started = time.perf_counter()
    written = write_corpus(articles, arguments.output)
    elapsed = time.perf_counter() - started
    size = arguments.output.stat().st_size
    print(
        f"{written:,} articles, {size / 1_000_000:.1f} MB, "
        f"{elapsed:.1f} s -> {arguments.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
