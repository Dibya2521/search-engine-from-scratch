"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from search_engine import __version__

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the top level argument parser."""
    parser = argparse.ArgumentParser(
        prog="search-engine",
        description="An inverted-index search engine built from scratch.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and return a process exit code.

    ``argv`` defaults to ``sys.argv[1:]``, which the console script relies on.
    """
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
