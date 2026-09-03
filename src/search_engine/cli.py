"""Command line entry point.

Phase 0 ships only ``--version`` and the help text. Its job is to prove the
packaging chain end to end: install the project, run the installed console
script, get output back. Real subcommands arrive with the index in Phase 1.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from search_engine import __version__

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the top level argument parser.

    Returns:
        A parser exposing the program name, description and ``--version``.
    """
    parser = argparse.ArgumentParser(
        prog="search-engine",
        description="An inverted-index search engine built from scratch.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]`` when ``None``,
            which is what the installed console script relies on.

    Returns:
        A process exit code, ``0`` on success.
    """
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
