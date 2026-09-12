"""Command line entry point.

Two subcommands, matching the two phases of using a search engine: build an
index once, then query it many times.

Exit codes follow the shell convention that zero means success, so this is
usable in a pipeline: 0 found results, 1 found none, 2 the input was wrong.

Results are printed three ways and the difference is who is reading. A person
gets the title and the passage that matched, with the matched words in bold
when the output is a terminal. A pipe gets the same text with no escape codes
in it, because that output ends up in a file. A caller passing ``--json`` gets
one object per line, which is what a program should be reading rather than
parsing columns out of prose.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
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
from search_engine.snippet import Snippet, extract
from search_engine.spelling import suggest
from search_engine.store import DocumentStore, StoreFormatError

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

ANSI_BOLD: Final = "\x1b[1m"
ANSI_RESET: Final = "\x1b[0m"

FALLBACK_COLUMNS: Final = 80
MINIMUM_ROOM: Final = 20
SCORE_ROOM: Final = 8
RANK_ROOM: Final = 5
INDENT: Final = "     "
ELLIPSIS: Final = "..."

# What a result carries when the index it came from has no text stored beside
# it. An index like that is still searchable, so this is absence, not failure.
NO_PASSAGE: Final = Snippet(text="", highlights=())

# Every C0 control, delete, and every C1 control, mapped to a space rather than
# removed: an escape sequence must not reach the terminal, and dropping the
# character would move every highlight offset after it.
_CONTROL: Final = dict.fromkeys([*range(0x20), 0x7F, *range(0x80, 0xA0)], " ")


@dataclass(frozen=True, slots=True)
class Options:
    """What one search was asked to do."""

    index: Path
    query: str
    limit: int = 10
    scorer: str = DEFAULT_SCORER
    snippets: bool = True
    as_json: bool = False


@dataclass(frozen=True, slots=True)
class Result:
    """One ranked document, as both a person and a program want it."""

    identifier: int
    title: str
    score: float
    snippet: Snippet


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
    find.add_argument(
        "--no-snippet",
        action="store_true",
        help="print identifiers and scores only, for a caller parsing them",
    )
    find.add_argument(
        "--json",
        action="store_true",
        help="print one JSON object per result and nothing else",
    )

    return parser


def build_index(corpus: Path, output: Path, *, on_disk: bool = False) -> int:
    """Index every document in a corpus file and write the result.

    The two formats differ in how they are read rather than in what they hold:
    one is decoded whole when it is opened, the other is mapped and decoded a
    term at a time. See docs/12-on-disk-index.md for which to prefer.

    The text is written beside the index so results can be shown rather than
    only scored. It is held in memory until the corpus has been read, because
    the store is written in identifier order and a corpus does not have to
    arrive in one. That is on top of an index already several times the size of
    its source, which is what makes this the path for a corpus that fits and
    `IndexWriter` the path for one that does not.
    """
    index = InvertedIndex()
    pages: dict[int, tuple[str, str]] = {}
    try:
        for document in read(corpus):
            index.add_document(document.identifier, document.indexable_text)
            pages[document.identifier] = (document.title, document.text)
    except (OSError, CorpusFormatError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT
    if on_disk:
        write_segment(index, output)
    else:
        save(index, output)
    _write_store(output, pages)
    print(
        f"indexed {index.document_count} documents, "
        f"{index.vocabulary_size} distinct terms -> {output}"
    )
    return EXIT_OK


def _write_store(output: Path, pages: dict[int, tuple[str, str]]) -> None:
    """Write the text beside the index, in the order a reader will assume.

    An ordinal is a position in the sorted identifiers, which is what both
    index formats hand back, so the two agree without either recording it.
    """
    with DocumentStore.writer(output.parent, output.name) as store:
        for _, (title, text) in sorted(pages.items()):
            store.add(title, text)


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


@contextmanager
def open_store(path: Path, documents: int) -> Generator[DocumentStore | None]:
    """Open the text stored beside an index, or nothing when there is none.

    An index written before the text was stored is still perfectly searchable,
    so a missing store costs the snippets rather than the search. That is the
    opposite of a merge, where carrying on without the text would destroy it.
    """
    try:
        store = DocumentStore.open(path.parent, path.name, documents=documents)
    except StoreFormatError:
        yield None
        return
    try:
        yield store
    finally:
        store.close()


def run_query(options: Options) -> int:
    """Search an index in either format and print ranked results."""
    try:
        with open_index(options.index) as index:
            ranked = _rank(index, options)
            if not ranked:
                return _report_nothing(index, options)
            with open_store(options.index, index.document_count) as store:
                results = _results(index, store, options.query, ranked)
                _report(results, options, stored=store is not None)
            return EXIT_OK
    except (OSError, IndexFormatError, SegmentFormatError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT


def _rank(index: ReadableIndex, options: Options) -> list[tuple[int, float]]:
    candidates = search(index, options.query)
    if not candidates:
        return []
    return SCORERS[options.scorer](index).rank(
        options.query, candidates, limit=options.limit
    )


def _report_nothing(index: ReadableIndex, options: Options) -> int:
    """Say that nothing matched, on the stream the caller is reading."""
    if not options.as_json:
        print("no matching documents")
    _suggest(index, options.query)
    return EXIT_NO_RESULTS


def _results(
    index: ReadableIndex,
    store: DocumentStore | None,
    query: str,
    ranked: Sequence[tuple[int, float]],
) -> list[Result]:
    """Pair each ranked document with its title and the passage that matched."""
    if store is None:
        return [Result(number, "", score, NO_PASSAGE) for number, score in ranked]
    ordinals = {
        document_id: ordinal
        for ordinal, document_id in enumerate(sorted(index.document_ids))
    }
    return [
        Result(
            number,
            store.title(ordinals[number]),
            score,
            extract(store.text(ordinals[number]), query),
        )
        for number, score in ranked
    ]


def _report(results: list[Result], options: Options, *, stored: bool) -> None:
    if options.as_json:
        _print_json(results)
        return
    if options.snippets and stored:
        _print_passages(results)
        return
    if options.snippets:
        print(
            "note: this index holds no stored text, so there are no snippets",
            file=sys.stderr,
        )
    _print_terse(results)


def _print_json(results: list[Result]) -> None:
    """Print one object per line, which is what a program should be reading.

    `json.dumps` escapes every control character and every non-ASCII one, so
    this output cannot carry an escape sequence to a terminal whatever the
    corpus held.
    """
    for result in results:
        print(
            json.dumps(
                {
                    "identifier": result.identifier,
                    "title": result.title,
                    "score": result.score,
                    "snippet": result.snippet.text,
                    "highlights": [list(pair) for pair in result.snippet.highlights],
                }
            )
        )


def _print_terse(results: list[Result]) -> None:
    """Print one line per result, with the identifier a caller wants."""
    for rank, result in enumerate(results, start=1):
        print(f"{rank:>3}. {result.score:.4f}  document {result.identifier}")


def _print_passages(results: list[Result]) -> None:
    """Print the title and the matching passage for each result."""
    columns = _columns()
    colour = sys.stdout.isatty()
    for rank, result in enumerate(results, start=1):
        if rank > 1:
            print()
        print(_writable(_heading(rank, result, columns)))
        for line in _passage(result.snippet, columns, colour=colour):
            print(_writable(line))


def _heading(rank: int, result: Result, columns: int) -> str:
    """Return the rank, the title and the score on one line of the terminal."""
    room = max(columns - RANK_ROOM - SCORE_ROOM, MINIMUM_ROOM)
    title = _display(result.title) or f"document {result.identifier}"
    return f"{rank:>3}. {_truncate(title, room):<{room}}{result.score:>{SCORE_ROOM}.4f}"


def _passage(snippet: Snippet, columns: int, *, colour: bool) -> list[str]:
    """Return the passage as indented lines, with the matched words in bold."""
    text = _display(snippet.text)
    room = max(columns - len(INDENT), MINIMUM_ROOM)
    lines: list[str] = []
    for start, end in _fold(text, room):
        piece = text[start:end]
        if colour:
            piece = _mark(piece, _within(snippet.highlights, start, end))
        lines.append(INDENT + piece)
    return lines


def _fold(text: str, room: int) -> list[tuple[int, int]]:
    """Return each display line as a range into the text.

    Ranges rather than strings, because a highlight has to be clipped to the
    line it falls on and `textwrap` returns the strings without the offsets.
    Wrapping after inserting escape codes would count them towards the width.
    """
    lines: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        if len(text) - start <= room:
            lines.append((start, len(text)))
            break
        cut = text.rfind(" ", start + 1, start + room + 1)
        end = cut if cut > start else start + room
        lines.append((start, end))
        start = end + 1 if text[end : end + 1] == " " else end
    return lines


def _within(
    highlights: tuple[tuple[int, int], ...], start: int, end: int
) -> list[tuple[int, int]]:
    """Return the highlights falling on one line, clipped and made relative."""
    return [
        (max(left, start) - start, min(right, end) - start)
        for left, right in highlights
        if left < end and right > start
    ]


def _mark(text: str, ranges: Sequence[tuple[int, int]]) -> str:
    """Wrap each range in bold, from the end, so earlier offsets stay valid."""
    for left, right in reversed(ranges):
        text = f"{text[:left]}{ANSI_BOLD}{text[left:right]}{ANSI_RESET}{text[right:]}"
    return text


def _truncate(text: str, room: int) -> str:
    return text if len(text) <= room else text[: room - len(ELLIPSIS)] + ELLIPSIS


def _display(text: str) -> str:
    """Return text with every control character replaced by a space.

    One character for one, so an escape sequence in a corpus cannot reach the
    terminal and every highlight offset still points where it did.
    """
    return text.translate(_CONTROL)


def _writable(line: str) -> str:
    """Return a finished line the current output stream can actually encode.

    A console that cannot represent a character shows its escape rather than
    ending the process, which is what printing a title in a script the console
    does not support would otherwise do.
    """
    encoding = sys.stdout.encoding or "utf-8"
    return line.encode(encoding, "backslashreplace").decode(encoding)


def _columns() -> int:
    """Return the terminal width, or 80 when there is not one to ask."""
    size = shutil.get_terminal_size(fallback=(FALLBACK_COLUMNS, 24))
    return size.columns or FALLBACK_COLUMNS


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
            Options(
                index=arguments.index,
                query=arguments.query,
                limit=arguments.limit,
                scorer=arguments.scorer,
                snippets=not arguments.no_snippet,
                as_json=arguments.json,
            )
        )
    parser.print_help()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
