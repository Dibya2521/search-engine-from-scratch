"""Writing an index to a file and reading it back.

Building an index costs roughly a minute per hundred megabytes of corpus, so
doing it once and reloading is the difference between a usable tool and a demo.

The format is one line per term:

```text
term|docID:pos,pos,pos;docID:pos
```

Plain text rather than pickle or JSON, for three reasons. It is readable, so a
wrong index can be diagnosed with `head`. It streams, so neither writing nor
reading needs the whole file in memory. And unlike pickle it cannot execute
code when loaded, which matters the moment an index file arrives from anywhere
but your own machine.

No escaping is needed anywhere, and that is not luck: terms are maximal runs of
``[a-z0-9]``, so none of the delimiters can occur inside one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from search_engine.index import InvertedIndex

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

FORMAT_VERSION = 1
_MAGIC = "search-engine-index"
_HEADER = f"# {_MAGIC} v{FORMAT_VERSION}"
_DOCUMENTS_PREFIX = "# documents:"


class IndexFormatError(ValueError):
    """Raised when a file is not an index this version can read.

    Loading an index of the wrong shape would produce silently wrong results
    rather than an error, so the version is checked rather than assumed.
    """


def save(index: InvertedIndex, path: Path) -> None:
    """Write the index to a file, one line per term.

    Terms and document identifiers are written in sorted order so that
    indexing the same corpus twice produces byte-identical files, which is what
    makes the output diffable and the tests exact.
    """
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{_HEADER}\n")
        document_ids = ",".join(str(one) for one in sorted(index.document_ids))
        handle.write(f"{_DOCUMENTS_PREFIX} {document_ids}\n")
        for term in sorted(index.terms):
            entries = index.postings(term)
            body = ";".join(
                f"{document_id}:" + ",".join(str(p) for p in entries[document_id])
                for document_id in sorted(entries)
            )
            handle.write(f"{term}|{body}\n")


def load(path: Path) -> InvertedIndex:
    """Read an index back from a file.

    Raises:
        IndexFormatError: If the header is missing, the version is not one this
            build understands, or a line is malformed.
    """
    with path.open(encoding="utf-8") as handle:
        lines = iter(handle)
        _read_header(lines)
        document_ids = _read_document_ids(lines)
        postings = dict(_parse_line(line) for line in lines)
    return InvertedIndex.from_postings(document_ids, postings)


def _read_header(lines: Iterator[str]) -> None:
    header = next(lines, "").strip()
    if header != _HEADER:
        message = f"expected {_HEADER!r} on the first line, found {header!r}"
        raise IndexFormatError(message)


def _read_document_ids(lines: Iterator[str]) -> list[int]:
    line = next(lines, "").strip()
    if not line.startswith(_DOCUMENTS_PREFIX):
        message = f"expected a {_DOCUMENTS_PREFIX!r} line, found {line!r}"
        raise IndexFormatError(message)
    body = line.removeprefix(_DOCUMENTS_PREFIX).strip()
    if not body:
        return []
    try:
        return [int(one) for one in body.split(",")]
    except ValueError as error:
        message = f"document identifiers are not all integers: {body!r}"
        raise IndexFormatError(message) from error


def _parse_line(line: str) -> tuple[str, dict[int, list[int]]]:
    term, separator, body = line.rstrip("\n").partition("|")
    if not separator or not term:
        message = f"expected 'term|postings', found {line.rstrip()!r}"
        raise IndexFormatError(message)
    entries: dict[int, list[int]] = {}
    for chunk in body.split(";"):
        document_text, found, positions_text = chunk.partition(":")
        if not found:
            message = f"expected 'docID:positions' in {chunk!r} for term {term!r}"
            raise IndexFormatError(message)
        try:
            entries[int(document_text)] = [
                int(one) for one in positions_text.split(",")
            ]
        except ValueError as error:
            message = f"non-integer document id or position in {chunk!r}"
            raise IndexFormatError(message) from error
    return term, entries
