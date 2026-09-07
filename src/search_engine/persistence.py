"""Writing an index to a file and reading it back.

Building an index costs roughly a minute per hundred megabytes of corpus, so
doing it once and reloading is the difference between a usable tool and a demo.

Two formats exist. `save` writes version 2; `load` reads either, so an index
written by an older build still opens.

## Version 2, the current format

Binary, and compressed with the codecs in `search_engine.codecs`: document
identifiers and positions are both sorted, so they are stored as gaps and the
gaps are stored in as few bytes as each needs.

```text
header      the ASCII line "search-engine-index v2", newline terminated
analyzer    length-prefixed fingerprint of the analysis configuration
checksum    CRC32 of everything after it, as four big-endian bytes
documents   count, then delta and variable-byte encoded identifiers
terms       count, then for each term in sorted order:
              length-prefixed ASCII term
              count of documents, then their delta-encoded identifiers
              for each document: count of positions, then their deltas
```

Every number is self-delimiting, so nothing needs a byte-length prefix and the
body can be read straight through.

Three things this format adds beyond size.

**A checksum.** A silently corrupt index returns wrong answers indefinitely,
which is worse than failing to open.

**The analyzer fingerprint.** An index records the exact configuration that
produced its terms. Loading it with a different one raises, because a query
analyzed differently from the documents asks for terms the index never stored
and the engine returns nothing while appearing to work.

**A version, checked rather than assumed.** Reading a file of the wrong shape
would produce plausible garbage.

## Version 1, still readable

One line of text per term:

```text
term|docID:pos,pos,pos;docID:pos
```

Readable with `head`, which version 2 gives up. That was one of three reasons
the text format was chosen, and losing it is the real cost of compression. The
other two reasons still hold: neither format needs the whole file in memory to
parse a record, and neither can execute code on load, which `pickle` can.

No escaping is needed anywhere in version 1, and that is not luck: terms are
maximal runs of ``[a-z0-9]``, so no delimiter can occur inside one.
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING, Final

from search_engine.analysis import fingerprint
from search_engine.codecs import (
    CodecError,
    decode_at,
    decode_sorted_at,
    encode_number,
    encode_sorted,
)
from search_engine.index import InvertedIndex

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

FORMAT_VERSION = 2
_MAGIC = "search-engine-index"
_HEADER = f"# {_MAGIC} v1"
_DOCUMENTS_PREFIX = "# documents:"

_BINARY_HEADER: Final = f"{_MAGIC} v{FORMAT_VERSION}\n".encode("ascii")
_CHECKSUM_BYTES: Final = 4


class IndexFormatError(ValueError):
    """Raised when a file is not an index this version can read.

    Loading an index of the wrong shape would produce silently wrong results
    rather than an error, so the version is checked rather than assumed.
    """


class IndexCorruptError(IndexFormatError):
    """Raised when an index file's checksum does not match its contents."""


class AnalyzerMismatchError(IndexFormatError):
    """Raised when an index was built by a different analysis configuration.

    Its terms were produced by a tokenizer, stopword list or stemmer that no
    longer matches the one queries go through, so the two would silently fail
    to meet.
    """

    def __init__(self, stored: str, current: str) -> None:
        super().__init__(
            f"index was built with analyzer {stored}, this build uses {current}; "
            f"rebuild the index"
        )


def save(index: InvertedIndex, path: Path) -> None:
    """Write the index in the current format.

    Terms and document identifiers are written in sorted order, so indexing the
    same corpus twice produces byte-identical files. Delta encoding requires
    that order anyway.

    The body is built in memory before writing, because the checksum covers it
    and cannot be known until it is complete. The index is already in memory, so
    this roughly doubles peak usage during the write rather than changing what
    the process needs overall.
    """
    body = _encode_body(index)
    with path.open("wb") as handle:
        handle.write(_BINARY_HEADER)
        stamp = fingerprint().encode("ascii")
        handle.write(encode_number(len(stamp)))
        handle.write(stamp)
        handle.write(zlib.crc32(body).to_bytes(_CHECKSUM_BYTES, "big"))
        handle.write(body)


def save_text(index: InvertedIndex, path: Path) -> None:
    """Write the index in the version 1 text format.

    Kept so the two formats can be measured against each other on the same
    index rather than against a remembered number.
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
    """Read an index back from a file, in either format.

    Raises:
        IndexFormatError: If the header is missing, the version is not one this
            build understands, or the contents are malformed.
        IndexCorruptError: If a version 2 file fails its checksum.
        AnalyzerMismatchError: If a version 2 file was built by a different
            analysis configuration.
    """
    data = path.read_bytes()
    if data.startswith(_BINARY_HEADER):
        return _load_binary(data)
    return _load_text(path)


def _encode_body(index: InvertedIndex) -> bytes:
    parts: list[bytes] = []
    document_ids = sorted(index.document_ids)
    parts.append(encode_number(len(document_ids)))
    parts.append(encode_sorted(document_ids))
    terms = sorted(index.terms)
    parts.append(encode_number(len(terms)))
    for term in terms:
        encoded = term.encode("utf-8")
        parts.append(encode_number(len(encoded)))
        parts.append(encoded)
        entries = index.postings(term)
        ordered = sorted(entries)
        parts.append(encode_number(len(ordered)))
        parts.append(encode_sorted(ordered))
        for document_id in ordered:
            positions = entries[document_id]
            parts.append(encode_number(len(positions)))
            parts.append(encode_sorted(list(positions)))
    return b"".join(parts)


def _load_binary(data: bytes) -> InvertedIndex:
    try:
        stamp_length, cursor = decode_at(data, len(_BINARY_HEADER))
    except CodecError as error:
        message = "index file ends before the analyzer fingerprint"
        raise IndexFormatError(message) from error
    stored = data[cursor : cursor + stamp_length].decode("utf-8", errors="replace")
    current = fingerprint()
    if stored != current:
        raise AnalyzerMismatchError(stored, current)
    cursor += stamp_length

    checksum = int.from_bytes(data[cursor : cursor + _CHECKSUM_BYTES], "big")
    cursor += _CHECKSUM_BYTES
    body = data[cursor:]
    if zlib.crc32(body) != checksum:
        message = "index file is corrupt: the checksum does not match its contents"
        raise IndexCorruptError(message)
    return _decode_body(body)


def _decode_body(body: bytes) -> InvertedIndex:
    reader = _Reader(body)
    try:
        document_ids = reader.sorted_values(reader.number())
        postings: dict[str, dict[int, list[int]]] = {}
        for _ in range(reader.number()):
            term = reader.text()
            entries: dict[int, list[int]] = {}
            for document_id in reader.sorted_values(reader.number()):
                entries[document_id] = reader.sorted_values(reader.number())
            postings[term] = entries
    except (CodecError, UnicodeDecodeError) as error:
        message = f"index file is malformed: {error}"
        raise IndexFormatError(message) from error
    return InvertedIndex.from_postings(document_ids, postings)


class _Reader:
    """Pulls numbers and length-prefixed text out of an encoded body in order."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._at = 0

    def number(self) -> int:
        value, self._at = decode_at(self._data, self._at)
        return value

    def text(self) -> str:
        length = self.number()
        chunk = self._data[self._at : self._at + length]
        if len(chunk) != length:
            message = "index file ends in the middle of a term"
            raise CodecError(message)
        self._at += length
        return chunk.decode("utf-8")

    def sorted_values(self, count: int) -> list[int]:
        """Read `count` gaps and return the ascending values they encode."""
        values, self._at = decode_sorted_at(self._data, self._at, count)
        return values


def _load_text(path: Path) -> InvertedIndex:
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
