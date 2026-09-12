"""The source text, kept beside a segment and addressed by ordinal.

A segment holds terms, the documents they occur in and where, and nothing that
could reconstruct the source. That is what makes it a fraction of the size of
what it indexes, and it is also why a result list built from one can name a
document and score it but say nothing about what it says.

So the text is stored separately, as two files beside the segment:

```text
segment-00000004.dat   each document as a length-prefixed title followed by a
                       length-prefixed body, both UTF-8
segment-00000004.idx   one big-endian uint64 offset per document, in ordinal
                       order, so ordinal n begins at byte 8n
```

**The offsets are fixed width and live in a file of their own** so that the
record for an ordinal is found by arithmetic rather than by looking for it. One
seek reaches any document, with no scan of the data file and no table held in
memory. Offsets written among the documents would themselves have to be found
before they could be used, which is the problem they exist to solve. It is the
same reason a segment's footer is fixed width.

**Ordinals rather than identifiers.** Ordinals are dense and identifiers are
arbitrary, so a file keyed by identifier would be as large as the identifier
space rather than as large as the segment. The segment already maps identifier
to ordinal for its tombstones, so the mapping costs nothing.

**The alternative was to seek into the original corpus file**, which would store
no bytes at all. It was rejected because it makes an index depend on a file it
does not own and cannot check: a corpus that is moved, edited or recompressed
turns every stored offset into a wrong answer rather than an error. What owning
the text costs is disk, and `benchmarks/document_store.py` measures it.

The title is stored apart from the body because a result list wants it on its
own, and recovering it from the body afterwards would be guesswork.

Neither file carries a header or a checksum. A header would put the offsets at
`8n + k` rather than `8n`, and a checksum would mean reading the whole store to
open it, which is what mapping it avoids. Damage is caught structurally
instead: the offset file must be a whole number of offsets and must agree with
the segment's document count, and every length prefix must fit inside what the
data file actually holds.
"""

from __future__ import annotations

import mmap
from typing import TYPE_CHECKING, Final, Self

from search_engine.codecs import CodecError, decode_at, encode_number

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO

    from search_engine.codecs import Bytelike

SEGMENT_SUFFIX: Final = ".seg"
DATA_SUFFIX: Final = ".dat"
OFFSET_SUFFIX: Final = ".idx"

OFFSET_BYTES: Final = 8
_BYTE_ORDER: Final = "big"


class StoreFormatError(ValueError):
    """Raised when a document store is missing or is not well formed."""


class StoreCorruptError(StoreFormatError):
    """Raised when a store's bytes do not hold what its offsets promise."""


def data_name(segment: str) -> str:
    """Return the store data file name that belongs to a segment file name."""
    return segment.removesuffix(SEGMENT_SUFFIX) + DATA_SUFFIX


def offset_name(segment: str) -> str:
    """Return the store offset file name that belongs to a segment file name."""
    return segment.removesuffix(SEGMENT_SUFFIX) + OFFSET_SUFFIX


class DocumentStoreWriter:
    """Appends documents to a store, one ordinal at a time.

    Documents must arrive in the segment's ordinal order. Nothing checks that
    they do, because the ordinal a document is given here is simply the
    position it was written at, and the segment is the only thing that knows
    what that position should mean.
    """

    def __init__(self, data: IO[bytes], offsets: IO[bytes]) -> None:
        """Take the two open binary sinks the store is written to."""
        self._data = data
        self._offsets = offsets
        self._offset = 0

    def add(self, title: str, text: str) -> None:
        """Append one document at the next ordinal."""
        record = _encode_string(title) + _encode_string(text)
        self._offsets.write(self._offset.to_bytes(OFFSET_BYTES, _BYTE_ORDER))
        self._data.write(record)
        self._offset += len(record)

    def close(self) -> None:
        """Close both files. Safe to call more than once."""
        self._data.close()
        self._offsets.close()

    def __enter__(self) -> Self:
        """Return the writer, so it can be used as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close both files when the block ends."""
        self.close()


class DocumentStore:
    """One segment's document text, addressed by ordinal.

    Both files stay mapped for as long as the store is open. A mapped file
    cannot be deleted or replaced on Windows, so a merge has to close every
    input store before it removes the files it merged.
    """

    def __init__(
        self, data: Path, offsets: Path, *, documents: int | None = None
    ) -> None:
        """Map an existing pair of store files.

        ``documents`` is the count the segment says it holds. Passing it checks
        that the store belongs to that segment; omitting it trusts the offset
        file's own length, which is all a reader holding no segment can do.

        Raises:
            StoreFormatError: If either file is missing, if the offset file is
                not a whole number of offsets, or if it covers a different
                number of documents than the segment does.
        """
        self._offsets = _Mapping(offsets)
        try:
            self._documents = _count_documents(self._offsets.view, offsets, documents)
            self._data = _Mapping(data)
        except StoreFormatError:
            self._offsets.close()
            raise

    @classmethod
    def writer(cls, directory: Path, name: str) -> DocumentStoreWriter:
        """Create the store belonging to a segment, replacing any already there."""
        return DocumentStoreWriter(
            (directory / data_name(name)).open("wb"),
            (directory / offset_name(name)).open("wb"),
        )

    @classmethod
    def open(
        cls, directory: Path, name: str, *, documents: int | None = None
    ) -> DocumentStore:
        """Open the store belonging to a segment.

        Raises:
            StoreFormatError: If the store is missing or is not well formed.
        """
        return cls(
            directory / data_name(name),
            directory / offset_name(name),
            documents=documents,
        )

    def title(self, ordinal: int) -> str:
        """Return one document's title.

        Raises:
            IndexError: If the store holds no such ordinal.
            StoreCorruptError: If the record does not lie inside the file.
        """
        title, _ = _decode_string(self._data.view, self._start(ordinal))
        return title

    def text(self, ordinal: int) -> str:
        """Return one document's body.

        Raises:
            IndexError: If the store holds no such ordinal.
            StoreCorruptError: If the record does not lie inside the file.
        """
        _, after_title = _decode_string(self._data.view, self._start(ordinal))
        text, _ = _decode_string(self._data.view, after_title)
        return text

    def close(self) -> None:
        """Release both mappings. Safe to call more than once."""
        self._data.close()
        self._offsets.close()

    def __enter__(self) -> Self:
        """Return the store, so it can be used as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close both mappings when the block ends."""
        self.close()

    @property
    def documents(self) -> int:
        """Return how many documents the store holds."""
        return self._documents

    def _start(self, ordinal: int) -> int:
        if not 0 <= ordinal < self._documents:
            message = f"no ordinal {ordinal} in a store of {self._documents} documents"
            raise IndexError(message)
        at = ordinal * OFFSET_BYTES
        return int.from_bytes(self._offsets.view[at : at + OFFSET_BYTES], _BYTE_ORDER)


class _Mapping:
    """One store file, mapped for reading.

    An empty file cannot be mapped at all and an empty segment produces exactly
    that, so it is represented as an empty view rather than refused.
    """

    __slots__ = ("_file", "_map", "view")

    def __init__(self, path: Path) -> None:
        """Open and map a file.

        Raises:
            StoreFormatError: If the file cannot be opened.
        """
        try:
            self._file = path.open("rb")
        except OSError as error:
            message = f"document store file cannot be read: {path}"
            raise StoreFormatError(message) from error
        try:
            self._map: mmap.mmap | None = mmap.mmap(
                self._file.fileno(), 0, access=mmap.ACCESS_READ
            )
        except ValueError:
            self._map = None
        self.view = memoryview(b"") if self._map is None else memoryview(self._map)

    def close(self) -> None:
        """Release the mapping and the file. Safe to call more than once."""
        self.view.release()
        if self._map is not None:
            self._map.close()
        self._file.close()


def _count_documents(view: Bytelike, path: Path, expected: int | None) -> int:
    """Return how many documents an offset file covers.

    Raises:
        StoreFormatError: If it is not a whole number of offsets, or covers a
            different count than the segment it belongs to.
    """
    if len(view) % OFFSET_BYTES:
        message = f"offset file is not a whole number of offsets: {path}"
        raise StoreFormatError(message)
    documents = len(view) // OFFSET_BYTES
    if expected is not None and expected != documents:
        message = (
            f"store covers {documents} documents and its segment holds "
            f"{expected}: {path}"
        )
        raise StoreFormatError(message)
    return documents


def _encode_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return encode_number(len(encoded)) + encoded


def _decode_string(data: Bytelike, at: int) -> tuple[str, int]:
    """Return the string starting at an offset, and where the next one starts.

    Raises:
        StoreCorruptError: If the length prefix, the bytes it promises or their
            encoding are not there. A store cut short mid-document has to raise
            rather than hand back the part of the text that survived.
    """
    try:
        length, start = decode_at(data, at)
    except CodecError as error:
        message = f"document store ends inside a length prefix at byte {at}"
        raise StoreCorruptError(message) from error
    end = start + length
    if end > len(data):
        message = (
            f"document store promises {length} bytes at {start} "
            f"and holds {len(data) - start}"
        )
        raise StoreCorruptError(message)
    try:
        return bytes(data[start:end]).decode("utf-8"), end
    except UnicodeDecodeError as error:
        message = f"document store holds bytes at {start} that are not UTF-8"
        raise StoreCorruptError(message) from error
