"""The write path: buffer documents in memory, flush them as a segment.

Until now the only way to add a document was to rebuild the whole index.
`add_document` refuses an identifier it has already seen and a ranker refuses an
index that changed under it, both deliberately, so a corpus that grows means
building it again from the start. At the measured 1.1 MB/s a 300 MB corpus is
about four and a half minutes for one new document.

This is the write path of a log-structured merge tree at its simplest.
Documents accumulate in an ordinary in-memory index, and when enough have
arrived the buffer is written out as an immutable segment and a fresh one
started. Nothing about analysis or postings differs between the buffered form
and the written one, because the buffer is the same `InvertedIndex` the rest of
the project uses.

A document is searchable once it has been flushed. Making it searchable before
that means querying the buffer as well, which is a later concern.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, Self

from search_engine.index import InvertedIndex
from search_engine.segment import write_segment

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

DEFAULT_BUFFER_DOCUMENTS: Final = 10_000

SEGMENT_SUFFIX: Final = ".seg"
_SEGMENT_PATTERN: Final = re.compile(r"segment-(\d{8})\.seg\Z")


def segment_name(number: int) -> str:
    """Return the file name for a segment number."""
    return f"segment-{number:08d}{SEGMENT_SUFFIX}"


def next_segment_number(directory: Path) -> int:
    """Return the lowest segment number not already used in a directory.

    Numbers are never reused, even once a segment has been merged away and its
    file deleted. Reusing one would let a reader holding an old mapping believe
    it is looking at a file that now means something else.
    """
    used = [
        int(match.group(1))
        for path in directory.glob(f"segment-*{SEGMENT_SUFFIX}")
        if (match := _SEGMENT_PATTERN.fullmatch(path.name))
    ]
    return max(used, default=-1) + 1


class IndexWriter:
    """Accumulates documents in memory and writes them out as segments.

    The buffer is flushed automatically once it holds ``buffer_documents``
    documents, and on close. A larger buffer means fewer, larger segments and
    more memory held while they fill.
    """

    def __init__(
        self,
        directory: Path,
        buffer_documents: int = DEFAULT_BUFFER_DOCUMENTS,
    ) -> None:
        """Open a writer over a directory, creating it if it does not exist.

        Raises:
            ValueError: If the buffer would hold no documents.
        """
        if buffer_documents < 1:
            message = f"buffer_documents must be at least 1, got {buffer_documents}"
            raise ValueError(message)
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = directory
        self._buffer_documents = buffer_documents
        self._buffer = InvertedIndex()
        self._next = next_segment_number(directory)
        self._segments: list[str] = []

    def add(self, document_id: int, text: str) -> None:
        """Buffer a document, flushing first if the buffer is already full.

        Raises:
            DuplicateDocumentError: If this identifier is already in the buffer.
        """
        if self._buffer.document_count >= self._buffer_documents:
            self.flush()
        self._buffer.add_document(document_id, text)

    def flush(self) -> str | None:
        """Write the buffer as a segment and return its name.

        Returns None and writes nothing when the buffer is empty, so flushing
        twice or closing an untouched writer costs nothing.

        The segment is written but not yet published: nothing records that it
        exists, which is what the manifest is for.
        """
        if self._buffer.document_count == 0:
            return None
        name = segment_name(self._next)
        write_segment(self._buffer, self._directory / name)
        self._next += 1
        self._segments.append(name)
        self._buffer = InvertedIndex()
        return name

    def close(self) -> None:
        """Flush anything still buffered."""
        self.flush()

    def __enter__(self) -> Self:
        """Return the writer, so it can be used as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Flush on the way out, so nothing accepted is silently dropped."""
        self.close()

    @property
    def buffered(self) -> int:
        """Return how many documents are waiting to be written."""
        return self._buffer.document_count

    @property
    def segments(self) -> tuple[str, ...]:
        """Return the names of the segments this writer has written."""
        return tuple(self._segments)
