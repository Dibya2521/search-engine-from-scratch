"""The write path: buffer documents, flush them as a segment, publish them.

Until now the only way to add a document was to rebuild the whole index.
`add_document` refuses an identifier it has already seen and a ranker refuses an
index that changed under it, both deliberately, so a corpus that grows meant
building it again from the start. At the measured 1.1 MB/s a 300 MB corpus is
about four and a half minutes for one new document.

This is the write path of a log-structured merge tree at its simplest.
Documents accumulate in memory, and when enough have arrived they are analysed
into an ordinary `InvertedIndex` and written out as an immutable segment.
Nothing about analysis or postings differs between what is buffered and what is
written.

**Writing is an upsert.** Ingestion in any real system is at-least-once, so the
same document arrives twice and indexing has to be idempotent.
`DuplicateDocumentError` is exactly the wrong behaviour there, and it is kept
where it is right: on `InvertedIndex`, where a repeated identifier in a
single-shot build is a bug in the corpus. Tolerating repeats belongs at this
layer, the one that knows how to make an older copy invisible.

An older copy cannot be removed from an immutable segment, so it is tombstoned:
recorded as deleted beside the segment and filtered out when it is read. The
space comes back at merge time.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, Self

from search_engine import manifest as manifest_file
from search_engine.index import InvertedIndex
from search_engine.manifest import Manifest, SegmentInfo
from search_engine.segment import SegmentReader, stored_checksum, write_segment
from search_engine.tombstones import Tombstones, tombstone_name

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

    The buffer is flushed once it holds ``buffer_documents`` documents, and on
    close. A larger buffer means fewer, larger segments and more memory held
    while they fill.

    Documents are held as text until a flush, so replacing one before it is
    written costs nothing and analysis happens once rather than once per
    revision.
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
        self._buffer: dict[int, str] = {}
        self._manifest = manifest_file.read(directory)
        # Orphans left by a crash are invisible, and their numbers are still
        # spent: reusing one would give a stale mapping a file that has changed
        # meaning underneath it.
        self._next = max(self._manifest.next_segment, next_segment_number(directory))
        self._locations = self._locate_documents()
        self._pending: dict[str, set[int]] = {}

    def _locate_documents(self) -> dict[int, tuple[str, int]]:
        """Map every published identifier to its segment and its ordinal there.

        Reading each segment's identifiers costs one small section per file and
        no postings at all. Without it an upsert would have no way to find the
        copy it has to replace.
        """
        located: dict[int, tuple[str, int]] = {}
        for segment in self._manifest.segments:
            path = self._directory / segment.name
            marks = Tombstones.read(
                self._directory / tombstone_name(segment.name), segment.documents
            )
            with SegmentReader(path, verify=False) as reader:
                for ordinal, document_id in enumerate(reader.document_ids):
                    if not marks.is_deleted(ordinal):
                        located[document_id] = (segment.name, ordinal)
        return located

    def add(self, document_id: int, text: str) -> None:
        """Index a document, replacing any earlier copy of the same identifier.

        Adding the same identifier twice is a no-op rather than an error, which
        is what makes replaying an at-least-once stream safe.
        """
        if (
            document_id not in self._buffer
            and len(self._buffer) >= self._buffer_documents
        ):
            self.flush()
        self.delete(document_id)
        self._buffer[document_id] = text

    def delete(self, document_id: int) -> bool:
        """Mark a document deleted, and return whether it was there to delete.

        Deleting one that does not exist is not an error, for the same reason
        that adding one twice is not.
        """
        if self._buffer.pop(document_id, None) is not None:
            return True
        found = self._locations.pop(document_id, None)
        if found is None:
            return False
        name, ordinal = found
        self._pending.setdefault(name, set()).add(ordinal)
        return True

    def flush(self) -> str | None:
        """Write anything buffered as a segment, publish it, and return its name.

        Returns None when there is nothing to write, so flushing twice or
        closing an untouched writer costs nothing.

        The segment file is written first and the manifest second, and that
        order is the whole of the crash safety. A process that dies between the
        two leaves a file nothing names, which the next open ignores. Publishing
        first would name a file that does not exist.
        """
        deletions = self._write_tombstones()
        if not self._buffer:
            if deletions:
                self._publish(deletions)
            return None

        name = segment_name(self._next)
        path = self._directory / name
        index = self._analyse()
        write_segment(index, path)
        info = SegmentInfo(
            name=name,
            documents=index.document_count,
            checksum=stored_checksum(path.read_bytes()),
        )
        for ordinal, document_id in enumerate(sorted(self._buffer)):
            self._locations[document_id] = (name, ordinal)
        self._next += 1
        self._buffer = {}
        published = deletions or list(self._manifest.segments)
        self._publish([*published, info])
        return name

    def _analyse(self) -> InvertedIndex:
        index = InvertedIndex()
        for document_id, text in sorted(self._buffer.items()):
            index.add_document(document_id, text)
        return index

    def _write_tombstones(self) -> list[SegmentInfo]:
        """Apply pending deletions to the tombstone file beside each segment.

        Returns every segment as it now stands when anything changed, and an
        empty list when nothing did, so a caller can tell one from the other.
        """
        updated: list[SegmentInfo] = []
        changed = False
        for segment in self._manifest.segments:
            ordinals = self._pending.pop(segment.name, set())
            if not ordinals:
                updated.append(segment)
                continue
            path = self._directory / tombstone_name(segment.name)
            marks = Tombstones.read(path, segment.documents)
            for ordinal in ordinals:
                marks.delete(ordinal)
            marks.write(path)
            updated.append(
                SegmentInfo(
                    name=segment.name,
                    documents=segment.documents,
                    checksum=segment.checksum,
                    deleted=marks.deleted_count,
                )
            )
            changed = True
        return updated if changed else []

    def _publish(self, segments: list[SegmentInfo]) -> None:
        self._manifest = self._manifest.with_segments(segments, self._next)
        manifest_file.publish(self._directory, self._manifest)

    def close(self) -> None:
        """Flush anything still buffered or deleted."""
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
        return len(self._buffer)

    @property
    def segments(self) -> tuple[str, ...]:
        """Return the names of the published segments, oldest first."""
        return tuple(segment.name for segment in self._manifest.segments)

    @property
    def manifest(self) -> Manifest:
        """Return the manifest as it now stands."""
        return self._manifest
