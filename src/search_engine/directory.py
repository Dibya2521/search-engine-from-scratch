"""Every published segment in a directory, read as one index.

A segmented index is many files, and a query has to see them as one. This
opens every segment the manifest names, merges what they hold, and hides what
the tombstones say is gone.

It satisfies the same protocol as an index held in memory or a single segment
on disk, so querying and ranking work against it without knowing that more than
one file is involved.

**A segment the manifest does not name is not opened**, whatever is sitting in
the directory. That is what makes a half-written segment left by a crash
harmless rather than corrupting.

The generation the directory was opened at is recorded and never changes.
Publishing a new one leaves this reader looking at the old files, so a query in
flight cannot see a directory change underneath it and a cache keyed by
generation can never be stale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from search_engine import manifest as manifest_file
from search_engine.segment import SegmentReader
from search_engine.tombstones import Tombstones, tombstone_name

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path
    from types import TracebackType


class _Part:
    """One segment, and the record of what has been deleted from it."""

    __slots__ = ("deleted", "live_ids", "reader")

    def __init__(self, reader: SegmentReader, deleted: Tombstones) -> None:
        self.reader = reader
        self.deleted = deleted
        self.live_ids = frozenset(
            document_id
            for ordinal, document_id in enumerate(reader.document_ids)
            if not deleted.is_deleted(ordinal)
        )


class DirectoryIndex:
    """The published segments of a directory, searched as a single index."""

    def __init__(self, directory: Path, *, verify: bool = True) -> None:
        """Open every segment the manifest names.

        Raises:
            ManifestFormatError: If the manifest is malformed.
            SegmentFormatError: If a segment it names is missing or damaged.
        """
        self._manifest = manifest_file.read(directory)
        self._parts: list[_Part] = []
        try:
            for segment in self._manifest.segments:
                reader = SegmentReader(directory / segment.name, verify=verify)
                marks = Tombstones.read(
                    directory / tombstone_name(segment.name), segment.documents
                )
                self._parts.append(_Part(reader, marks))
        except Exception:
            self.close()
            raise
        self._lengths = {
            document_id: part.reader.document_length(document_id)
            for part in self._parts
            for document_id in part.live_ids
        }
        self._vocabulary: frozenset[str] | None = None

    def close(self) -> None:
        """Close every segment. Safe to call more than once."""
        for part in self._parts:
            part.reader.close()
        self._parts = []

    def __enter__(self) -> Self:
        """Return the index, so it can be used as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the segments when the block ends."""
        self.close()

    def postings(self, term: str) -> Mapping[int, Sequence[int]]:
        """Return the documents containing a term, across every segment.

        Deleted documents are filtered out here rather than at any later stage,
        because a caller that saw them would count them and every weight
        derived from that count would be wrong.
        """
        return {
            document_id: positions
            for part in self._parts
            for document_id, positions in part.reader.postings(term).items()
            if document_id in part.live_ids
        }

    def document_frequency(self, term: str) -> int:
        """Return how many live documents contain a term.

        A segment with nothing deleted can answer from its dictionary without
        reading a posting, which is the common case and worth the branch.
        """
        total = 0
        for part in self._parts:
            if part.deleted.any_deleted:
                total += sum(
                    1
                    for document_id in part.reader.postings(term)
                    if document_id in part.live_ids
                )
            else:
                total += part.reader.document_frequency(term)
        return total

    def document_length(self, document_id: int) -> int:
        """Return a document's length in analysed tokens."""
        return self._lengths.get(document_id, 0)

    def __contains__(self, term: str) -> bool:
        """Return whether any live document holds this term."""
        return self.document_frequency(term) > 0

    @property
    def generation(self) -> int:
        """Return the manifest generation this reader was opened at."""
        return self._manifest.generation

    @property
    def segment_count(self) -> int:
        """Return how many segments are open."""
        return len(self._parts)

    @property
    def document_count(self) -> int:
        """Return how many documents are visible, deleted ones excluded."""
        return len(self._lengths)

    @property
    def vocabulary_size(self) -> int:
        """Return how many distinct terms the segments hold between them.

        This counts what the files contain, so a term left behind by a deleted
        document is still counted until a merge drops it. Establishing that a
        term has no live document left would mean reading its postings, and
        that is a high price for a number nothing scores with.

        Counting the union means visiting every term of every segment, so the
        answer is computed once and kept. It does not change, because the
        segments this reader holds do not.
        """
        return len(self._terms())

    @property
    def terms(self) -> Iterable[str]:
        """Return every distinct term across the segments, in sorted order."""
        return sorted(self._terms())

    @property
    def document_ids(self) -> Iterable[int]:
        """Return every visible document identifier."""
        return tuple(self._lengths)

    def _terms(self) -> frozenset[str]:
        if self._vocabulary is None:
            self._vocabulary = frozenset(
                term for part in self._parts for term in part.reader.terms
            )
        return self._vocabulary
