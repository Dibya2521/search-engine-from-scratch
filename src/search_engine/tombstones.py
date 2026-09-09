"""Deleted documents, recorded beside a segment rather than removed from it.

A segment is immutable, so a delete cannot take a posting out of it. What it can
do is record that a document is no longer visible, and let queries filter it
out. The space comes back when segments are merged and the deleted documents are
simply not carried across. This is why deleting from a real search engine does
not free disk immediately.

The bitset is over each segment's document *ordinals*, meaning the position of
an identifier in that segment's sorted list, rather than over the identifiers
themselves. Identifiers are arbitrary and may be sparse, so a bitset over them
could be enormous for a segment holding three documents. Ordinals are dense by
construction, so the bitset is always one bit per document.

This file is the one mutable thing in a design built on immutability. That is a
real exception and worth naming: it never changes what a posting means, only
whether a document is visible, so none of the properties immutability buys are
lost. It is still replaced atomically rather than edited in place.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pathlib import Path

TOMBSTONE_SUFFIX: Final = ".del"

_BITS_PER_BYTE: Final = 8


class TombstoneFormatError(ValueError):
    """Raised when a tombstone file is not the size its segment requires."""


def tombstone_name(segment: str) -> str:
    """Return the tombstone file name that belongs to a segment file name."""
    return segment.removesuffix(".seg") + TOMBSTONE_SUFFIX


class Tombstones:
    """A bitset of deleted documents for one segment."""

    def __init__(self, bits: bytearray, document_count: int) -> None:
        """Take the raw bitset and the number of documents it covers."""
        self._bits = bits
        self._count = document_count

    @classmethod
    def empty(cls, document_count: int) -> Tombstones:
        """Return a set with nothing deleted."""
        return cls(bytearray(_byte_length(document_count)), document_count)

    @classmethod
    def read(cls, path: Path, document_count: int) -> Tombstones:
        """Read a segment's tombstones, or an empty set if the file is absent.

        Raises:
            TombstoneFormatError: If the file is not the length the segment's
                document count requires, which means the two do not belong
                together.
        """
        if not path.is_file():
            return cls.empty(document_count)
        data = path.read_bytes()
        expected = _byte_length(document_count)
        if len(data) != expected:
            message = (
                f"tombstone file is {len(data)} bytes, but a segment of "
                f"{document_count} documents needs {expected}"
            )
            raise TombstoneFormatError(message)
        return cls(bytearray(data), document_count)

    def write(self, path: Path) -> None:
        """Replace the tombstone file atomically.

        Rewritten in place is what this is not. A reader arriving during a
        partial write would see a bitset that is half one state and half
        another, and would hide or reveal the wrong documents.
        """
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(self._bits)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def delete(self, ordinal: int) -> bool:
        """Mark a document deleted, and return whether that changed anything.

        Deleting one that is already deleted is not an error. Ingestion is
        at-least-once, so the same instruction arriving twice has to be safe.

        Raises:
            IndexError: If the ordinal is not in this segment.
        """
        self._check(ordinal)
        index, mask = divmod(ordinal, _BITS_PER_BYTE)
        bit = 1 << mask
        if self._bits[index] & bit:
            return False
        self._bits[index] |= bit
        return True

    def is_deleted(self, ordinal: int) -> bool:
        """Return whether a document has been deleted.

        Raises:
            IndexError: If the ordinal is not in this segment.
        """
        self._check(ordinal)
        index, mask = divmod(ordinal, _BITS_PER_BYTE)
        return bool(self._bits[index] & (1 << mask))

    def _check(self, ordinal: int) -> None:
        if not 0 <= ordinal < self._count:
            message = f"ordinal {ordinal} is not in a segment of {self._count}"
            raise IndexError(message)

    @property
    def deleted_count(self) -> int:
        """Return how many documents are deleted."""
        return sum(byte.bit_count() for byte in self._bits)

    @property
    def document_count(self) -> int:
        """Return how many documents this covers, deleted or not."""
        return self._count

    @property
    def any_deleted(self) -> bool:
        """Return whether anything at all is deleted.

        Worth asking before filtering a query, because the common case is that
        nothing is and the filter can be skipped entirely.
        """
        return any(self._bits)

    def __bytes__(self) -> bytes:
        """Return the raw bitset."""
        return bytes(self._bits)


def _byte_length(document_count: int) -> int:
    return (document_count + _BITS_PER_BYTE - 1) // _BITS_PER_BYTE
