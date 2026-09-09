"""What has been accepted but is not yet in a segment.

A document sitting in the writer's buffer exists only in memory. If the process
dies it is gone, with no error, no log line and nothing to notice: the worst
class of failure a data system has, because the loss is silent and can go
unnoticed for months.

So a write is recorded here before it is acknowledged, and the log is discarded
only once the documents it holds are durable in a published segment. Recovery
is replaying whatever the log still contains.

**Replaying is safe because indexing is an upsert.** A crash between publishing
a segment and discarding the log replays documents that are already indexed,
and each one simply replaces itself. That is not luck; it is why upsert was
built before this was.

The record format puts the length first and the checksum last:

    vbyte(payload length)
    payload
    uint32 checksum of the payload

The reader needs the length to know how much to read, and the checksum has to
cover everything it just read. A record whose length runs past the end of the
file, or whose checksum does not match, is the torn tail of a crash, and replay
stops there rather than guessing.
"""

from __future__ import annotations

import os
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Final

from search_engine.codecs import CodecError, decode_at, encode_number

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import TracebackType
    from typing import IO, Self

_CHECKSUM_BYTES: Final = 4

LOG_NAME: Final = "wal.log"


class Kind(IntEnum):
    """What a log record says happened."""

    ADD = 0
    DELETE = 1


@dataclass(frozen=True, slots=True)
class Record:
    """One accepted operation, waiting to reach a segment."""

    kind: Kind
    document_id: int
    text: str = ""


class WriteAheadLog:
    """Records accepted operations before they are durable in a segment."""

    def __init__(self, path: Path, *, sync: bool = True) -> None:
        """Open the log for appending, creating it if it does not exist.

        ``sync`` forces each record to the disk before the call returns, which
        is what durability actually costs. Turning it off makes writing much
        faster and loses recent records on a crash, so it is for measuring the
        difference rather than for running an index.
        """
        self._path = path
        self._sync = sync
        self._handle: IO[bytes] | None = None

    def append(self, record: Record) -> None:
        """Record one operation, and do not return until it is durable."""
        payload = _encode(record)
        framed = encode_number(len(payload)) + payload
        framed += zlib.crc32(payload).to_bytes(_CHECKSUM_BYTES, "big")
        handle = self._open()
        handle.write(framed)
        handle.flush()
        if self._sync:
            os.fsync(handle.fileno())

    def replay(self) -> Iterator[Record]:
        """Yield every intact record, stopping at the first torn one.

        A partial record is what a crash mid-append leaves, and everything
        before it is still good. Stopping is the only safe response: the bytes
        after a torn record cannot be located, because the length that would
        locate them is the part that is missing.
        """
        if not self._path.is_file():
            return
        data = self._path.read_bytes()
        cursor = 0
        while cursor < len(data):
            record, cursor = _decode_one(data, cursor)
            if record is None:
                return
            yield record

    def truncate(self) -> None:
        """Discard the log, once everything in it is durable elsewhere."""
        self.close()
        self._path.unlink(missing_ok=True)

    def close(self) -> None:
        """Close the file. Safe to call more than once."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> Self:
        """Return the log, so it can be used as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the file when the block ends."""
        self.close()

    def _open(self) -> IO[bytes]:
        if self._handle is None:
            self._handle = self._path.open("ab")
        return self._handle


def _encode(record: Record) -> bytes:
    parts = [encode_number(int(record.kind)), encode_number(record.document_id)]
    if record.kind is Kind.ADD:
        encoded = record.text.encode("utf-8")
        parts.append(encode_number(len(encoded)))
        parts.append(encoded)
    return b"".join(parts)


def _decode_one(data: bytes, cursor: int) -> tuple[Record | None, int]:
    """Return the record starting at a cursor, or None if the tail is torn."""
    try:
        length, cursor = decode_at(data, cursor)
    except CodecError:
        return None, cursor
    end = cursor + length
    if end + _CHECKSUM_BYTES > len(data):
        return None, cursor
    payload = data[cursor:end]
    stored = int.from_bytes(data[end : end + _CHECKSUM_BYTES], "big")
    if zlib.crc32(payload) != stored:
        return None, cursor
    return _decode_payload(payload), end + _CHECKSUM_BYTES


def _decode_payload(payload: bytes) -> Record | None:
    try:
        kind, at = decode_at(payload, 0)
        document_id, at = decode_at(payload, at)
        if kind == Kind.DELETE:
            return Record(Kind.DELETE, document_id)
        if kind != Kind.ADD:
            return None
        length, at = decode_at(payload, at)
        text = payload[at : at + length]
        if len(text) != length:
            return None
    except (CodecError, UnicodeDecodeError):
        return None
    return Record(Kind.ADD, document_id, text.decode("utf-8"))
