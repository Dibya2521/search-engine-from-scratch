"""An index written as one immutable file, searchable without loading it.

The format this module writes exists to answer one question: where are the
postings for a term, without reading anything else? An index saved by
``persistence`` answers it only by decoding the whole file, which is why loading
one costs time and memory proportional to the index rather than to the query.

The file is laid out so that a reader can start at the end and work backwards:

    header       a magic string naming the format and its version
    fingerprint  the analysis configuration that produced the index
    postings     one block per term, in sorted term order
    dictionary   the terms, front coded, in blocks of TERMS_PER_BLOCK
    block index  the first term of each block and where the block starts
    documents    the document identifiers, and each one's length in tokens
    footer       fixed-width offsets to every section, then a checksum

The footer is last and fixed width so that a reader can seek to a known
distance from the end of the file, read it, and from there know where
everything else is. Putting it at the head would mean knowing the size of every
section before writing any of them, which forces either two passes over the
index or holding the whole file in memory.

Terms are sorted, so consecutive terms tend to share a prefix, and the
dictionary stores only what changed from the term before. That is cheap and it
makes a term unfindable without decoding every term before it, so the
dictionary is cut into blocks and each block starts afresh. A lookup binary
searches the block index, then scans one block of at most TERMS_PER_BLOCK
terms.

Two fields exist for work that comes later and are written now because adding
them afterwards would mean a new format version. Each document's length in
tokens is stored because a length-normalising scorer otherwise recovers it by
walking every posting of every term, which would read the whole file on the
first query. Each term's highest frequency in any one document is stored
because it gives an upper bound on that term's contribution to a score without
reading its postings at all.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from search_engine.analysis import fingerprint
from search_engine.codecs import (
    CodecError,
    decode_at,
    decode_sorted_at,
    encode_number,
    encode_sorted,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from search_engine.index import InvertedIndex

SEGMENT_HEADER: Final = b"search-engine-segment v1\n"

TERMS_PER_BLOCK: Final = 16

_FOOTER_FIELDS: Final = 8
_FIELD_BYTES: Final = 8
_CHECKSUM_BYTES: Final = 4
FOOTER_SIZE: Final = _FOOTER_FIELDS * _FIELD_BYTES + _CHECKSUM_BYTES


class SegmentFormatError(ValueError):
    """Raised when a segment file is not in the expected format."""


class SegmentCorruptError(SegmentFormatError):
    """Raised when a segment's checksum does not match its contents."""


class SegmentAnalyzerMismatchError(SegmentFormatError):
    """Raised when a segment was written by a different analysis configuration."""

    def __init__(self, stored: str, current: str) -> None:
        """Name both fingerprints, since the difference is the whole problem."""
        super().__init__(
            f"segment was built by analyzer {stored}, but this build is {current}"
        )
        self.stored = stored
        self.current = current


@dataclass(frozen=True, slots=True)
class TermEntry:
    """Where one term's postings are, and what can be known without reading them."""

    offset: int
    length: int
    document_frequency: int
    max_term_frequency: int


@dataclass(frozen=True, slots=True)
class Footer:
    """The offsets a reader needs before it can read anything else."""

    dictionary_offset: int
    dictionary_length: int
    block_index_offset: int
    block_index_length: int
    documents_offset: int
    documents_length: int
    document_count: int
    vocabulary_size: int


def write_segment(index: InvertedIndex, path: Path) -> None:
    """Write an in-memory index to a segment file."""
    path.write_bytes(encode_segment(index))


def encode_segment(index: InvertedIndex) -> bytes:
    """Return the bytes of a segment file for this index."""
    prologue = _encode_prologue()
    postings, entries = _encode_postings(index, len(prologue))
    dictionary, block_index = _encode_dictionary(entries)

    dictionary_offset = len(prologue) + len(postings)
    block_index_offset = dictionary_offset + len(dictionary)
    documents = _encode_documents(index)
    documents_offset = block_index_offset + len(block_index)

    footer = Footer(
        dictionary_offset=dictionary_offset,
        dictionary_length=len(dictionary),
        block_index_offset=block_index_offset,
        block_index_length=len(block_index),
        documents_offset=documents_offset,
        documents_length=len(documents),
        document_count=index.document_count,
        vocabulary_size=len(entries),
    )
    body = prologue + postings + dictionary + block_index + documents
    body += _encode_footer(footer)
    return body + zlib.crc32(body).to_bytes(_CHECKSUM_BYTES, "big")


def _encode_prologue() -> bytes:
    stamp = fingerprint().encode("ascii")
    return SEGMENT_HEADER + encode_number(len(stamp)) + stamp


def _encode_postings(
    index: InvertedIndex, start: int
) -> tuple[bytes, list[tuple[str, TermEntry]]]:
    """Encode every term's postings, and record where each one landed."""
    parts: list[bytes] = []
    entries: list[tuple[str, TermEntry]] = []
    offset = start
    for term in sorted(index.terms):
        postings = index.postings(term)
        block = _encode_term_postings(postings)
        parts.append(block)
        entries.append(
            (
                term,
                TermEntry(
                    offset=offset,
                    length=len(block),
                    document_frequency=len(postings),
                    max_term_frequency=max(
                        len(positions) for positions in postings.values()
                    ),
                ),
            )
        )
        offset += len(block)
    return b"".join(parts), entries


def _encode_term_postings(postings: Mapping[int, Sequence[int]]) -> bytes:
    """Encode one term's postings the way the whole-file format already does."""
    ordered = sorted(postings)
    parts = [encode_number(len(ordered)), encode_sorted(ordered)]
    for document_id in ordered:
        positions = list(postings[document_id])
        parts.append(encode_number(len(positions)))
        parts.append(encode_sorted(positions))
    return b"".join(parts)


def decode_term_postings(data: bytes, entry: TermEntry) -> dict[int, list[int]]:
    """Decode one term's postings from the bytes its entry points at.

    Reads only the entry's own range, which is the property the whole format
    exists to provide.

    Raises:
        SegmentFormatError: If the range does not hold well-formed postings.
    """
    try:
        return _decode_term_postings(data, entry)
    except CodecError as error:
        message = f"segment postings are malformed: {error}"
        raise SegmentFormatError(message) from error


def _decode_term_postings(data: bytes, entry: TermEntry) -> dict[int, list[int]]:
    cursor = entry.offset
    count, cursor = decode_at(data, cursor)
    document_ids, cursor = decode_sorted_at(data, cursor, count)
    postings: dict[int, list[int]] = {}
    for document_id in document_ids:
        length, cursor = decode_at(data, cursor)
        positions, cursor = decode_sorted_at(data, cursor, length)
        postings[document_id] = positions
    return postings


def decode_documents(data: bytes, footer: Footer) -> tuple[list[int], list[int]]:
    """Return the document identifiers and their lengths in tokens, in step.

    Raises:
        SegmentFormatError: If the documents section is malformed.
    """
    try:
        cursor = footer.documents_offset
        count, cursor = decode_at(data, cursor)
        document_ids, cursor = decode_sorted_at(data, cursor, count)
        lengths: list[int] = []
        for _ in document_ids:
            length, cursor = decode_at(data, cursor)
            lengths.append(length)
    except CodecError as error:
        message = f"segment documents section is malformed: {error}"
        raise SegmentFormatError(message) from error
    return document_ids, lengths


def _encode_dictionary(
    entries: Sequence[tuple[str, TermEntry]],
) -> tuple[bytes, bytes]:
    """Front code the terms in blocks, and index the blocks by their first term."""
    blocks: list[bytes] = []
    starts: list[tuple[bytes, int]] = []
    offset = 0
    for start in range(0, len(entries), TERMS_PER_BLOCK):
        block = _encode_block(entries[start : start + TERMS_PER_BLOCK])
        starts.append((entries[start][0].encode("utf-8"), offset))
        blocks.append(block)
        offset += len(block)
    return b"".join(blocks), _encode_block_index(starts)


def _encode_block(entries: Sequence[tuple[str, TermEntry]]) -> bytes:
    """Encode one block, whose first entry stores no shared prefix.

    That is what lets a block be decoded without decoding the block before it,
    which is what makes the block index binary searchable.
    """
    parts: list[bytes] = []
    previous_term = b""
    previous_offset = 0
    for term, entry in entries:
        encoded = term.encode("utf-8")
        shared = _shared_prefix(previous_term, encoded)
        parts.append(encode_number(shared))
        parts.append(encode_number(len(encoded) - shared))
        parts.append(encoded[shared:])
        parts.append(encode_number(entry.offset - previous_offset))
        parts.append(encode_number(entry.length))
        parts.append(encode_number(entry.document_frequency))
        parts.append(encode_number(entry.max_term_frequency))
        previous_term = encoded
        previous_offset = entry.offset
    return b"".join(parts)


def _shared_prefix(left: bytes, right: bytes) -> int:
    """Return how many leading bytes two terms have in common.

    Counted in bytes rather than characters, because a prefix measured in
    characters can end in the middle of a multi-byte code point.
    """
    limit = min(len(left), len(right))
    shared = 0
    while shared < limit and left[shared] == right[shared]:
        shared += 1
    return shared


def _encode_block_index(starts: Sequence[tuple[bytes, int]]) -> bytes:
    parts = [encode_number(len(starts))]
    for term, offset in starts:
        parts.append(encode_number(len(term)))
        parts.append(term)
        parts.append(encode_number(offset))
    return b"".join(parts)


def _encode_documents(index: InvertedIndex) -> bytes:
    document_ids = sorted(index.document_ids)
    lengths = _document_lengths(index)
    parts = [encode_number(len(document_ids)), encode_sorted(document_ids)]
    parts.extend(
        encode_number(lengths.get(document_id, 0)) for document_id in document_ids
    )
    return b"".join(parts)


def _document_lengths(index: InvertedIndex) -> dict[int, int]:
    """Return each document's length in analysed tokens.

    Costs one pass over the index at write time so that a reader never has to
    make the same pass at query time.
    """
    lengths: dict[int, int] = {}
    for term in index.terms:
        for document_id, positions in index.postings(term).items():
            lengths[document_id] = lengths.get(document_id, 0) + len(positions)
    return lengths


def _encode_footer(footer: Footer) -> bytes:
    fields = (
        footer.dictionary_offset,
        footer.dictionary_length,
        footer.block_index_offset,
        footer.block_index_length,
        footer.documents_offset,
        footer.documents_length,
        footer.document_count,
        footer.vocabulary_size,
    )
    return b"".join(value.to_bytes(_FIELD_BYTES, "big") for value in fields)


def read_footer(data: bytes) -> Footer:
    """Read the footer of a segment file, after checking the file is one.

    Raises:
        SegmentFormatError: If the file is too short, or does not start with the
            expected header.
        SegmentCorruptError: If the checksum does not match the contents.
        SegmentAnalyzerMismatchError: If a different analysis configuration
            produced the file.
    """
    if not data.startswith(SEGMENT_HEADER):
        message = "not a segment file: the header is missing"
        raise SegmentFormatError(message)
    if len(data) < len(SEGMENT_HEADER) + FOOTER_SIZE:
        message = "segment file ends before its footer"
        raise SegmentFormatError(message)

    stored_checksum = int.from_bytes(data[-_CHECKSUM_BYTES:], "big")
    if zlib.crc32(data[:-_CHECKSUM_BYTES]) != stored_checksum:
        message = "segment file is corrupt: the checksum does not match its contents"
        raise SegmentCorruptError(message)

    _check_fingerprint(data)
    start = len(data) - FOOTER_SIZE
    fields = [
        int.from_bytes(
            data[start + n * _FIELD_BYTES : start + (n + 1) * _FIELD_BYTES], "big"
        )
        for n in range(_FOOTER_FIELDS)
    ]
    return Footer(*fields)


def _check_fingerprint(data: bytes) -> None:
    try:
        length, cursor = decode_at(data, len(SEGMENT_HEADER))
    except CodecError as error:
        message = "segment file ends before the analyzer fingerprint"
        raise SegmentFormatError(message) from error
    stored = bytes(data[cursor : cursor + length]).decode("utf-8", errors="replace")
    current = fingerprint()
    if stored != current:
        raise SegmentAnalyzerMismatchError(stored, current)


class TermDictionary:
    """Maps a term to the location of its postings, without loading them.

    The block index is decoded when the dictionary is opened, because every
    lookup needs it and it holds one entry per TERMS_PER_BLOCK terms. The
    dictionary blocks themselves are decoded one at a time, only when a lookup
    lands in them.
    """

    def __init__(self, data: bytes, footer: Footer) -> None:
        """Take the whole segment and the footer that says where things are."""
        self._data = data
        self._footer = footer
        self._starts = _decode_block_index(
            data, footer.block_index_offset, footer.block_index_length
        )

    @classmethod
    def read(cls, data: bytes) -> TermDictionary:
        """Open the dictionary of a segment file.

        Raises:
            SegmentFormatError: If the file is not a well-formed segment.
        """
        return cls(data, read_footer(data))

    def __len__(self) -> int:
        """Return the number of terms."""
        return self._footer.vocabulary_size

    def lookup(self, term: str) -> TermEntry | None:
        """Return where a term's postings are, or None if it is not present."""
        encoded = term.encode("utf-8")
        block = self._find_block(encoded)
        if block is None:
            return None
        for candidate, entry in self._decode_block(block):
            if candidate == encoded:
                return entry
        return None

    def terms(self) -> Iterator[str]:
        """Yield every term in sorted order."""
        for block in range(len(self._starts)):
            for encoded, _ in self._decode_block(block):
                yield encoded.decode("utf-8")

    def _find_block(self, encoded: bytes) -> int | None:
        """Return the last block whose first term is at or before this one."""
        low, high = 0, len(self._starts)
        while low < high:
            middle = (low + high) // 2
            if self._starts[middle][0] <= encoded:
                low = middle + 1
            else:
                high = middle
        return low - 1 if low else None

    def _bounds(self, block: int) -> tuple[int, int]:
        start = self._footer.dictionary_offset + self._starts[block][1]
        if block + 1 < len(self._starts):
            end = self._footer.dictionary_offset + self._starts[block + 1][1]
        else:
            end = self._footer.dictionary_offset + self._footer.dictionary_length
        return start, end

    def _decode_block(self, block: int) -> list[tuple[bytes, TermEntry]]:
        cursor, end = self._bounds(block)
        try:
            return _decode_entries(self._data, cursor, end)
        except (CodecError, IndexError) as error:
            message = f"segment dictionary is malformed: {error}"
            raise SegmentFormatError(message) from error


def _decode_entry(
    data: bytes, cursor: int, previous_term: bytes, previous_offset: int
) -> tuple[bytes, TermEntry, int]:
    """Decode one front-coded entry, given the term and offset before it."""
    shared, cursor = decode_at(data, cursor)
    suffix_length, cursor = decode_at(data, cursor)
    term = previous_term[:shared] + bytes(data[cursor : cursor + suffix_length])
    cursor += suffix_length
    delta, cursor = decode_at(data, cursor)
    length, cursor = decode_at(data, cursor)
    frequency, cursor = decode_at(data, cursor)
    highest, cursor = decode_at(data, cursor)
    entry = TermEntry(previous_offset + delta, length, frequency, highest)
    return term, entry, cursor


def _decode_entries(
    data: bytes, cursor: int, end: int
) -> list[tuple[bytes, TermEntry]]:
    entries: list[tuple[bytes, TermEntry]] = []
    term = b""
    offset = 0
    while cursor < end:
        term, entry, cursor = _decode_entry(data, cursor, term, offset)
        offset = entry.offset
        entries.append((term, entry))
    return entries


def _decode_block_index(
    data: bytes, offset: int, length: int
) -> list[tuple[bytes, int]]:
    cursor = offset
    end = offset + length
    try:
        count, cursor = decode_at(data, cursor)
        starts: list[tuple[bytes, int]] = []
        for _ in range(count):
            term_length, cursor = decode_at(data, cursor)
            term = bytes(data[cursor : cursor + term_length])
            cursor += term_length
            block_offset, cursor = decode_at(data, cursor)
            starts.append((term, block_offset))
    except CodecError as error:
        message = f"segment block index is malformed: {error}"
        raise SegmentFormatError(message) from error
    if cursor > end:
        message = "segment block index runs past its own length"
        raise SegmentFormatError(message)
    return starts
