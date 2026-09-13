"""An index written as one immutable file, searchable without loading it.

The format this module writes exists to answer one question: where are the
postings for a term, without reading anything else? An index saved by
``persistence`` answers it only by decoding the whole file, which is why loading
one costs time and memory proportional to the index rather than to the query.

The file is laid out so that a reader can start at the end and work backwards:

    header       a magic string naming the format and its version
    fingerprint  the analysis configuration that produced the index
    postings     one section per term, in sorted term order, cut into blocks
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

A term's postings are cut into blocks of POSTINGS_BLOCK_SIZE documents, and a
table ahead of them holds one row per block: the block's last document
identifier, how many documents it holds, the highest term frequency inside it,
and its length in bytes. The table is why this format has a second version. It
makes a block reachable without decoding the blocks before it, because delta
encoding restarts at every block boundary, and it gives a scorer a bound per
block instead of one for the whole term. A single bound is set by the term's
most extreme document, which is far too loose to prune with when that document
is an outlier.

Version 1 carried a separate skip list of isqrt(n) entries instead. Block
boundaries answer the same question, so the skip list was removed rather than
kept beside them, and version 1 files are refused rather than read.
"""

from __future__ import annotations

import io
import mmap
import zlib
from bisect import bisect_left
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NoReturn, Self

from search_engine.analysis import fingerprint
from search_engine.cache import MISSING, POSTINGS_CACHE_SIZE, LruCache
from search_engine.codecs import (
    Bytelike,
    CodecError,
    decode_at,
    decode_sorted_at,
    encode_number,
    encode_sorted,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from pathlib import Path
    from typing import IO

    from search_engine.index import InvertedIndex
from search_engine.metrics import REGISTRY

SEGMENT_MAGIC: Final = b"search-engine-segment v"
SEGMENT_VERSION: Final = b"2"
SEGMENT_HEADER: Final = SEGMENT_MAGIC + SEGMENT_VERSION + b"\n"

TERMS_PER_BLOCK: Final = 16

# Lucene's default, and the same trade either way: larger blocks mean a smaller
# table and looser per-block bounds, smaller blocks the reverse.
POSTINGS_BLOCK_SIZE: Final = 128

SEGMENTS_OPEN: Final = REGISTRY.gauge(
    "segments_open", "Segment files mapped and not yet closed."
)
POSTINGS_CACHE_HITS: Final = REGISTRY.counter(
    "postings_cache_hits_total", "Postings answered from a segment's own cache."
)
POSTINGS_CACHE_MISSES: Final = REGISTRY.counter(
    "postings_cache_misses_total", "Postings that had to be decoded from the file."
)

_FOOTER_FIELDS: Final = 8
_FIELD_BYTES: Final = 8
_CHECKSUM_BYTES: Final = 4
FOOTER_SIZE: Final = _FOOTER_FIELDS * _FIELD_BYTES + _CHECKSUM_BYTES


class SegmentFormatError(ValueError):
    """Raised when a segment file is not in the expected format."""


class SegmentCorruptError(SegmentFormatError):
    """Raised when a segment's checksum does not match its contents."""


class SegmentVersionError(SegmentFormatError):
    """Raised when a segment was written in a different version of the format."""

    def __init__(self, stored: str, current: str) -> None:
        """Name both versions, since the only way forward is to rebuild."""
        super().__init__(
            f"segment is format version {stored}, but this build reads "
            f"version {current}: rebuild the index"
        )
        self.stored = stored
        self.current = current


class SegmentAnalyzerMismatchError(SegmentFormatError):
    """Raised when a segment was written by a different analysis configuration."""

    def __init__(self, stored: str, current: str) -> None:
        """Name both fingerprints, since the difference is what has to be fixed."""
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
class PostingsBlock:
    """One block of a term's postings, and what is known without decoding it."""

    last_document: int
    document_count: int
    max_term_frequency: int
    offset: int
    length: int


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


class SegmentBuilder:
    """Writes a segment one term at a time, in ascending term order.

    Postings go to the sink as each term arrives, so building a segment costs
    memory proportional to the vocabulary and the document count rather than to
    the postings, which are the overwhelming majority of the bytes. That is what
    lets a merge combine segments far larger than memory.

    Terms must arrive sorted, because the dictionary is front coded against the
    term before and binary searched afterwards. Nothing checks it on every call;
    a merge produces them in order by construction and a caller that does not is
    writing a file it cannot read back.
    """

    def __init__(self, sink: IO[bytes], document_ids: Sequence[int]) -> None:
        """Take an open binary sink and the identifiers the segment will hold."""
        self._sink = sink
        self._document_ids = sorted(document_ids)
        self._lengths: dict[int, int] = {}
        self._entries: list[tuple[str, TermEntry]] = []
        self._checksum = 0
        self._offset = 0
        self._write(_encode_prologue())

    def add(self, term: str, postings: Mapping[int, Sequence[int]]) -> None:
        """Append one term's postings and record where they landed."""
        encoded = _encode_term_postings(postings)
        self._entries.append(
            (
                term,
                TermEntry(
                    offset=self._offset,
                    length=len(encoded),
                    document_frequency=len(postings),
                    max_term_frequency=max(
                        len(positions) for positions in postings.values()
                    ),
                ),
            )
        )
        for document_id, positions in postings.items():
            self._lengths[document_id] = self._lengths.get(document_id, 0) + len(
                positions
            )
        self._write(encoded)

    def finish(self) -> None:
        """Write the dictionary, the documents and the footer."""
        dictionary, block_index = _encode_dictionary(self._entries)
        dictionary_offset = self._offset
        self._write(dictionary)
        block_index_offset = self._offset
        self._write(block_index)
        documents_offset = self._offset
        documents = _encode_document_section(self._document_ids, self._lengths)
        self._write(documents)
        self._write(
            _encode_footer(
                Footer(
                    dictionary_offset=dictionary_offset,
                    dictionary_length=len(dictionary),
                    block_index_offset=block_index_offset,
                    block_index_length=len(block_index),
                    documents_offset=documents_offset,
                    documents_length=len(documents),
                    document_count=len(self._document_ids),
                    vocabulary_size=len(self._entries),
                )
            )
        )
        self._sink.write(self._checksum.to_bytes(_CHECKSUM_BYTES, "big"))

    def _write(self, data: bytes) -> None:
        self._sink.write(data)
        self._checksum = zlib.crc32(data, self._checksum)
        self._offset += len(data)


def write_segment(index: InvertedIndex, path: Path) -> None:
    """Write an in-memory index to a segment file."""
    with path.open("wb") as sink:
        _build(index, sink)


def encode_segment(index: InvertedIndex) -> bytes:
    """Return the bytes of a segment file for this index."""
    sink: IO[bytes] = io.BytesIO()
    _build(index, sink)
    sink.seek(0)
    return sink.read()


def _build(index: InvertedIndex, sink: IO[bytes]) -> None:
    builder = SegmentBuilder(sink, sorted(index.document_ids))
    for term in sorted(index.terms):
        builder.add(term, index.postings(term))
    builder.finish()


def _encode_prologue() -> bytes:
    stamp = fingerprint().encode("ascii")
    return SEGMENT_HEADER + encode_number(len(stamp)) + stamp


def _encode_term_postings(postings: Mapping[int, Sequence[int]]) -> bytes:
    """Encode one term's postings as a table of blocks followed by the blocks."""
    ordered = sorted(postings)
    groups = [
        ordered[at : at + POSTINGS_BLOCK_SIZE]
        for at in range(0, len(ordered), POSTINGS_BLOCK_SIZE)
    ]
    payloads = [_encode_postings_block(group, postings) for group in groups]
    return _encode_block_table(groups, payloads, postings) + b"".join(payloads)


def _encode_block_table(
    groups: Sequence[Sequence[int]],
    payloads: Sequence[bytes],
    postings: Mapping[int, Sequence[int]],
) -> bytes:
    """Encode one row per block: last identifier, size, highest frequency, bytes.

    The last identifier is what makes the table a skip list. Without it a reader
    would have to decode a block to find out whether the document it wants is
    inside, which is the cost the table exists to remove.
    """
    parts = [encode_number(len(groups))]
    previous = 0
    for group, payload in zip(groups, payloads, strict=True):
        parts.append(encode_number(group[-1] - previous))
        parts.append(encode_number(len(group)))
        parts.append(
            encode_number(max(len(postings[document_id]) for document_id in group))
        )
        parts.append(encode_number(len(payload)))
        previous = group[-1]
    return b"".join(parts)


def _encode_postings_block(
    group: Sequence[int], postings: Mapping[int, Sequence[int]]
) -> bytes:
    """Encode one block's identifiers and then its positions.

    Delta encoding restarts here rather than carrying on from the block before,
    which is what lets a reader jump straight to this block.
    """
    parts = [encode_sorted(group)]
    for document_id in group:
        positions = list(postings[document_id])
        parts.append(encode_number(len(positions)))
        parts.append(encode_sorted(positions))
    return b"".join(parts)


def read_blocks(data: Bytelike, entry: TermEntry) -> list[PostingsBlock]:
    """Return one term's block table, without decoding any postings.

    Raises:
        SegmentFormatError: If the table is malformed.
    """
    try:
        return _read_blocks(data, entry)
    except CodecError as error:
        message = f"segment postings are malformed: {error}"
        raise SegmentFormatError(message) from error


def _read_blocks(data: Bytelike, entry: TermEntry) -> list[PostingsBlock]:
    count, cursor = decode_at(data, entry.offset)
    rows: list[tuple[int, int, int, int]] = []
    last = 0
    for _ in range(count):
        gap, cursor = decode_at(data, cursor)
        documents, cursor = decode_at(data, cursor)
        highest, cursor = decode_at(data, cursor)
        length, cursor = decode_at(data, cursor)
        last += gap
        rows.append((last, documents, highest, length))
    return _locate_blocks(rows, cursor)


def _locate_blocks(
    rows: Sequence[tuple[int, int, int, int]], offset: int
) -> list[PostingsBlock]:
    """Give each table row the byte offset its payload starts at."""
    blocks: list[PostingsBlock] = []
    for last_document, documents, highest, length in rows:
        blocks.append(PostingsBlock(last_document, documents, highest, offset, length))
        offset += length
    return blocks


def block_identifiers(data: Bytelike, block: PostingsBlock) -> list[int]:
    """Return one block's document identifiers, leaving its positions alone."""
    identifiers, _ = decode_sorted_at(data, block.offset, block.document_count)
    return identifiers


def advance_to(data: Bytelike, entry: TermEntry, target: int) -> int | None:
    """Return the first document identifier at or after target, or None.

    Skips whole blocks on the table's last-identifier column, then binary
    searches inside the one block that can hold the answer.

    Raises:
        SegmentFormatError: If the postings are malformed.
    """
    try:
        return _advance_to(data, entry, target)
    except CodecError as error:
        message = f"segment postings are malformed: {error}"
        raise SegmentFormatError(message) from error


def _advance_to(data: Bytelike, entry: TermEntry, target: int) -> int | None:
    for block in _read_blocks(data, entry):
        if block.last_document >= target:
            identifiers = block_identifiers(data, block)
            return identifiers[bisect_left(identifiers, target)]
    return None


def decode_term_postings(data: Bytelike, entry: TermEntry) -> dict[int, list[int]]:
    """Decode every block of one term's postings.

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


def _decode_term_postings(data: Bytelike, entry: TermEntry) -> dict[int, list[int]]:
    postings: dict[int, list[int]] = {}
    for block in _read_blocks(data, entry):
        postings.update(_decode_postings_block(data, block))
    return postings


def _decode_postings_block(
    data: Bytelike, block: PostingsBlock
) -> dict[int, list[int]]:
    document_ids, cursor = decode_sorted_at(data, block.offset, block.document_count)
    postings: dict[int, list[int]] = {}
    for document_id in document_ids:
        length, cursor = decode_at(data, cursor)
        positions, cursor = decode_sorted_at(data, cursor, length)
        postings[document_id] = positions
    return postings


def decode_documents(data: Bytelike, footer: Footer) -> tuple[list[int], list[int]]:
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


def _encode_document_section(
    document_ids: Sequence[int], lengths: Mapping[int, int]
) -> bytes:
    """Encode the identifiers and each one's length in analysed tokens.

    Lengths are stored so that a scorer with a length prior never has to
    recover them by walking every posting, which against a mapped file would
    read the whole thing on the first query.
    """
    parts = [encode_number(len(document_ids)), encode_sorted(list(document_ids))]
    parts.extend(
        encode_number(lengths.get(document_id, 0)) for document_id in document_ids
    )
    return b"".join(parts)


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


def read_footer(data: Bytelike, *, verify: bool = True) -> Footer:
    """Read the footer of a segment file, after checking the file is one.

    Verifying the checksum reads every byte of the file, which for a
    memory-mapped segment means faulting in every page it was opened to avoid
    touching. Pass ``verify=False`` to skip it and check integrity separately,
    accepting that a corrupt file is then found when a query stumbles into the
    damage rather than when the file is opened.

    Raises:
        SegmentFormatError: If the file is too short, or does not start with the
            expected header.
        SegmentCorruptError: If the checksum does not match the contents.
        SegmentAnalyzerMismatchError: If a different analysis configuration
            produced the file.
    """
    if data[: len(SEGMENT_HEADER)] != SEGMENT_HEADER:
        _refuse_header(data)
    if len(data) < len(SEGMENT_HEADER) + FOOTER_SIZE:
        message = "segment file ends before its footer"
        raise SegmentFormatError(message)

    if verify:
        verify_checksum(data)

    _check_fingerprint(data)
    start = len(data) - FOOTER_SIZE
    fields = [
        int.from_bytes(
            data[start + n * _FIELD_BYTES : start + (n + 1) * _FIELD_BYTES], "big"
        )
        for n in range(_FOOTER_FIELDS)
    ]
    return Footer(*fields)


def _refuse_header(data: Bytelike) -> NoReturn:
    """Refuse a file this reader cannot read, saying which of the two it is.

    Raises:
        SegmentVersionError: If it is a segment written in another version.
        SegmentFormatError: If it is not a segment file at all.
    """
    if data[: len(SEGMENT_MAGIC)] != SEGMENT_MAGIC:
        message = "not a segment file: the header is missing"
        raise SegmentFormatError(message)
    stored = bytes(data[len(SEGMENT_MAGIC) : len(SEGMENT_HEADER)])
    raise SegmentVersionError(
        stored.decode("ascii", errors="replace").strip(),
        SEGMENT_VERSION.decode("ascii"),
    )


def stored_checksum(data: Bytelike) -> int:
    """Return the checksum a segment file carries, without recomputing it."""
    return int.from_bytes(data[-_CHECKSUM_BYTES:], "big")


def verify_checksum(data: Bytelike) -> None:
    """Check a segment against its stored checksum.

    Reads the whole file, so it is a deliberate act rather than something a
    reader does on every open.

    Raises:
        SegmentCorruptError: If the checksum does not match the contents.
    """
    stored = int.from_bytes(data[-_CHECKSUM_BYTES:], "big")
    if zlib.crc32(data[:-_CHECKSUM_BYTES]) != stored:
        message = "segment file is corrupt: the checksum does not match its contents"
        raise SegmentCorruptError(message)


def _check_fingerprint(data: Bytelike) -> None:
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

    def __init__(self, data: Bytelike, footer: Footer) -> None:
        """Take the whole segment and the footer that says where things are."""
        self._data = data
        self._footer = footer
        self._starts = _decode_block_index(
            data, footer.block_index_offset, footer.block_index_length
        )

    @classmethod
    def read(cls, data: Bytelike) -> TermDictionary:
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
    data: Bytelike, cursor: int, previous_term: bytes, previous_offset: int
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
    data: Bytelike, cursor: int, end: int
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
    data: Bytelike, offset: int, length: int
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


class SegmentReader:
    """Reads postings out of a segment file without loading it.

    The block index and the document identifiers are read when the segment is
    opened, because every query needs them and both are small: the block index
    holds one entry per TERMS_PER_BLOCK terms. Postings are decoded on demand,
    so a query faults in only the pages holding the terms it actually asked
    about.

    Verifying the checksum reads every page of the file and so undoes that, and
    it is the only way to know the file is intact before trusting an answer, so
    the choice is left to the caller.

    The file stays open for the life of the mapping. Closing it first works on
    Unix and fails on Windows, and a mapped file cannot be replaced or deleted
    on Windows at all, so a reader that outlives its usefulness blocks the next
    write.
    """

    def __init__(self, path: Path, *, verify: bool = True, cache: bool = True) -> None:
        """Open and map a segment file.

        Caching decoded postings pays when terms repeat and is pure
        overhead when they do not. A merge reads every term exactly once,
        so it opens its inputs with caching off.

        Raises:
            SegmentFormatError: If the file is not a well-formed segment.
        """
        self._file = path.open("rb")
        try:
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:
            self._file.close()
            message = f"segment file is empty: {path}"
            raise SegmentFormatError(message) from None
        self._view = memoryview(self._map)
        # Counted here rather than once the reader is built, so a constructor
        # that fails after this point and calls close() balances the gauge.
        SEGMENTS_OPEN.increment()
        try:
            self._footer = read_footer(self._view, verify=verify)
            self._dictionary = TermDictionary(self._view, self._footer)
            ids, lengths = decode_documents(self._view, self._footer)
        except SegmentFormatError:
            self.close()
            raise
        self._document_ids = ids
        self._lengths = dict(zip(ids, lengths, strict=True))
        self._cache: LruCache[str, dict[int, list[int]]] | None = (
            LruCache(POSTINGS_CACHE_SIZE) if cache else None
        )

    def close(self) -> None:
        """Release the mapping and the file. Safe to call more than once."""
        if self._map.closed:
            return
        self._view.release()
        self._map.close()
        self._file.close()
        SEGMENTS_OPEN.decrement()

    def __enter__(self) -> Self:
        """Return the reader, so it can be used as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the reader when the block ends."""
        self.close()

    def verify(self) -> None:
        """Check the whole file against its checksum.

        Raises:
            SegmentCorruptError: If the checksum does not match the contents.
        """
        verify_checksum(self._view)

    def postings(self, term: str) -> Mapping[int, Sequence[int]]:
        """Return the documents containing a term, and where in each it occurs.

        A term nobody holds caches its empty result too, so a query for a word
        the segment does not have pays one dictionary lookup rather than one
        per repeat.
        """
        if self._cache is None:
            return self._decode(term)
        cached = self._cache.get(term)
        if cached is not MISSING:
            POSTINGS_CACHE_HITS.increment()
            return cached
        POSTINGS_CACHE_MISSES.increment()
        decoded = self._decode(term)
        self._cache.put(term, decoded)
        return decoded

    def _decode(self, term: str) -> dict[int, list[int]]:
        entry = self._dictionary.lookup(term)
        if entry is None:
            return {}
        return decode_term_postings(self._view, entry)

    def document_frequency(self, term: str) -> int:
        """Return how many documents contain a term."""
        entry = self._dictionary.lookup(term)
        return 0 if entry is None else entry.document_frequency

    def max_term_frequency(self, term: str) -> int:
        """Return the term's highest frequency in any one document.

        Read from the dictionary, so it costs no postings read. It bounds what
        the term can contribute to a document's score.
        """
        entry = self._dictionary.lookup(term)
        return 0 if entry is None else entry.max_term_frequency

    def document_length(self, document_id: int) -> int:
        """Return a document's length in analysed tokens."""
        return self._lengths.get(document_id, 0)

    def __contains__(self, term: str) -> bool:
        """Return whether a term is in the vocabulary."""
        return self._dictionary.lookup(term) is not None

    @property
    def document_count(self) -> int:
        """Return how many documents the segment holds."""
        return self._footer.document_count

    @property
    def vocabulary_size(self) -> int:
        """Return how many distinct terms the segment holds."""
        return self._footer.vocabulary_size

    @property
    def terms(self) -> Iterable[str]:
        """Yield every term, in sorted order."""
        return self._dictionary.terms()

    @property
    def document_ids(self) -> Iterable[int]:
        """Return every document identifier, in ascending order."""
        return tuple(self._document_ids)

    @property
    def postings_cache(self) -> LruCache[str, dict[int, list[int]]] | None:
        """Return the postings cache, or None when this reader caches nothing.

        Exposed so a benchmark can publish the hit rate.
        """
        return self._cache

    @property
    def raw(self) -> memoryview:
        """Return the mapped bytes, for a caller decoding postings itself.

        Early termination walks postings without materialising them, which the
        postings mapping cannot express.
        """
        return self._view

    @property
    def dictionary(self) -> TermDictionary:
        """Return the term dictionary, for a caller that needs entries directly.

        An entry carries the document frequency and the highest term frequency,
        which is what an upper bound on a term's score is computed from.
        """
        return self._dictionary
