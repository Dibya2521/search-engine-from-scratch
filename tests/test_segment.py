"""Tests for the on-disk segment format and its term dictionary."""

from __future__ import annotations

import zlib
from bisect import bisect_left
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from search_engine.analysis import fingerprint
from search_engine.cache import POSTINGS_CACHE_SIZE
from search_engine.codecs import encode_number, encode_sorted
from search_engine.index import InvertedIndex
from search_engine.segment import (
    FOOTER_SIZE,
    POSTINGS_BLOCK_SIZE,
    SEGMENT_HEADER,
    SEGMENT_MAGIC,
    TERMS_PER_BLOCK,
    Footer,
    SegmentAnalyzerMismatchError,
    SegmentCorruptError,
    SegmentFormatError,
    SegmentReader,
    SegmentVersionError,
    TermDictionary,
    TermEntry,
    advance_to,
    block_identifiers,
    decode_documents,
    decode_term_postings,
    encode_segment,
    read_blocks,
    read_footer,
    write_segment,
)

if TYPE_CHECKING:
    from pathlib import Path


def index_of(*texts: str) -> InvertedIndex:
    index = InvertedIndex()
    for document_id, text in enumerate(texts):
        index.add_document(document_id, text)
    return index


def index_of_terms(terms: list[str]) -> InvertedIndex:
    """Build an index whose vocabulary is exactly these terms, one per document.

    Goes through from_postings rather than add_document so the terms survive
    analysis unchanged, which is what a dictionary test wants to control.
    """
    postings = {term: {number: [0]} for number, term in enumerate(sorted(set(terms)))}
    return InvertedIndex.from_postings(list(range(len(postings))), postings)


def test_every_term_is_findable_and_its_postings_survive() -> None:
    index = index_of(
        "the quick brown fox jumps over the lazy dog",
        "quick brown foxes are quick",
        "a lazy dog sleeps",
    )
    data = encode_segment(index)
    dictionary = TermDictionary.read(data)

    assert len(dictionary) == index.vocabulary_size
    for term in index.terms:
        entry = dictionary.lookup(term)
        assert entry is not None
        assert entry.document_frequency == index.document_frequency(term)
        decoded = decode_term_postings(data, entry)
        assert decoded == {
            document_id: list(positions)
            for document_id, positions in index.postings(term).items()
        }


def test_a_term_that_is_not_there_returns_none() -> None:
    dictionary = TermDictionary.read(encode_segment(index_of("alpha beta")))
    assert dictionary.lookup("gamma") is None


def test_a_term_before_every_block_returns_none() -> None:
    """The binary search has no block to land in, which is a separate path."""
    dictionary = TermDictionary.read(encode_segment(index_of_terms(["m", "n", "o"])))
    assert dictionary.lookup("a") is None


def test_an_empty_index_writes_a_valid_segment() -> None:
    dictionary = TermDictionary.read(encode_segment(InvertedIndex()))
    assert len(dictionary) == 0
    assert dictionary.lookup("anything") is None
    assert list(dictionary.terms()) == []


def test_a_single_term_index_round_trips() -> None:
    dictionary = TermDictionary.read(encode_segment(index_of_terms(["solo"])))
    assert list(dictionary.terms()) == ["solo"]
    assert dictionary.lookup("solo") is not None


@pytest.mark.parametrize(
    "count", [TERMS_PER_BLOCK - 1, TERMS_PER_BLOCK, TERMS_PER_BLOCK + 1]
)
def test_block_boundaries(count: int) -> None:
    """An off-by-one at a block boundary is the likeliest bug in this format."""
    terms = [f"term{number:04d}" for number in range(count)]
    data = encode_segment(index_of_terms(terms))
    dictionary = TermDictionary.read(data)
    assert list(dictionary.terms()) == terms
    for term in terms:
        assert dictionary.lookup(term) is not None


def test_max_term_frequency_is_the_highest_in_any_one_document() -> None:
    index = index_of("alpha alpha alpha beta", "alpha beta beta")
    dictionary = TermDictionary.read(encode_segment(index))
    alpha = dictionary.lookup("alpha")
    beta = dictionary.lookup("beta")
    assert alpha is not None
    assert beta is not None
    assert alpha.max_term_frequency == 3
    assert beta.max_term_frequency == 2


def test_document_lengths_are_stored() -> None:
    index = index_of("alpha beta gamma", "alpha")
    data = encode_segment(index)
    document_ids, lengths = decode_documents(data, _footer_of(data))
    assert document_ids == [0, 1]
    assert lengths == [3, 1]


def test_a_document_with_no_terms_has_a_length_of_zero() -> None:
    index = InvertedIndex.from_postings([0, 1], {"alpha": {0: [0]}})
    data = encode_segment(index)
    _, lengths = decode_documents(data, _footer_of(data))
    assert lengths == [1, 0]


def test_terms_sharing_long_prefixes_round_trip() -> None:
    """Front coding stores only what changed, so deep sharing is the stress case."""
    terms = ["a" * length for length in range(1, 201)]
    dictionary = TermDictionary.read(encode_segment(index_of_terms(terms)))
    assert list(dictionary.terms()) == sorted(terms)
    for term in terms:
        assert dictionary.lookup(term) is not None


def test_multi_byte_terms_round_trip() -> None:
    """A shared prefix counted in characters would split a code point."""
    base = chr(0x00E9) + chr(0x4E2D) + chr(0x1F600)
    terms = [base[:cut] + suffix for cut in (1, 2, 3) for suffix in ("x", "y")]
    dictionary = TermDictionary.read(encode_segment(index_of_terms(terms)))
    assert list(dictionary.terms()) == sorted(terms)
    for term in terms:
        assert dictionary.lookup(term) is not None


def test_a_truncated_file_is_refused() -> None:
    data = encode_segment(index_of("alpha beta"))
    with pytest.raises(SegmentFormatError):
        TermDictionary.read(data[: len(SEGMENT_HEADER) + FOOTER_SIZE - 1])


def test_a_file_without_the_header_is_refused() -> None:
    with pytest.raises(SegmentFormatError, match="header is missing"):
        TermDictionary.read(b"not a segment at all" + bytes(FOOTER_SIZE))


def test_a_flipped_byte_is_caught_by_the_checksum() -> None:
    data = bytearray(encode_segment(index_of("alpha beta gamma delta")))
    data[len(SEGMENT_HEADER) + 20] ^= 0xFF
    with pytest.raises(SegmentCorruptError):
        TermDictionary.read(bytes(data))


def test_a_different_analyzer_is_refused() -> None:
    data = bytearray(encode_segment(index_of("alpha beta")))
    stamp_at = len(SEGMENT_HEADER) + 1
    data[stamp_at] = ord("0") if data[stamp_at] != ord("0") else ord("1")
    repaired = _with_checksum(data)
    with pytest.raises(SegmentAnalyzerMismatchError):
        TermDictionary.read(repaired)


def test_write_segment_produces_the_same_bytes(tmp_path: Path) -> None:
    index = index_of("alpha beta", "beta gamma")
    path = tmp_path / "one.seg"
    write_segment(index, path)
    assert path.read_bytes() == encode_segment(index)


@settings(deadline=None)
@given(
    st.lists(
        st.text(alphabet="abcdefghij", min_size=1, max_size=6),
        min_size=1,
        max_size=60,
        unique=True,
    )
)
def test_lookup_agrees_with_a_plain_dictionary(terms: list[str]) -> None:
    index = index_of_terms(terms)
    data = encode_segment(index)
    dictionary = TermDictionary.read(data)
    expected = {term: index.document_frequency(term) for term in index.terms}

    assert list(dictionary.terms()) == sorted(expected)
    for term, frequency in expected.items():
        entry = dictionary.lookup(term)
        assert entry is not None
        assert entry.document_frequency == frequency
    assert dictionary.lookup("zzzz") is None


def craft_segment(
    *,
    dictionary: bytes = b"",
    block_index: bytes = b"",
    documents: bytes = b"",
    dictionary_length: int | None = None,
    block_index_length: int | None = None,
) -> bytes:
    """Assemble a segment from arbitrary sections, with a correct checksum.

    A file damaged by cutting bytes out of it fails the checksum long before any
    section is parsed, so reaching a section's own error handling needs a file
    that is malformed and correctly checksummed.
    """
    stamp = fingerprint().encode("ascii")
    prologue = SEGMENT_HEADER + encode_number(len(stamp)) + stamp
    dictionary_offset = len(prologue)
    block_index_offset = dictionary_offset + len(dictionary)
    documents_offset = block_index_offset + len(block_index)
    fields = (
        dictionary_offset,
        len(dictionary) if dictionary_length is None else dictionary_length,
        block_index_offset,
        len(block_index) if block_index_length is None else block_index_length,
        documents_offset,
        len(documents),
        0,
        0,
    )
    body = (
        prologue
        + dictionary
        + block_index
        + documents
        + b"".join(value.to_bytes(8, "big") for value in fields)
    )
    return body + zlib.crc32(body).to_bytes(4, "big")


def test_postings_pointing_past_the_file_are_refused() -> None:
    data = encode_segment(index_of("alpha beta"))
    entry = TermEntry(
        offset=len(data), length=4, document_frequency=1, max_term_frequency=1
    )
    with pytest.raises(SegmentFormatError, match="postings are malformed"):
        decode_term_postings(data, entry)


def test_a_documents_section_pointing_past_the_file_is_refused() -> None:
    data = encode_segment(index_of("alpha beta"))
    footer = _footer_of(data)
    broken = replace(footer, documents_offset=len(data))
    with pytest.raises(SegmentFormatError, match="documents section is malformed"):
        decode_documents(data, broken)


def test_a_file_ending_before_the_fingerprint_is_refused() -> None:
    """Every byte after the header continues a number that never terminates.

    The checksum is part of the file, so the padding length is chosen to give a
    checksum with no high bit set in any of its four bytes.
    """
    for padding in range(FOOTER_SIZE, FOOTER_SIZE + 64):
        body = SEGMENT_HEADER + b"\x00" * padding
        checksum = zlib.crc32(body).to_bytes(4, "big")
        if all(byte < 0x80 for byte in checksum):
            with pytest.raises(SegmentFormatError, match="ends before"):
                TermDictionary.read(body + checksum)
            return
    pytest.fail("no padding length produced a checksum free of high bits")


def test_a_dictionary_entry_claiming_more_than_the_file_holds_is_refused() -> None:
    dictionary = encode_number(0) + encode_number(200)
    data = craft_segment(
        dictionary=dictionary,
        block_index=encode_number(1) + encode_number(1) + b"a" + encode_number(0),
    )
    with pytest.raises(SegmentFormatError, match="dictionary is malformed"):
        TermDictionary.read(data).lookup("a")


def test_a_block_index_claiming_more_than_the_file_holds_is_refused() -> None:
    data = craft_segment(block_index=encode_number(1) + encode_number(200))
    with pytest.raises(SegmentFormatError, match="block index is malformed"):
        TermDictionary.read(data)


def test_a_block_index_running_past_its_own_length_is_refused() -> None:
    block_index = encode_number(1) + encode_number(1) + b"a" + encode_number(0)
    data = craft_segment(block_index=block_index, block_index_length=2)
    with pytest.raises(SegmentFormatError, match="runs past its own length"):
        TermDictionary.read(data)


def _footer_of(data: bytes) -> Footer:
    return read_footer(data)


def _with_checksum(data: bytearray) -> bytes:
    body = bytes(data[:-4])
    return body + zlib.crc32(body).to_bytes(4, "big")


def write(index: InvertedIndex, path: Path) -> Path:
    write_segment(index, path)
    return path


def test_a_reader_answers_the_same_as_the_index_it_came_from(tmp_path: Path) -> None:
    index = index_of(
        "the quick brown fox jumps over the lazy dog",
        "quick brown foxes are quick and brown",
        "a lazy dog sleeps all day",
    )
    with SegmentReader(write(index, tmp_path / "a.seg")) as reader:
        assert reader.document_count == index.document_count
        assert reader.vocabulary_size == index.vocabulary_size
        assert list(reader.terms) == sorted(index.terms)
        assert list(reader.document_ids) == sorted(index.document_ids)
        for term in index.terms:
            assert term in reader
            assert reader.document_frequency(term) == index.document_frequency(term)
            assert reader.postings(term) == {
                document_id: list(positions)
                for document_id, positions in index.postings(term).items()
            }


def test_an_absent_term_reads_as_empty(tmp_path: Path) -> None:
    with SegmentReader(write(index_of("alpha beta"), tmp_path / "a.seg")) as reader:
        assert "gamma" not in reader
        assert reader.postings("gamma") == {}
        assert reader.document_frequency("gamma") == 0
        assert reader.max_term_frequency("gamma") == 0


def test_postings_are_cached_after_the_first_read(tmp_path: Path) -> None:
    with SegmentReader(write(index_of("alpha beta"), tmp_path / "a.seg")) as reader:
        assert reader.postings("alpha") is reader.postings("alpha")


def test_the_cache_is_bounded_and_counted(tmp_path: Path) -> None:
    """The reader holds the bounded cache, not a dictionary that only grows."""
    with SegmentReader(write(index_of("alpha beta"), tmp_path / "a.seg")) as reader:
        cache = reader.postings_cache
        assert cache is not None
        assert cache.capacity == POSTINGS_CACHE_SIZE
        reader.postings("alpha")
        reader.postings("alpha")
        assert (cache.hits, cache.misses) == (1, 1)


def test_a_reader_opened_without_a_cache_holds_none(tmp_path: Path) -> None:
    """A merge reads every term once, so caching would retain what it passed."""
    path = write(index_of("alpha beta"), tmp_path / "a.seg")
    with SegmentReader(path, cache=False) as reader:
        assert reader.postings_cache is None
        assert reader.postings("alpha") == {0: [0]}
        assert reader.postings("alpha") is not reader.postings("alpha")


def test_a_term_the_segment_lacks_is_cached_as_empty(tmp_path: Path) -> None:
    """Otherwise every repeat of a query for an unknown word costs a lookup."""
    with SegmentReader(write(index_of("alpha"), tmp_path / "a.seg")) as reader:
        assert reader.postings("absent") == {}
        assert reader.postings("absent") == {}
        cache = reader.postings_cache
        assert cache is not None
        assert cache.hits == 1


def test_document_lengths_and_frequencies_come_from_the_dictionary(
    tmp_path: Path,
) -> None:
    index = index_of("alpha alpha alpha beta", "alpha beta beta gamma")
    with SegmentReader(write(index, tmp_path / "a.seg")) as reader:
        assert reader.document_length(0) == 4
        assert reader.document_length(1) == 4
        assert reader.document_length(99) == 0
        assert reader.max_term_frequency("alpha") == 3


def test_closing_twice_is_safe(tmp_path: Path) -> None:
    reader = SegmentReader(write(index_of("alpha"), tmp_path / "a.seg"))
    reader.close()
    reader.close()


def test_a_closed_reader_refuses_to_read(tmp_path: Path) -> None:
    reader = SegmentReader(write(index_of("alpha beta"), tmp_path / "a.seg"))
    reader.close()
    with pytest.raises(ValueError, match="released"):
        reader.postings("alpha")


def test_a_corrupt_file_is_refused_and_left_unmapped(tmp_path: Path) -> None:
    """Windows cannot delete a mapped file, so deleting it proves it was closed."""
    path = tmp_path / "a.seg"
    data = bytearray(encode_segment(index_of("alpha beta gamma")))
    data[len(SEGMENT_HEADER) + 20] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(SegmentCorruptError):
        SegmentReader(path)
    path.unlink()


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "a.seg"
    path.write_bytes(b"")
    with pytest.raises(SegmentFormatError, match="empty"):
        SegmentReader(path)
    path.unlink()


def test_a_corrupt_file_opens_unverified_and_verify_still_catches_it(
    tmp_path: Path,
) -> None:
    """Skipping the checksum is what makes opening cheap, and what it costs."""
    path = tmp_path / "a.seg"
    data = bytearray(encode_segment(index_of("alpha beta gamma")))
    data[len(SEGMENT_HEADER) + 20] ^= 0xFF
    path.write_bytes(bytes(data))
    with SegmentReader(path, verify=False) as reader:
        assert reader.document_count == 1
        with pytest.raises(SegmentCorruptError):
            reader.verify()


def test_a_term_whose_postings_span_many_pages(tmp_path: Path) -> None:
    """Offsets have to survive a term far larger than one memory page."""
    postings = {"common": {number: [0, 5] for number in range(20_000)}}
    index = InvertedIndex.from_postings(list(range(20_000)), postings)
    with SegmentReader(write(index, tmp_path / "big.seg")) as reader:
        decoded = reader.postings("common")
        assert len(decoded) == 20_000
        assert decoded[19_999] == [0, 5]


@settings(deadline=None, max_examples=25)
@given(
    st.lists(
        st.text(alphabet="abcde", min_size=1, max_size=4),
        min_size=1,
        max_size=25,
        unique=True,
    )
)
def test_a_reader_matches_the_index_for_any_vocabulary(
    tmp_path_factory: pytest.TempPathFactory, terms: list[str]
) -> None:
    """Writes a file per example, so the deadline measures the disk not the code."""
    index = index_of_terms(terms)
    path = tmp_path_factory.mktemp("seg") / "a.seg"
    with SegmentReader(write(index, path)) as reader:
        for term in index.terms:
            assert reader.postings(term) == {
                document_id: list(positions)
                for document_id, positions in index.postings(term).items()
            }
        assert reader.postings("zzzzz") == {}


def postings_index(document_ids: list[int]) -> InvertedIndex:
    """An index with one term present in exactly these documents."""
    return InvertedIndex.from_postings(
        sorted(document_ids), {"t": {number: [0] for number in document_ids}}
    )


def entry_for(data: bytes, term: str) -> TermEntry:
    entry = TermDictionary.read(data).lookup(term)
    assert entry is not None
    return entry


def test_a_block_starts_with_identifiers_the_whole_file_format_would_write() -> None:
    """A block begins at its own identifiers, so the two encodings stay the same."""
    ordered = [1, 4, 9, 300, 70_000]
    data = encode_segment(postings_index(ordered))
    entry = entry_for(data, "t")
    block = read_blocks(data, entry)[0]
    expected = encode_sorted(ordered)
    assert bytes(data[block.offset : block.offset + len(expected)]) == expected


def test_a_short_postings_list_is_a_single_block() -> None:
    ordered = list(range(POSTINGS_BLOCK_SIZE - 1))
    blocks = read_blocks(*_term(ordered))
    assert len(blocks) == 1
    assert blocks[0].document_count == len(ordered)
    assert blocks[0].last_document == ordered[-1]


def test_a_long_postings_list_is_cut_at_the_block_size() -> None:
    ordered = list(range(0, 4_000, 2))
    blocks = read_blocks(*_term(ordered))
    assert len(blocks) == -(-len(ordered) // POSTINGS_BLOCK_SIZE)
    assert [block.document_count for block in blocks[:-1]] == [POSTINGS_BLOCK_SIZE] * (
        len(blocks) - 1
    )
    assert sum(block.document_count for block in blocks) == len(ordered)
    assert [block.last_document for block in blocks] == sorted(
        block.last_document for block in blocks
    )


def test_every_block_reports_the_identifiers_it_holds() -> None:
    """The table is only a skip list if its last identifier is the real one."""
    ordered = list(range(0, 1_000, 3))
    data, entry = _term(ordered)
    seen: list[int] = []
    for block in read_blocks(data, entry):
        identifiers = block_identifiers(data, block)
        assert len(identifiers) == block.document_count
        assert identifiers[-1] == block.last_document
        seen += identifiers
    assert seen == ordered


def test_block_maxima_match_the_postings_inside_each_block() -> None:
    """The bound a scorer prunes with is wrong if this is wrong."""
    postings = {
        "t": {
            document_id: list(range(1 + (document_id % 7)))
            for document_id in range(500)
        }
    }
    index = InvertedIndex.from_postings(list(range(500)), postings)
    data = encode_segment(index)
    entry = entry_for(data, "t")
    decoded = decode_term_postings(data, entry)
    blocks = read_blocks(data, entry)
    assert len(blocks) > 1
    for block in blocks:
        inside = block_identifiers(data, block)
        assert block.max_term_frequency == max(len(decoded[at]) for at in inside)
    assert entry.max_term_frequency == max(block.max_term_frequency for block in blocks)


def test_the_block_table_and_its_blocks_fill_the_term_exactly() -> None:
    """A byte unaccounted for here is a byte the next term would misread."""
    ordered = list(range(0, 900, 2))
    data, entry = _term(ordered)
    blocks = read_blocks(data, entry)
    table = blocks[0].offset - entry.offset
    assert table + sum(block.length for block in blocks) == entry.length


def _term(ordered: list[int]) -> tuple[bytes, TermEntry]:
    """Return a segment holding one term with these identifiers, and its entry."""
    data = encode_segment(postings_index(ordered))
    return data, entry_for(data, "t")


@pytest.mark.parametrize("count", [10, POSTINGS_BLOCK_SIZE, 1_000])
def test_advance_to_agrees_with_a_linear_search(count: int) -> None:
    ordered = [number * 3 for number in range(count)]
    data = encode_segment(postings_index(ordered))
    entry = entry_for(data, "t")
    for target in range(-1, ordered[-1] + 3):
        at = bisect_left(ordered, target)
        expected = ordered[at] if at < len(ordered) else None
        assert advance_to(data, entry, target) == expected


def test_advance_to_past_the_end_returns_none() -> None:
    ordered = list(range(0, 2_000, 2))
    data = encode_segment(postings_index(ordered))
    assert advance_to(data, entry_for(data, "t"), 10_000) is None


def test_advance_to_on_malformed_postings_is_refused() -> None:
    data = encode_segment(postings_index([1, 2, 3]))
    broken = TermEntry(len(data), 4, 1, 1)
    with pytest.raises(SegmentFormatError, match="postings are malformed"):
        advance_to(data, broken, 1)


def test_dense_and_sparse_identifiers_both_skip_correctly() -> None:
    """A block holds a fixed count of documents, not a fixed number of bytes.

    Dense identifiers encode to one byte each and identifiers spaced beyond
    2 ** 28 to five, so blocks of the same size cover very different byte
    distances. Only the stored length keeps a reader in step with them.
    """
    for ordered in (
        list(range(1_000)),
        [number * (1 << 28) for number in range(1, 1_001)],
    ):
        data = encode_segment(postings_index(ordered))
        entry = entry_for(data, "t")
        probes = (ordered[0], ordered[len(ordered) // 2], ordered[-1])
        for target in (*probes, *(value - 1 for value in probes)):
            at = bisect_left(ordered, target)
            expected = ordered[at] if at < len(ordered) else None
            assert advance_to(data, entry, target) == expected
        assert advance_to(data, entry, ordered[-1] + 1) is None


@settings(deadline=None, max_examples=30)
@given(
    st.lists(st.integers(min_value=0, max_value=5_000), min_size=1, unique=True),
    st.integers(min_value=-1, max_value=5_001),
)
def test_advance_to_matches_bisect_for_any_list(
    document_ids: list[int], target: int
) -> None:
    ordered = sorted(document_ids)
    data = encode_segment(postings_index(ordered))
    at = bisect_left(ordered, target)
    expected = ordered[at] if at < len(ordered) else None
    assert advance_to(data, entry_for(data, "t"), target) == expected


def test_a_reader_exposes_its_buffer_and_dictionary(tmp_path: Path) -> None:
    """Early termination decodes postings itself rather than through a mapping."""
    index = index_of("alpha beta gamma")
    with SegmentReader(write(index, tmp_path / "a.seg")) as reader:
        entry = reader.dictionary.lookup("alpha")
        assert entry is not None
        assert decode_term_postings(reader.raw, entry) == {0: [0]}
        assert len(reader.raw) == (tmp_path / "a.seg").stat().st_size


def test_a_segment_from_another_format_version_is_refused() -> None:
    """The postings layout changed, so an older file cannot simply be read.

    The header is checked before the checksum, so the version is reported
    rather than the corruption that patching the header also causes.
    """
    data = bytearray(encode_segment(index_of("alpha beta")))
    data[len(SEGMENT_MAGIC)] = ord("1")
    with pytest.raises(SegmentVersionError, match="rebuild the index") as caught:
        read_footer(bytes(data))
    assert caught.value.stored == "1"
    assert caught.value.current == "2"
    assert "version 1" in str(caught.value)
    assert "version 2" in str(caught.value)


def test_a_malformed_block_table_is_refused() -> None:
    data = encode_segment(postings_index([1, 2, 3]))
    broken = TermEntry(len(data), 4, 1, 1)
    with pytest.raises(SegmentFormatError, match="postings are malformed"):
        read_blocks(data, broken)
