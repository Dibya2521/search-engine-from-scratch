"""Tests for the on-disk segment format and its term dictionary."""

from __future__ import annotations

import zlib
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from search_engine.analysis import fingerprint
from search_engine.codecs import encode_number
from search_engine.index import InvertedIndex
from search_engine.segment import (
    FOOTER_SIZE,
    SEGMENT_HEADER,
    TERMS_PER_BLOCK,
    Footer,
    SegmentAnalyzerMismatchError,
    SegmentCorruptError,
    SegmentFormatError,
    TermDictionary,
    TermEntry,
    decode_documents,
    decode_term_postings,
    encode_segment,
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
