"""Tests for the compressed binary index format."""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from search_engine.analysis import fingerprint
from search_engine.codecs import encode_number, encode_sorted
from search_engine.index import InvertedIndex
from search_engine.persistence import (
    AnalyzerMismatchError,
    IndexCorruptError,
    IndexFormatError,
    load,
    save,
    save_text,
)
from search_engine.query import search
from search_engine.tokenizer import tokenize

if TYPE_CHECKING:
    from pathlib import Path

HEADER = b"search-engine-index v2\n"


def index_of(texts: dict[int, str]) -> InvertedIndex:
    index = InvertedIndex()
    for document_id, text in texts.items():
        index.add_document(document_id, text)
    return index


def test_the_header_is_readable_even_though_the_body_is_not() -> None:
    """Enough of the file stays legible to identify what it is."""
    assert HEADER.decode("ascii").startswith("search-engine-index")


def test_the_file_starts_with_the_header(tmp_path: Path) -> None:
    path = tmp_path / "i.index"
    save(index_of({1: "web search"}), path)
    assert path.read_bytes().startswith(HEADER)


def test_the_fingerprint_is_stored(tmp_path: Path) -> None:
    path = tmp_path / "i.index"
    save(index_of({1: "web search"}), path)
    assert fingerprint().encode("ascii") in path.read_bytes()


def test_the_binary_format_is_smaller_than_the_text_one(tmp_path: Path) -> None:
    """The reason the format changed at all."""
    index = index_of({one: f"term{one % 9} common word {one}" for one in range(300)})
    binary = tmp_path / "i.index"
    text = tmp_path / "i.text"
    save(index, binary)
    save_text(index, text)
    assert binary.stat().st_size < text.stat().st_size


def test_a_version_one_file_still_loads(tmp_path: Path) -> None:
    """An index written by an older build must not stop opening."""
    index = index_of({1: "web search", 2: "web index"})
    path = tmp_path / "i.index"
    save_text(index, path)
    restored = load(path)
    assert set(restored.document_ids) == {1, 2}
    assert restored.postings("web") == {1: [0], 2: [0]}


def test_a_corrupted_body_is_detected(tmp_path: Path) -> None:
    """A silently corrupt index returns wrong answers indefinitely."""
    path = tmp_path / "i.index"
    save(index_of({1: "web search", 2: "web index"}), path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(IndexCorruptError, match="corrupt"):
        load(path)


def test_a_truncated_file_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "i.index"
    save(index_of({1: "web search engine", 2: "web index"}), path)
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(IndexFormatError):
        load(path)


def test_a_file_ending_before_the_fingerprint_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "i.index"
    path.write_bytes(HEADER + b"\x01")
    with pytest.raises(IndexFormatError, match="ends before"):
        load(path)


def test_an_index_from_a_different_analyzer_is_refused(tmp_path: Path) -> None:
    """The silent failure the fingerprint exists to prevent.

    An index whose terms came from a different tokenizer, stopword list or
    stemmer cannot be queried by this build. The query would ask for terms the
    index never stored, and the engine would return nothing while appearing to
    work perfectly.
    """
    path = tmp_path / "i.index"
    save(index_of({1: "web search"}), path)
    data = bytearray(path.read_bytes())
    digit = len(HEADER) + 1
    data[digit] = ord("0") if data[digit] != ord("0") else ord("1")
    path.write_bytes(bytes(data))
    with pytest.raises(AnalyzerMismatchError, match="rebuild the index"):
        load(path)


def test_the_mismatch_error_names_both_fingerprints() -> None:
    error = AnalyzerMismatchError("aaaa", "bbbb")
    assert "aaaa" in str(error)
    assert "bbbb" in str(error)


# Writes a file per generated example, so the timing measures the disk rather
# than the code. Hypothesis deadlines catch code that goes pathologically slow
# on some input, which is a real signal for a pure function and noise here.
@settings(deadline=None)
@given(st.lists(st.text(), min_size=1, max_size=6))
def test_any_index_round_trips_unchanged(
    tmp_path_factory: pytest.TempPathFactory, texts: list[str]
) -> None:
    """The falsifying direction: generated corpora rather than chosen ones."""
    index = index_of(dict(enumerate(texts)))
    path = tmp_path_factory.mktemp("rt") / "i.index"
    save(index, path)
    restored = load(path)
    assert set(restored.document_ids) == set(index.document_ids)
    assert set(restored.terms) == set(index.terms)
    for term in index.terms:
        assert restored.postings(term) == index.postings(term)


# Writes a file per generated example; see the note above.
@settings(deadline=None)
@given(st.lists(st.text(), min_size=1, max_size=6))
def test_queries_answer_the_same_after_a_round_trip(
    tmp_path_factory: pytest.TempPathFactory, texts: list[str]
) -> None:
    """What actually matters: the reloaded index answers identically."""
    index = index_of(dict(enumerate(texts)))
    path = tmp_path_factory.mktemp("rt") / "i.index"
    save(index, path)
    restored = load(path)
    for token in {token for text in texts for token in tokenize(text)}:
        assert search(restored, token) == search(index, token)


# Writes a file per generated example; see the note above.
@settings(deadline=None)
@given(st.lists(st.text(), min_size=1, max_size=6))
def test_both_formats_load_to_the_same_index(
    tmp_path_factory: pytest.TempPathFactory, texts: list[str]
) -> None:
    """The comparison is only fair if the two formats mean the same thing."""
    index = index_of(dict(enumerate(texts)))
    directory = tmp_path_factory.mktemp("rt")
    binary, text = directory / "i.index", directory / "i.text"
    save(index, binary)
    save_text(index, text)
    from_binary, from_text = load(binary), load(text)
    assert set(from_binary.document_ids) == set(from_text.document_ids)
    assert set(from_binary.terms) == set(from_text.terms)
    for term in from_text.terms:
        assert from_binary.postings(term) == from_text.postings(term)


def write_body(path: Path, body: bytes) -> None:
    """Write a well-formed header and checksum around an arbitrary body.

    A file truncated by cutting bytes off the end fails the checksum before the
    body is ever parsed, so reaching the body's own error handling needs a body
    that is malformed and correctly checksummed.
    """
    stamp = fingerprint().encode("ascii")
    path.write_bytes(
        HEADER
        + encode_number(len(stamp))
        + stamp
        + zlib.crc32(body).to_bytes(4, "big")
        + body
    )


def test_a_body_claiming_more_documents_than_it_holds(tmp_path: Path) -> None:
    path = tmp_path / "i.index"
    write_body(path, encode_number(5) + encode_sorted([1, 2]))
    with pytest.raises(IndexFormatError, match="malformed"):
        load(path)


def test_a_body_claiming_a_longer_term_than_it_holds(tmp_path: Path) -> None:
    path = tmp_path / "i.index"
    body = (
        encode_number(0)  # no documents
        + encode_number(1)  # one term
        + encode_number(50)  # claiming fifty bytes of term
        + b"web"  # supplying three
    )
    write_body(path, body)
    with pytest.raises(IndexFormatError, match="middle of a term"):
        load(path)


def test_a_body_with_an_undecodable_term(tmp_path: Path) -> None:
    """Terms are ASCII by construction, so this can only be a corrupt file."""
    path = tmp_path / "i.index"
    body = encode_number(0) + encode_number(1) + encode_number(2) + b"\xff\xfe"
    write_body(path, body)
    with pytest.raises(IndexFormatError, match="malformed"):
        load(path)
