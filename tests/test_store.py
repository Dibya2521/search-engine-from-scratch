"""Tests for the source text kept beside a segment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from search_engine.store import (
    OFFSET_BYTES,
    DocumentStore,
    StoreCorruptError,
    StoreFormatError,
    data_name,
    offset_name,
)

if TYPE_CHECKING:
    from pathlib import Path

NAME = "segment-00000007.seg"

PAGES = [
    ("Information retrieval", "ranked retrieval returns documents in order"),
    ("Inverted index", "a map from each term to the documents holding it"),
    ("Tokenization", "splitting text into the units an index is built from"),
]

TEN_MEGABYTES = 10 * 1024 * 1024


def write(directory: Path, documents: list[tuple[str, str]]) -> None:
    """Write a store holding these documents, at these ordinals."""
    with DocumentStore.writer(directory, NAME) as store:
        for title, text in documents:
            store.add(title, text)


def offsets_in(directory: Path) -> list[int]:
    """Return the offset file decoded the way the store addresses it."""
    raw = (directory / offset_name(NAME)).read_bytes()
    return [
        int.from_bytes(raw[at : at + OFFSET_BYTES], "big")
        for at in range(0, len(raw), OFFSET_BYTES)
    ]


def test_every_document_comes_back_as_it_went_in(tmp_path: Path) -> None:
    write(tmp_path, PAGES)
    with DocumentStore.open(tmp_path, NAME) as store:
        assert store.documents == len(PAGES)
        for ordinal, (title, text) in enumerate(PAGES):
            assert store.title(ordinal) == title
            assert store.text(ordinal) == text


def test_the_files_are_named_after_the_segment() -> None:
    assert data_name(NAME) == "segment-00000007.dat"
    assert offset_name(NAME) == "segment-00000007.idx"


def test_an_ordinal_costs_one_fixed_width_entry(tmp_path: Path) -> None:
    """The whole argument for a separate offset file is this arithmetic."""
    write(tmp_path, PAGES)
    raw = (tmp_path / offset_name(NAME)).read_bytes()
    assert len(raw) == OFFSET_BYTES * len(PAGES)

    offsets = offsets_in(tmp_path)
    assert offsets[0] == 0
    assert offsets == sorted(offsets)
    assert offsets[-1] < (tmp_path / data_name(NAME)).stat().st_size


def test_an_empty_segment_has_an_empty_store(tmp_path: Path) -> None:
    """A file of no bytes cannot be mapped at all, which is the case to cover."""
    write(tmp_path, [])
    with DocumentStore.open(tmp_path, NAME, documents=0) as store:
        assert store.documents == 0


def test_an_ordinal_the_store_does_not_hold_is_refused(tmp_path: Path) -> None:
    write(tmp_path, PAGES)
    with DocumentStore.open(tmp_path, NAME) as store:
        with pytest.raises(IndexError, match="no ordinal 3"):
            store.title(3)
        with pytest.raises(IndexError, match="no ordinal -1"):
            store.text(-1)


def test_closing_twice_is_safe(tmp_path: Path) -> None:
    write(tmp_path, PAGES)
    store = DocumentStore.open(tmp_path, NAME)
    store.close()
    store.close()


def test_a_store_that_was_never_written_is_refused(tmp_path: Path) -> None:
    with pytest.raises(StoreFormatError, match="cannot be read"):
        DocumentStore.open(tmp_path, NAME)


def test_a_store_missing_its_data_file_is_refused(tmp_path: Path) -> None:
    """The offset file opens first, so this is the path that has to close it."""
    write(tmp_path, PAGES)
    (tmp_path / data_name(NAME)).unlink()
    with pytest.raises(StoreFormatError, match="cannot be read"):
        DocumentStore.open(tmp_path, NAME)


def test_an_offset_file_of_partial_offsets_is_refused(tmp_path: Path) -> None:
    write(tmp_path, PAGES)
    path = tmp_path / offset_name(NAME)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(StoreFormatError, match="whole number of offsets"):
        DocumentStore.open(tmp_path, NAME)


def test_a_store_that_disagrees_with_its_segment_is_refused(tmp_path: Path) -> None:
    write(tmp_path, PAGES)
    with pytest.raises(StoreFormatError, match="covers 3 documents"):
        DocumentStore.open(tmp_path, NAME, documents=len(PAGES) + 1)


def test_a_store_cut_short_mid_document_raises(tmp_path: Path) -> None:
    """A partial string returned as if it were whole is the failure to avoid."""
    write(tmp_path, PAGES)
    path = tmp_path / data_name(NAME)
    path.write_bytes(path.read_bytes()[: offsets_in(tmp_path)[-1] + 1])
    with DocumentStore.open(tmp_path, NAME) as store:
        assert store.title(0) == PAGES[0][0]
        with pytest.raises(StoreCorruptError, match="promises"):
            store.title(len(PAGES) - 1)


def test_a_store_cut_short_at_a_record_boundary_raises(tmp_path: Path) -> None:
    """Nothing is left of the last record, not even the length that sizes it."""
    write(tmp_path, PAGES)
    path = tmp_path / data_name(NAME)
    path.write_bytes(path.read_bytes()[: offsets_in(tmp_path)[-1]])
    with (
        DocumentStore.open(tmp_path, NAME) as store,
        pytest.raises(StoreCorruptError, match="length prefix"),
    ):
        store.text(len(PAGES) - 1)


def test_bytes_that_are_not_text_are_refused(tmp_path: Path) -> None:
    write(tmp_path, [("title", "hello")])
    path = tmp_path / data_name(NAME)
    path.write_bytes(path.read_bytes()[:-1] + b"\xff")
    with DocumentStore.open(tmp_path, NAME) as store:
        assert store.title(0) == "title"
        with pytest.raises(StoreCorruptError, match="not UTF-8"):
            store.text(0)


def test_a_document_with_no_text_round_trips(tmp_path: Path) -> None:
    write(tmp_path, [("", ""), ("only a title", "")])
    with DocumentStore.open(tmp_path, NAME) as store:
        assert store.title(0) == ""
        assert store.text(0) == ""
        assert store.title(1) == "only a title"
        assert store.text(1) == ""


def test_a_nul_and_a_four_byte_code_point_survive(tmp_path: Path) -> None:
    """Neither is a character the format reserves, and neither may be dropped."""
    text = "before\x00after \U0001f600 é"
    write(tmp_path, [("t\x00itle", text)])
    with DocumentStore.open(tmp_path, NAME) as store:
        assert store.title(0) == "t\x00itle"
        assert store.text(0) == text


def test_a_ten_megabyte_document_round_trips(tmp_path: Path) -> None:
    """A length prefix of more than one byte, and a record no buffer sizes for."""
    text = "x" * TEN_MEGABYTES
    write(tmp_path, [("large", text), ("after it", "still addressable")])
    with DocumentStore.open(tmp_path, NAME) as store:
        assert len(store.text(0)) == TEN_MEGABYTES
        assert store.title(1) == "after it"
        assert store.text(1) == "still addressable"


# Each example writes two files, so the deadline would be timing the disk.
@settings(deadline=None)
@given(
    documents=st.lists(
        st.tuples(st.text(max_size=100), st.text(max_size=1000)), max_size=20
    )
)
def test_any_documents_come_back_in_the_order_they_were_written(
    tmp_path_factory: pytest.TempPathFactory, documents: list[tuple[str, str]]
) -> None:
    directory = tmp_path_factory.mktemp("store")
    write(directory, documents)
    with DocumentStore.open(directory, NAME, documents=len(documents)) as store:
        assert [
            (store.title(ordinal), store.text(ordinal))
            for ordinal in range(store.documents)
        ] == documents
