"""Tests for the file that decides which segments exist."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.analysis import fingerprint
from search_engine.manifest import (
    MANIFEST_NAME,
    TEMPORARY_NAME,
    Manifest,
    ManifestFormatError,
    SegmentInfo,
    empty,
    publish,
    read,
)
from search_engine.segment import SegmentAnalyzerMismatchError
from search_engine.writer import IndexWriter, segment_name

if TYPE_CHECKING:
    from pathlib import Path


def write_raw(directory: Path, data: object) -> None:
    (directory / MANIFEST_NAME).write_text(json.dumps(data), encoding="utf-8")


def sample(generation: int = 0) -> Manifest:
    return Manifest(
        generation=generation,
        next_segment=2,
        analyzer=fingerprint(),
        segments=(
            SegmentInfo("segment-00000000.seg", 3, 1234),
            SegmentInfo("segment-00000001.seg", 5, 5678),
        ),
    )


def test_a_directory_with_no_manifest_reads_as_empty(tmp_path: Path) -> None:
    """No manifest means no index, which is a state rather than a failure."""
    manifest = read(tmp_path)
    assert manifest.segments == ()
    assert manifest.generation == 0
    assert manifest.document_count == 0


def test_a_published_manifest_reads_back_unchanged(tmp_path: Path) -> None:
    publish(tmp_path, sample())
    assert read(tmp_path) == sample()


def test_publishing_leaves_no_temporary_behind(tmp_path: Path) -> None:
    publish(tmp_path, sample())
    assert not (tmp_path / TEMPORARY_NAME).exists()


def test_each_publication_advances_the_generation() -> None:
    """Every cache entry is keyed by this, so it must never repeat."""
    first = empty()
    second = first.with_segments([SegmentInfo("a.seg", 1, 1)], 1)
    third = second.with_segments([SegmentInfo("a.seg", 1, 1)], 1)
    assert (first.generation, second.generation, third.generation) == (0, 1, 2)


def test_document_count_totals_the_segments() -> None:
    assert sample().document_count == 8


def test_a_manifest_from_a_different_analyzer_is_refused(tmp_path: Path) -> None:
    """Terms produced by a different pipeline would silently fail to match."""
    publish(tmp_path, Manifest(1, 1, "0000000000000000", ()))
    with pytest.raises(SegmentAnalyzerMismatchError):
        read(tmp_path)


@pytest.mark.parametrize(
    "data",
    [
        [],
        "a string",
        {"format": 1},
        {
            "format": 99,
            "generation": 0,
            "next_segment": 0,
            "analyzer": "x",
            "segments": [],
        },
        {
            "format": 1,
            "generation": "one",
            "next_segment": 0,
            "analyzer": "x",
            "segments": [],
        },
        {
            "format": True,
            "generation": 0,
            "next_segment": 0,
            "analyzer": "x",
            "segments": [],
        },
        {
            "format": 1,
            "generation": 0,
            "next_segment": 0,
            "analyzer": "x",
            "segments": {},
        },
        {
            "format": 1,
            "generation": 0,
            "next_segment": 0,
            "analyzer": "x",
            "segments": ["not an object"],
        },
        {
            "format": 1,
            "generation": 0,
            "next_segment": 0,
            "analyzer": "x",
            "segments": [{"name": "a.seg", "documents": 1}],
        },
    ],
)
def test_a_malformed_manifest_is_refused(tmp_path: Path, data: object) -> None:
    """Valid JSON of the wrong shape must raise a manifest error, not a JSON one."""
    write_raw(tmp_path, data)
    with pytest.raises(ManifestFormatError):
        read(tmp_path)


def test_a_manifest_that_is_not_json_is_refused(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestFormatError, match="not readable JSON"):
        read(tmp_path)


def test_a_manifest_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_NAME).write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ManifestFormatError, match="not readable JSON"):
        read(tmp_path)


def test_a_flushed_segment_is_named_by_the_manifest(tmp_path: Path) -> None:
    with IndexWriter(tmp_path, buffer_documents=1) as writer:
        writer.add(0, "alpha beta")
        writer.add(1, "gamma")
    manifest = read(tmp_path)
    assert [segment.name for segment in manifest.segments] == [
        segment_name(0),
        segment_name(1),
    ]
    assert manifest.document_count == 2
    assert manifest.next_segment == 2


def test_a_segment_written_but_never_published_is_invisible(tmp_path: Path) -> None:
    """The crash that matters: the file exists and nothing names it.

    Simulated by writing a real segment with one writer and then discarding the
    manifest it published, which leaves exactly the state a process that died
    between the two steps would leave.
    """
    with IndexWriter(tmp_path, buffer_documents=1) as writer:
        writer.add(0, "alpha")
    orphan = tmp_path / segment_name(0)
    (tmp_path / MANIFEST_NAME).unlink()
    assert orphan.is_file()

    assert read(tmp_path).segments == ()
    with IndexWriter(tmp_path, buffer_documents=1) as writer:
        writer.add(1, "beta")
    assert [segment.name for segment in read(tmp_path).segments] == [segment_name(1)]


def test_an_abandoned_temporary_manifest_is_ignored(tmp_path: Path) -> None:
    """A crash during the write leaves this, and the old manifest still stands."""
    publish(tmp_path, sample())
    (tmp_path / TEMPORARY_NAME).write_text("{half written", encoding="utf-8")
    assert read(tmp_path) == sample()


def test_a_reused_segment_number_is_never_handed_out(tmp_path: Path) -> None:
    """An orphan's number is spent, even though the orphan itself is invisible."""
    with IndexWriter(tmp_path, buffer_documents=1) as writer:
        writer.add(0, "alpha")
    (tmp_path / MANIFEST_NAME).unlink()
    with IndexWriter(tmp_path, buffer_documents=1) as writer:
        writer.add(1, "beta")
    assert (tmp_path / segment_name(1)).is_file()


@given(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000),
    st.lists(
        st.tuples(
            st.text(alphabet="abcdef", min_size=1, max_size=8),
            st.integers(min_value=0, max_value=10_000),
            st.integers(min_value=0, max_value=2**32 - 1),
        ),
        max_size=8,
    ),
)
def test_a_manifest_survives_a_round_trip_through_its_dict(
    generation: int, next_segment: int, rows: list[tuple[str, int, int]]
) -> None:
    manifest = Manifest(
        generation=generation,
        next_segment=next_segment,
        analyzer=fingerprint(),
        segments=tuple(SegmentInfo(*row) for row in rows),
    )
    restored: dict[str, Any] = manifest.to_dict()
    assert Manifest.from_dict(restored) == manifest
