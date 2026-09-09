"""Tests for the buffered write path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from search_engine.index import DuplicateDocumentError, InvertedIndex
from search_engine.segment import SegmentReader
from search_engine.writer import (
    IndexWriter,
    next_segment_number,
    segment_name,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

TEXTS = (
    "the quick brown fox",
    "quick brown foxes run",
    "a lazy dog sleeps",
    "dogs and foxes",
    "sleeping dogs lie",
)


def read_all(directory: Path) -> dict[str, dict[int, list[int]]]:
    """Return the union of every segment's postings in a directory."""
    union: dict[str, dict[int, list[int]]] = {}
    for path in sorted(directory.glob("segment-*.seg")):
        with SegmentReader(path) as reader:
            for term in reader.terms:
                union.setdefault(term, {}).update(
                    {
                        document_id: list(positions)
                        for document_id, positions in reader.postings(term).items()
                    }
                )
    return union


def in_memory(texts: Mapping[int, str]) -> dict[str, dict[int, list[int]]]:
    index = InvertedIndex()
    for document_id, text in texts.items():
        index.add_document(document_id, text)
    return {
        term: {
            document_id: list(positions)
            for document_id, positions in index.postings(term).items()
        }
        for term in index.terms
    }


def test_nothing_is_written_before_the_buffer_fills(tmp_path: Path) -> None:
    with IndexWriter(tmp_path, buffer_documents=10) as writer:
        for document_id, text in enumerate(TEXTS):
            writer.add(document_id, text)
        assert writer.buffered == len(TEXTS)
        assert list(tmp_path.glob("segment-*.seg")) == []


def test_closing_flushes_what_is_buffered(tmp_path: Path) -> None:
    with IndexWriter(tmp_path, buffer_documents=10) as writer:
        writer.add(0, "alpha beta")
    assert len(list(tmp_path.glob("segment-*.seg"))) == 1


def test_a_full_buffer_flushes_itself(tmp_path: Path) -> None:
    with IndexWriter(tmp_path, buffer_documents=2) as writer:
        for document_id, text in enumerate(TEXTS):
            writer.add(document_id, text)
    assert len(writer.segments) == 3
    assert len(list(tmp_path.glob("segment-*.seg"))) == 3


def test_flushing_an_empty_buffer_writes_nothing(tmp_path: Path) -> None:
    with IndexWriter(tmp_path) as writer:
        assert writer.flush() is None
        assert writer.flush() is None
    assert list(tmp_path.glob("segment-*.seg")) == []


def test_the_segments_hold_exactly_what_was_added(tmp_path: Path) -> None:
    texts = dict(enumerate(TEXTS))
    with IndexWriter(tmp_path, buffer_documents=2) as writer:
        for document_id, text in texts.items():
            writer.add(document_id, text)
    assert read_all(tmp_path) == in_memory(texts)


def test_segment_numbers_continue_across_writers(tmp_path: Path) -> None:
    """A number is never reused, or a stale mapping would mean something else."""
    with IndexWriter(tmp_path, buffer_documents=1) as writer:
        writer.add(0, "alpha")
    with IndexWriter(tmp_path, buffer_documents=1) as writer:
        writer.add(1, "beta")
    assert sorted(path.name for path in tmp_path.glob("segment-*.seg")) == [
        segment_name(0),
        segment_name(1),
    ]


def test_a_directory_is_created_if_it_is_missing(tmp_path: Path) -> None:
    nested = tmp_path / "one" / "two"
    with IndexWriter(nested) as writer:
        writer.add(0, "alpha")
    assert nested.is_dir()


def test_a_duplicate_in_one_buffer_is_still_refused(tmp_path: Path) -> None:
    """Upsert belongs to the writer once it can tombstone. Until then this holds."""
    with IndexWriter(tmp_path) as writer:
        writer.add(0, "alpha")
        with pytest.raises(DuplicateDocumentError):
            writer.add(0, "beta")


def test_a_buffer_of_no_documents_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        IndexWriter(tmp_path, buffer_documents=0)


def test_unrelated_files_do_not_disturb_the_numbering(tmp_path: Path) -> None:
    """The adversarial case: names that look like segments and are not."""
    (tmp_path / "segment-0000000A.seg").write_bytes(b"")
    (tmp_path / "segment-1.seg").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    assert next_segment_number(tmp_path) == 0


def test_a_document_that_analyses_to_nothing_still_counts(tmp_path: Path) -> None:
    with IndexWriter(tmp_path, buffer_documents=2) as writer:
        writer.add(0, "the and of")
        writer.add(1, "alpha")
    with SegmentReader(tmp_path / segment_name(0)) as reader:
        assert reader.document_count == 2
        assert reader.document_length(0) == 0


@settings(deadline=None, max_examples=25)
@given(
    st.lists(st.sampled_from(TEXTS), min_size=1, max_size=12),
    st.integers(min_value=1, max_value=50),
)
def test_any_buffer_size_stores_the_same_documents(
    tmp_path_factory: pytest.TempPathFactory, texts: list[str], size: int
) -> None:
    """Writes files per example, so the deadline would measure the disk."""
    directory = tmp_path_factory.mktemp("writer")
    documents = dict(enumerate(texts))
    with IndexWriter(directory, buffer_documents=size) as writer:
        for document_id, text in documents.items():
            writer.add(document_id, text)
    assert read_all(directory) == in_memory(documents)


def test_the_writer_exposes_the_manifest_it_published(tmp_path: Path) -> None:
    """A full buffer is flushed when the next document arrives, not before."""
    with IndexWriter(tmp_path, buffer_documents=1) as writer:
        writer.add(0, "alpha")
        assert writer.manifest.generation == 0
        writer.add(1, "beta")
        assert writer.manifest.generation == 1
        assert writer.manifest.document_count == 1
    assert writer.manifest.generation == 2
    assert writer.manifest.document_count == 2
    assert writer.manifest.next_segment == 2
