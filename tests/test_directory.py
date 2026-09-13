"""Tests for reading every published segment as one index."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from search_engine.bm25 import BM25Ranker
from search_engine.directory import DirectoryIndex
from search_engine.manifest import MANIFEST_NAME
from search_engine.query import search
from search_engine.ranking import Ranker
from search_engine.segment import SEGMENT_HEADER, SegmentCorruptError
from search_engine.writer import IndexWriter, segment_name
from tests.conftest import assert_same_postings, build_index, write_segments

if TYPE_CHECKING:
    from pathlib import Path

TEXTS = {
    0: "web search engines rank documents",
    1: "an inverted index maps terms to documents",
    2: "ranking documents by relevance is the point of search",
    3: "a lazy dog sleeps all day",
}


def test_a_directory_answers_the_same_as_one_index(tmp_path: Path) -> None:
    """Many files, one set of answers."""
    write_segments(tmp_path, TEXTS, buffer=2)
    expected = build_index(TEXTS)
    with DirectoryIndex(tmp_path) as index:
        assert index.segment_count == 2
        assert index.vocabulary_size == expected.vocabulary_size
        assert sorted(index.document_ids) == sorted(expected.document_ids)
        assert_same_postings(index, expected)
        for term in expected.terms:
            assert term in index
            assert index.document_frequency(term) == expected.document_frequency(term)


def test_an_empty_directory_is_an_empty_index(tmp_path: Path) -> None:
    with DirectoryIndex(tmp_path) as index:
        assert index.document_count == 0
        assert index.vocabulary_size == 0
        assert index.postings("anything") == {}
        assert "anything" not in index
        assert index.generation == 0


def test_querying_a_directory_finds_the_right_document(tmp_path: Path) -> None:
    write_segments(tmp_path, TEXTS, buffer=2)
    with DirectoryIndex(tmp_path) as index:
        assert search(index, "inverted") == {1}
        assert search(index, '"lazy dog"') == {3}


@pytest.mark.parametrize("scorer", [Ranker, BM25Ranker])
def test_both_scorers_rank_a_directory(
    tmp_path: Path, scorer: type[Ranker | BM25Ranker]
) -> None:
    """Ranking depends on the protocol, so it cannot tell how many files there are."""
    write_segments(tmp_path, TEXTS, buffer=2)
    with DirectoryIndex(tmp_path) as index:
        candidates = search(index, "search engines")
        ranked = scorer(index).rank("search engines", candidates, limit=5)
    assert ranked
    assert ranked[0][0] == 0


def test_a_deleted_document_stops_being_found(tmp_path: Path) -> None:
    """Its postings are still in the file, and it must not appear anyway."""
    write_segments(tmp_path, TEXTS, buffer=2)
    with IndexWriter(tmp_path) as writer:
        assert writer.delete(1) is True
    with DirectoryIndex(tmp_path) as index:
        assert index.document_count == 3
        assert search(index, "inverted") == set()
        assert 1 not in index.document_ids
        assert 1 not in index.postings("documents")


def test_a_replaced_document_is_found_only_in_its_new_form(tmp_path: Path) -> None:
    write_segments(tmp_path, TEXTS, buffer=2)
    with IndexWriter(tmp_path) as writer:
        writer.add(1, "completely different wording")
    with DirectoryIndex(tmp_path) as index:
        assert index.document_count == 4
        assert search(index, "inverted") == set()
        assert search(index, "wording") == {1}


def test_document_frequency_excludes_deleted_documents(tmp_path: Path) -> None:
    """A count including the dead would make every term weight wrong."""
    write_segments(tmp_path, TEXTS, buffer=2)
    with DirectoryIndex(tmp_path) as index:
        before = index.document_frequency("document")
    with IndexWriter(tmp_path) as writer:
        writer.delete(0)
    with DirectoryIndex(tmp_path) as index:
        assert index.document_frequency("document") == before - 1


def test_deleting_everything_leaves_an_empty_index(tmp_path: Path) -> None:
    write_segments(tmp_path, TEXTS, buffer=2)
    with IndexWriter(tmp_path) as writer:
        for document_id in TEXTS:
            writer.delete(document_id)
    with DirectoryIndex(tmp_path) as index:
        assert index.document_count == 0
        assert index.postings("documents") == {}
        assert search(index, "documents") == set()


def test_a_segment_the_manifest_does_not_name_is_not_opened(tmp_path: Path) -> None:
    """A crash between writing a segment and publishing it leaves exactly this."""
    write_segments(tmp_path, TEXTS, buffer=2)
    with DirectoryIndex(tmp_path) as index:
        published = index.document_count
    (tmp_path / MANIFEST_NAME).unlink()
    with DirectoryIndex(tmp_path) as index:
        assert index.document_count == 0
    assert published == 4


def test_a_corrupt_segment_is_refused_and_leaves_nothing_open(
    tmp_path: Path,
) -> None:
    write_segments(tmp_path, TEXTS, buffer=2)
    path = tmp_path / segment_name(0)
    data = bytearray(path.read_bytes())
    data[len(SEGMENT_HEADER) + 20] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(SegmentCorruptError):
        DirectoryIndex(tmp_path)
    path.unlink()


def test_the_generation_is_the_one_it_opened_at(tmp_path: Path) -> None:
    """A reader does not see a directory change under it."""
    write_segments(tmp_path, TEXTS, buffer=2)
    with DirectoryIndex(tmp_path) as index:
        first = index.generation
        with IndexWriter(tmp_path) as writer:
            writer.add(99, "added later")
        assert index.generation == first
        assert 99 not in index.document_ids
    with DirectoryIndex(tmp_path) as index:
        assert index.generation > first
        assert 99 in index.document_ids


def test_closing_twice_is_safe(tmp_path: Path) -> None:
    write_segments(tmp_path, TEXTS, buffer=2)
    index = DirectoryIndex(tmp_path)
    index.close()
    index.close()


@settings(deadline=None, max_examples=20)
@given(
    st.lists(st.sampled_from(sorted(TEXTS.values())), min_size=1, max_size=8),
    st.integers(min_value=1, max_value=4),
)
def test_any_split_across_segments_reads_as_one_index(
    tmp_path_factory: pytest.TempPathFactory, texts: list[str], buffer: int
) -> None:
    """However the documents fall across files, the answers must not change."""
    directory = tmp_path_factory.mktemp("directory")
    documents = dict(enumerate(texts))
    write_segments(directory, documents, buffer=buffer)
    expected = build_index(documents)
    with DirectoryIndex(directory) as index:
        assert_same_postings(index, expected)
