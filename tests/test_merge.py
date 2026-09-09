"""Tests for combining segments into larger ones."""

from __future__ import annotations

import tracemalloc
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from search_engine.directory import DirectoryIndex
from search_engine.manifest import SegmentInfo, publish
from search_engine.manifest import read as read_manifest
from search_engine.merge import (
    MERGE_FACTOR,
    merge_once,
    merge_segments,
    select_merge,
    tier_of,
)
from search_engine.tombstones import tombstone_name
from search_engine.writer import IndexWriter, segment_name
from tests.conftest import assert_same_postings, build_index, write_segments

if TYPE_CHECKING:
    from pathlib import Path


def info(name: str, documents: int, deleted: int = 0) -> SegmentInfo:
    return SegmentInfo(name=name, documents=documents, checksum=0, deleted=deleted)


@pytest.mark.parametrize(
    ("documents", "tier"),
    [(0, 0), (1, 0), (3, 0), (4, 1), (15, 1), (16, 2), (63, 2), (64, 3)],
)
def test_tiers_are_powers_of_the_merge_factor(documents: int, tier: int) -> None:
    assert tier_of(documents) == tier


def test_a_tier_below_the_factor_is_not_merged() -> None:
    segments = [info(f"{n}.seg", 1) for n in range(MERGE_FACTOR - 1)]
    assert select_merge(segments) == []


def test_a_full_tier_is_selected() -> None:
    segments = [info(f"{n}.seg", 1) for n in range(MERGE_FACTOR)]
    assert select_merge(segments) == segments


def test_the_smallest_full_tier_is_selected_first() -> None:
    small = [info(f"s{n}.seg", 1) for n in range(MERGE_FACTOR)]
    large = [info(f"l{n}.seg", 100) for n in range(MERGE_FACTOR)]
    assert select_merge([*large, *small]) == small


def test_a_tier_is_judged_on_live_documents_not_stored_ones() -> None:
    """A segment whose documents are mostly deleted is a small segment."""
    segments = [info(f"{n}.seg", 100, deleted=99) for n in range(MERGE_FACTOR)]
    assert tier_of(segments[0].live) == 0
    assert select_merge(segments) == segments


def test_merging_produces_the_union_of_its_inputs(tmp_path: Path) -> None:
    texts = {n: f"document number {n} about search and ranking" for n in range(6)}
    write_segments(tmp_path, texts)
    manifest = read_manifest(tmp_path)
    merge_segments(tmp_path, manifest.segments, "merged.seg")

    expected = build_index(texts)
    published = manifest.with_segments(
        [info("merged.seg", len(texts))], manifest.next_segment + 1
    )
    publish(tmp_path, published)
    with DirectoryIndex(tmp_path) as index:
        assert index.segment_count == 1
        assert_same_postings(index, expected)


def test_deleted_documents_are_not_carried_across(tmp_path: Path) -> None:
    """This is where the space a tombstone marked actually comes back."""
    texts = {n: f"alpha beta document {n}" for n in range(MERGE_FACTOR)}
    with IndexWriter(tmp_path, buffer_documents=1, merge=False) as writer:
        for document_id, text in texts.items():
            writer.add(document_id, text)
        writer.delete(0)

    manifest = read_manifest(tmp_path)
    merge_once(tmp_path, manifest)
    with DirectoryIndex(tmp_path) as index:
        assert index.segment_count == 1
        assert index.document_count == MERGE_FACTOR - 1
        assert 0 not in index.document_ids
        assert 0 not in index.postings("alpha")
    assert not (tmp_path / tombstone_name(segment_name(0))).exists()


def test_the_inputs_are_deleted_only_after_the_manifest_is_published(
    tmp_path: Path,
) -> None:
    texts = {n: f"document {n}" for n in range(MERGE_FACTOR)}
    write_segments(tmp_path, texts)
    before = {segment.name for segment in read_manifest(tmp_path).segments}
    merge_once(tmp_path, read_manifest(tmp_path))
    after = {segment.name for segment in read_manifest(tmp_path).segments}

    assert len(after) == 1
    assert after.isdisjoint(before)
    for name in before:
        assert not (tmp_path / name).exists()


def test_nothing_is_merged_when_no_tier_is_full(tmp_path: Path) -> None:
    write_segments(tmp_path, {n: f"document {n}" for n in range(MERGE_FACTOR - 1)})
    assert merge_once(tmp_path, read_manifest(tmp_path)) is None


def test_a_merge_of_wholly_deleted_segments_produces_an_empty_one(
    tmp_path: Path,
) -> None:
    """Every document deleted means an empty output, which must still be valid.

    It also shows what a merge reclaims. Before it, the terms of deleted
    documents are still in the files and still counted; afterwards they are
    gone, which is the space a tombstone only promised.
    """
    texts = {n: f"document {n}" for n in range(MERGE_FACTOR + 1)}
    with IndexWriter(tmp_path, buffer_documents=1, merge=False) as writer:
        for document_id, text in texts.items():
            writer.add(document_id, text)
        for document_id in texts:
            writer.delete(document_id)

    with DirectoryIndex(tmp_path) as index:
        assert index.document_count == 0
        assert index.vocabulary_size > 0

    assert merge_once(tmp_path, read_manifest(tmp_path)) is not None
    with DirectoryIndex(tmp_path) as index:
        assert index.document_count == 0
        assert index.vocabulary_size == 0


def test_merging_repeats_until_no_tier_is_full(tmp_path: Path) -> None:
    """One merge fills the tier above, so the policy is applied until it stops."""
    texts = {n: f"document {n}" for n in range(MERGE_FACTOR * MERGE_FACTOR)}
    with IndexWriter(tmp_path, buffer_documents=1) as writer:
        for document_id, text in texts.items():
            writer.add(document_id, text)
    with DirectoryIndex(tmp_path) as index:
        assert index.segment_count < MERGE_FACTOR
        assert index.document_count == len(texts)


def test_segments_sharing_every_term_and_sharing_none(tmp_path: Path) -> None:
    """A wrong merge key passes one of these and fails the other.

    Sharing every term exercises the common path where postings combine;
    sharing none exercises the path where each term comes from one input only.
    """
    shared = dict.fromkeys(range(MERGE_FACTOR), "alpha beta gamma")
    write_segments(tmp_path, shared)
    merge_once(tmp_path, read_manifest(tmp_path))
    with DirectoryIndex(tmp_path) as index:
        assert sorted(index.terms) == ["alpha", "beta", "gamma"]
        assert index.document_frequency("alpha") == MERGE_FACTOR


def test_segments_with_no_terms_in_common(tmp_path: Path) -> None:
    distinct = {n: f"word{n}" for n in range(MERGE_FACTOR)}
    write_segments(tmp_path, distinct)
    merge_once(tmp_path, read_manifest(tmp_path))
    with DirectoryIndex(tmp_path) as index:
        assert sorted(index.terms) == sorted(f"word{n}" for n in range(MERGE_FACTOR))
        for n in range(MERGE_FACTOR):
            assert index.document_frequency(f"word{n}") == 1


def test_merging_holds_far_less_than_building_the_index(tmp_path: Path) -> None:
    """Streaming is the point: memory tracks the vocabulary, not the postings.

    The comparison is against the thing this design exists to avoid, which is
    materialising the merged index before writing it. Both sides are measured
    the same way in the same run, so the ratio means something a fixed
    threshold would not.
    """
    texts = {
        n: " ".join(f"term{value}" for value in range(n % 50, n % 50 + 200))
        for n in range(MERGE_FACTOR * 60)
    }
    write_segments(tmp_path, texts, buffer=60)
    manifest = read_manifest(tmp_path)

    tracemalloc.start()
    merged = merge_segments(tmp_path, manifest.segments, "merged.seg")
    _, streaming = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    materialised = build_index(texts)
    _, in_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert materialised.document_count == len(texts)

    assert merged.documents == len(texts)
    assert streaming < in_memory / 3


@settings(deadline=None, max_examples=15)
@given(
    st.lists(
        st.sampled_from(["alpha beta", "beta gamma", "gamma delta", "delta alpha"]),
        min_size=MERGE_FACTOR,
        max_size=12,
    )
)
def test_a_merged_segment_answers_the_same_as_its_inputs(
    tmp_path_factory: pytest.TempPathFactory, texts: list[str]
) -> None:
    directory = tmp_path_factory.mktemp("merge")
    documents = dict(enumerate(texts))
    write_segments(directory, documents)

    with DirectoryIndex(directory) as index:
        before = {term: dict(index.postings(term)) for term in index.terms}
    merge_once(directory, read_manifest(directory))
    with DirectoryIndex(directory) as index:
        after = {term: dict(index.postings(term)) for term in index.terms}
    assert after == before


def test_a_failed_merge_closes_what_it_already_opened(tmp_path: Path) -> None:
    """Windows cannot delete a mapped file, so deleting proves it was closed."""
    texts = {n: f"document {n}" for n in range(MERGE_FACTOR)}
    write_segments(tmp_path, texts)
    manifest = read_manifest(tmp_path)
    missing = SegmentInfo(name="segment-99999999.seg", documents=1, checksum=0)

    with pytest.raises(OSError, match="segment-99999999"):
        merge_segments(tmp_path, [*manifest.segments, missing], "merged.seg")

    for segment in manifest.segments:
        (tmp_path / segment.name).unlink()
