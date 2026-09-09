"""Tests for the bitset that records which documents are no longer visible."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from search_engine.tombstones import (
    TombstoneFormatError,
    Tombstones,
    tombstone_name,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_a_new_set_has_nothing_deleted() -> None:
    marks = Tombstones.empty(10)
    assert marks.deleted_count == 0
    assert marks.any_deleted is False
    assert all(not marks.is_deleted(ordinal) for ordinal in range(10))


def test_deleting_marks_only_that_document() -> None:
    marks = Tombstones.empty(10)
    assert marks.delete(3) is True
    assert marks.is_deleted(3)
    assert marks.deleted_count == 1
    assert marks.any_deleted is True
    assert not marks.is_deleted(2)
    assert not marks.is_deleted(4)


def test_deleting_twice_changes_nothing_the_second_time() -> None:
    """At-least-once delivery means the same instruction can arrive twice."""
    marks = Tombstones.empty(10)
    assert marks.delete(3) is True
    assert marks.delete(3) is False
    assert marks.deleted_count == 1


@pytest.mark.parametrize("count", [1, 7, 8, 9, 16, 17])
def test_every_ordinal_can_be_deleted(count: int) -> None:
    """Byte boundaries are where a bitset goes wrong, so cross several."""
    marks = Tombstones.empty(count)
    for ordinal in range(count):
        assert marks.delete(ordinal) is True
    assert marks.deleted_count == count
    assert all(marks.is_deleted(ordinal) for ordinal in range(count))


@pytest.mark.parametrize("ordinal", [-1, 10, 999])
def test_an_ordinal_outside_the_segment_is_refused(ordinal: int) -> None:
    marks = Tombstones.empty(10)
    with pytest.raises(IndexError):
        marks.delete(ordinal)
    with pytest.raises(IndexError):
        marks.is_deleted(ordinal)


def test_a_set_over_no_documents_holds_nothing() -> None:
    marks = Tombstones.empty(0)
    assert marks.deleted_count == 0
    assert bytes(marks) == b""
    with pytest.raises(IndexError):
        marks.is_deleted(0)


def test_a_written_set_reads_back_unchanged(tmp_path: Path) -> None:
    marks = Tombstones.empty(20)
    marks.delete(0)
    marks.delete(19)
    path = tmp_path / "a.del"
    marks.write(path)
    restored = Tombstones.read(path, 20)
    assert bytes(restored) == bytes(marks)
    assert restored.deleted_count == 2
    assert restored.is_deleted(0)
    assert restored.is_deleted(19)


def test_writing_leaves_no_temporary_behind(tmp_path: Path) -> None:
    path = tmp_path / "a.del"
    Tombstones.empty(8).write(path)
    assert [item.name for item in tmp_path.iterdir()] == ["a.del"]


def test_a_missing_file_reads_as_nothing_deleted(tmp_path: Path) -> None:
    """A segment nobody has deleted from has no tombstone file at all."""
    marks = Tombstones.read(tmp_path / "absent.del", 12)
    assert marks.deleted_count == 0
    assert marks.document_count == 12


def test_a_file_of_the_wrong_length_is_refused(tmp_path: Path) -> None:
    """The adversarial case: a tombstone file paired with the wrong segment.

    Reading it against a different document count would silently hide or reveal
    the wrong documents, which is worse than refusing to open it.
    """
    path = tmp_path / "a.del"
    Tombstones.empty(8).write(path)
    with pytest.raises(TombstoneFormatError, match="needs"):
        Tombstones.read(path, 200)


def test_the_name_follows_the_segment_it_belongs_to() -> None:
    assert tombstone_name("segment-00000007.seg") == "segment-00000007.del"


@settings(deadline=None)
@given(
    st.integers(min_value=1, max_value=200),
    st.data(),
)
def test_the_bitset_agrees_with_a_plain_set(count: int, data: st.DataObject) -> None:
    ordinals = data.draw(
        st.lists(st.integers(min_value=0, max_value=count - 1), max_size=60)
    )
    marks = Tombstones.empty(count)
    expected: set[int] = set()
    for ordinal in ordinals:
        assert marks.delete(ordinal) is (ordinal not in expected)
        expected.add(ordinal)
    assert marks.deleted_count == len(expected)
    assert all(
        marks.is_deleted(ordinal) == (ordinal in expected) for ordinal in range(count)
    )
