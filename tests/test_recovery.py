"""What survives a crash, tested where it is experienced rather than built.

Every failure mode here is already tested in the module that implements it. None
of those tests prove the property the design exists to provide, because recovery
is spread across the writer, the manifest, the tombstones and the log, and each
part is correct on its own while the whole could still lose a document.

So these tests build the exact state a crash leaves on disk and then open the
directory. No new behaviour is added by this file: if a scenario cannot be made
to pass, the fix belongs in the module that owns it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from search_engine.directory import DirectoryIndex
from search_engine.manifest import MANIFEST_NAME, TEMPORARY_NAME
from search_engine.manifest import read as read_manifest
from search_engine.merge import merge_once
from search_engine.query import search
from search_engine.segment import SEGMENT_HEADER, SegmentCorruptError
from search_engine.wal import LOG_NAME, Kind, Record, WriteAheadLog
from search_engine.writer import IndexWriter, segment_name

if TYPE_CHECKING:
    from pathlib import Path

DOCUMENTS = {
    1: "web search engines rank documents by relevance",
    2: "an inverted index maps every term to the documents holding it",
    3: "a lazy dog sleeps through the afternoon",
    4: "ranking is what separates a search engine from a grep",
}


def visible(directory: Path) -> set[int]:
    with DirectoryIndex(directory) as index:
        return set(index.document_ids)


def test_a_crash_after_writing_a_segment_and_before_publishing(
    tmp_path: Path,
) -> None:
    """The segment file exists and nothing names it, so it does not exist.

    Nothing is lost, because the documents it holds are still in the log: the
    log is truncated only after the manifest names them.
    """
    with WriteAheadLog(tmp_path / LOG_NAME) as log:
        for document_id, text in DOCUMENTS.items():
            log.append(Record(Kind.ADD, document_id, text))
    orphan = tmp_path / segment_name(0)
    with IndexWriter(tmp_path, buffer_documents=1_000, merge=False) as writer:
        writer.flush()
    published = read_manifest(tmp_path)

    assert orphan.is_file()
    assert [segment.name for segment in published.segments] == [orphan.name]
    assert visible(tmp_path) == set(DOCUMENTS)


def test_an_orphan_segment_is_ignored_and_its_number_is_not_reused(
    tmp_path: Path,
) -> None:
    with IndexWriter(tmp_path, buffer_documents=1_000, merge=False) as writer:
        writer.add(1, "alpha")
    (tmp_path / MANIFEST_NAME).unlink()
    orphan = tmp_path / segment_name(0)
    assert orphan.is_file()

    assert visible(tmp_path) == set()
    with IndexWriter(tmp_path, buffer_documents=1_000, merge=False) as writer:
        writer.add(2, "beta")
    assert (tmp_path / segment_name(1)).is_file()
    assert visible(tmp_path) == {2}


def test_a_crash_during_the_manifest_write_leaves_the_old_one_standing(
    tmp_path: Path,
) -> None:
    """A temporary file is what a crash mid-publish leaves behind."""
    with IndexWriter(tmp_path, buffer_documents=1_000, merge=False) as writer:
        writer.add(1, "alpha beta")
    before = read_manifest(tmp_path)
    (tmp_path / TEMPORARY_NAME).write_text("{half written", encoding="utf-8")

    assert read_manifest(tmp_path) == before
    assert visible(tmp_path) == {1}


def test_a_crash_mid_append_keeps_every_intact_record(tmp_path: Path) -> None:
    """The torn record is dropped and everything written before it survives."""
    with WriteAheadLog(tmp_path / LOG_NAME) as log:
        for document_id, text in DOCUMENTS.items():
            log.append(Record(Kind.ADD, document_id, text))
        log.append(Record(Kind.ADD, 5, "a record that never finished being written"))
    path = tmp_path / LOG_NAME
    data = path.read_bytes()
    path.write_bytes(data[:-12])

    with IndexWriter(tmp_path, buffer_documents=1_000, merge=False):
        pass
    assert visible(tmp_path) == set(DOCUMENTS)


def test_a_crash_after_publishing_and_before_truncating_changes_nothing(
    tmp_path: Path,
) -> None:
    """Replay re-adds documents already indexed, and upsert makes that a no-op.

    The count is what matters. Asserting only that it does not raise would pass
    just as well if every document were indexed twice.
    """
    with IndexWriter(tmp_path, buffer_documents=1_000, merge=False) as writer:
        for document_id, text in DOCUMENTS.items():
            writer.add(document_id, text)
    with WriteAheadLog(tmp_path / LOG_NAME) as log:
        for document_id, text in DOCUMENTS.items():
            log.append(Record(Kind.ADD, document_id, text))

    with IndexWriter(tmp_path, buffer_documents=1_000, merge=False):
        pass
    with DirectoryIndex(tmp_path) as index:
        assert index.document_count == len(DOCUMENTS)
        assert set(index.document_ids) == set(DOCUMENTS)
        assert search(index, "inverted") == {2}


def test_a_delete_replayed_after_publishing_stays_deleted(tmp_path: Path) -> None:
    """The other direction: recovery must not undo a delete it already applied."""
    with IndexWriter(tmp_path, buffer_documents=1_000, merge=False) as writer:
        for document_id, text in DOCUMENTS.items():
            writer.add(document_id, text)
        writer.delete(3)
    with WriteAheadLog(tmp_path / LOG_NAME) as log:
        log.append(Record(Kind.DELETE, 3))

    with IndexWriter(tmp_path, buffer_documents=1_000, merge=False):
        pass
    assert visible(tmp_path) == set(DOCUMENTS) - {3}


def test_a_corrupt_segment_named_by_the_manifest_is_refused(tmp_path: Path) -> None:
    """Wrong answers forever are worse than a failure to open."""
    with IndexWriter(tmp_path, buffer_documents=1_000, merge=False) as writer:
        for document_id, text in DOCUMENTS.items():
            writer.add(document_id, text)
    path = tmp_path / segment_name(0)
    data = bytearray(path.read_bytes())
    data[len(SEGMENT_HEADER) + 30] ^= 0xFF
    path.write_bytes(bytes(data))

    with pytest.raises(SegmentCorruptError):
        DirectoryIndex(tmp_path)
    path.unlink()


def test_a_segment_the_manifest_names_but_the_disk_lacks_is_refused(
    tmp_path: Path,
) -> None:
    with IndexWriter(tmp_path, buffer_documents=1_000, merge=False) as writer:
        writer.add(1, "alpha")
    (tmp_path / segment_name(0)).unlink()

    with pytest.raises(OSError, match=segment_name(0)):
        DirectoryIndex(tmp_path)


def test_a_crash_between_merging_and_deleting_the_inputs(tmp_path: Path) -> None:
    """The merged output is published and the inputs are still on the disk.

    They are orphans that nothing names, so the answers are already correct and
    the leftover files are only wasted space.
    """
    texts = {n: f"document number {n} about ranking" for n in range(8)}
    with IndexWriter(tmp_path, buffer_documents=1, merge=False) as writer:
        for document_id, text in texts.items():
            writer.add(document_id, text)

    before = {segment.name for segment in read_manifest(tmp_path).segments}
    merged = merge_once(tmp_path, read_manifest(tmp_path))
    assert merged is not None
    for name in before:
        (tmp_path / name).write_bytes(b"an orphan nothing names")

    with DirectoryIndex(tmp_path) as index:
        assert set(index.document_ids) == set(texts)


def test_nothing_accepted_is_lost_across_a_crash_at_any_point(
    tmp_path: Path,
) -> None:
    """The property the whole design exists for, stated once as a test.

    Every prefix of a stream of operations is written to the log and nothing
    else, which is the state a crash before any flush leaves. Recovery then has
    to produce exactly what that prefix asked for, no matter where it stopped.
    """
    operations: list[tuple[Kind, int, str]] = [
        (Kind.ADD, 1, DOCUMENTS[1]),
        (Kind.ADD, 2, DOCUMENTS[2]),
        (Kind.DELETE, 1, ""),
        (Kind.ADD, 3, DOCUMENTS[3]),
        (Kind.ADD, 2, "replaced text for document two"),
        (Kind.DELETE, 3, ""),
    ]
    for cut in range(1, len(operations) + 1):
        directory = tmp_path / f"crash-{cut}"
        directory.mkdir()
        expected: set[int] = set()
        with WriteAheadLog(directory / LOG_NAME) as log:
            for kind, document_id, text in operations[:cut]:
                log.append(Record(kind, document_id, text))
                if kind is Kind.ADD:
                    expected.add(document_id)
                else:
                    expected.discard(document_id)

        with IndexWriter(directory, buffer_documents=2, merge=False):
            pass
        assert visible(directory) == expected
