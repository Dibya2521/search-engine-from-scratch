"""Tests for the log that makes an accepted document survive a crash."""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from search_engine.codecs import encode_number
from search_engine.directory import DirectoryIndex
from search_engine.wal import LOG_NAME, Kind, Record, WriteAheadLog
from search_engine.writer import IndexWriter

if TYPE_CHECKING:
    from pathlib import Path


def log_at(tmp_path: Path) -> WriteAheadLog:
    return WriteAheadLog(tmp_path / LOG_NAME)


def test_a_record_replays_exactly_as_it_was_written(tmp_path: Path) -> None:
    with log_at(tmp_path) as log:
        log.append(Record(Kind.ADD, 7, "alpha beta"))
        log.append(Record(Kind.DELETE, 3))
    assert list(log_at(tmp_path).replay()) == [
        Record(Kind.ADD, 7, "alpha beta"),
        Record(Kind.DELETE, 3),
    ]


def test_a_missing_log_replays_as_nothing(tmp_path: Path) -> None:
    assert list(log_at(tmp_path).replay()) == []


def test_truncating_discards_the_log(tmp_path: Path) -> None:
    with log_at(tmp_path) as log:
        log.append(Record(Kind.ADD, 1, "alpha"))
        log.truncate()
    assert not (tmp_path / LOG_NAME).exists()
    assert list(log_at(tmp_path).replay()) == []


def test_closing_twice_is_safe(tmp_path: Path) -> None:
    log = log_at(tmp_path)
    log.append(Record(Kind.ADD, 1, "alpha"))
    log.close()
    log.close()


@pytest.mark.parametrize("cut", [1, 3, 6, 9])
def test_a_torn_tail_stops_replay_and_keeps_what_came_before(
    tmp_path: Path, cut: int
) -> None:
    """A crash mid-append leaves exactly this, and everything before it is good."""
    with log_at(tmp_path) as log:
        log.append(Record(Kind.ADD, 1, "first"))
        log.append(Record(Kind.ADD, 2, "second"))
        log.append(Record(Kind.ADD, 3, "third and longest record"))
    path = tmp_path / LOG_NAME
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - cut])

    replayed = list(log_at(tmp_path).replay())
    assert replayed[:2] == [
        Record(Kind.ADD, 1, "first"),
        Record(Kind.ADD, 2, "second"),
    ]
    assert len(replayed) == 2


def test_a_flipped_byte_stops_replay_at_that_record(tmp_path: Path) -> None:
    with log_at(tmp_path) as log:
        log.append(Record(Kind.ADD, 1, "first"))
        log.append(Record(Kind.ADD, 2, "second"))
    path = tmp_path / LOG_NAME
    data = bytearray(path.read_bytes())
    data[-3] ^= 0xFF
    path.write_bytes(bytes(data))

    assert list(log_at(tmp_path).replay()) == [Record(Kind.ADD, 1, "first")]


def test_a_record_of_an_unknown_kind_stops_replay(tmp_path: Path) -> None:
    """The adversarial case: a well-formed record this build does not understand."""
    payload = encode_number(99) + encode_number(4)
    framed = encode_number(len(payload)) + payload
    framed += zlib.crc32(payload).to_bytes(4, "big")
    (tmp_path / LOG_NAME).write_bytes(framed)

    assert list(log_at(tmp_path).replay()) == []


def test_awkward_text_survives_the_round_trip(tmp_path: Path) -> None:
    """Newlines, tabs, markup, a NUL and a code point outside the BMP."""
    text = "line\nbreak\ttab <page> " + chr(0) + chr(0x1F600) + chr(0x00E9)
    with log_at(tmp_path) as log:
        log.append(Record(Kind.ADD, 1, text))
    assert list(log_at(tmp_path).replay()) == [Record(Kind.ADD, 1, text)]


def test_writing_without_syncing_still_replays(tmp_path: Path) -> None:
    """Turning off the sync is for measuring its cost, not for changing meaning."""
    log = WriteAheadLog(tmp_path / LOG_NAME, sync=False)
    log.append(Record(Kind.ADD, 1, "alpha"))
    log.close()
    assert list(log_at(tmp_path).replay()) == [Record(Kind.ADD, 1, "alpha")]


@settings(deadline=None)
@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=10_000),
            st.text(max_size=40),
        ),
        max_size=25,
    )
)
def test_any_sequence_of_records_round_trips(
    tmp_path_factory: pytest.TempPathFactory, rows: list[tuple[int, str]]
) -> None:
    directory = tmp_path_factory.mktemp("wal")
    records = [Record(Kind.ADD, document_id, text) for document_id, text in rows]
    with WriteAheadLog(directory / LOG_NAME) as log:
        for record in records:
            log.append(record)
    assert list(WriteAheadLog(directory / LOG_NAME).replay()) == records


def test_a_writer_that_dies_before_flushing_loses_nothing(tmp_path: Path) -> None:
    """The crash that matters: accepted, buffered, never written, process gone.

    The state a crash leaves is a log with records and a directory with no
    segments, so that state is built directly rather than by abandoning a
    writer object, which would only be testing the garbage collector.
    """
    with WriteAheadLog(tmp_path / LOG_NAME) as log:
        log.append(Record(Kind.ADD, 1, "alpha beta"))
        log.append(Record(Kind.ADD, 2, "gamma delta"))
        log.append(Record(Kind.DELETE, 1))

    with DirectoryIndex(tmp_path) as index:
        assert index.document_count == 0

    with IndexWriter(tmp_path, buffer_documents=1_000) as recovered:
        assert recovered.buffered == 1
    with DirectoryIndex(tmp_path) as index:
        assert index.document_count == 1
        assert 2 in index.document_ids
        assert 1 not in index.document_ids


def test_the_log_is_discarded_once_its_documents_are_published(
    tmp_path: Path,
) -> None:
    with IndexWriter(tmp_path, buffer_documents=1_000) as writer:
        writer.add(1, "alpha")
    assert not (tmp_path / LOG_NAME).exists()


def test_replaying_documents_already_published_changes_nothing(
    tmp_path: Path,
) -> None:
    """A crash after publishing and before truncating replays what is indexed.

    That is harmless only because adding is an upsert, which is why upsert was
    built before this was. Simulated by leaving the log in place afterwards.
    """
    with IndexWriter(tmp_path, buffer_documents=1_000) as writer:
        writer.add(1, "alpha")
        writer.add(2, "beta")
    with WriteAheadLog(tmp_path / LOG_NAME) as log:
        log.append(Record(Kind.ADD, 1, "alpha"))
        log.append(Record(Kind.ADD, 2, "beta"))

    with IndexWriter(tmp_path, buffer_documents=1_000):
        pass
    with DirectoryIndex(tmp_path) as index:
        assert index.document_count == 2
        assert sorted(index.document_ids) == [1, 2]


def frame(payload: bytes) -> bytes:
    """Wrap a payload the way the log does, with a correct checksum.

    A log damaged by cutting bytes off fails its checksum before the payload is
    ever parsed, so reaching the payload's own error handling needs a record
    that is malformed and correctly checksummed.
    """
    return (
        encode_number(len(payload)) + payload + zlib.crc32(payload).to_bytes(4, "big")
    )


def test_a_log_ending_inside_a_length_stops_replay(tmp_path: Path) -> None:
    """One byte with no terminator: the length that would locate the record."""
    (tmp_path / LOG_NAME).write_bytes(b"\x00")
    assert list(log_at(tmp_path).replay()) == []


def test_a_record_claiming_more_text_than_it_holds_stops_replay(
    tmp_path: Path,
) -> None:
    payload = encode_number(int(Kind.ADD)) + encode_number(1) + encode_number(50)
    (tmp_path / LOG_NAME).write_bytes(frame(payload + b"abc"))
    assert list(log_at(tmp_path).replay()) == []


def test_a_record_ending_before_its_text_length_stops_replay(
    tmp_path: Path,
) -> None:
    payload = encode_number(int(Kind.ADD)) + encode_number(1)
    (tmp_path / LOG_NAME).write_bytes(frame(payload))
    assert list(log_at(tmp_path).replay()) == []
