"""Tests for the structured log lines and the identifier that ties them together."""

from __future__ import annotations

import io
import json
import logging
import threading
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from search_engine.logs import LOGGER_NAME, configure, correlated
from search_engine.query import search
from tests.conftest import CORPUS, build_index

AWKWARD = 'a "quoted" phrase\nwith a newline and a \\ backslash'


def emitted(stream: io.StringIO) -> list[dict[str, Any]]:
    """Return every record written so far, parsed."""
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def logger() -> logging.Logger:
    """Return a logger under the configured package logger."""
    return logging.getLogger(f"{LOGGER_NAME}.test")


def test_a_record_is_one_line_of_json(log_stream: io.StringIO) -> None:
    logger().info("started", extra={"documents": 7})
    assert len(log_stream.getvalue().splitlines()) == 1
    record = emitted(log_stream)[0]
    assert record["event"] == "started"
    assert record["level"] == "INFO"
    assert record["documents"] == 7
    assert record["time"].endswith("+00:00")


def test_awkward_text_cannot_break_the_one_object_per_line_rule(
    log_stream: io.StringIO,
) -> None:
    """A newline in a query would otherwise split one record into two."""
    logger().info("query", extra={"query": AWKWARD})
    assert len(log_stream.getvalue().splitlines()) == 1
    assert emitted(log_stream)[0]["query"] == AWKWARD


def test_the_identifier_is_on_every_record_inside_the_block_and_none_outside(
    log_stream: io.StringIO,
) -> None:
    logger().info("before")
    with correlated() as identifier:
        logger().info("during")
        logger().info("also during")
    logger().info("after")

    records = emitted(log_stream)
    assert "correlation_id" not in records[0]
    assert records[1]["correlation_id"] == identifier
    assert records[2]["correlation_id"] == identifier
    assert "correlation_id" not in records[3]


def test_a_nested_block_restores_the_one_outside_it(log_stream: io.StringIO) -> None:
    with correlated("outer") as outer:
        with correlated("inner"):
            logger().info("deep")
        logger().info("shallow")
    assert outer == "outer"
    assert [record["correlation_id"] for record in emitted(log_stream)] == [
        "inner",
        "outer",
    ]


def test_an_identifier_that_was_supplied_is_the_one_used(
    log_stream: io.StringIO,
) -> None:
    """A server given one in a request header must not invent another."""
    with correlated("from-a-header"):
        logger().info("served")
    assert emitted(log_stream)[0]["correlation_id"] == "from-a-header"


def test_a_generated_identifier_is_not_the_same_twice() -> None:
    with correlated() as first, correlated() as second:
        assert first != second


def fail() -> None:
    """Raise, so the traceback in the record has something real in it."""
    message = "deliberate"
    raise RuntimeError(message)


def test_an_exception_carries_its_traceback(log_stream: io.StringIO) -> None:
    try:
        fail()
    except RuntimeError:
        logger().exception("failed")
    record = emitted(log_stream)[0]
    assert record["level"] == "ERROR"
    assert "RuntimeError: deliberate" in record["traceback"]


def test_a_value_json_cannot_hold_is_rendered_rather_than_raising(
    log_stream: io.StringIO,
) -> None:
    """A log call must not be able to break the thing it is reporting on."""
    logger().info("odd", extra={"path": object()})
    assert "object object at" in emitted(log_stream)[0]["path"]


def test_a_record_from_a_thread_that_never_correlated_does_not_raise(
    log_stream: io.StringIO,
) -> None:
    """A context variable is per context, and a fresh thread has an empty one."""
    with correlated("main"):
        thread = threading.Thread(target=lambda: logger().info("from a thread"))
        thread.start()
        thread.join()
    records = emitted(log_stream)
    assert records[0]["event"] == "from a thread"
    assert "correlation_id" not in records[0]


def test_configuring_twice_does_not_print_every_line_twice(
    log_stream: io.StringIO,
) -> None:
    configure(log_stream)
    logger().info("once")
    assert len(log_stream.getvalue().splitlines()) == 1


def test_the_query_text_is_absent_when_it_is_not_to_be_kept(
    log_stream: io.StringIO,
) -> None:
    """What a person searched for is regulated data in most places."""
    configure(log_stream, log_queries=False)
    logger().info("query", extra={"query": "something private", "terms": 2})
    record = emitted(log_stream)[0]
    assert "query" not in record
    assert record["terms"] == 2


def test_nothing_is_emitted_until_logging_is_configured() -> None:
    """A library that writes to stderr uninvited has to be silenced to be used."""
    package = logging.getLogger(LOGGER_NAME)
    assert package.handlers == []
    logger().info("into the void")


def test_a_search_reports_what_it_did(log_stream: io.StringIO) -> None:
    index = build_index(CORPUS)
    with correlated("one-search"):
        search(index, "index")
    record = next(row for row in emitted(log_stream) if row["event"] == "query")
    assert record["correlation_id"] == "one-search"
    assert record["query"] == "index"
    assert record["terms"] == 1
    assert record["phrase"] is False
    assert record["candidates"] == 3
    assert record["duration_ms"] >= 0


def test_a_search_never_logs_the_documents_it_found(log_stream: io.StringIO) -> None:
    """The result set is unbounded and is the fastest way to fill a disk."""
    search(build_index(CORPUS), "index")
    record = next(row for row in emitted(log_stream) if row["event"] == "query")
    assert set(record) == {
        "time",
        "level",
        "event",
        "query",
        "terms",
        "phrase",
        "candidates",
        "duration_ms",
    }


@given(text=st.text(max_size=200))
def test_any_query_text_survives_as_one_line(text: str) -> None:
    stream = io.StringIO()
    configure(stream)
    try:
        logging.getLogger(f"{LOGGER_NAME}.property").info(
            "query", extra={"query": text}
        )
        lines = stream.getvalue().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["query"] == text
    finally:
        package = logging.getLogger(LOGGER_NAME)
        for handler in list(package.handlers):
            package.removeHandler(handler)
        package.propagate = True


def test_a_level_below_the_threshold_is_not_written(log_stream: io.StringIO) -> None:
    configure(log_stream, level=logging.WARNING)
    logger().info("quiet")
    logger().warning("loud")
    assert [record["event"] for record in emitted(log_stream)] == ["loud"]


def test_an_empty_identifier_is_replaced_by_a_made_one(
    log_stream: io.StringIO,
) -> None:
    """An empty string is not an identifier, so it cannot stand as one."""
    with correlated("") as identifier:
        logger().info("worked")
    assert identifier
    assert emitted(log_stream)[0]["correlation_id"] == identifier


def test_an_identifier_that_looks_odd_is_still_the_one_used(
    log_stream: io.StringIO,
) -> None:
    """Only emptiness counts as absent, so a caller's odd choice is honoured."""
    with correlated("0") as identifier:
        logger().info("worked")
    assert identifier == "0"
    assert emitted(log_stream)[0]["correlation_id"] == "0"
