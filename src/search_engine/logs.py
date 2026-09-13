"""One line of JSON per event, and one identifier tying them together.

"The query was slow" is not something anyone can act on. Which query, how many
candidates it touched, how long each phase took, and which other lines belong to
the same request: those are what turn a pile of lines into a trace, and none of
them survive being formatted into a sentence.

**Built on `logging` rather than beside it.** The standard library already has
levels, hierarchy, handlers and an operator who knows how to configure them.
What it does not have is a formatter that emits JSON, so that is all this adds.

**One object per line, always.** `json.dumps` escapes every newline, quote and
backslash it is given, so a query containing any of them cannot break the
guarantee that a line is a record. That is the whole reason the text is not
formatted into the message.

**A `ContextVar` rather than a thread local**, because it is correct under
threads and under coroutines alike. The serving layer is threaded now and does
not have to stay that way.

**Never log a result set.** It is unbounded, it is the largest thing in the
request, and writing it is the fastest way to fill a disk. One line about the
result, never the result.

**Logging a query is logging what a person searched for.** In most places that
is regulated data with a retention policy attached, so `configure` takes a
switch that drops the text from every record.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import IO

LOGGER_NAME: Final = "search_engine"

correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")

# Everything `logging` puts on a record itself. Anything else came from a
# caller's `extra` and belongs in the object.
_RESERVED: Final = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """Formats records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the record as a single line of JSON.

        Values that JSON cannot represent are rendered with `str` rather than
        raising. A log call must not be able to break the thing it is
        reporting on.
        """
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in _RESERVED
            }
        )
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class Correlate(logging.Filter):
    """Stamps each record with the identifier in force where it was made."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add the identifier if there is one, and never drop a record."""
        identifier = correlation_id.get()
        if identifier:
            record.correlation_id = identifier
        return True


class RedactQuery(logging.Filter):
    """Removes the query text from every record.

    A filter rather than a flag the call sites read, so the decision lives in
    the configuration where an operator can see it, and no code that logs has
    to know what the policy is.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Drop the query text if the record carries any, and keep the record."""
        record.__dict__.pop("query", None)
        return True


def configure(
    stream: IO[str], level: int = logging.INFO, *, log_queries: bool = True
) -> logging.Logger:
    """Install the JSON formatter on the package logger and return it.

    Any handler an earlier call installed is removed first, so configuring
    twice does not print every line twice.

    Nothing is emitted until this is called. A library that logs to standard
    error whether or not it was asked to is a library that has to be silenced
    before it can be used.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(Correlate())
    if not log_queries:
        handler.addFilter(RedactQuery())
    logger.addHandler(handler)
    logger.setLevel(level)
    # The package handles its own records, so a root handler configured by an
    # application does not print each of them a second time.
    logger.propagate = False
    return logger


@contextmanager
def correlated(identifier: str | None = None) -> Generator[str]:
    """Bind a correlation identifier for the duration of a block.

    Yields the identifier in force, generating one when the caller has none to
    supply, which is what a command line has and what a server given a request
    header does not.

    Restores whatever was bound before, so a nested block does not leave the
    outer one unidentified.
    """
    chosen = identifier or uuid.uuid4().hex
    token = correlation_id.set(chosen)
    try:
        yield chosen
    finally:
        correlation_id.reset(token)
