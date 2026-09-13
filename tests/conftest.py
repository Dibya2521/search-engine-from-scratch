"""Shared fixtures and the corpora they are built from.

Fixtures live here rather than beside the tests that use them so that a test
function's parameter does not shadow a module-level function of the same name,
which is what pytest's fixture mechanism would otherwise force. For the same
reason no fixture here depends on another: each builds what it needs.
"""

from __future__ import annotations

import io
import logging
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hypothesis import settings

from search_engine.cli import main as cli_main
from search_engine.index import InvertedIndex, ReadableIndex
from search_engine.logs import LOGGER_NAME, configure
from search_engine.ranking import Ranker
from search_engine.segment import SegmentReader
from search_engine.store import DocumentStore
from search_engine.writer import IndexWriter

if TYPE_CHECKING:
    from collections.abc import Iterator

# The heaviest property example in this suite was measured at 27.5 ms at best
# and 197.8 ms at worst over thirty runs of identical work, against a default
# deadline of 200 ms. A deadline that close to the spread of the machine
# measures the machine. Two seconds is ten times the worst observed and still
# catches an example that has gone quadratic, which is what a deadline is for.
# A test that writes a file per example sets deadline=None for itself.
DEADLINE = timedelta(seconds=2)

settings.register_profile("project", deadline=DEADLINE)
settings.load_profile("project")

SAMPLE_CORPUS = Path(__file__).parent / "fixtures" / "sample_corpus.xml"

CORPUS = {
    1: "Web search engines build an inverted index",
    2: "The index maps terms to documents and positions",
    3: "Searching the web requires an index",
}

# Chosen so the same two words appear adjacent, apart, and in both orders.
PHRASE_CORPUS = {
    1: "The computer science department offers courses",
    2: "Science and computer engineering are related",
    3: "The department of computer science is large",
    4: "Web search engines use an inverted index",
    5: "computer science computer science",
}

# Every ranking figure in the documentation was measured against this.
RANKING_CORPUS = {
    1: "the computer science department teaches computer science",
    2: "a short note about computers",
    3: "science journal about biology and chemistry",
    4: "computer computer computer computer computer",
    5: "unrelated text about gardening",
}


def build_index(corpus: dict[int, str]) -> InvertedIndex:
    """Index a mapping of identifier to text."""
    built = InvertedIndex()
    for document_id, text in corpus.items():
        built.add_document(document_id, text)
    return built


@pytest.fixture
def index() -> InvertedIndex:
    """A three document index, small enough to assert postings by hand."""
    return build_index(CORPUS)


@pytest.fixture
def phrase_index() -> InvertedIndex:
    """An index built for exercising adjacency and term order."""
    return build_index(PHRASE_CORPUS)


@pytest.fixture
def ranking_index() -> InvertedIndex:
    """The corpus every documented ranking number came from."""
    return build_index(RANKING_CORPUS)


@pytest.fixture
def ranker() -> Ranker:
    """A ranker over a finished index, the only valid state to build one in."""
    return Ranker(build_index(RANKING_CORPUS))


def unconfigure_logging() -> None:
    """Return the package logger to writing nowhere.

    `configure` installs a handler on a process-wide logger, so a test that did
    not undo it would leave later tests writing into a stream nobody reads.
    """
    package = logging.getLogger(LOGGER_NAME)
    for handler in list(package.handlers):
        package.removeHandler(handler)
    package.propagate = True


@pytest.fixture
def log_stream() -> Iterator[io.StringIO]:
    """Send the package's log records here, and leave the logger as it was."""
    stream = io.StringIO()
    configure(stream)
    yield stream
    unconfigure_logging()


@pytest.fixture
def quiet_logging() -> Iterator[None]:
    """Leave the package logger writing nowhere, whatever the test configured."""
    yield
    unconfigure_logging()


@pytest.fixture
def built_index(tmp_path: Path) -> Path:
    """An index file built from the committed sample corpus, through the CLI."""
    path = tmp_path / "sample.index"
    assert cli_main(["index", str(SAMPLE_CORPUS), str(path)]) == 0
    return path


def write_segments(directory: Path, texts: dict[int, str], buffer: int = 1) -> None:
    """Write documents into a directory as segments, without merging them."""
    with IndexWriter(directory, buffer_documents=buffer, merge=False) as writer:
        for document_id, text in texts.items():
            writer.add(document_id, text)


def assert_stored_text(
    directory: Path, name: str, expected: dict[int, tuple[str, str]]
) -> None:
    """Assert a segment's store holds this title and body at each of its ordinals.

    Shared by the write path and the merge path, which have to agree that an
    ordinal means the same thing in a segment and in the store beside it.
    """
    with (
        SegmentReader(directory / name) as reader,
        DocumentStore.open(directory, name, documents=len(expected)) as store,
    ):
        assert list(reader.document_ids) == sorted(expected)
        for ordinal, document_id in enumerate(reader.document_ids):
            assert (store.title(ordinal), store.text(ordinal)) == expected[document_id]


def assert_same_postings(actual: ReadableIndex, expected: InvertedIndex) -> None:
    """Assert an index holds exactly what a single in-memory index would.

    Shared by the segmented and the merged readers, which have to agree with
    the same reference and would otherwise each carry a copy of this.
    """
    assert actual.document_count == expected.document_count
    assert sorted(actual.terms) == sorted(expected.terms)
    for term in expected.terms:
        assert actual.postings(term) == {
            document_id: list(positions)
            for document_id, positions in expected.postings(term).items()
        }
