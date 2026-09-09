"""Shared fixtures and the corpora they are built from.

Fixtures live here rather than beside the tests that use them so that a test
function's parameter does not shadow a module-level function of the same name,
which is what pytest's fixture mechanism would otherwise force. For the same
reason no fixture here depends on another: each builds what it needs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from search_engine.cli import main as cli_main
from search_engine.index import InvertedIndex, ReadableIndex
from search_engine.ranking import Ranker
from search_engine.writer import IndexWriter

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
