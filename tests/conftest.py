"""Shared fixtures.

Fixtures live here rather than beside the tests that use them so that a test
function's parameter does not shadow a module-level function of the same name,
which is what pytest's fixture mechanism would otherwise force.
"""

from __future__ import annotations

import pytest

from search_engine.index import InvertedIndex

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


def _build(corpus: dict[int, str]) -> InvertedIndex:
    built = InvertedIndex()
    for document_id, text in corpus.items():
        built.add_document(document_id, text)
    return built


@pytest.fixture
def index() -> InvertedIndex:
    """A three document index, small enough to assert postings by hand."""
    return _build(CORPUS)


@pytest.fixture
def phrase_index() -> InvertedIndex:
    """An index built for exercising adjacency and term order."""
    return _build(PHRASE_CORPUS)
