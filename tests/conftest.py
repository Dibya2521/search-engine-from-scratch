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


@pytest.fixture
def index() -> InvertedIndex:
    """A three document index, small enough to assert postings by hand."""
    built = InvertedIndex()
    for document_id, text in CORPUS.items():
        built.add_document(document_id, text)
    return built
