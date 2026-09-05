"""The inverted index: a map from each term to where it occurs.

A forward index answers "what is in document 5", which is the wrong question.
Searching asks "which documents contain this term", so the mapping is inverted:
term first, documents second. That turns a scan of every document into one
dictionary lookup.

Each term maps to postings recording not only *which* documents contain it but
at *which positions*. Positions cost memory and buy phrase queries: without
them there is no way to know that `computer` and `science` were adjacent rather
than merely both present.

Postings are keyed by document identifier rather than held as a sorted list.
Both shapes appear in the literature; a mapping is chosen because every query
operation needed here is a set operation over document identifiers, and
dictionary keys give those directly.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from search_engine.analysis import analyze_positioned

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

_NO_POSTINGS: Mapping[int, Sequence[int]] = MappingProxyType({})


class DuplicateDocumentError(ValueError):
    """Raised when a document identifier is added twice.

    Adding one twice would append a second set of positions to the first,
    silently corrupting every position already stored for it.
    """

    def __init__(self, document_id: int) -> None:
        super().__init__(f"document {document_id} is already indexed")


class InvertedIndex:
    """An in-memory inverted index with positional postings.

    Construct it empty, then add documents one at a time.
    """

    def __init__(self) -> None:
        self._postings: dict[str, dict[int, list[int]]] = {}
        self._document_ids: set[int] = set()

    @classmethod
    def from_postings(
        cls,
        document_ids: Iterable[int],
        postings: Mapping[str, Mapping[int, Sequence[int]]],
    ) -> InvertedIndex:
        """Rebuild an index from stored postings, bypassing analysis.

        The load path. Stored terms are already analyzed, and re-analyzing them
        would change them, because stemming is not idempotent.

        Document identifiers are passed separately because a document holding
        no terms appears in no postings list, yet must still count towards the
        total used to weight rare terms.
        """
        index = cls()
        index._document_ids = set(document_ids)
        index._postings = {
            term: {
                document_id: list(positions)
                for document_id, positions in entries.items()
            }
            for term, entries in postings.items()
        }
        return index

    def add_document(self, document_id: int, text: str) -> None:
        """Analyze a document and record every term occurrence in it.

        Raises:
            DuplicateDocumentError: If this identifier was already added.
        """
        if document_id in self._document_ids:
            raise DuplicateDocumentError(document_id)
        self._document_ids.add(document_id)
        for position, term in analyze_positioned(text):
            self._postings.setdefault(term, {}).setdefault(document_id, []).append(
                position
            )

    @property
    def document_count(self) -> int:
        """Number of documents indexed, including any with no terms."""
        return len(self._document_ids)

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct terms, which is how many dimensions a document has."""
        return len(self._postings)

    @property
    def terms(self) -> Iterable[str]:
        """Every distinct term, in the order first encountered."""
        return self._postings.keys()

    @property
    def document_ids(self) -> Iterable[int]:
        """Every indexed identifier, including documents holding no terms.

        Those cannot be recovered from the postings, since they appear in none,
        so anything saving or iterating the corpus needs them from here.
        """
        return frozenset(self._document_ids)

    def postings(self, term: str) -> Mapping[int, Sequence[int]]:
        """Return document identifiers to increasing positions, empty if unknown.

        Returning an empty mapping rather than raising lets callers treat an
        absent term as a term with no matches, which is what it is.
        """
        found = self._postings.get(term)
        return _NO_POSTINGS if found is None else MappingProxyType(found)

    def document_frequency(self, term: str) -> int:
        """How many documents contain the term, not how often it occurs."""
        return len(self._postings.get(term, ()))

    def __contains__(self, term: str) -> bool:
        """Whether any document holds this term. Terms, not document ids."""
        return term in self._postings
