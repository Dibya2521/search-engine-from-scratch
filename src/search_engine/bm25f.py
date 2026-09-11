"""BM25F: BM25 over fields that are weighted differently.

A term in a title says more about a document than the same term in its fortieth
paragraph. The index records which field an occurrence came from; this is the
scorer that uses it.

**The ordering is the whole content of the idea.** BM25F does not score each
field separately and add the results. It combines the per-field frequencies
first, and saturates once over the total::

    combined(term, doc) = sum over fields f of
            weight(f) * frequency(term, doc, f)
            / (1 - b + b * length(doc, f) / average_length(f))

    score(term, doc) = idf(term) * combined * (k1 + 1) / (k1 + combined)

Saturating each field and then adding lets a term appearing twice in the title
and twice in the body out-score one appearing four times in the title, because
each field runs up its own saturation curve. Combining first keeps one curve
over the document as a whole, which is what saturation is for.

**The fields here overlap, and textbook BM25F assumes they do not.** This index
stores the whole document under unqualified terms, so that every query written
before fields existed still works, and stores the title a second time under
qualified ones. So an occurrence in the title is counted by the document field
*and* by the title field, and a title weight of 3.0 gives a title term an
effective weight of 4.0 against a body term's 1.0. That is a boost rather than a
partition, it is what this index can support without a format change, and it is
said here rather than left for a reader to deduce from a surprising number.

**Weighting only the document field reproduces plain BM25 exactly**, which is
the strongest correctness check available and is a property test over generated
indexes.

**The 3.0 title weight is the conventional starting point, not a measurement.**
Tuning it needs a test collection whose queries are hard enough for a ranking
change to show, and this project does not have one: see `docs/06-ranking.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from search_engine.bm25 import K1, B, bm25_inverse_document_frequency
from search_engine.index import (
    FIELD_SEPARATOR,
    TITLE_FIELD,
    is_field_term,
)
from search_engine.ranking import BaseRanker

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from search_engine.index import ReadableIndex

# The unqualified term, whose frequency covers the whole document.
DOCUMENT_FIELD: Final = ""

DEFAULT_FIELD_WEIGHTS: Final = {DOCUMENT_FIELD: 1.0, TITLE_FIELD: 3.0}


class BM25FRanker(BaseRanker):
    """BM25 over weighted fields, combining field frequencies before saturating.

    Args:
        index: A finished index, whose documents were added with their fields.
        weights: How much each field counts. A field weighted 0.0 contributes
            nothing, and a field the index does not hold contributes nothing.
        k1: Term frequency saturation, as in BM25.
        b: Length correction, applied per field against that field's own
            average length.
    """

    def __init__(
        self,
        index: ReadableIndex,
        weights: Mapping[str, float] = DEFAULT_FIELD_WEIGHTS,
        k1: float = K1,
        b: float = B,
    ) -> None:
        super().__init__(index)
        self._weights = dict(weights)
        self._k1 = k1
        self._b = b
        # BM25's inverse document frequency, and only for terms a query can
        # produce: a qualified term is a copy of content counted elsewhere.
        self._idf = {
            term: bm25_inverse_document_frequency(index, term)
            for term in index.terms
            if not is_field_term(term)
        }
        self._lengths = {field: self._field_lengths(field) for field in self._weights}
        self._averages = {
            field: self._mean(lengths) for field, lengths in self._lengths.items()
        }

    def query_weights(self, terms: Sequence[str]) -> dict[str, float]:
        """Return the inverse document frequency of each distinct query term."""
        return {term: self._idf.get(term, 0.0) for term in terms}

    def score(self, weights: Mapping[str, float], document_id: int) -> float:
        """Return the BM25F score of a document against a weighted query.

        Raises:
            StaleRankerError: If the index changed after this ranker was built.
        """
        self._check_fresh()
        total = 0.0
        for term, weight in weights.items():
            combined = sum(
                self._contribution(field, term, document_id) for field in self._weights
            )
            if combined:
                total += weight * combined * (self._k1 + 1) / (self._k1 + combined)
        return total

    def _contribution(self, field: str, term: str, document_id: int) -> float:
        """Return one field's share of a term's combined frequency.

        Length correction uses that field's own average, so a long title is
        measured against other titles rather than against whole documents.
        """
        positions = self._index.postings(_key(field, term)).get(document_id)
        if positions is None:
            return 0.0
        length = self._lengths[field].get(document_id, 0)
        correction = 1 - self._b + self._b * length / self._averages[field]
        return self._weights[field] * len(positions) / correction

    def _field_lengths(self, field: str) -> dict[int, int]:
        """Return each document's length within one field.

        The document field is already stored per document. Any other field is
        recovered by walking its terms, which is one pass over the vocabulary
        when the ranker is built rather than one per document scored.
        """
        if field == DOCUMENT_FIELD:
            return {
                document_id: self._index.document_length(document_id)
                for document_id in self._index.document_ids
            }
        prefix = f"{field}{FIELD_SEPARATOR}"
        lengths: dict[int, int] = {}
        for term in self._index.terms:
            if not term.startswith(prefix):
                continue
            for document_id, positions in self._index.postings(term).items():
                lengths[document_id] = lengths.get(document_id, 0) + len(positions)
        return lengths

    def _mean(self, lengths: Mapping[int, int]) -> float:
        """Return a field's average length, or 0.0 when no document has one.

        Documents with nothing in the field count towards the average, because
        they are still documents and a field most of them lack should not be
        normalised as though every document had one.
        """
        if self._built_for == 0:
            return 0.0
        return sum(lengths.values()) / self._built_for


def _key(field: str, term: str) -> str:
    """Return the index key a term has within one field."""
    if field == DOCUMENT_FIELD:
        return term
    return f"{field}{FIELD_SEPARATOR}{term}"
