"""BM25 scoring, the probabilistic alternative to TF-IDF.

BM25 corrects two properties of TF-IDF that do not match how relevance behaves.

Term frequency saturates. Under TF-IDF a document containing a term twenty
times scores twenty times one containing it once. BM25 divides by a quantity
that grows with the term frequency, so each additional occurrence contributes
less than the one before and the score approaches a limit.

Document length is corrected explicitly. Each document's length is compared
against the corpus average, so a term occurring three times in a short document
counts for more than the same three occurrences in a long one.

The score of a document for a query is the sum over query terms of::

    idf(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))

where ``tf`` is the term's frequency in the document, ``dl`` the document's
length in terms and ``avgdl`` the corpus average. ``k1`` controls how quickly
term frequency saturates and ``b`` how strongly length is corrected.

Both quantities BM25 needs are already stored by the index: term frequency is
the length of a position list, and document length is the sum of a document's
position list lengths.

Because the score rises with term frequency and falls with document length,
the most a term can contribute to any document is fixed by the highest
frequency it reaches and the shortest document in the corpus. `upper_bound`
returns that value, and a caller holding one per query term can rule a document
out before scoring it. The bound belongs here rather than with the caller so
that it stays derived from the same `k1` and `b` the score itself uses.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

from search_engine.ranking import BaseRanker

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from search_engine.index import ReadableIndex

# Conventional defaults from the literature. k1 is usually quoted as a range of
# 1.2 to 2.0 and b as 0.75; these are the values most implementations ship.
K1: Final = 1.2
B: Final = 0.75


def bm25_inverse_document_frequency(index: ReadableIndex, term: str) -> float:
    """Return the BM25 inverse document frequency of a term.

    ``log(1 + (N - df + 0.5) / (df + 0.5))``. The original formulation omits
    the ``1 +``, which makes the value negative for any term present in more
    than half the documents. A negative weight means a document is penalised
    for containing a common term, so a document containing every query term can
    rank below one containing fewer. The ``1 +`` keeps the value positive for
    every possible document frequency and is what current implementations use.

    An unindexed term returns 0.0, matching the TF-IDF function.
    """
    frequency = index.document_frequency(term)
    if frequency == 0:
        return 0.0
    total = index.document_count
    return math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))


class BM25Ranker(BaseRanker):
    """Scores documents with BM25.

    Args:
        index: A finished index. Adding documents afterwards invalidates the
            precomputed statistics and is detected rather than ignored.
        k1: Term frequency saturation. Larger values make repeated occurrences
            count for more; 0.0 reduces the term frequency component to a
            constant, leaving only the inverse document frequency.
        b: Length correction, from 0.0 for none to 1.0 for full.
    """

    def __init__(
        self,
        index: ReadableIndex,
        k1: float = K1,
        b: float = B,
        *,
        proximity: bool = False,
    ) -> None:
        super().__init__(index, proximity=proximity)
        self._k1 = k1
        self._b = b
        # BM25 needs its own inverse document frequency, not the TF-IDF one the
        # base class computes, because the two formulas differ.
        self._idf = {
            term: bm25_inverse_document_frequency(index, term) for term in index.terms
        }
        self._lengths = self._document_lengths()
        self._average_length = self._mean_length()
        self._minimum_length = min(self._lengths.values(), default=0)

    def _document_lengths(self) -> dict[int, int]:
        """Return each document's length in terms, asked of the index.

        An index read from a file has these stored and answers each in one
        lookup. Recovering them here by walking every posting instead would
        read the whole file on the first query.
        """
        return {
            document_id: self._index.document_length(document_id)
            for document_id in self._index.document_ids
        }

    def _mean_length(self) -> float:
        """Return the mean document length, or 0.0 for an empty corpus.

        Documents holding no terms are included in the average, since they are
        part of the corpus and the index counts them.
        """
        if self._built_for == 0:
            return 0.0
        return sum(self._lengths.values()) / self._built_for

    def query_weights(self, terms: Sequence[str]) -> dict[str, float]:
        """Return the inverse document frequency of each distinct query term.

        Repetition within the query is ignored. The full formulation includes a
        query term frequency factor controlled by a third parameter, which
        matters only for queries long enough to repeat a term, and typed
        queries are not.
        """
        return {term: self._idf.get(term, 0.0) for term in terms}

    def score(self, weights: Mapping[str, float], document_id: int) -> float:
        """Return the BM25 score of a document against a weighted query.

        Raises:
            StaleRankerError: If the index changed after this ranker was built.
        """
        self._check_fresh()
        length = self._lengths.get(document_id, 0)
        total = 0.0
        for term, weight in weights.items():
            positions = self._index.postings(term).get(document_id)
            if positions is not None:
                total += weight * self._saturate(len(positions), length)
        return total

    def upper_bound(self, term: str, max_frequency: int) -> float:
        """Return the most this term can contribute to any document's score.

        `max_frequency` is the term's highest frequency in any one document,
        which a segment stores per term. The shortest document in the corpus is
        used rather than the shortest document containing the term, because the
        two need no postings read and the first is never an underestimate.

        Call this only for a term some document holds. A term with no postings
        has no frequency to bound, and on an empty corpus there is no average
        length to correct against.
        """
        return self._idf.get(term, 0.0) * self._saturate(
            max_frequency, self._minimum_length
        )

    def _saturate(self, frequency: int, length: int) -> float:
        """Return the length-corrected, saturating term frequency component.

        The average length cannot be zero here: this is reached only for a
        document that contains the term, and such a document has a non-zero
        length, so the corpus average is non-zero too.
        """
        correction = 1 - self._b + self._b * length / self._average_length
        return (frequency * (self._k1 + 1)) / (frequency + self._k1 * correction)
