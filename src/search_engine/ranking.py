"""Scoring documents by relevance, using TF-IDF and cosine similarity.

Matching says *whether* a document qualifies. Ranking says *how well*, which is
what makes results useful rather than merely correct.

Two forces, multiplied:

- **Term frequency.** A document mentioning a term ten times is more about it
  than one mentioning it once.
- **Inverse document frequency.** A term in every document separates nothing,
  so it is worth almost nothing. ``log(N / df)`` reaches zero when a term
  appears everywhere.

Documents and queries become vectors over the vocabulary, and relevance is the
cosine of the angle between them. Dividing each vector by its own length stops
long documents winning merely by being long, and turns the cosine into a plain
dot product.

Word order is discarded here, which is the bag-of-words assumption. Phrase
queries keep order; ranking does not.

Scoring is done through `Ranker`, which precomputes the two things that would
otherwise be recomputed per query: every term's inverse document frequency, and
every document's vector length. Without that precomputation, scoring one
candidate costs a pass over the whole vocabulary. Measured on a 4,000 term
index, that was 1.64 ms per candidate and projected to over 700 seconds on a
realistic vocabulary.
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

from search_engine.analysis import analyze

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from search_engine.index import InvertedIndex


def inverse_document_frequency(index: InvertedIndex, term: str) -> float:
    """Return how much knowing a document holds this term tells you.

    ``log(N / df)``. Zero when the term is in every document, larger the rarer
    it is. An unknown term returns 0.0 rather than dividing by zero, which is
    also the right answer: a term nobody indexed cannot discriminate.
    """
    frequency = index.document_frequency(term)
    if frequency == 0:
        return 0.0
    return math.log(index.document_count / frequency)


def normalize(weights: Mapping[str, float]) -> dict[str, float]:
    """Scale a vector to unit length, so document size stops mattering.

    An all-zero vector, which happens when every term is in every document, is
    returned unchanged: its length is zero and there is nothing to scale.
    """
    length = math.sqrt(sum(weight * weight for weight in weights.values()))
    if length == 0.0:
        return dict(weights)
    return {term: weight / length for term, weight in weights.items()}


def cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Return the dot product of two unit vectors, which is their cosine.

    Iterates the smaller vector, since a term absent from one contributes
    nothing and both are sparse.
    """
    if len(right) < len(left):
        left, right = right, left
    return sum(weight * right.get(term, 0.0) for term, weight in left.items())


class StaleRankerError(RuntimeError):
    """Raised when the index gained documents after the ranker was built.

    Every score would be computed from an out-of-date document count and stale
    vector lengths, so the numbers would be quietly wrong rather than obviously
    broken.
    """

    def __init__(self, built_for: int, current: int) -> None:
        super().__init__(
            f"ranker was built for {built_for} documents, index now has {current}"
        )


class Ranker:
    """Scores documents against queries for one fixed state of an index.

    Build it after indexing is finished. It snapshots inverse document
    frequencies and document vector lengths, both of which depend on the whole
    corpus and change whenever a document is added.
    """

    def __init__(self, index: InvertedIndex) -> None:
        self._index = index
        self._built_for = index.document_count
        self._idf = {
            term: inverse_document_frequency(index, term) for term in index.terms
        }
        self._norms = self._document_norms()

    def _document_norms(self) -> dict[int, float]:
        """Compute every document's vector length in one pass over the postings.

        One pass over the index total, rather than one pass over the vocabulary
        per document scored. This is the difference between usable and not.
        """
        squares: dict[int, float] = {}
        for term, weight in self._idf.items():
            for document_id, positions in self._index.postings(term).items():
                contribution = (len(positions) * weight) ** 2
                squares[document_id] = squares.get(document_id, 0.0) + contribution
        return {document_id: math.sqrt(total) for document_id, total in squares.items()}

    def _check_fresh(self) -> None:
        current = self._index.document_count
        if current != self._built_for:
            raise StaleRankerError(self._built_for, current)

    def query_weights(self, terms: Sequence[str]) -> dict[str, float]:
        """Return the unit-length TF-IDF vector for a query.

        Built exactly as a document vector is, because the two must occupy the
        same space to be comparable.
        """
        counts: dict[str, int] = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
        return normalize(
            {term: count * self._idf.get(term, 0.0) for term, count in counts.items()}
        )

    def document_vector(self, document_id: int) -> dict[str, float]:
        """Return one document's unit-length TF-IDF vector.

        Only terms the document holds appear; every other component is zero and
        storing them would mean a vector as wide as the vocabulary.
        """
        weights = {
            term: len(positions) * self._idf[term]
            for term in self._index.terms
            if (positions := self._index.postings(term).get(document_id)) is not None
        }
        return normalize(weights)

    def score(self, weights: Mapping[str, float], document_id: int) -> float:
        """Score one document against an already-weighted query.

        Costs one postings lookup per *query* term, not per vocabulary term,
        because a term missing from the query contributes zero to the dot
        product no matter what the document says.
        """
        self._check_fresh()
        norm = self._norms.get(document_id, 0.0)
        if norm == 0.0:
            return 0.0
        total = 0.0
        for term, query_weight in weights.items():
            positions = self._index.postings(term).get(document_id)
            if positions is not None:
                total += query_weight * (len(positions) * self._idf[term]) / norm
        return total

    def rank(
        self, query: str, candidates: Iterable[int], limit: int = 10
    ) -> list[tuple[int, float]]:
        """Score candidates against the query text, best first.

        Returns ``(document_id, score)`` pairs, omitting anything scoring zero.
        Ties break on the lower document identifier, so the order is
        deterministic and two runs are comparable.

        Uses a heap rather than a full sort: taking the best few of many
        candidates is O(n log k) instead of O(n log n).
        """
        terms = analyze(query)
        if not terms or limit <= 0:
            return []
        weights = self.query_weights(terms)
        scored = (
            (score, -document_id)
            for document_id in candidates
            if (score := self.score(weights, document_id)) > 0
        )
        return [
            (-negated_id, score) for score, negated_id in heapq.nlargest(limit, scored)
        ]
