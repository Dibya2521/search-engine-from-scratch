"""TF-IDF scoring over the vector space model.

Matching decides whether a document qualifies. Ranking decides how well, by
combining two quantities:

- Term frequency: how often a term occurs in the document.
- Inverse document frequency: how rare the term is across the corpus. A term
  present in every document has an inverse document frequency of zero.

Documents and queries are represented as vectors over the vocabulary, and
relevance is the cosine of the angle between them. Scaling each vector to unit
length removes the advantage a long document would otherwise have, and reduces
the cosine to a dot product.

Word order is discarded, which is the bag-of-words assumption. Phrase queries
preserve order; ranking does not.

`Ranker` precomputes every term's inverse document frequency and every
document's vector length. Without that, scoring one candidate costs a pass over
the whole vocabulary: measured at 1.64 ms per candidate on a 4,000 term index,
which projects to over 700 seconds per query on a realistic vocabulary.

`BaseRanker` holds the parts that do not depend on the scoring formula, so that
a second formula sees the same candidates, document frequencies and selection
rule. Any difference in the results is then caused by the formula alone.

`TopK` holds the same selection rule for a caller that produces scores one at a
time instead of all at once, and that needs the score of the lowest result held
so far while it works.

Any ranker can be built with `proximity=True`, which multiplies each score by
how close together the query terms appear in that document. It is **off by
default**, because on the collection this project can measure against it cannot
be shown to help. See `search_engine.proximity` and `docs/06-ranking.md`.
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING, Final

from search_engine.analysis import analyze
from search_engine.index import is_field_term
from search_engine.metrics import REGISTRY
from search_engine.proximity import minimum_span, proximity_boost

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from search_engine.access import Permit
    from search_engine.index import ReadableIndex

DOCUMENTS_SCORED: Final = REGISTRY.counter(
    "documents_scored_total", "Candidate documents a scorer actually scored."
)


def inverse_document_frequency(index: ReadableIndex, term: str) -> float:
    """Return ``log(N / df)`` for a term.

    The result is zero for a term present in every document and grows as the
    term becomes rarer. An unindexed term returns 0.0 rather than dividing by
    zero, which is also correct: a term nobody indexed cannot discriminate.
    """
    frequency = index.document_frequency(term)
    if frequency == 0:
        return 0.0
    return math.log(index.document_count / frequency)


def normalize(weights: Mapping[str, float]) -> dict[str, float]:
    """Scale a vector to unit length.

    An all-zero vector, which occurs when every term is present in every
    document, is returned unchanged because its length is zero.
    """
    length = math.sqrt(sum(weight * weight for weight in weights.values()))
    if length == 0.0:
        return dict(weights)
    return {term: weight / length for term, weight in weights.items()}


def cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Return the dot product of two unit vectors.

    Iterates the smaller vector, since a term absent from one contributes
    nothing to the product and both are sparse.
    """
    if len(right) < len(left):
        left, right = right, left
    return sum(weight * right.get(term, 0.0) for term, weight in left.items())


class StaleRankerError(RuntimeError):
    """Raised when the index gained documents after the ranker was built.

    Scores would be computed from an out-of-date document count and stale
    vector lengths, making them silently wrong rather than obviously broken.
    """

    def __init__(self, built_for: int, current: int) -> None:
        super().__init__(
            f"ranker was built for {built_for} documents, index now has {current}"
        )


class TopK:
    """The best scored documents so far, and the score a new one has to beat.

    Selection matches `BaseRanker.rank`: highest score first, ties broken on the
    lower document identifier. Scores arrive one at a time, so only the results
    kept are held, which is what makes the threshold available before the last
    document has been seen.
    """

    __slots__ = ("_heap", "_limit")

    def __init__(self, limit: int) -> None:
        """Hold at most `limit` results.

        Raises:
            ValueError: If limit is not positive.
        """
        if limit <= 0:
            message = f"limit must be positive, got {limit}"
            raise ValueError(message)
        self._limit = limit
        self._heap: list[tuple[float, int]] = []

    def push(self, document_id: int, score: float) -> None:
        """Keep a scored document if it belongs in the results."""
        entry = (score, -document_id)
        if len(self._heap) < self._limit:
            heapq.heappush(self._heap, entry)
        elif entry > self._heap[0]:
            heapq.heapreplace(self._heap, entry)

    @property
    def threshold(self) -> float:
        """Return the score of the lowest result held, or 0.0 until full.

        Zero until the limit is reached, because until then any positive score
        belongs in the results and nothing can be pruned.
        """
        if len(self._heap) < self._limit:
            return 0.0
        return self._heap[0][0]

    def best(self) -> list[tuple[int, float]]:
        """Return the results as ``(document_id, score)``, highest first."""
        return [
            (-negated_id, score)
            for score, negated_id in sorted(self._heap, reverse=True)
        ]


class BaseRanker:
    """Shared machinery for scorers over one fixed state of an index.

    Provides inverse document frequencies, the staleness check and top-k
    selection. Subclasses supply the two parts that vary between scoring
    formulas: how a query becomes weighted terms, and how a document scores
    against them.

    Construct any ranker after indexing is complete. The statistics it
    snapshots depend on the whole corpus and change when a document is added.
    """

    def __init__(self, index: ReadableIndex, *, proximity: bool = False) -> None:
        self._index = index
        self._built_for = index.document_count
        self._proximity = proximity
        self._idf = {
            term: inverse_document_frequency(index, term) for term in index.terms
        }

    @property
    def proximity(self) -> bool:
        """Return whether `rank` multiplies scores by a proximity boost."""
        return self._proximity

    def _check_fresh(self) -> None:
        current = self._index.document_count
        if current != self._built_for:
            raise StaleRankerError(self._built_for, current)

    def query_weights(self, terms: Sequence[str]) -> dict[str, float]:
        """Return the per-term query weights this scorer uses."""
        raise NotImplementedError

    def score(self, weights: Mapping[str, float], document_id: int) -> float:
        """Return the score of one document against an already-weighted query."""
        raise NotImplementedError

    def rank(
        self,
        query: str,
        candidates: Collection[int],
        limit: int = 10,
        permit: Permit | None = None,
    ) -> list[tuple[int, float]]:
        """Return the best candidates for a query as ``(document_id, score)``.

        Documents scoring zero are omitted. Ties break on the lower document
        identifier, so the order is deterministic and two runs are comparable.

        Selection uses a heap rather than a full sort, which is O(n log k)
        rather than O(n log n) for k results out of n candidates.

        ``permit`` says which documents this caller may see, and is applied
        before scoring rather than to the finished list. Filtering afterwards
        returns fewer than ``limit`` results, and the gaps tell the caller that
        documents they cannot see exist.
        """
        terms = analyze(query)
        if not terms or limit <= 0:
            return []
        allowed = (
            candidates
            if permit is None
            else [document_id for document_id in candidates if permit(document_id)]
        )
        # Every permitted candidate is scored, because selecting the best k
        # consumes the whole generator. Early termination counts separately.
        DOCUMENTS_SCORED.increment(len(allowed))
        weights = self.query_weights(terms)
        distinct = tuple(dict.fromkeys(terms))
        scored = (
            (score * self._nearness(distinct, document_id), -document_id)
            for document_id in allowed
            if (score := self.score(weights, document_id)) > 0
        )
        return [
            (-negated_id, score) for score, negated_id in heapq.nlargest(limit, scored)
        ]

    def _nearness(self, terms: Sequence[str], document_id: int) -> float:
        """Return the proximity multiplier for one document, or 1.0 when off.

        Costs one postings lookup per query term, the same as scoring does, so
        it is bounded by the same cache that keeps scoring linear.
        """
        if not self._proximity:
            return 1.0
        positions = [self._index.postings(term).get(document_id, ()) for term in terms]
        return proximity_boost(minimum_span(positions), len(terms))


class Ranker(BaseRanker):
    """Scores documents by TF-IDF cosine similarity."""

    def __init__(self, index: ReadableIndex, *, proximity: bool = False) -> None:
        super().__init__(index, proximity=proximity)
        self._norms = self._document_norms()

    def _document_norms(self) -> dict[int, float]:
        """Return every document's vector length, computed in one pass.

        One pass over the whole index, rather than one pass over the vocabulary
        for each document scored.
        """
        squares: dict[int, float] = {}
        for term in self._content_terms():
            weight = self._idf[term]
            for document_id, positions in self._index.postings(term).items():
                contribution = (len(positions) * weight) ** 2
                squares[document_id] = squares.get(document_id, 0.0) + contribution
        return {document_id: math.sqrt(total) for document_id, total in squares.items()}

    def query_weights(self, terms: Sequence[str]) -> dict[str, float]:
        """Return the unit-length TF-IDF vector for a query.

        Constructed exactly as a document vector is, because the two must
        occupy the same space to be comparable.
        """
        counts: dict[str, int] = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
        return normalize(
            {term: count * self._idf.get(term, 0.0) for term, count in counts.items()}
        )

    def document_vector(self, document_id: int) -> dict[str, float]:
        """Return one document's unit-length TF-IDF vector.

        Only terms the document contains are present. Every other component is
        zero, and storing them would make the vector as wide as the vocabulary.
        """
        weights = {
            term: len(positions) * self._idf[term]
            for term in self._content_terms()
            if (positions := self._index.postings(term).get(document_id)) is not None
        }
        return normalize(weights)

    def _content_terms(self) -> list[str]:
        """Return the terms that describe the document's own content.

        A field-qualified term repeats content already counted under its
        unqualified form, so a vector including both would count a title twice
        and no longer be the vector the score is defined against.
        """
        return [term for term in self._idf if not is_field_term(term)]

    def score(self, weights: Mapping[str, float], document_id: int) -> float:
        """Return the cosine similarity of a document to a weighted query.

        Costs one postings lookup per query term rather than per vocabulary
        term, because a term absent from the query contributes zero to the dot
        product regardless of the document.

        Raises:
            StaleRankerError: If the index changed after this ranker was built.
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
