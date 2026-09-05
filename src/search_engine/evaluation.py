"""Measuring whether the results are any good, not merely correct.

Every test elsewhere asks whether the engine does what it says. These metrics
ask a different question: given what a human considers relevant, how well does
the ranking agree? That is the only question a search engine is ultimately
judged on, and no amount of passing unit tests answers it.

All three metrics need judgements, meaning a set of document identifiers a
person has marked relevant for a query. They cannot be derived from the corpus.

- **Precision@k**: of the top k results, what fraction are relevant? Punishes
  returning rubbish.
- **Recall@k**: of everything relevant, what fraction made the top k? Punishes
  missing things.
- **Average precision**: precision recomputed at each relevant hit and averaged,
  which rewards ranking relevant documents *higher* rather than merely
  including them. This is the one that distinguishes two engines returning the
  same set in a different order.

Precision and recall trade against each other. Returning every document gives
perfect recall and useless precision; returning one certain hit gives the
reverse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from collections.abc import Set as AbstractSet


def precision_at_k(
    retrieved: Sequence[int], relevant: AbstractSet[int], k: int
) -> float:
    """Fraction of the top k results that are relevant.

    Divided by the number actually returned rather than by k, so an engine that
    returns three good results out of a possible ten scores 1.0 instead of 0.3.
    Returns 0.0 when nothing was returned.
    """
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for document_id in top if document_id in relevant) / len(top)


def recall_at_k(retrieved: Sequence[int], relevant: AbstractSet[int], k: int) -> float:
    """Fraction of all relevant documents that reached the top k.

    Returns 0.0 when nothing is relevant, which is a convention rather than a
    truth: the quantity is undefined there, and 0.0 keeps averages finite.
    """
    if not relevant or k <= 0:
        return 0.0
    found = sum(1 for document_id in retrieved[:k] if document_id in relevant)
    return found / len(relevant)


def average_precision(retrieved: Sequence[int], relevant: AbstractSet[int]) -> float:
    """Mean of the precision measured at each relevant document's rank.

    Rewards putting relevant documents near the top. Two engines returning the
    same three documents in different orders get different scores here, which
    neither precision@k nor recall@k can see.

    Divided by the total number of relevant documents, so failing to retrieve
    one costs the same as ranking it last.
    """
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def mean_average_precision(
    results: Iterable[tuple[Sequence[int], AbstractSet[int]]],
) -> float:
    """Average of the average precisions across several queries.

    The standard single number for comparing two retrieval systems, because it
    accounts for both what was returned and the order it came back in.
    """
    scores = [average_precision(retrieved, relevant) for retrieved, relevant in results]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
