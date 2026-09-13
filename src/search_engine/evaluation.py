"""Measuring whether the results are any good, not merely correct.

Every test elsewhere asks whether the engine does what it says. These metrics
ask a different question: given what a human considers relevant, how well does
the ranking agree? No test of the mechanism answers it.

All three metrics need judgements, meaning a set of document identifiers a
person has marked relevant for a query. They cannot be derived from the corpus.

- **Precision@k**: of the top k results, what fraction are relevant? Falls when
  irrelevant documents are returned.
- **Recall@k**: of everything relevant, what fraction made the top k? Falls when
  relevant documents are missed.
- **Average precision**: precision recomputed at each relevant hit and averaged,
  which rewards ranking relevant documents *higher* rather than merely
  including them. This is the one that distinguishes two engines returning the
  same set in a different order.
- **Reciprocal rank**: one divided by the position of the first relevant result.
  The right measure when a user needs one answer rather than a set of them.
- **Normalized discounted cumulative gain**: the only metric here that uses
  *graded* judgements, so that a directly relevant document at position one
  beats a marginally relevant one there.

Precision and recall trade against each other. Returning every document gives
perfect recall and useless precision; returning one certain hit gives the
reverse.

Every function takes results already ranked and judgements already made. None of
them look at the index, which is deliberate: a metric that could see the engine
would be measuring itself.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
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


def reciprocal_rank(retrieved: Sequence[int], relevant: AbstractSet[int]) -> float:
    """One over the rank of the first relevant result, or 0.0 if there is none.

    The measure to use when a query has one right answer and the user stops
    reading as soon as they find it. It ignores everything after the first hit.
    """
    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    results: Iterable[tuple[Sequence[int], AbstractSet[int]]],
) -> float:
    """Average of the reciprocal ranks across several queries."""
    scores = [reciprocal_rank(retrieved, relevant) for retrieved, relevant in results]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def discounted_cumulative_gain(
    retrieved: Sequence[int], grades: Mapping[int, int], k: int
) -> float:
    """Sum each result's grade, discounted by the logarithm of its rank.

    Two ideas in one formula. *Gain*: a document graded 3 is worth more than one
    graded 1, which a binary metric cannot express. *Discount*: a result at rank
    10 is worth less than the same result at rank 1, because fewer people look
    at it. Dividing by ``log2(rank + 1)`` gives rank 1 a divisor of 1 and decays
    slowly after that.

    Documents with no judgement contribute nothing, which treats unjudged as
    not relevant. That is the standard convention and it is a real assumption:
    an unjudged document might have been excellent.
    """
    if k <= 0:
        return 0.0
    return sum(
        grades.get(document_id, 0) / math.log2(rank + 1)
        for rank, document_id in enumerate(retrieved[:k], start=1)
    )


def normalized_discounted_cumulative_gain(
    retrieved: Sequence[int], grades: Mapping[int, int], k: int
) -> float:
    """Gain as a fraction of the best gain any ranking of these grades could get.

    Raw gain is not comparable between queries, because a query with six
    relevant documents can score higher than one with two no matter how well
    both are ranked. Dividing by the ideal ranking's gain puts every query on
    the same 0 to 1 scale, which is what makes averaging across queries mean
    anything.

    The ideal ranking is built from *all* judged documents, including ones the
    engine failed to return, so that missing a relevant document costs
    something.
    """
    ideal_order = sorted(grades, key=lambda document_id: -grades[document_id])
    ideal = discounted_cumulative_gain(ideal_order, grades, k)
    if ideal == 0.0:
        return 0.0
    return discounted_cumulative_gain(retrieved, grades, k) / ideal


def sign_test(differences: Iterable[float]) -> float:
    """Return the two-sided p-value of a sign test over paired differences.

    Answers whether one system beating another across a query set could
    plausibly be chance. Counts how many queries each system won, ignoring
    ties, and asks how likely a split at least this lopsided would be if the
    two systems were equally good and each query were a coin toss.

    A p-value near 1.0 means the result is indistinguishable from chance. The
    conventional threshold for reporting a difference is 0.05.

    The test uses only the sign of each difference, never its size, so it
    cannot be distorted by one query with a huge swing. That insensitivity is
    also its weakness: a system that wins narrowly on many queries and loses
    heavily on a few scores the same as one that does the reverse.

    Returns 1.0 when every query is a tie, since there is nothing to explain.
    """
    signs = [one for one in differences if one != 0]
    total = len(signs)
    if total == 0:
        return 1.0
    wins = sum(1 for one in signs if one > 0)
    extreme = max(wins, total - wins)
    tail = sum(math.comb(total, count) for count in range(extreme, total + 1))
    return min(1.0, 2 * tail / 2**total)
