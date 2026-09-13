"""Two ways an unguarded search engine hurts the people running it.

Neither of these is a feature. Both are defects that happen not to have been
exercised yet.

**A query's cost is unbounded.** A phrase of fifty terms, or a query built from
the fifty commonest terms in the corpus, does work proportional to the whole
index. One client can then occupy the engine indefinitely, which is a denial of
service that needs no exploit and no privilege.

**Judged by estimated cost, never by shape.** A list of forbidden patterns is
always one query behind whoever is writing them, and it forbids legitimate
queries that happen to look alarming. A cost estimate is about the work, which
is the thing actually being rationed.

The estimate is free. Document frequency is already in each term's dictionary
entry, put there so a scorer would not have to count postings, so summing it
touches no postings at all.

**A filter applied after selection is a data leak.** Where different people may
see different documents, filtering the top k after choosing it tells the caller
that documents exist: ten asked for, seven returned, and the three gaps are now
known to be there. Worse with early termination, where the discarded documents
also set the threshold that pruned the ones the caller could have seen. So the
permit is consulted before a document is scored and before it can affect
anything, which is why it is passed down rather than wrapped around.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from search_engine.index import ReadableIndex

# A query longer than this is a machine's, and no person's query needs it.
MAX_QUERY_TERMS: Final = 32

# Roughly a second of scoring at the measured 1.7 microseconds per candidate.
MAX_ESTIMATED_CANDIDATES: Final = 5_000_000

type Permit = Callable[[int], bool]


class QueryTooExpensiveError(ValueError):
    """Raised when a query would cost more than the limits allow."""


def allow_all(_: int) -> bool:
    """Permit every document, which is what having no access control means."""
    return True


def estimate_cost(index: ReadableIndex, terms: Sequence[str]) -> int:
    """Return roughly how many postings a query would touch.

    Sums document frequency over the distinct terms, which every index already
    holds beside the term, so the estimate reads no postings at all. Repeats
    are counted once because the postings behind them are read once.

    It is an upper bound on the candidate set rather than its size: a document
    holding two of the terms is counted twice. The work being rationed is per
    posting, so an upper bound on postings is the quantity to limit.
    """
    return sum(index.document_frequency(term) for term in dict.fromkeys(terms))


def refuse_expensive(
    index: ReadableIndex,
    terms: Sequence[str],
    max_terms: int = MAX_QUERY_TERMS,
    max_candidates: int = MAX_ESTIMATED_CANDIDATES,
) -> None:
    """Refuse a query that would cost more than the limits allow.

    The term count is checked first because it costs nothing and because a
    query with thousands of terms is expensive to estimate as well as to run.

    Raises:
        QueryTooExpensiveError: If the query has too many terms, or would touch
            too many postings.
    """
    if len(terms) > max_terms:
        message = (
            f"query has {len(terms)} terms and the limit is {max_terms}; "
            f"ask for less of the corpus at once"
        )
        raise QueryTooExpensiveError(message)
    cost = estimate_cost(index, terms)
    if cost > max_candidates:
        message = (
            f"query would touch about {cost} postings and the limit is "
            f"{max_candidates}; use rarer terms or a phrase"
        )
        raise QueryTooExpensiveError(message)
