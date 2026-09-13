"""Tests for the cost limit and for filtering before the top k is chosen."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from search_engine.access import (
    MAX_ESTIMATED_CANDIDATES,
    MAX_QUERY_TERMS,
    Permit,
    QueryTooExpensiveError,
    allow_all,
    estimate_cost,
    refuse_expensive,
)
from search_engine.analysis import analyze
from search_engine.bm25 import BM25Ranker
from search_engine.index import InvertedIndex
from search_engine.query import parse, search
from search_engine.wand import rank_wand
from tests.conftest import build_index

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

CORPUS_SIZE = 40


class NoPostings(InvertedIndex):
    """An index that refuses to hand over postings, to prove none were read."""

    def postings(self, term: str) -> Mapping[int, Sequence[int]]:
        """Fail, because reading a posting here would be the defect."""
        message = f"the estimate read the postings of {term}"
        raise RuntimeError(message)


def skewed(size: int = CORPUS_SIZE) -> dict[int, str]:
    """A corpus with one term nearly everywhere and another in a few documents.

    The shape early termination exists for, so a permit has something to
    interact with rather than a flat corpus where nothing is ever pruned.
    """
    return {
        number: " ".join(
            ["alpha"] * (1 + number % 3) + (["beta"] * 4 if number % 5 == 0 else [])
        )
        for number in range(size)
    }


def forbidding(forbidden: frozenset[int]) -> Permit:
    """Return a permit that hides exactly these documents."""

    def permit(document_id: int) -> bool:
        return document_id not in forbidden

    return permit


def test_the_estimate_is_the_sum_of_the_document_frequencies() -> None:
    index = build_index({1: "alpha beta", 2: "alpha gamma", 3: "alpha"})
    assert estimate_cost(index, ["alpha"]) == 3
    assert estimate_cost(index, ["alpha", "beta"]) == 4
    assert estimate_cost(index, ["absent"]) == 0


def test_a_repeated_term_is_counted_once() -> None:
    """Its postings are read once, so counting it twice would overstate the work."""
    index = build_index({1: "alpha beta", 2: "alpha gamma"})
    assert estimate_cost(index, ["alpha", "alpha", "alpha"]) == 2


def test_the_estimate_reads_no_postings_at_all() -> None:
    """It is free only because document frequency sits beside the term already."""
    index = NoPostings()
    index.add_document(1, "alpha beta")
    index.add_document(2, "alpha gamma")
    assert estimate_cost(index, ["alpha", "beta", "absent"]) == 3


def test_a_query_of_too_many_terms_is_refused() -> None:
    index = build_index({1: "alpha"})
    terms = [f"term{number}" for number in range(MAX_QUERY_TERMS + 1)]
    with pytest.raises(QueryTooExpensiveError, match="33 terms and the limit is 32"):
        refuse_expensive(index, terms)


def test_a_query_that_would_touch_too_many_postings_is_refused() -> None:
    index = build_index(dict.fromkeys(range(10), "alpha"))
    with pytest.raises(QueryTooExpensiveError, match="about 10 postings"):
        refuse_expensive(index, ["alpha"], max_candidates=9)


def test_a_query_inside_both_limits_is_allowed() -> None:
    index = build_index(skewed())
    refuse_expensive(index, ["alpha", "beta"])
    assert estimate_cost(index, ["alpha", "beta"]) < MAX_ESTIMATED_CANDIDATES


def test_the_limit_is_enforced_where_the_postings_would_be_read() -> None:
    """Judged on cost, so a legitimate query of the same shape still runs."""
    index = build_index(skewed())
    long_query = " ".join(f"term{number}" for number in range(MAX_QUERY_TERMS + 1))
    with pytest.raises(QueryTooExpensiveError):
        search(index, long_query)
    assert search(index, "alpha beta")


def test_a_phrase_is_judged_on_its_own_terms() -> None:
    index = build_index(skewed())
    quoted = '"' + " ".join(f"term{number}" for number in range(MAX_QUERY_TERMS + 1))
    with pytest.raises(QueryTooExpensiveError):
        search(index, f'{quoted}"')
    assert parse(f'{quoted}"').is_phrase


def test_allowing_everything_is_what_no_access_control_means() -> None:
    assert allow_all(0) is True
    assert allow_all(-1) is True


def test_a_permit_is_applied_before_the_top_k_is_chosen() -> None:
    """Filtering afterwards returns fewer than the limit, and the gaps are visible."""
    index = build_index(skewed())
    ranker = BM25Ranker(index)
    candidates = search(index, "alpha beta")
    hidden = frozenset(
        document_id for document_id in candidates if document_id % 5 == 0
    )

    ranked = ranker.rank("alpha beta", candidates, 5, forbidding(hidden))
    assert len(ranked) == 5
    assert not {document_id for document_id, _ in ranked} & hidden


def test_a_permit_that_forbids_everything_returns_nothing() -> None:
    index = build_index(skewed())
    candidates = search(index, "alpha beta")
    assert (
        BM25Ranker(index).rank(
            "alpha", candidates, 5, forbidding(frozenset(candidates))
        )
        == []
    )
    assert (
        rank_wand(
            index,
            BM25Ranker(index),
            analyze("alpha"),
            5,
            forbidding(frozenset(candidates)),
        )
        == []
    )


def test_a_permit_that_raises_is_not_swallowed_into_an_empty_result() -> None:
    """An empty result would look like a legitimate answer and be acted on."""
    index = build_index(skewed())
    candidates = search(index, "alpha beta")

    def broken(document_id: int) -> bool:
        message = f"cannot decide about {document_id}"
        raise PermissionError(message)

    with pytest.raises(PermissionError):
        BM25Ranker(index).rank("alpha beta", candidates, 5, broken)
    with pytest.raises(PermissionError):
        rank_wand(index, BM25Ranker(index), analyze("alpha beta"), 5, broken)


def test_a_permit_that_changes_its_mind_leaves_nothing_broken() -> None:
    """Undefined which documents come back, defined that the next call is right."""
    index = build_index(skewed())
    ranker = BM25Ranker(index)
    candidates = search(index, "alpha beta")
    seen: list[int] = []

    def alternating(document_id: int) -> bool:
        seen.append(document_id)
        return len(seen) % 2 == 0

    rank_wand(index, ranker, analyze("alpha beta"), 5, alternating)
    assert rank_wand(index, ranker, analyze("alpha beta"), 5) == ranker.rank(
        "alpha beta", candidates, 5
    )


@settings(deadline=None, max_examples=25)
@given(
    forbidden=st.frozensets(st.integers(min_value=0, max_value=CORPUS_SIZE - 1)),
    limit=st.integers(min_value=1, max_value=10),
)
def test_early_termination_and_a_full_scan_agree_under_any_permit(
    forbidden: frozenset[int], limit: int
) -> None:
    """Early termination must not see a permit differently from a full scan.

    The bug this exists for ships silently: a threshold set by a document the
    caller cannot see prunes one they can.
    """
    index = build_index(skewed())
    ranker = BM25Ranker(index)
    candidates = search(index, "alpha beta")
    permit = forbidding(forbidden)

    expected = ranker.rank("alpha beta", candidates, limit, permit)
    assert rank_wand(index, ranker, analyze("alpha beta"), limit, permit) == expected
