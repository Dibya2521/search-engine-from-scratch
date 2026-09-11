"""Tests for spelling suggestions.

Two tests carry this file. `test_the_distance_matches_a_full_matrix` checks the
two-row implementation against the textbook one on generated pairs, because the
two-row version is an optimisation and only a differential test catches it
drifting. `test_a_tree_search_finds_exactly_what_a_scan_finds` checks the
triangle-inequality pruning against brute force, because pruning that is
slightly too aggressive is invisible except on the examples it wrongly skipped.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.index import InvertedIndex
from search_engine.spelling import (
    MAX_EDIT_DISTANCE,
    BKTree,
    edit_distance,
    suggest,
)

WORDS = st.text(alphabet="abcd", min_size=0, max_size=7)
COMBINING_ACUTE = chr(0x301)


def reference_distance(left: str, right: str) -> int:
    """The full-matrix Levenshtein, written for clarity rather than for speed."""
    matrix = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for row in range(len(left) + 1):
        matrix[row][0] = row
    for column in range(len(right) + 1):
        matrix[0][column] = column
    for row in range(1, len(left) + 1):
        for column in range(1, len(right) + 1):
            change = 0 if left[row - 1] == right[column - 1] else 1
            matrix[row][column] = min(
                matrix[row - 1][column] + 1,
                matrix[row][column - 1] + 1,
                matrix[row - 1][column - 1] + change,
            )
    return matrix[-1][-1]


def by_scanning(
    vocabulary: list[str], term: str, distance: int
) -> list[tuple[str, int]]:
    """Return what a tree search should return, computed the obvious way."""
    found = [
        (candidate, edit_distance(term, candidate))
        for candidate in dict.fromkeys(vocabulary)
    ]
    return sorted(
        [(word, seen) for word, seen in found if seen <= distance],
        key=lambda match: (match[1], match[0]),
    )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("kitten", "sitting", 3),
        ("", "", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("same", "same", 0),
        ("recieve", "receive", 2),
        ("a", "b", 1),
    ],
)
def test_the_distance_on_known_pairs(left: str, right: str, expected: int) -> None:
    """`kitten` to `sitting` is the textbook example and is three."""
    assert edit_distance(left, right) == expected


@given(WORDS, WORDS)
def test_the_distance_matches_a_full_matrix(left: str, right: str) -> None:
    """The two-row optimisation against the definition it optimises."""
    assert edit_distance(left, right) == reference_distance(left, right)


@given(WORDS, WORDS)
def test_the_distance_is_symmetric(left: str, right: str) -> None:
    assert edit_distance(left, right) == edit_distance(right, left)


@given(WORDS, WORDS, WORDS)
def test_the_distance_obeys_the_triangle_inequality(
    first: str, second: str, third: str
) -> None:
    """The property the whole tree depends on. Without it the pruning is wrong."""
    assert edit_distance(first, third) <= edit_distance(first, second) + edit_distance(
        second, third
    )


@given(st.lists(WORDS, max_size=25), WORDS, st.integers(min_value=0, max_value=3))
def test_a_tree_search_finds_exactly_what_a_scan_finds(
    vocabulary: list[str], term: str, distance: int
) -> None:
    """Pruning must skip only subtrees that could not have held a match."""
    tree = BKTree.build(vocabulary)
    assert tree.search(term, distance) == by_scanning(vocabulary, term, distance)


@given(st.lists(WORDS, max_size=25))
def test_a_tree_holds_each_distinct_term_once(vocabulary: list[str]) -> None:
    assert len(BKTree.build(vocabulary)) == len(set(vocabulary))


def test_an_empty_vocabulary_searches_without_raising() -> None:
    tree = BKTree.build([])
    assert len(tree) == 0
    assert tree.search("anything") == []


def test_a_single_term_vocabulary() -> None:
    tree = BKTree.build(["alpha"])
    assert tree.search("alpha") == [("alpha", 0)]
    assert tree.search("alphb") == [("alpha", 1)]
    assert tree.search("entirely-different") == []


def test_a_term_already_present_is_returned_at_distance_zero() -> None:
    assert BKTree.build(["alpha", "beta"]).search("alpha") == [("alpha", 0)]


def test_a_repeated_term_is_added_once() -> None:
    tree = BKTree.build(["alpha", "alpha", "alpha"])
    assert len(tree) == 1
    assert tree.search("alpha") == [("alpha", 0)]


def test_two_terms_differing_only_by_a_combining_mark_are_one_apart() -> None:
    """Adversarial: identical on screen, one code point apart in the index.

    The distance is over code points, not over what a reader sees, so these are
    two different terms exactly one edit from each other.
    """
    plain = "cafe"
    marked = f"cafe{COMBINING_ACUTE}"
    assert edit_distance(plain, marked) == 1
    assert BKTree.build([plain, marked]).search(plain, 1) == [
        (plain, 0),
        (marked, 1),
    ]


def test_a_term_longer_than_every_entry_still_searches() -> None:
    """Adversarial: nothing can be within range, and nothing may raise."""
    tree = BKTree.build(["a", "b", "c"])
    assert tree.search("a" * 50) == []


def test_a_vocabulary_all_at_exactly_the_maximum_distance() -> None:
    """Adversarial: the boundary the triangle-inequality pruning gets wrong.

    Every term is exactly `MAX_EDIT_DISTANCE` from the query, so every one is a
    match and any off-by-one in the range check drops some of them.
    """
    query = "aaaa"
    vocabulary = ["bbaa", "aabb", "baab", "abba"]
    for candidate in vocabulary:
        assert edit_distance(query, candidate) == MAX_EDIT_DISTANCE
    found = BKTree.build(vocabulary).search(query)
    assert [term for term, _ in found] == sorted(vocabulary)


def indexed(*documents: str) -> InvertedIndex:
    index = InvertedIndex()
    for document_id, text in enumerate(documents):
        index.add_document(document_id, text)
    return index


def test_a_term_the_index_holds_needs_no_correction() -> None:
    index = indexed("search engine")
    assert suggest(index, "search") == []


def test_suggestions_are_ranked_by_distance_first() -> None:
    """A nearer term outranks a commoner one, whatever their frequencies."""
    index = indexed("alphb", "alzhz", "alzhz", "alzhz", "alzhz")
    assert edit_distance("alpha", "alphb") == 1
    assert edit_distance("alpha", "alzhz") == 2
    assert index.document_frequency("alzhz") > index.document_frequency("alphb")
    assert suggest(index, "alpha") == ["alphb", "alzhz"]


def test_a_more_common_word_wins_at_the_same_distance() -> None:
    """Distance decides first; document frequency breaks the tie."""
    index = indexed(
        "alpht alpht", "alphz", "alpht", "filler filler", "alphz other", "alpht"
    )
    assert suggest(index, "alphx", limit=2) == ["alpht", "alphz"]
    assert index.document_frequency("alpht") > index.document_frequency("alphz")


def test_a_limit_that_is_not_positive_suggests_nothing() -> None:
    assert suggest(indexed("alpha"), "alphb", limit=0) == []


def test_nothing_within_range_suggests_nothing() -> None:
    assert suggest(indexed("alpha beta"), "completely-unrelated") == []
