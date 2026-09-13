"""Tests for query-time synonym expansion and query segmentation.

The load-bearing tests are the two that pin what "equivalence" means:
`test_expansion_is_symmetric` and `test_groups_sharing_a_term_merge`. A table
that expanded one way, or that kept two overlapping groups apart, would still
look right on a single example and be wrong on the next one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.index import InvertedIndex
from search_engine.query import search
from search_engine.synonyms import (
    SynonymFormatError,
    SynonymTable,
    segment_query,
)

if TYPE_CHECKING:
    from pathlib import Path

GROUPS = ["laptop, notebook", "car, automobile, vehicle"]


def table_of(*lines: str) -> SynonymTable:
    return SynonymTable.parse(lines)


def test_a_term_in_no_group_expands_to_itself() -> None:
    """So a caller can expand every query term without checking first."""
    assert table_of(*GROUPS).expand("bicycl") == frozenset({"bicycl"})


def test_an_empty_table_expands_nothing() -> None:
    assert SynonymTable.empty().expand("laptop") == frozenset({"laptop"})
    assert len(SynonymTable.empty()) == 0


def test_expansion_is_symmetric() -> None:
    """Groups are equivalences, so direction cannot matter."""
    table = table_of(*GROUPS)
    assert table.expand("laptop") == table.expand("notebook")


def test_every_member_of_a_group_expands_to_the_whole_group() -> None:
    table = table_of(*GROUPS)
    whole = frozenset({"car", "automobil", "vehicl"})
    for member in whole:
        assert table.expand(member) == whole


def test_members_are_stored_as_index_terms() -> None:
    """Expansion has to produce terms the index actually holds."""
    table = table_of("car, automobile, vehicle")
    assert table.expand("car") == frozenset({"car", "automobil", "vehicl"})


def test_groups_sharing_a_term_merge() -> None:
    """Equivalence is transitive, so the result cannot depend on line order."""
    forward = table_of("laptop, notebook", "notebook, ultrabook")
    backward = table_of("notebook, ultrabook", "laptop, notebook")
    whole = frozenset({"laptop", "notebook", "ultrabook"})
    assert forward.expand("laptop") == whole
    assert backward.expand("laptop") == whole
    assert forward.expand("ultrabook") == backward.expand("ultrabook") == whole


def test_three_groups_chained_through_shared_terms_all_merge() -> None:
    table = table_of("a1, b1", "b1, c1", "c1, d1")
    assert table.expand("a1") == frozenset({"a1", "b1", "c1", "d1"})


@pytest.mark.parametrize("line", ["", "   ", "# a comment", "   # indented comment"])
def test_blank_and_comment_lines_are_skipped(line: str) -> None:
    assert len(table_of(line, "laptop, notebook")) == 2


def test_a_multi_word_member_is_refused_with_its_line_number() -> None:
    """Matching one would mean matching a phrase, which expansion cannot produce.

    Accepting it and expanding to the words separately would make `portable
    computer` match any document containing `computer`, which is worse than
    refusing it.
    """
    with pytest.raises(SynonymFormatError, match="line 2") as caught:
        table_of("laptop, notebook", "laptop, portable computer")
    assert caught.value.line_number == 2
    assert "single term" in str(caught.value)


def test_a_member_that_analyses_to_nothing_is_refused() -> None:
    """A stopword or punctuation leaves no term to be equivalent to."""
    with pytest.raises(SynonymFormatError, match="single term"):
        table_of("laptop, the")


def test_a_group_of_fewer_than_two_distinct_terms_is_refused() -> None:
    with pytest.raises(SynonymFormatError, match="at least two"):
        table_of("lonely")


def test_a_group_whose_members_stem_together_is_refused_rather_than_empty() -> None:
    """Adversarial: two surface words, one index term, so the group is a no-op.

    It must not crash and must not produce a group of one, which would expand a
    term to itself while claiming to be a synonym group.
    """
    with pytest.raises(SynonymFormatError, match="at least two"):
        table_of("running, running")
    with pytest.raises(SynonymFormatError, match="at least two"):
        table_of("compute, computing")


def test_a_table_loads_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / "synonyms.txt"
    path.write_text("# comment\nlaptop, notebook\n", encoding="utf-8")
    assert SynonymTable.load(path).expand("laptop") == frozenset({"laptop", "notebook"})


def test_a_missing_file_raises_an_os_error(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="No such file"):
        SynonymTable.load(tmp_path / "absent.txt")


@given(st.lists(st.sampled_from(["a1", "b1", "c1", "d1"]), min_size=2, max_size=4))
def test_expansion_always_contains_the_term_itself(members: list[str]) -> None:
    """Whatever the groups, a term never expands to something excluding itself."""
    distinct = sorted(set(members))
    table = table_of(", ".join(distinct)) if len(distinct) > 1 else SynonymTable.empty()
    for term in ("a1", "b1", "c1", "d1", "unrelated"):
        assert term in table.expand(term)


def test_segmentation_prefers_the_longest_match() -> None:
    """Otherwise `new york times` becomes `new york` followed by `times`."""
    phrases = {"new york times", "new york"}
    assert segment_query("new york times review", phrases) == [
        '"new york times"',
        "review",
    ]


def test_a_phrase_that_is_a_prefix_of_another_still_matches_alone() -> None:
    """Adversarial: the shorter phrase must survive where the longer does not."""
    phrases = {"new york times", "new york"}
    assert segment_query("new york bagel", phrases) == ['"new york"', "bagel"]


def test_text_with_no_known_phrase_is_left_as_words() -> None:
    assert segment_query("plain ordinary query", {"new york"}) == [
        "plain",
        "ordinary",
        "query",
    ]


def test_an_empty_phrase_set_segments_nothing() -> None:
    assert segment_query("new york times", set()) == ["new", "york", "times"]


def test_a_phrase_is_matched_however_it_is_capitalised() -> None:
    """The set names entities, and an entity is the same entity in any case."""
    assert segment_query("New York Times", {"new york times"}) == ['"New York Times"']


def test_a_phrase_at_the_end_of_the_text_is_matched() -> None:
    """The window must not run past the end and quietly stop matching."""
    assert segment_query("read the new york times", {"new york times"}) == [
        "read",
        "the",
        '"new york times"',
    ]


def test_empty_text_segments_to_nothing() -> None:
    assert segment_query("", {"new york"}) == []


def test_expansion_widens_what_a_query_matches() -> None:
    """A document that never says `laptop` is found by a search for it."""
    index = InvertedIndex()
    index.add_document(1, "a portable notebook for travel")
    index.add_document(2, "a heavy desktop machine")
    index.add_document(3, "laptop reviews")
    table = table_of("laptop, notebook")
    assert search(index, "laptop") == {3}
    assert search(index, "laptop", table) == {1, 3}


def test_a_phrase_query_is_never_expanded() -> None:
    """A phrase matches on spacing, and a stand-in term occupies no position."""
    index = InvertedIndex()
    index.add_document(1, "a portable notebook here")
    table = table_of("laptop, notebook")
    assert search(index, '"portable laptop"', table) == set()
    assert search(index, '"portable notebook"', table) == {1}
