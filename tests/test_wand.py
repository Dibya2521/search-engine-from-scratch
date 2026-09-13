"""Tests for early termination.

The load-bearing test is `test_wand_returns_exactly_what_a_full_scan_returns`.
WAND is exact rather than approximate, so it is only worth having if it returns
what scoring every candidate returns, identifiers and scores alike. The property
test compares the two on generated corpora and queries, and every other test
here covers a shape generation reaches rarely or not at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.analysis import analyze
from search_engine.bm25 import BM25Ranker
from search_engine.directory import DirectoryIndex
from search_engine.index import InvertedIndex
from search_engine.segment import SegmentReader, write_segment
from search_engine.wand import rank_wand, search_wand
from search_engine.writer import IndexWriter
from tests.conftest import build_index

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from search_engine.index import ReadableIndex

WORDS = ("alpha", "beta", "gamma", "delta", "epsilon")


def candidates_of(index: ReadableIndex, terms: Sequence[str]) -> set[int]:
    """Return every document holding a query term, which is what a scan sees."""
    return {document_id for term in terms for document_id in index.postings(term)}


def assert_matches_full_scan(
    index: ReadableIndex, ranker: BM25Ranker, query: str, limit: int
) -> int:
    """Assert early termination agrees with scoring everything, and count scores."""
    terms = analyze(query)
    expected = ranker.rank(query, candidates_of(index, terms), limit)
    result = search_wand(index, ranker, terms, limit)
    assert result.documents == expected
    return result.scored


def skewed_corpus() -> dict[int, str]:
    """One term in every document, a second in a few, and lengths that vary.

    The shape early termination exists for: the rare term decides the results
    while the common term supplies most of the candidates.
    """
    corpus: dict[int, str] = {}
    for document_id in range(200):
        words = ["alpha"] * (1 + document_id % 3)
        if document_id % 25 == 0:
            words += ["beta"] * 4
        words += ["filler"] * (document_id % 11)
        corpus[document_id] = " ".join(words)
    return corpus


@given(
    documents=st.lists(
        st.lists(st.sampled_from(WORDS), max_size=10).map(" ".join),
        min_size=1,
        max_size=12,
    ),
    query_terms=st.lists(st.sampled_from((*WORDS, "absent")), min_size=1, max_size=3),
    limit=st.integers(min_value=1, max_value=5),
)
def test_wand_returns_exactly_what_a_full_scan_returns(
    documents: list[str], query_terms: list[str], limit: int
) -> None:
    """Any difference at all is a bug, because the algorithm is exact."""
    index = build_index(dict(enumerate(documents)))
    ranker = BM25Ranker(index)
    assert_matches_full_scan(index, ranker, " ".join(query_terms), limit)


def test_a_single_term_query_is_exact(ranking_index: InvertedIndex) -> None:
    """One cursor, so the pivot is always the first and nothing is skipped."""
    ranker = BM25Ranker(ranking_index)
    assert assert_matches_full_scan(ranking_index, ranker, "computer", 10) == 3


def test_a_query_of_absent_terms_returns_nothing(ranking_index: InvertedIndex) -> None:
    ranker = BM25Ranker(ranking_index)
    assert rank_wand(ranking_index, ranker, ["absent"], 10) == []


def test_an_empty_query_returns_nothing(ranking_index: InvertedIndex) -> None:
    ranker = BM25Ranker(ranking_index)
    assert rank_wand(ranking_index, ranker, [], 10) == []


@pytest.mark.parametrize("limit", [0, -1])
def test_a_limit_that_is_not_positive_is_rejected(
    ranking_index: InvertedIndex, limit: int
) -> None:
    ranker = BM25Ranker(ranking_index)
    with pytest.raises(ValueError, match="limit must be positive"):
        rank_wand(ranking_index, ranker, analyze("computer"), limit)


def test_a_limit_larger_than_the_matches_returns_them_all(
    ranking_index: InvertedIndex,
) -> None:
    ranker = BM25Ranker(ranking_index)
    assert len(rank_wand(ranking_index, ranker, analyze("computer"), 50)) == 3


def test_the_tightest_limit_scores_the_fewest_documents() -> None:
    """Measured: 8 of 200 candidates scored to return the single best."""
    index = build_index(skewed_corpus())
    ranker = BM25Ranker(index)
    assert assert_matches_full_scan(index, ranker, "alpha beta", 1) == 8


def test_a_low_threshold_prunes_nothing() -> None:
    """Only 8 documents hold the rare term, so the tenth result is a weak one.

    The threshold it sets is below what any candidate could reach, and every
    candidate is scored. Early termination costs rather than pays here.
    """
    index = build_index(skewed_corpus())
    ranker = BM25Ranker(index)
    assert assert_matches_full_scan(index, ranker, "alpha beta", 10) == 200


def test_bounds_that_cannot_prune_are_still_exact() -> None:
    """Every document holds every query term at the same frequency.

    The upper bounds are then as tight as they can be and no document can be
    ruled out. An implementation that prunes here returns the wrong documents.
    """
    corpus = {
        document_id: " ".join(["alpha", "beta"] + ["filler"] * (document_id % 5))
        for document_id in range(40)
    }
    index = build_index(corpus)
    ranker = BM25Ranker(index)
    assert assert_matches_full_scan(index, ranker, "alpha beta", 10) == 40


def test_identical_documents_are_separated_by_identifier() -> None:
    """Every score is equal, so the tie-break alone decides the result.

    The search stops once the limit is held, because no document still to come
    can beat a threshold its own bound only equals, and any that tied it would
    lose the tie-break to one already scored.
    """
    index = build_index(dict.fromkeys(range(40), "alpha beta gamma"))
    ranker = BM25Ranker(index)
    assert assert_matches_full_scan(index, ranker, "alpha beta", 5) == 5
    results = rank_wand(index, ranker, analyze("alpha beta"), 5)
    assert [document_id for document_id, _ in results] == [0, 1, 2, 3, 4]


def test_a_segment_is_walked_through_its_own_postings(tmp_path: Path) -> None:
    """The segment path decodes identifiers from the mapped file directly."""
    path = tmp_path / "one.seg"
    write_segment(build_index(skewed_corpus()), path)
    with SegmentReader(path) as reader:
        ranker = BM25Ranker(reader)
        assert assert_matches_full_scan(reader, ranker, "alpha beta", 1) == 8


def test_a_term_no_segment_holds_opens_no_cursor(tmp_path: Path) -> None:
    path = tmp_path / "one.seg"
    write_segment(build_index({0: "alpha", 1: "beta"}), path)
    with SegmentReader(path) as reader:
        ranker = BM25Ranker(reader)
        assert rank_wand(reader, ranker, ["absent"], 5) == []


def test_a_deleted_document_never_reaches_the_results(tmp_path: Path) -> None:
    """A directory index is walked through `postings`, which hides tombstones.

    Reading the segment bytes underneath it would return the deleted document,
    because nothing below `postings` applies the tombstones.
    """
    texts = {0: "alpha beta", 1: "alpha", 2: "alpha alpha"}
    with IndexWriter(tmp_path, buffer_documents=1, merge=False) as writer:
        for document_id, text in texts.items():
            writer.add(document_id, text)
        assert writer.delete(2)
    with DirectoryIndex(tmp_path) as index:
        ranker = BM25Ranker(index)
        assert_matches_full_scan(index, ranker, "alpha", 10)
        results = rank_wand(index, ranker, analyze("alpha"), 10)
    assert {document_id for document_id, _ in results} == {0, 1}


def outlier_index(size: int, at: int, spike: int) -> InvertedIndex:
    """One term everywhere, once per document except for one that repeats it.

    Built through `from_postings` so the frequencies are exact rather than
    whatever analysis would make of generated text.
    """
    postings = {
        "alpha": {
            document_id: list(range(spike if document_id == at else 1))
            for document_id in range(size)
        }
    }
    return InvertedIndex.from_postings(list(range(size)), postings)


def test_block_bounds_skip_the_blocks_a_single_bound_cannot(tmp_path: Path) -> None:
    """The case a bound for the whole term cannot handle.

    One document holds the term 500 times and 4,999 hold it once. The term's
    own bound is set by that one document, so it stays above every threshold
    and prunes nothing. A bound per block prunes every block the outlier is not
    in, leaving only the block it sits in to be scored.

    Measured: 5,000 documents scored through the term bound, 129 through the
    block bounds, and the same result from both.
    """
    index = outlier_index(5_000, 2_500, 500)
    without_blocks = search_wand(index, BM25Ranker(index), ["alpha"], 1)
    assert without_blocks.scored == 5_000

    path = tmp_path / "outlier.seg"
    write_segment(index, path)
    with SegmentReader(path) as reader:
        with_blocks = search_wand(reader, BM25Ranker(reader), ["alpha"], 1)
    assert with_blocks.scored == 129
    assert with_blocks.documents == without_blocks.documents


def test_only_the_block_holding_the_outlier_is_scored(tmp_path: Path) -> None:
    """Scoring stops at one document plus the block the outlier sits in.

    The trailing block holds 8 of the 5,000 documents, so putting the outlier
    there leaves 9 documents scored rather than 129. The count follows the
    block the outlier lands in, not the size of the corpus.
    """
    index = outlier_index(5_000, 4_999, 500)
    expected = search_wand(index, BM25Ranker(index), ["alpha"], 1)
    path = tmp_path / "tail.seg"
    write_segment(index, path)
    with SegmentReader(path) as reader:
        result = search_wand(reader, BM25Ranker(reader), ["alpha"], 1)
    assert result.scored == 9
    assert result.documents == expected.documents
    assert [document_id for document_id, _ in result.documents] == [4_999]


def test_early_termination_refuses_a_proximity_ranker(
    ranking_index: InvertedIndex,
) -> None:
    """It cannot reproduce a boost that depends on where terms sit.

    The bounds here are per term and know nothing about position, so pruning
    would happen against one scale and scores be reported on another.
    """
    ranker = BM25Ranker(ranking_index, proximity=True)
    with pytest.raises(ValueError, match="cannot reproduce a proximity boost"):
        rank_wand(ranking_index, ranker, analyze("computer"), 10)
