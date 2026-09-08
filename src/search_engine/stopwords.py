"""Words too common to help, and the means to find more of them.

A term appearing in most documents separates almost nothing, so its postings
list is long and worth little. Dropping such terms shrinks the index where it
is fattest.

Two sources are offered. The default list is small, published and fixed, so
behaviour is reproducible. `derive` computes a list from a real index instead,
which is the honest way to decide what is common *in this corpus* rather than
in English generally.

Filtering happens on raw tokens, before stemming, so entries here are surface
words. Stemming first would require the list to hold stems (`wa` rather than
`was`), which nobody can read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from search_engine.index import ReadableIndex

# The 25 term list from Manning, Raghavan and Schutze, "Introduction to
# Information Retrieval", table 2.2. Chosen because it is published and small:
# a longer list removes more but starts discarding words that carry meaning in
# some queries.
DEFAULT_STOPWORDS: Final = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)


def derive(index: ReadableIndex, threshold: float = 0.5) -> frozenset[str]:
    """Return terms appearing in at least `threshold` of the indexed documents.

    Returns stems, not surface words, because that is what an index stores.
    Useful for measuring what a corpus considers common; not interchangeable
    with `DEFAULT_STOPWORDS`, which is applied before stemming.

    An empty index yields an empty set rather than dividing by zero.
    """
    if index.document_count == 0:
        return frozenset()
    minimum = threshold * index.document_count
    return frozenset(
        term for term in index.terms if index.document_frequency(term) >= minimum
    )
