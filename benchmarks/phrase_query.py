"""Check whether the phrase query optimisations actually pay for themselves.

The implementation intersects the smallest position set first and stops as soon
as the overlap empties. Both are standard advice. Advice is not evidence, so
this compares against a deliberately naive version that intersects in query
order and always finishes.

Cases are chosen to be fair to the optimisation rather than convenient: short
and long phrases, a very frequent term in different positions, and phrases that
fail at the last term, where stopping early should help most.

Run with ``uv run python benchmarks/phrase_query.py``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from search_engine.analysis import analyze
from search_engine.index import InvertedIndex
from search_engine.query import search

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

DOCUMENT_COUNT = 2_000
REPEATS = 20
FILLER = "the " * 100
MARKERS = "alpha beta gamma delta"


def naive_phrase(index: InvertedIndex, terms: Sequence[str]) -> set[int]:
    """Intersect in query order, and never stop early."""
    postings = [index.postings(term) for term in terms]
    if any(not entry for entry in postings):
        return set()
    candidates = set(postings[0])
    for entry in postings[1:]:
        candidates.intersection_update(entry)
    matches: set[int] = set()
    for document_id in candidates:
        shifted = [
            {position - offset for position in postings[offset][document_id]}
            for offset in range(len(terms))
        ]
        common = shifted[0]
        for other in shifted[1:]:
            common &= other
        if common:
            matches.add(document_id)
    return matches


def build() -> InvertedIndex:
    """A rare marker phrase buried in a long run of one very frequent word.

    ``zeta`` sits at the far end, so a phrase ending in it shares a long prefix
    with a real phrase but fails on the final term.
    """
    index = InvertedIndex()
    for document_id in range(DOCUMENT_COUNT):
        index.add_document(document_id, f"{FILLER}{MARKERS} {FILLER} zeta")
    return index


def time_it(
    label: str, run: Callable[[], set[int]], expected: set[int] | None = None
) -> float:
    """Time a zero-argument callable over several repeats."""
    result = run()
    start = time.perf_counter()
    for _ in range(REPEATS):
        run()
    elapsed = (time.perf_counter() - start) / REPEATS
    if expected is not None and result != expected:
        message = f"{label} gave {len(result)} matches, expected {len(expected)}"
        raise AssertionError(message)
    print(f"  {label:32} {elapsed * 1000:8.2f} ms  ({len(result):,} matches)")
    return elapsed


def main() -> None:
    """Compare the two strategies across phrase shapes."""
    index = build()
    for term in ("the", "alpha", "zeta"):
        stemmed = analyze(term)[0]
        print(f"'{stemmed}' df={index.document_frequency(stemmed):,}")
    print(f"{DOCUMENT_COUNT:,} documents, 200 positions of 'the' each")
    print()

    cases = (
        ("2 terms, frequent first", '"the alpha"'),
        ("2 terms, both rare", '"alpha beta"'),
        ("4 rare terms", '"alpha beta gamma delta"'),
        ("4 terms, frequent first", '"the alpha beta gamma"'),
        ("4 terms, fails on the last", '"alpha beta gamma zeta"'),
        ("fails on the first pair", '"delta alpha"'),
    )
    for label, query in cases:
        print(f"{label}: {query}")
        terms = tuple(analyze(query))
        expected = search(index, query)
        ours = time_it("shortest first, early exit", lambda q=query: search(index, q))
        theirs = time_it(
            "query order, no early exit",
            lambda t=terms: naive_phrase(index, t),
            expected,
        )
        verdict = "optimised wins" if theirs > ours else "naive wins"
        print(f"  {'ratio naive/optimised':32} {theirs / ours:8.2f}x  {verdict}")
        print()


if __name__ == "__main__":
    main()
