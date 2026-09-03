"""Reading a query and finding the documents that satisfy it.

Three query forms, distinguished by how the text is written:

- ``search`` matches documents containing that term.
- ``web search engine`` matches documents containing *any* of those terms.
- ``"web search engine"`` matches only documents where those terms appear
  adjacent, in that order.

Results are unordered. Every matching document is equally a match until
something scores them, so a set is the honest return type.

Query text goes through the same analysis as documents. It has to: a query
analyzed differently would ask for terms the index never stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from search_engine.analysis import analyze

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from search_engine.index import InvertedIndex


@dataclass(frozen=True, slots=True)
class Query:
    """A parsed query: the analyzed terms, and whether order matters."""

    terms: tuple[str, ...]
    is_phrase: bool


def parse(text: str) -> Query:
    """Parse query text into terms and a phrase flag.

    A query counts as a phrase only when the whole of it is wrapped in double
    quotes. Mixing a phrase with loose terms is not supported, and in that case
    the quotes are simply separators like any other punctuation.
    """
    stripped = text.strip()
    # The last clause stops a lone quote character satisfying both ends.
    is_phrase = stripped.startswith('"') and stripped.endswith('"') and stripped != '"'
    return Query(terms=tuple(analyze(stripped)), is_phrase=is_phrase)


def _matching_any_term(index: InvertedIndex, terms: Sequence[str]) -> set[int]:
    """Union the postings, which is what a one-word or free-text query wants."""
    matches: set[int] = set()
    for term in terms:
        matches.update(index.postings(term))
    return matches


def _contains_phrase_at_any_position(
    postings: Sequence[Mapping[int, Sequence[int]]], document_id: int
) -> bool:
    """Test whether one document holds the terms adjacent and in order.

    The trick is to subtract each term's offset in the phrase from its
    positions. A three word phrase starting at position 7 gives positions 7, 8
    and 9, which after subtracting 0, 1 and 2 all become 7. So the phrase
    occurs exactly where the shifted position sets overlap, and adjacency
    becomes a plain set intersection.

    The shifted sets are built one at a time and the loop stops as soon as the
    overlap empties, so a phrase that fails early never pays to build the rest.
    Sorting them by size first was measurably slower: sorting has to build
    every set before it can compare their lengths, which is the expensive part.
    """
    common = set(postings[0][document_id])
    for offset in range(1, len(postings)):
        if not common:
            return False
        common &= {position - offset for position in postings[offset][document_id]}
    return bool(common)


def _matching_phrase(index: InvertedIndex, terms: Sequence[str]) -> set[int]:
    """Find documents holding every term, adjacent and in the given order."""
    if not terms:
        return set()
    postings = [index.postings(term) for term in terms]
    if any(not entry for entry in postings):
        return set()
    # Start from the rarest term, so the candidate set begins as small as it
    # can. Skipping that entry avoids intersecting it with itself.
    rarest = min(range(len(postings)), key=lambda offset: len(postings[offset]))
    candidates = set(postings[rarest])
    for offset, entry in enumerate(postings):
        if offset == rarest:
            continue
        candidates.intersection_update(entry)
        if not candidates:
            return set()
    return {
        document_id
        for document_id in candidates
        if _contains_phrase_at_any_position(postings, document_id)
    }


def execute(index: InvertedIndex, query: Query) -> set[int]:
    """Run a parsed query and return the identifiers of matching documents."""
    if query.is_phrase:
        return _matching_phrase(index, query.terms)
    return _matching_any_term(index, query.terms)


def search(index: InvertedIndex, text: str) -> set[int]:
    """Parse and run query text in one step."""
    return execute(index, parse(text))
