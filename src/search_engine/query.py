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

from search_engine.analysis import analyze_positioned

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from search_engine.index import ReadableIndex


@dataclass(frozen=True, slots=True)
class Query:
    """A parsed query.

    ``offsets[i]`` is where ``terms[i]`` sat in the original token stream, so a
    dropped stopword leaves a gap. Phrase matching subtracts these rather than
    consecutive integers, which is what lets a phrase containing a stopword
    still match.
    """

    terms: tuple[str, ...]
    offsets: tuple[int, ...]
    is_phrase: bool


def parse(text: str) -> Query:
    """Parse query text into terms, their offsets, and a phrase flag.

    A query counts as a phrase only when the whole of it is wrapped in double
    quotes. Mixing a phrase with loose terms is not supported, and in that case
    the quotes are simply separators like any other punctuation.
    """
    stripped = text.strip()
    # The last clause stops a lone quote character satisfying both ends.
    is_phrase = stripped.startswith('"') and stripped.endswith('"') and stripped != '"'
    positioned = analyze_positioned(stripped)
    return Query(
        terms=tuple(term for _, term in positioned),
        offsets=tuple(offset for offset, _ in positioned),
        is_phrase=is_phrase,
    )


def _matching_any_term(index: ReadableIndex, terms: Sequence[str]) -> set[int]:
    """Union the postings, which is what a one-word or free-text query wants."""
    matches: set[int] = set()
    for term in terms:
        matches.update(index.postings(term))
    return matches


def _contains_phrase(
    postings: Sequence[Mapping[int, Sequence[int]]],
    offsets: Sequence[int],
    document_id: int,
) -> bool:
    """Test whether one document holds the terms at the queried spacing.

    Subtract each term's query offset from its document positions. A phrase
    occurring at document position ``p`` puts the term queried at offset ``i``
    at ``p + i``, so subtracting ``i`` maps every term of a real occurrence
    onto the same ``p``. Adjacency, and any gap a dropped stopword left,
    becomes a plain set intersection.

    Sets are built one at a time and the loop stops the moment the overlap
    empties, so a phrase failing early never pays to build the rest. Sorting
    them by size first measured slower: sorting must build every set before it
    can compare lengths, which is the expensive part.
    """
    common = {position - offsets[0] for position in postings[0][document_id]}
    for index in range(1, len(postings)):
        if not common:
            return False
        common &= {
            position - offsets[index] for position in postings[index][document_id]
        }
    return bool(common)


def _matching_phrase(
    index: ReadableIndex, terms: Sequence[str], offsets: Sequence[int]
) -> set[int]:
    """Find documents holding every term at the spacing the query asked for."""
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
        if _contains_phrase(postings, offsets, document_id)
    }


def execute(index: ReadableIndex, query: Query) -> set[int]:
    """Run a parsed query and return the identifiers of matching documents."""
    if query.is_phrase:
        return _matching_phrase(index, query.terms, query.offsets)
    return _matching_any_term(index, query.terms)


def search(index: ReadableIndex, text: str) -> set[int]:
    """Parse and run query text in one step."""
    return execute(index, parse(text))
