"""The inverted index: a map from each term to where it occurs.

A forward index answers "what is in document 5", which is the wrong question.
Searching asks "which documents contain this term", so the mapping is inverted:
term first, documents second. That turns a scan of every document into one
dictionary lookup.

Each term maps to postings recording not only *which* documents contain it but
at *which positions*. Positions cost memory and buy phrase queries: without
them there is no way to know that `computer` and `science` were adjacent rather
than merely both present.

Postings are keyed by document identifier rather than held as a sorted list.
Both shapes appear in the literature; a mapping is chosen because every query
operation needed here is a set operation over document identifiers, and
dictionary keys give those directly.

A term can also be qualified by the field it came from, so `title:python` sits
in the same dictionary beside `python`. That is the cheapest of the three ways
to record fields, and deliberately so: per-field postings maps or a field
bitmask beside each position would both change the on-disk format, and this
changes nothing. The cost is that the vocabulary grows and a title's terms are
stored twice, which is measured rather than assumed.

**A qualified term cannot be forged from document text.** The separator is not a
word character, so the tokenizer never emits it: a body containing the literal
text `title:python` produces the two terms `titl` and `python` and never the
qualified one.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from search_engine.analysis import analyze_positioned

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

_NO_POSTINGS: Mapping[int, Sequence[int]] = MappingProxyType({})

# Not a word character, so no token can contain it and no document text can be
# mistaken for a qualified term.
FIELD_SEPARATOR: Final = ":"
TITLE_FIELD: Final = "title"


def field_term(field: str, term: str) -> str:
    """Return a term qualified by the field it occurred in."""
    return f"{field}{FIELD_SEPARATOR}{term}"


def is_field_term(term: str) -> bool:
    """Return whether a term is qualified by a field."""
    return FIELD_SEPARATOR in term


class ReadableIndex(Protocol):
    """The read side of an index, satisfied by the in-memory and on-disk forms.

    Querying and ranking depend on this and nothing more, which is what lets an
    index read from a file be searched by the same code that searches one held
    in memory.
    """

    def postings(self, term: str) -> Mapping[int, Sequence[int]]:
        """Return document identifiers to increasing positions."""
        ...

    def document_frequency(self, term: str) -> int:
        """Return how many documents contain the term."""
        ...

    def document_length(self, document_id: int) -> int:
        """Return a document's length in analysed tokens."""
        ...

    def __contains__(self, term: str) -> bool:
        """Return whether any document holds this term."""
        ...

    @property
    def document_count(self) -> int:
        """Return how many documents the index holds."""
        ...

    @property
    def vocabulary_size(self) -> int:
        """Return how many distinct terms the index holds."""
        ...

    @property
    def terms(self) -> Iterable[str]:
        """Return every distinct term."""
        ...

    @property
    def document_ids(self) -> Iterable[int]:
        """Return every indexed document identifier."""
        ...


class DuplicateDocumentError(ValueError):
    """Raised when a document identifier is added twice.

    Adding one twice would append a second set of positions to the first,
    silently corrupting every position already stored for it.
    """

    def __init__(self, document_id: int) -> None:
        super().__init__(f"document {document_id} is already indexed")


class InvertedIndex:
    """An in-memory inverted index with positional postings.

    Construct it empty, then add documents one at a time.
    """

    def __init__(self) -> None:
        self._postings: dict[str, dict[int, list[int]]] = {}
        self._document_ids: set[int] = set()
        self._lengths: dict[int, int] | None = None

    @classmethod
    def from_postings(
        cls,
        document_ids: Iterable[int],
        postings: Mapping[str, Mapping[int, Sequence[int]]],
    ) -> InvertedIndex:
        """Rebuild an index from stored postings, bypassing analysis.

        The load path. Stored terms are already analyzed, and re-analyzing them
        would change them, because stemming is not idempotent.

        Document identifiers are passed separately because a document holding
        no terms appears in no postings list, yet must still count towards the
        total used to weight rare terms.
        """
        index = cls()
        index._document_ids = set(document_ids)
        index._postings = {
            term: {
                document_id: list(positions)
                for document_id, positions in entries.items()
            }
            for term, entries in postings.items()
        }
        return index

    def add_document(
        self, document_id: int, text: str, fields: Mapping[str, str] | None = None
    ) -> None:
        """Analyze a document and record every term occurrence in it.

        `fields` names parts of the same document worth telling apart, such as
        its title. Each is indexed a second time under terms qualified by the
        field name, so `title:python` can be asked for without changing
        anything a search for `python` already returns.

        Raises:
            DuplicateDocumentError: If this identifier was already added.
        """
        if document_id in self._document_ids:
            raise DuplicateDocumentError(document_id)
        self._document_ids.add(document_id)
        self._lengths = None
        self._record(document_id, text, "")
        for field, content in (fields or {}).items():
            self._record(document_id, content, field)

    def _record(self, document_id: int, text: str, field: str) -> None:
        """Add one text's terms, qualified by a field name when there is one."""
        for position, term in analyze_positioned(text):
            key = field_term(field, term) if field else term
            self._postings.setdefault(key, {}).setdefault(document_id, []).append(
                position
            )

    @property
    def document_count(self) -> int:
        """Number of documents indexed, including any with no terms."""
        return len(self._document_ids)

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct terms, which is how many dimensions a document has."""
        return len(self._postings)

    @property
    def terms(self) -> Iterable[str]:
        """Every distinct term, in the order first encountered."""
        return self._postings.keys()

    @property
    def document_ids(self) -> Iterable[int]:
        """Every indexed identifier, including documents holding no terms.

        Those cannot be recovered from the postings, since they appear in none,
        so anything saving or iterating the corpus needs them from here.
        """
        return frozenset(self._document_ids)

    def postings(self, term: str) -> Mapping[int, Sequence[int]]:
        """Return document identifiers to increasing positions, empty if unknown.

        Returning an empty mapping rather than raising lets callers treat an
        absent term as a term with no matches, which is what it is.
        """
        found = self._postings.get(term)
        return _NO_POSTINGS if found is None else MappingProxyType(found)

    def document_frequency(self, term: str) -> int:
        """How many documents contain the term, not how often it occurs."""
        return len(self._postings.get(term, ()))

    def document_length(self, document_id: int) -> int:
        """Length in analysed tokens, so removed stopwords do not count.

        Recovering this means one pass over every posting, so the pass is made
        once and kept until a document is added. A scorer with a length prior
        asks for it per document and should not pay for the pass each time.
        """
        if self._lengths is None:
            self._lengths = self._compute_lengths()
        return self._lengths.get(document_id, 0)

    def _compute_lengths(self) -> dict[int, int]:
        """Count each occurrence once, ignoring the field-qualified copies.

        A qualified term repeats content already counted under its unqualified
        form, so counting both would inflate every length a length-normalising
        scorer divides by, and a document with a long title would look longer
        than it is.
        """
        lengths: dict[int, int] = {}
        for term, entries in self._postings.items():
            if is_field_term(term):
                continue
            for document_id, positions in entries.items():
                lengths[document_id] = lengths.get(document_id, 0) + len(positions)
        return lengths

    def __contains__(self, term: str) -> bool:
        """Whether any document holds this term. Terms, not document ids."""
        return term in self._postings
