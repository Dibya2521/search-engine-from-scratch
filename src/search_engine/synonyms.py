"""Query-time synonym expansion, and grouping adjacent words into one entity.

Most relevance gains in real systems come from understanding the query rather
than from a better scorer. A document that only says `notebook` should be found
by a search for `laptop`.

**Expansion happens at query time, never at index time.** Storing every synonym
in the postings would work, and it has two costs that make it the wrong answer.
The synonym list is the thing that changes most often, and index-time expansion
means no synonym can be added without reindexing every document. It also
inflates document frequency for every term in a group, which corrupts the
inverse document frequency of terms that were never searched for. Query-time
expansion changes one query and nothing else.

**Groups are equivalences, not directed rules.** A directed rule needs a
direction decided for every pair, and there is rarely a principled answer. An
equivalence needs none: any member matches any other. Groups that share a term
are merged into one, because equivalence is transitive and pretending otherwise
would make the result depend on the order lines happen to appear in.

**The file format**, one group per line:

```text
# Groups are bidirectional: any member matches any other.
laptop, notebook
car, automobile, vehicle
```

Members are analysed exactly as documents are, so what is stored are index
terms. **A member that analyses to more than one term is refused**, naming the
line. Matching a multi-word synonym means matching a phrase, which needs the
expansion to produce phrases rather than terms, and that is not built here.
Accepting such a member and expanding to its words separately would make
`portable computer` match any document containing `computer`, which is worse
than refusing it.

**Expansion changes scores, and there is no down-weighting yet.** An expanded
query matches on more terms, so its scores are not comparable with the
unexpanded query's. The conventional answer is to weight an expanded term below
the original, commonly at 0.7, which is a convention and not a measurement. It
needs the ranker to accept weighted query terms, and it does not, so the
weighting is not implemented rather than being faked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from search_engine.analysis import analyze

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from collections.abc import Set as AbstractSet
    from pathlib import Path

COMMENT = "#"
SEPARATOR = ","

# A group of one expands a term to itself, which is what having no group does.
SMALLEST_GROUP = 2


class SynonymFormatError(ValueError):
    """Raised when a synonym file cannot be read as equivalence groups.

    Names the line, because a synonym file is edited by hand and often by
    someone who did not write the parser.
    """

    def __init__(self, line_number: int, reason: str) -> None:
        super().__init__(f"synonyms line {line_number}: {reason}")
        self.line_number = line_number


class SynonymTable:
    """Bidirectional equivalence groups, applied at query time."""

    __slots__ = ("_groups",)

    def __init__(self, groups: Mapping[str, frozenset[str]]) -> None:
        """Take a mapping from each term to the whole group it belongs to."""
        self._groups = dict(groups)

    @classmethod
    def empty(cls) -> SynonymTable:
        """Return a table that expands nothing, for a caller with no file."""
        return cls({})

    @classmethod
    def load(cls, path: Path) -> SynonymTable:
        """Read equivalence groups from a file.

        Raises:
            SynonymFormatError: If a line is not a group of single terms.
            OSError: If the file cannot be read.
        """
        return cls.parse(path.read_text(encoding="utf-8").splitlines())

    @classmethod
    def parse(cls, lines: Iterable[str]) -> SynonymTable:
        """Read equivalence groups from lines, so a caller needs no file.

        Raises:
            SynonymFormatError: If a line is not a group of single terms.
        """
        groups = [
            _group(number, line)
            for number, line in enumerate(lines, start=1)
            if _is_group(line)
        ]
        return cls(_merge(groups))

    def expand(self, term: str) -> frozenset[str]:
        """Return the term and every term equivalent to it.

        A term in no group expands to itself, so a caller can apply this to
        every query term without checking first.
        """
        return self._groups.get(term, frozenset({term}))

    def __len__(self) -> int:
        """Return how many terms belong to some group."""
        return len(self._groups)


def _is_group(line: str) -> bool:
    """Return whether a line carries a group rather than a comment or a blank."""
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith(COMMENT)


def _group(number: int, line: str) -> frozenset[str]:
    """Return one line's members as index terms.

    Raises:
        SynonymFormatError: If the line holds fewer than two members, or a
            member that is not exactly one term after analysis.
    """
    members = [member.strip() for member in line.split(SEPARATOR)]
    terms: list[str] = []
    for member in members:
        analysed = analyze(member)
        if len(analysed) != 1:
            reason = f"{member!r} is not a single term after analysis"
            raise SynonymFormatError(number, reason)
        terms.append(analysed[0])
    if len(set(terms)) < SMALLEST_GROUP:
        reason = "a group needs at least two distinct terms"
        raise SynonymFormatError(number, reason)
    return frozenset(terms)


def _merge(groups: Iterable[frozenset[str]]) -> dict[str, frozenset[str]]:
    """Combine every pair of groups sharing a term, since equivalence is transitive."""
    merged: list[set[str]] = []
    for group in groups:
        touching = [existing for existing in merged if existing & group]
        merged = [existing for existing in merged if not existing & group]
        merged.append(set(group).union(*touching))
    return {term: frozenset(group) for group in merged for term in group}


def segment_query(text: str, phrases: AbstractSet[str]) -> list[str]:
    """Group adjacent words that form a known phrase into one quoted term.

    Longest match wins, so ``new york times`` segments as one entity rather than
    as ``new york`` followed by ``times``. A phrase set is matched on the words
    as typed, lowercased, because it names entities rather than index terms.

    **Learning the phrase set from data is the missing half.** The statistic
    that would do it is pointwise mutual information over adjacent term pairs,
    which needs a corpus large enough for the counts to mean something. Until
    then the set is supplied by the caller.
    """
    words = text.split()
    longest = max((len(phrase.split()) for phrase in phrases), default=1)
    segments: list[str] = []
    at = 0
    while at < len(words):
        length = _longest_phrase_at(words, at, phrases, longest)
        window = words[at : at + length]
        segments.append(f'"{" ".join(window)}"' if length > 1 else window[0])
        at += length
    return segments


def _longest_phrase_at(
    words: list[str], at: int, phrases: AbstractSet[str], longest: int
) -> int:
    """Return how many words from `at` form the longest known phrase, or one."""
    for length in range(min(longest, len(words) - at), 1, -1):
        if " ".join(words[at : at + length]).lower() in phrases:
            return length
    return 1
