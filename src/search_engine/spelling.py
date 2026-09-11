"""Suggesting corrections for a query that found nothing.

A query returning nothing is the worst answer a search engine can give, and a
misspelling is the commonest reason for it. The engine already knows every term
it holds, so it can offer the near ones.

**The structure is a BK-tree**, a metric tree over edit distance. It works
because Levenshtein distance obeys the triangle inequality: for any three
strings, ``d(a, c) <= d(a, b) + d(b, c)``. So if the query is at distance ``d``
from some node, a term reachable through that node at distance ``k`` from it can
be no closer to the query than ``d - k`` and no further than ``d + k``. Only
children whose distance from the node lies in ``[d - max, d + max]`` can hold a
match, and every other subtree is skipped without one distance being computed
inside it.

That is the whole of the idea. A vocabulary of tens of thousands of terms is
searched by computing a few hundred distances rather than all of them.

**The tree is built from the vocabulary, not from the postings**, so it costs
one pass over the term dictionary and reads no postings at all. Building it is
still proportional to the vocabulary, so a caller answering many queries should
build one tree and call `BKTree.search` directly rather than calling `suggest`
repeatedly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable

    from search_engine.index import ReadableIndex

MAX_EDIT_DISTANCE: Final = 2
DEFAULT_SUGGESTIONS: Final = 3


def edit_distance(left: str, right: str) -> int:
    """Return the Levenshtein distance: insertions, deletions, substitutions.

    Two rows rather than the full matrix, so memory is proportional to the
    shorter of the two strings rather than to their product. The shorter string
    is placed on the inner axis to make that so.

    The distance is exact and takes no limit. A BK-tree needs the true distance
    at every node it visits, because the range of children worth descending into
    is computed from it, and a distance capped at some threshold would make the
    tree prune the wrong subtrees.
    """
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, from_character in enumerate(left, start=1):
        current = [row]
        for column, to_character in enumerate(right, start=1):
            change = 0 if from_character == to_character else 1
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + change,
                )
            )
        previous = current
    return previous[-1]


@dataclass(slots=True)
class _Node:
    """One term, and the children keyed by their distance from it."""

    term: str
    children: dict[int, _Node] = field(default_factory=dict["int", "_Node"])


class BKTree:
    """A metric tree over edit distance, for finding near matches in a vocabulary."""

    __slots__ = ("_root", "_size")

    def __init__(self) -> None:
        """Build an empty tree. Use `build` to fill one from a vocabulary."""
        self._root: _Node | None = None
        self._size = 0

    @classmethod
    def build(cls, terms: Iterable[str]) -> BKTree:
        """Build a tree over a vocabulary, ignoring any term already added."""
        tree = cls()
        for term in terms:
            tree.add(term)
        return tree

    def add(self, term: str) -> None:
        """Insert one term, doing nothing if the tree already holds it.

        A repeat would be inserted at distance zero from a node, which is the
        one slot that cannot hold a distinct term, so repeats are dropped rather
        than allowed to overwrite.
        """
        if self._root is None:
            self._root = _Node(term)
            self._size = 1
            return
        node = self._root
        while True:
            distance = edit_distance(term, node.term)
            if distance == 0:
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _Node(term)
                self._size += 1
                return
            node = child

    def search(
        self, term: str, max_distance: int = MAX_EDIT_DISTANCE
    ) -> list[tuple[str, int]]:
        """Return every term within max_distance, nearest first.

        Ties at equal distance are broken alphabetically, so the order does not
        depend on how the tree happened to be built.
        """
        found: list[tuple[str, int]] = []
        pending = [] if self._root is None else [self._root]
        while pending:
            node = pending.pop()
            distance = edit_distance(term, node.term)
            if distance <= max_distance:
                found.append((node.term, distance))
            pending.extend(
                child
                for offset, child in node.children.items()
                if distance - max_distance <= offset <= distance + max_distance
            )
        found.sort(key=lambda match: (match[1], match[0]))
        return found

    def __len__(self) -> int:
        """Return how many distinct terms the tree holds."""
        return self._size


def suggest(
    index: ReadableIndex, term: str, limit: int = DEFAULT_SUGGESTIONS
) -> list[str]:
    """Return likely corrections for a term the index does not contain.

    Ranked by edit distance first and document frequency second, because at the
    same distance a term many documents use is a more likely intended spelling
    than one a single document uses.

    A term the index already holds needs no correction and returns nothing. This
    builds a tree over the whole vocabulary on every call, which is why a caller
    answering many queries should keep one and call `BKTree.search` itself.
    """
    if limit <= 0 or term in index:
        return []
    matches = BKTree.build(index.terms).search(term)
    ranked = sorted(
        matches,
        key=lambda match: (match[1], -index.document_frequency(match[0]), match[0]),
    )
    return [candidate for candidate, _ in ranked[:limit]]
