"""The one text pipeline, shared by indexing and querying.

Both paths must call this. If they ever diverge, queries produce terms the
index never stored and the engine silently returns nothing, which is the single
easiest way to break a search engine.

Order is tokenize, drop stopwords, stem. Stopwords are matched on raw tokens
because the list holds readable surface words.

Positions are the position in the *unfiltered* token stream, so a dropped
stopword leaves a gap. That gap is what keeps `"king of england"` and
`"king england"` different queries.

Stemming is not idempotent, so this must be applied exactly once per text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from search_engine.stemmer import stem
from search_engine.stopwords import DEFAULT_STOPWORDS
from search_engine.tokenizer import tokenize

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet


def analyze_positioned(
    text: str, stopwords: AbstractSet[str] = DEFAULT_STOPWORDS
) -> list[tuple[int, str]]:
    """Return surviving terms paired with their original token positions."""
    return [
        (position, stem(token))
        for position, token in enumerate(tokenize(text))
        if token not in stopwords
    ]


def analyze(text: str, stopwords: AbstractSet[str] = DEFAULT_STOPWORDS) -> list[str]:
    """Return the ordered terms, for callers that do not need positions."""
    return [term for _, term in analyze_positioned(text, stopwords)]
