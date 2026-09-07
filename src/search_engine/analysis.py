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

`fingerprint` identifies the exact configuration that produced a set of terms.
An index records it, and loading refuses an index whose fingerprint differs from
the current one. Without that check, changing the tokenizer, the stopword list
or the stemmer leaves old documents analyzed one way and new queries analyzed
another, and the engine returns wrong results with no error anywhere. It is the
most common way a search system breaks quietly.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Final

from search_engine.stemmer import ALGORITHM_VERSION, stem
from search_engine.stopwords import DEFAULT_STOPWORDS
from search_engine.tokenizer import TOKEN_PATTERN, tokenize

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

# Bump when a change alters the terms produced but is not visible in the
# pattern, the stopwords or the stemmer version, such as reordering the
# pipeline stages.
_PIPELINE_VERSION: Final = "1"
_FINGERPRINT_LENGTH: Final = 16


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


def fingerprint(stopwords: AbstractSet[str] = DEFAULT_STOPWORDS) -> str:
    """Return a short digest identifying this exact analysis configuration.

    Built from everything that determines the terms produced: the pipeline
    version, the token pattern, the stemmer variant and the stopword list. Two
    configurations agree on the digest only if they agree on all four.

    Truncated to 16 hex characters. The digest is not a security boundary, only
    a change detector, and a full digest in every index header is noise.
    """
    material = "\0".join(
        [
            _PIPELINE_VERSION,
            TOKEN_PATTERN.pattern,
            ALGORITHM_VERSION,
            *sorted(stopwords),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:_FINGERPRINT_LENGTH]
