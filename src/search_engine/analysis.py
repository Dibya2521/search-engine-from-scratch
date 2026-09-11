"""The one text pipeline, shared by indexing and querying.

Both paths must call this. If they ever diverge, queries produce terms the
index never stored and the engine silently returns nothing, which is the single
easiest way to break a search engine.

Order is normalize, tokenize, drop stopwords, stem. Stopwords are matched on
raw tokens because the list holds readable surface words.

The normalization form is a parameter for the same reason the stopword list is:
it changes every term produced, so it is part of the configuration the
fingerprint identifies rather than a fixed property of the code. Index with a
form and you must query with it.

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
from search_engine.tokenizer import DEFAULT_FORM, TOKEN_PATTERN, tokenize

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from search_engine.tokenizer import NormalForm

# Bump when a change alters the terms produced but is not visible in the
# pattern, the stopwords or the stemmer version, such as reordering the
# pipeline stages. Version 2 added Unicode normalization, which changes the
# terms of every document containing a composed or decomposed letter.
_PIPELINE_VERSION: Final = "2"
_FINGERPRINT_LENGTH: Final = 16


def analyze_positioned(
    text: str,
    stopwords: AbstractSet[str] = DEFAULT_STOPWORDS,
    form: NormalForm = DEFAULT_FORM,
) -> list[tuple[int, str]]:
    """Return surviving terms paired with their original token positions."""
    return [
        (position, stem(token))
        for position, token in enumerate(tokenize(text, form))
        if token not in stopwords
    ]


def analyze(
    text: str,
    stopwords: AbstractSet[str] = DEFAULT_STOPWORDS,
    form: NormalForm = DEFAULT_FORM,
) -> list[str]:
    """Return the ordered terms, for callers that do not need positions."""
    return [term for _, term in analyze_positioned(text, stopwords, form)]


def fingerprint(
    stopwords: AbstractSet[str] = DEFAULT_STOPWORDS,
    form: NormalForm = DEFAULT_FORM,
) -> str:
    """Return a short digest identifying this exact analysis configuration.

    Built from everything that determines the terms produced: the pipeline
    version, the token pattern, the stemmer variant, the normalization form and
    the stopword list. Two configurations agree on the digest only if they agree
    on all five.

    Anything that indexed with a non-default form must pass the same form here,
    exactly as it must pass the same stopword list. Neither is recovered from
    the index.

    Truncated to 16 hex characters. The digest is not a security boundary, only
    a change detector, and a full digest in every index header is noise.
    """
    material = "\0".join(
        [
            _PIPELINE_VERSION,
            TOKEN_PATTERN.pattern,
            ALGORITHM_VERSION,
            form,
            *sorted(stopwords),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:_FINGERPRINT_LENGTH]
