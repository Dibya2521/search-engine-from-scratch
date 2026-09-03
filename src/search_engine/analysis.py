"""The one text pipeline, shared by indexing and querying.

Both paths must call this. If they ever diverge, queries produce terms the
index never stored and the engine silently returns nothing, which is the single
easiest way to break a search engine.

Stemming is not idempotent, so this must be applied exactly once per text.
"""

from __future__ import annotations

from search_engine.stemmer import stem
from search_engine.tokenizer import tokenize


def analyze(text: str) -> list[str]:
    """Turn raw text into the ordered terms the index stores.

    One term out per token in, so a term's position is also its token
    position. Phrase queries depend on that correspondence.
    """
    return [stem(token) for token in tokenize(text)]
