"""Typed adapter around the independent stemmer used as a differential oracle.

The reference library ships no type information, so strict checking is switched
off for this one small module and a precisely typed function is exposed
instead. Nothing else in the test suite imports it directly.
"""
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from nltk.stem.porter import PorterStemmer

_REFERENCE = PorterStemmer(mode=PorterStemmer.MARTIN_EXTENSIONS)


def reference_stem(word: str) -> str:
    """Stem with the reference implementation, without re-lowercasing."""
    stemmed: str = _REFERENCE.stem(word, to_lowercase=False)
    return stemmed
