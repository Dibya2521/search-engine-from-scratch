"""Split text into the tokens the index stores.

Normalize, lowercase, then keep maximal runs of ``[a-z0-9]``. ASCII-only by
design, so non-Latin text yields no tokens at all.

Normalizing comes first because one word can be written as more than one
sequence of code points. ``cafe`` with a composed acute accent is a single
non-ASCII code point and the whole letter is dropped; the same word with a
combining accent is an ASCII ``e`` followed by a mark, and only the mark is
dropped. The two render identically and used to produce different tokens, so a
document stored in one form could never match a query typed in the other.

Documents and queries must both go through this function. If they diverge,
queries produce terms the index never stored.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final, Literal

type NormalForm = Literal["NFC", "NFKC"]

# Applied after lowercasing, hence no A-Z in the class. Public because the
# analyzer fingerprint is built from it: changing this changes every term.
TOKEN_PATTERN: Final = re.compile(r"[a-z0-9]+")

DEFAULT_FORM: Final[NormalForm] = "NFC"


def tokenize(text: str, form: NormalForm = DEFAULT_FORM) -> list[str]:
    """Return tokens in order of appearance.

    Normalizing must come before lowercasing, and lowercasing before filtering:
    some code points lowercase *into* ASCII, such as U+212A becoming a plain
    ``k``.

    ``NFC`` merges spellings that render identically and changes nothing else.
    ``NFKC`` also folds compatibility forms, which recovers tokens that would
    otherwise be dropped whole, and in doing so destroys distinctions: a
    superscript two and an ordinary two become the same token. Anything indexed
    with a form must be queried with it, and `analysis.fingerprint` takes the
    form for exactly that reason.
    """
    normalized = unicodedata.normalize(form, text)
    return [match.group() for match in TOKEN_PATTERN.finditer(normalized.lower())]
