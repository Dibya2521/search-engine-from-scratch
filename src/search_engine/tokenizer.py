"""Split text into the tokens the index stores.

Lowercase, then keep maximal runs of ``[a-z0-9]``. ASCII-only by design, so
non-Latin text yields no tokens at all.

Documents and queries must both go through this function. If they diverge,
queries produce terms the index never stored.
"""

from __future__ import annotations

import re
from typing import Final

# Applied after lowercasing, hence no A-Z in the class. Public because the
# analyzer fingerprint is built from it: changing this changes every term.
TOKEN_PATTERN: Final = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Return tokens in order of appearance.

    Lowercasing must come before filtering: some code points lowercase *into*
    ASCII, such as U+212A becoming a plain ``k``.
    """
    return [match.group() for match in TOKEN_PATTERN.finditer(text.lower())]
