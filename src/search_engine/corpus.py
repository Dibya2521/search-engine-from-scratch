"""Streaming documents out of a corpus file.

The expected shape is a sequence of page records:

```text
<page><id>42</id><title>Some title</title><text>Body...</text></page>
```

Read a page at a time rather than the whole file. A corpus is hundreds of
megabytes and the index built from it already costs several times the source in
memory, so the reader must not add to that.

Deliberately a tolerant scanner rather than an XML parser, for two reasons.
Real dumps are not always wrapped in a single root element, which a strict
parser rejects outright. And an XML parser is vulnerable to entity expansion
attacks, the billion-laughs class, whereas a scanner that never expands an
entity cannot be. Named entities are unescaped explicitly, which is the only
part of XML semantics this needs.

The cost of that choice is real: no namespaces, no attribute parsing, no
validation. Anything beyond these three fields needs a proper parser.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_PAGE: Final = re.compile(r"<page>(.*?)</page>", re.DOTALL)
# Matched inside one page, so the first id found is the page's own rather than
# one belonging to a nested revision or contributor.
_ID: Final = re.compile(r"<id>\s*(\d+)\s*</id>")
_TITLE: Final = re.compile(r"<title>(.*?)</title>", re.DOTALL)
# Real dumps write <text xml:space="preserve">, so attributes must be tolerated.
_TEXT: Final = re.compile(r"<text[^>]*>(.*?)</text>", re.DOTALL)

CHUNK_SIZE: Final = 1 << 20


class CorpusFormatError(ValueError):
    """Raised when a page record cannot be understood.

    Skipping a malformed record silently would mean an index quietly missing
    documents, and nothing downstream could detect that.
    """


@dataclass(frozen=True, slots=True)
class Document:
    """One corpus record: its identifier, its title, and its body."""

    identifier: int
    title: str
    text: str

    @property
    def indexable_text(self) -> str:
        """Title joined to body, which is what actually gets indexed.

        The title is indexed because it is the most concentrated description a
        document has. Joining with a space keeps the last title word and the
        first body word from merging into one token.
        """
        return f"{self.title} {self.text}"


def read(path: Path, chunk_size: int = CHUNK_SIZE) -> Iterator[Document]:
    """Yield documents from a corpus file, holding one page at a time.

    Raises:
        CorpusFormatError: If a page record has no identifier.
    """
    buffer = ""
    with path.open(encoding="utf-8", errors="replace") as handle:
        while chunk := handle.read(chunk_size):
            buffer += chunk
            while (match := _PAGE.search(buffer)) is not None:
                yield _parse(match.group(1))
                buffer = buffer[match.end() :]
            # Nothing here can start a page, so holding it would grow the
            # buffer without limit on a file that contains no records at all.
            if "<page>" not in buffer:
                buffer = buffer[-len("<page>") :]


def _parse(body: str) -> Document:
    identifier = _ID.search(body)
    if identifier is None:
        message = f"page record has no <id>: {body[:80]!r}"
        raise CorpusFormatError(message)
    title = _TITLE.search(body)
    text = _TEXT.search(body)
    return Document(
        identifier=int(identifier.group(1)),
        title=html.unescape(title.group(1)).strip() if title else "",
        text=html.unescape(text.group(1)) if text else "",
    )
