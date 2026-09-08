"""Converting a Wikipedia dump into the corpus format this engine reads.

A dump is a single XML file of several hundred megabytes compressed and several
gigabytes expanded, so every part of this streams. Nothing here holds more than
one page in memory.

The dump's page records look like this, with much else omitted:

```text
<page>
  <title>Inverted index</title>
  <ns>0</ns>
  <id>12345</id>
  <redirect title="Search engine" />
  <revision><id>67890</id><text>An inverted index is...</text></revision>
</page>
```

Three of those fields decide whether a page is worth keeping.

`ns` is the namespace: 0 is an article, and everything else is a talk page, a
user page, a template or a category. Indexing those makes the corpus larger and
worse.

`redirect` marks a page that is only a pointer to another one. Its body is a
single directive, so it contributes nothing but noise.

The first `id` inside a page is the page's own. A later one belongs to the
revision, which is why order matters when scanning rather than parsing.

Scanned with regular expressions rather than an XML parser, for the same two
reasons the corpus reader is: a parser is vulnerable to entity expansion, and
dumps are not always well formed at the boundaries of a truncated download. The
patterns here have no nested quantifiers and no alternation over overlapping
branches, so none of them can backtrack pathologically.

Compression is handled by extension, so a dump can be read without expanding it
first.
"""

from __future__ import annotations

import bz2
import gzip
import html
import re
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_PAGE: Final = re.compile(r"<page[^>]*>(.*?)</page>", re.DOTALL)
_TITLE: Final = re.compile(r"<title>(.*?)</title>", re.DOTALL)
_NAMESPACE: Final = re.compile(r"<ns>\s*(-?\d+)\s*</ns>")
_IDENTIFIER: Final = re.compile(r"<id>\s*(\d+)\s*</id>")
_TEXT: Final = re.compile(r"<text[^>]*>(.*?)</text>", re.DOTALL)
_REDIRECT: Final = re.compile(r"<redirect[^>]*/?>")

# Wikitext markup that carries no searchable content. Applied in this order.
_COMMENT: Final = re.compile(r"<!--.*?-->", re.DOTALL)
_TABLE: Final = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_TEMPLATE: Final = re.compile(r"\{\{[^{}]*\}\}")
_REFERENCE: Final = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL)
_TAG: Final = re.compile(r"<[^>]{0,200}>")
_LINK_TARGET: Final = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]|]*)\]\]")
_EXTERNAL_LINK: Final = re.compile(r"\[https?://\S+\s([^\]]*)\]")
_HEADING: Final = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.MULTILINE)
_APOSTROPHES: Final = re.compile(r"'{2,}")
_WHITESPACE: Final = re.compile(r"\s+")

ARTICLE_NAMESPACE: Final = 0
CHUNK_SIZE: Final = 1 << 22
# Templates nest, and each pass removes one level. Five clears almost everything
# in practice; the guard is that this terminates rather than looping on input
# designed to nest without limit.
_TEMPLATE_PASSES: Final = 5


@dataclass(frozen=True, slots=True)
class Article:
    """One article kept from a dump."""

    identifier: int
    title: str
    text: str


def read_dump(path: Path, chunk_size: int = CHUNK_SIZE) -> Iterator[Article]:
    """Yield the articles in a dump, skipping redirects and non-articles.

    Holds one page at a time regardless of how large the dump is.
    """
    buffer = ""
    with _open(path) as handle:
        while chunk := handle.read(chunk_size):
            buffer += chunk
            while (match := _PAGE.search(buffer)) is not None:
                article = _parse(match.group(1))
                if article is not None:
                    yield article
                buffer = buffer[match.end() :]
            # Nothing left can begin a page, so keeping it would grow the
            # buffer without limit on a file holding no records at all.
            if "<page" not in buffer:
                buffer = buffer[-len("<page") :]


def write_corpus(articles: Iterator[Article], path: Path) -> int:
    """Write articles in the corpus format, one record per line.

    Returns the number written. Escapes the three characters that would
    otherwise close a tag early; nothing else needs escaping, because the
    reader only looks for these element names.
    """
    written = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for article in articles:
            title = _escape(article.title)
            text = _escape(article.text)
            handle.write(
                f"<page><id>{article.identifier}</id>"
                f"<title>{title}</title>"
                f"<text>{text}</text></page>\n"
            )
            written += 1
    return written


def clean(wikitext: str) -> str:
    """Strip wikitext markup, leaving the prose a reader would see.

    Not a wikitext parser, and not trying to be. The aim is to remove the
    markup that would otherwise become tokens: templates, tables, references
    and link syntax. What survives is close enough to article prose for
    retrieval, and the parts it gets wrong are a small fraction of any article.
    """
    text = _COMMENT.sub(" ", wikitext)
    text = _REFERENCE.sub(" ", text)
    text = _TABLE.sub(" ", text)
    # Innermost templates only match once their contents hold no braces, so
    # repeating the substitution unwraps one level of nesting per pass.
    for _ in range(_TEMPLATE_PASSES):
        text, count = _TEMPLATE.subn(" ", text)
        if count == 0:
            break
    text = _LINK_TARGET.sub(r"\1", text)
    text = _EXTERNAL_LINK.sub(r"\1", text)
    text = _HEADING.sub(r"\1", text)
    text = _TAG.sub(" ", text)
    text = _APOSTROPHES.sub("", text)
    return _WHITESPACE.sub(" ", html.unescape(text)).strip()


def _open(path: Path) -> IO[str]:
    """Open a dump, decompressing by extension."""
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _parse(body: str) -> Article | None:
    """Return the article a page record describes, or None to skip it."""
    if _REDIRECT.search(body) is not None:
        return None
    namespace = _NAMESPACE.search(body)
    if namespace is not None and int(namespace.group(1)) != ARTICLE_NAMESPACE:
        return None
    identifier = _IDENTIFIER.search(body)
    text = _TEXT.search(body)
    if identifier is None or text is None:
        return None
    body_text = clean(html.unescape(text.group(1)))
    if not body_text:
        return None
    title = _TITLE.search(body)
    return Article(
        identifier=int(identifier.group(1)),
        title=html.unescape(title.group(1)).strip() if title else "",
        text=body_text,
    )


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
