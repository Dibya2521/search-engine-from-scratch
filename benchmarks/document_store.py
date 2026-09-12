"""Measure what keeping the source text beside a segment costs on disk.

Run with ``uv run python benchmarks/document_store.py``.

An index holds no text, so a result can be scored but not shown. Storing the
text beside each segment is what makes a snippet possible, and the whole price
of it is disk: the alternative, seeking back into the corpus file, stores no
bytes at all and depends on a file the index does not own.

That price is the entire argument against the design, so it is measured here
rather than asserted anywhere.

The log is written without syncing. This measures the size of what is written
and not the cost of making it durable, and one fsync per document would
otherwise dominate the run.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from corpus import make_vocabulary, varied_length_corpus

from search_engine.store import DATA_SUFFIX, OFFSET_BYTES, OFFSET_SUFFIX
from search_engine.writer import SEGMENT_SUFFIX, IndexWriter

VOCABULARY_SIZE = 4_000
COUNTS = (1_000, 4_000, 16_000)
TITLE_WORDS = 6

MEGABYTE = 1024 * 1024


def title_of(text: str) -> str:
    """Take the opening words as the title, which is the length a real one has."""
    return " ".join(text.split()[:TITLE_WORDS])


def build(directory: Path, texts: list[str]) -> int:
    """Write every document into one segment, and return the source bytes."""
    source = 0
    with IndexWriter(
        directory, buffer_documents=len(texts) + 1, merge=False, sync=False
    ) as writer:
        for document_id, text in enumerate(texts):
            title = title_of(text)
            writer.add(document_id, text, title=title)
            source += len(title.encode("utf-8")) + len(text.encode("utf-8"))
    return source


def bytes_with(directory: Path, suffix: str) -> int:
    """Return the total size of every file in a directory with this suffix."""
    return sum(path.stat().st_size for path in directory.glob(f"*{suffix}"))


def main() -> None:
    """Run the measurement and print the report."""
    vocabulary = make_vocabulary(VOCABULARY_SIZE)
    print(f"vocabulary : {VOCABULARY_SIZE:,} terms")
    print(f"title      : the first {TITLE_WORDS} words of each document")
    print(f"offsets    : {OFFSET_BYTES} bytes per document, fixed")
    print()
    print(
        "| Documents | Source MB | Segment MB | Store MB | Offsets KB "
        "| Store against segment | Store against source |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for count in COUNTS:
        texts = list(varied_length_corpus(count, vocabulary))
        with TemporaryDirectory() as name:
            directory = Path(name)
            source = build(directory, texts)
            segment = bytes_with(directory, SEGMENT_SUFFIX)
            data = bytes_with(directory, DATA_SUFFIX)
            offsets = bytes_with(directory, OFFSET_SUFFIX)
        store = data + offsets
        print(
            f"| {count:,} | {source / MEGABYTE:.2f} | {segment / MEGABYTE:.2f} "
            f"| {store / MEGABYTE:.2f} | {offsets / 1024:.1f} "
            f"| {store / segment:.2f}x | {store / source:.2f}x |"
        )


if __name__ == "__main__":
    main()
