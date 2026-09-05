"""Tests for the corpus reader."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from search_engine.corpus import CorpusFormatError, Document, read

if TYPE_CHECKING:
    from collections.abc import Sequence

FIXTURE = Path(__file__).parent / "fixtures" / "sample_corpus.xml"

# Printable characters excluding the markup delimiters. Control characters and
# lone surrogates are not legal in XML at all, so generating them would test
# the file format rather than this reader; the Unicode cases that matter are
# asserted explicitly above.
_SAFE_ALPHABET = st.characters(
    min_codepoint=33,
    max_codepoint=0x2FFF,
    blacklist_characters="<>&",
    blacklist_categories=("Cs", "Cc", "Cf", "Zs", "Zl", "Zp"),
)


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "corpus.xml"
    path.write_text(content, encoding="utf-8")
    return path


def test_reading_the_committed_fixture() -> None:
    documents = list(read(FIXTURE))
    assert [document.identifier for document in documents] == [1, 2, 3, 4, 5, 6, 7]
    assert documents[1].title == "Computer science"
    assert documents[1].text.startswith("Computer science is the study")


def test_named_entities_are_unescaped() -> None:
    """The one piece of XML semantics a scanner still has to implement."""
    documents = {document.identifier: document for document in read(FIXTURE)}
    assert documents[6].title == "Special characters & entities"
    assert "&" in documents[6].text
    assert "<" in documents[6].text


def test_an_empty_body_is_a_document_with_no_text() -> None:
    """It still counts, because the document total drives every term weight."""
    documents = {document.identifier: document for document in read(FIXTURE)}
    assert documents[7].text == ""
    assert documents[7].title == "Empty body"


def test_the_title_is_indexed_with_the_body() -> None:
    """Joined with a space, so title and body words cannot merge into one token."""
    document = Document(identifier=1, title="Alpha Beta", text="gamma delta")
    assert document.indexable_text == "Alpha Beta gamma delta"


def test_a_missing_title_or_text_is_tolerated(tmp_path: Path) -> None:
    path = _write(tmp_path, "<page><id>5</id></page>")
    assert list(read(path)) == [Document(identifier=5, title="", text="")]


def test_a_page_without_an_identifier_is_refused(tmp_path: Path) -> None:
    """Skipping it silently would leave the index quietly missing a document."""
    path = _write(tmp_path, "<page><title>No id</title></page>")
    with pytest.raises(CorpusFormatError, match="no <id>"):
        list(read(path))


def test_text_tag_attributes_are_tolerated(tmp_path: Path) -> None:
    """Real dumps write <text xml:space="preserve">."""
    path = _write(
        tmp_path,
        '<page><id>1</id><text xml:space="preserve">body here</text></page>',
    )
    assert next(iter(read(path))).text == "body here"


def test_the_first_id_in_a_page_wins(tmp_path: Path) -> None:
    """Real dumps nest further ids inside revision and contributor elements."""
    path = _write(
        tmp_path,
        "<page><id>42</id><title>T</title>"
        "<revision><id>999</id></revision><text>body</text></page>",
    )
    assert next(iter(read(path))).identifier == 42


def test_content_outside_page_records_is_ignored(tmp_path: Path) -> None:
    """A dump wrapped in a root element must still read; hence a scanner."""
    path = _write(
        tmp_path,
        "<mediawiki><siteinfo>junk</siteinfo>"
        "<page><id>1</id><text>a</text></page>"
        "<page><id>2</id><text>b</text></page></mediawiki>",
    )
    assert [document.identifier for document in read(path)] == [1, 2]


def test_a_file_with_no_records_yields_nothing(tmp_path: Path) -> None:
    assert list(read(_write(tmp_path, "nothing to see here" * 500))) == []


def test_a_truncated_final_record_is_dropped(tmp_path: Path) -> None:
    """Half a document is worse than none, since its text would be incomplete."""
    path = _write(
        tmp_path,
        "<page><id>1</id><text>complete</text></page><page><id>2</id><text>trunc",
    )
    assert [document.identifier for document in read(path)] == [1]


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 64, 1 << 20])
def test_the_result_is_the_same_at_every_chunk_size(chunk_size: int) -> None:
    """Proves the streaming is correct: a record split across reads still parses."""
    expected = list(read(FIXTURE))
    assert list(read(FIXTURE, chunk_size=chunk_size)) == expected


# Writes a file per generated example, so the timing measures the disk rather
# than the code. Hypothesis deadlines catch code that goes pathologically slow
# on some input, which is a real signal for a pure function and noise here.
@settings(deadline=None)
@given(
    st.lists(
        st.tuples(
            st.integers(1, 10_000),
            st.text(alphabet=_SAFE_ALPHABET, max_size=30),
            st.text(alphabet=_SAFE_ALPHABET, max_size=60),
        ),
        min_size=1,
        max_size=8,
    ),
    st.integers(1, 200),
)
def test_any_generated_corpus_reads_back(
    tmp_path_factory: pytest.TempPathFactory,
    records: Sequence[tuple[int, str, str]],
    chunk_size: int,
) -> None:
    """Round-trips generated records at an arbitrary chunk size."""
    body = "".join(
        f"<page><id>{identifier}</id><title>{title}</title><text>{text}</text></page>"
        for identifier, title, text in records
    )
    path = tmp_path_factory.mktemp("corpus") / "c.xml"
    path.write_text(body, encoding="utf-8")
    documents = list(read(path, chunk_size=chunk_size))
    assert [document.identifier for document in documents] == [
        identifier for identifier, _, _ in records
    ]
    for document, (_, title, _) in zip(documents, records, strict=True):
        assert document.title == title
