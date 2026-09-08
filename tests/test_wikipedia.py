"""Tests for converting a Wikipedia dump into the corpus format."""

from __future__ import annotations

import bz2
import gzip
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from search_engine.corpus import read
from search_engine.wikipedia import Article, clean, read_dump, write_corpus

if TYPE_CHECKING:
    from pathlib import Path

PAGE = """<page>
  <title>Inverted index</title>
  <ns>0</ns>
  <id>42</id>
  <revision><id>99</id>
  <text xml:space="preserve">An inverted index maps terms.</text>
  </revision>
</page>"""

REDIRECT = """<page>
  <title>Inverted indices</title>
  <ns>0</ns>
  <id>43</id>
  <redirect title="Inverted index" />
  <revision><text>#REDIRECT [[Inverted index]]</text></revision>
</page>"""

TALK = """<page>
  <title>Talk:Inverted index</title>
  <ns>1</ns>
  <id>44</id>
  <revision><text>Discussion about the article.</text></revision>
</page>"""


def write(tmp_path: Path, text: str, name: str = "dump.xml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_an_article_is_kept(tmp_path: Path) -> None:
    articles = list(read_dump(write(tmp_path, PAGE)))
    assert articles == [
        Article(
            identifier=42, title="Inverted index", text="An inverted index maps terms."
        )
    ]


def test_the_page_id_is_taken_not_the_revision_id(tmp_path: Path) -> None:
    """The revision id comes second, so scanning order is what distinguishes them."""
    assert next(iter(read_dump(write(tmp_path, PAGE)))).identifier == 42


def test_a_redirect_is_skipped(tmp_path: Path) -> None:
    """Its body is one directive, so it contributes nothing but noise."""
    assert list(read_dump(write(tmp_path, REDIRECT))) == []


def test_a_page_outside_the_article_namespace_is_skipped(tmp_path: Path) -> None:
    assert list(read_dump(write(tmp_path, TALK))) == []


def test_a_page_with_no_namespace_is_kept(tmp_path: Path) -> None:
    """Some dumps omit it, and refusing them would discard the whole file."""
    page = "<page><title>T</title><id>7</id><text>Body text here.</text></page>"
    assert [one.identifier for one in read_dump(write(tmp_path, page))] == [7]


def test_a_page_with_no_identifier_is_skipped(tmp_path: Path) -> None:
    page = "<page><title>T</title><text>Body text.</text></page>"
    assert list(read_dump(write(tmp_path, page))) == []


def test_a_page_with_no_text_is_skipped(tmp_path: Path) -> None:
    page = "<page><title>T</title><id>7</id></page>"
    assert list(read_dump(write(tmp_path, page))) == []


def test_a_page_whose_text_is_only_markup_is_skipped(tmp_path: Path) -> None:
    """Cleaning can empty a page, and an empty document is not worth indexing."""
    page = "<page><title>T</title><id>7</id><text>{{stub}}</text></page>"
    assert list(read_dump(write(tmp_path, page))) == []


def test_a_page_with_no_title_still_works(tmp_path: Path) -> None:
    page = "<page><id>7</id><text>Body text here.</text></page>"
    assert next(iter(read_dump(write(tmp_path, page)))).title == ""


def test_several_pages_are_all_found(tmp_path: Path) -> None:
    dump = f"<mediawiki>{PAGE}{REDIRECT}{TALK}{PAGE.replace('42', '77')}</mediawiki>"
    assert [one.identifier for one in read_dump(write(tmp_path, dump))] == [42, 77]


def test_a_file_with_no_pages_terminates(tmp_path: Path) -> None:
    """The buffer must not grow without limit on input holding no records."""
    assert list(read_dump(write(tmp_path, "x" * 100_000), chunk_size=1024)) == []


def test_a_page_split_across_chunks_is_found(tmp_path: Path) -> None:
    """The falsifying case: a record straddling a read boundary."""
    dump = f"{'padding ' * 200}{PAGE}"
    assert [
        one.identifier for one in read_dump(write(tmp_path, dump), chunk_size=64)
    ] == [42]


@pytest.mark.parametrize("suffix", [".xml", ".gz", ".bz2"])
def test_a_dump_is_read_compressed_or_not(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"dump{suffix}"
    raw = PAGE.encode("utf-8")
    if suffix == ".gz":
        path.write_bytes(gzip.compress(raw))
    elif suffix == ".bz2":
        path.write_bytes(bz2.compress(raw))
    else:
        path.write_bytes(raw)
    assert [one.identifier for one in read_dump(path)] == [42]


class TestCleaning:
    """Wikitext markup that would otherwise become tokens."""

    @pytest.mark.parametrize(
        ("markup", "expected"),
        [
            ("plain text", "plain text"),
            ("<!-- hidden -->visible", "visible"),
            ("{{infobox|a=b}}kept", "kept"),
            ("{{outer|{{inner}}}}kept", "kept"),
            ("{|\n| a table |\n|}kept", "kept"),
            ("before<ref>a citation</ref>after", "before after"),
            ("before<ref name='x'/>after", "before after"),
            ("[[Inverted index]]", "Inverted index"),
            ("[[Inverted index|the index]]", "the index"),
            ("[https://example.com the label]", "the label"),
            ("== A heading ==", "A heading"),
            ("''italic'' and '''bold'''", "italic and bold"),
            ("<div>tagged</div>", "tagged"),
            ("&amp; unescaped", "& unescaped"),
            ("collapsed    whitespace", "collapsed whitespace"),
        ],
    )
    def test_markup_is_stripped(self, markup: str, expected: str) -> None:
        """Each case is markup that would otherwise become indexed tokens."""
        assert clean(markup) == expected

    def test_nesting_deeper_than_the_pass_limit_terminates(self) -> None:
        """It must stop rather than loop, even if some markup survives."""
        assert clean("{{" * 40 + "x" + "}}" * 40) is not None


def test_writing_produces_a_corpus_the_reader_understands(tmp_path: Path) -> None:
    """The contract that matters: the output is valid input to the engine."""
    articles = iter(
        [
            Article(identifier=1, title="First", text="Body of the first article."),
            Article(identifier=2, title="Second", text="Body of the second."),
        ]
    )
    path = tmp_path / "corpus.xml"
    assert write_corpus(articles, path) == 2
    documents = list(read(path))
    assert [one.identifier for one in documents] == [1, 2]
    assert documents[0].title == "First"
    assert documents[0].text == "Body of the first article."


def test_characters_that_would_close_a_tag_are_escaped(tmp_path: Path) -> None:
    """Otherwise a title containing a bracket truncates the record."""
    path = tmp_path / "corpus.xml"
    write_corpus(iter([Article(1, "a < b & c > d", "x </text> y")]), path)
    documents = list(read(path))
    assert documents[0].title == "a < b & c > d"
    assert documents[0].text == "x </text> y"


def test_writing_nothing_produces_an_empty_corpus(tmp_path: Path) -> None:
    path = tmp_path / "corpus.xml"
    assert write_corpus(iter([]), path) == 0
    assert list(read(path)) == []


# Writes a file per generated example, so the timing measures the disk rather
# than the code. Hypothesis deadlines catch code that goes pathologically slow
# on some input, which is a real signal for a pure function and noise here.
@settings(deadline=None)
@given(
    st.lists(
        st.tuples(
            st.integers(1, 10_000),
            st.text(max_size=40),
            st.text(min_size=1, max_size=80),
        ),
        max_size=8,
        unique_by=lambda row: row[0],
    )
)
def test_anything_written_can_be_read_back(
    tmp_path_factory: pytest.TempPathFactory,
    rows: list[tuple[int, str, str]],
) -> None:
    """The falsifying direction: generated titles and bodies, not chosen ones."""
    path = tmp_path_factory.mktemp("corpus") / "corpus.xml"
    write_corpus(iter([Article(one, title, text) for one, title, text in rows]), path)
    assert [one.identifier for one in read(path)] == [one for one, _, _ in rows]
