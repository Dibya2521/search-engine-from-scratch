"""Tests for saving and loading an index."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from search_engine.index import InvertedIndex
from search_engine.persistence import IndexFormatError, load, save
from search_engine.query import search
from search_engine.tokenizer import tokenize

if TYPE_CHECKING:
    from pathlib import Path


def _index_of(texts: dict[int, str]) -> InvertedIndex:
    index = InvertedIndex()
    for document_id, text in texts.items():
        index.add_document(document_id, text)
    return index


def test_the_written_format(tmp_path: Path) -> None:
    """Readable by design, so a wrong index can be diagnosed with `head`."""
    index = _index_of({1: "web search", 2: "web index"})
    path = tmp_path / "i.index"
    save(index, path)
    assert path.read_text(encoding="utf-8").splitlines() == [
        "# search-engine-index v1",
        "# documents: 1,2",
        "index|2:1",
        "search|1:1",
        "web|1:0;2:0",
    ]


def test_a_round_trip_preserves_everything(tmp_path: Path) -> None:
    original = _index_of(
        {
            1: "the computer science department teaches computer science",
            2: "an index maps terms to documents and positions",
            7: "...!!!",
        }
    )
    path = tmp_path / "i.index"
    save(original, path)
    restored = load(path)
    assert restored.document_count == original.document_count
    assert set(restored.document_ids) == set(original.document_ids)
    assert set(restored.terms) == set(original.terms)
    for term in original.terms:
        assert restored.postings(term) == original.postings(term)


def test_a_document_with_no_terms_survives_the_round_trip(tmp_path: Path) -> None:
    """It appears in no postings list, so only the header can carry it.

    Losing it would change the document count, which is the N that decides how
    rare every term is, so every score afterwards would be wrong.
    """
    index = _index_of({1: "web", 2: "...!!!"})
    path = tmp_path / "i.index"
    save(index, path)
    restored = load(path)
    assert restored.document_count == 2
    assert set(restored.document_ids) == {1, 2}


def test_an_empty_index_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "i.index"
    save(InvertedIndex(), path)
    restored = load(path)
    assert restored.document_count == 0
    assert restored.vocabulary_size == 0


def test_saving_twice_produces_identical_bytes(tmp_path: Path) -> None:
    """Sorted output, so the file is diffable and the tests can be exact."""
    index = _index_of({3: "gamma beta", 1: "alpha beta", 2: "beta"})
    first = tmp_path / "a.index"
    second = tmp_path / "b.index"
    save(index, first)
    save(index, second)
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", "expected '# search-engine-index v1'"),
        ("wrong header\n# documents: 1\n", "expected '# search-engine-index v1'"),
        ("# search-engine-index v99\n", "expected '# search-engine-index v1'"),
        ("# search-engine-index v1\n", "documents:"),
        ("# search-engine-index v1\n# documents: a,b\n", "not all integers"),
        ("# search-engine-index v1\n# documents: 1\nnopipe\n", "expected 'term|"),
        ("# search-engine-index v1\n# documents: 1\n|1:0\n", "expected 'term|"),
        ("# search-engine-index v1\n# documents: 1\nterm|nocolon\n", "docID:positions"),
        ("# search-engine-index v1\n# documents: 1\nterm|x:0\n", "non-integer"),
        ("# search-engine-index v1\n# documents: 1\nterm|1:z\n", "non-integer"),
    ],
)
def test_malformed_files_are_refused(
    tmp_path: Path, content: str, expected: str
) -> None:
    """A wrong index would give silently wrong results, so it is rejected."""
    path = tmp_path / "bad.index"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(IndexFormatError, match=expected.replace("|", r"\|")):
        load(path)


def test_a_version_bump_is_detected(tmp_path: Path) -> None:
    """Weights will change the format later; an old file must not load silently."""
    index = _index_of({1: "web"})
    path = tmp_path / "i.index"
    save(index, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = "# search-engine-index v2"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(IndexFormatError, match="v1"):
        load(path)


# Writes a file per generated example, so the timing measures the disk rather
# than the code. Hypothesis deadlines catch code that goes pathologically slow
# on some input, which is a real signal for a pure function and noise here.
@settings(deadline=None)
@given(st.lists(st.text(), min_size=1, max_size=6))
def test_any_index_round_trips_unchanged(
    tmp_path_factory: pytest.TempPathFactory, texts: list[str]
) -> None:
    """The falsifying direction: generated corpora rather than chosen ones."""
    index = _index_of(dict(enumerate(texts)))
    path = tmp_path_factory.mktemp("rt") / "i.index"
    save(index, path)
    restored = load(path)
    assert set(restored.document_ids) == set(index.document_ids)
    assert set(restored.terms) == set(index.terms)
    for term in index.terms:
        assert restored.postings(term) == index.postings(term)


# Writes a file per generated example, so the timing measures the disk rather
# than the code. Hypothesis deadlines catch code that goes pathologically slow
# on some input, which is a real signal for a pure function and noise here.
@settings(deadline=None)
@given(st.lists(st.text(), min_size=1, max_size=6))
def test_queries_behave_the_same_before_and_after_a_round_trip(
    tmp_path_factory: pytest.TempPathFactory, texts: list[str]
) -> None:
    """What actually matters: the reloaded index answers identically."""
    index = _index_of(dict(enumerate(texts)))
    path = tmp_path_factory.mktemp("rt") / "i.index"
    save(index, path)
    restored = load(path)
    for token in {token for text in texts for token in tokenize(text)}:
        assert search(restored, token) == search(index, token)
