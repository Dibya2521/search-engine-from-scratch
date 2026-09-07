"""Tests for saving and loading an index."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from search_engine.index import InvertedIndex
from search_engine.persistence import (
    IndexFormatError,
    load,
    save,
    save_text,
)

if TYPE_CHECKING:
    from pathlib import Path


def _index_of(texts: dict[int, str]) -> InvertedIndex:
    index = InvertedIndex()
    for document_id, text in texts.items():
        index.add_document(document_id, text)
    return index


def test_the_written_text_format(tmp_path: Path) -> None:
    """Version 1 is readable, so a wrong index can be diagnosed with `head`."""
    index = _index_of({1: "web search", 2: "web index"})
    path = tmp_path / "i.index"
    save_text(index, path)
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


def test_an_unknown_version_is_detected(tmp_path: Path) -> None:
    """A file of the wrong shape must not load and give plausible garbage."""
    index = _index_of({1: "web"})
    path = tmp_path / "i.index"
    save_text(index, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = "# search-engine-index v9"
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    with pytest.raises(IndexFormatError, match="v1"):
        load(path)


def test_a_version_one_file_with_no_documents(tmp_path: Path) -> None:
    """An empty corpus is a real case, and its documents line has no body."""
    path = tmp_path / "i.index"
    save_text(InvertedIndex(), path)
    # The prefix is written with its trailing space even when nothing follows.
    assert path.read_text(encoding="utf-8").splitlines() == [
        "# search-engine-index v1",
        "# documents: ",
    ]
    assert load(path).document_count == 0
