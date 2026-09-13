"""Tests for reading relevance judgement files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from search_engine.corpus import read
from search_engine.judgements import Judgement, JudgementFormatError, load

JUDGEMENTS = Path(__file__).parent / "fixtures" / "judgements.jsonl"
EVALUATION_CORPUS = Path(__file__).parent / "fixtures" / "evaluation_corpus.xml"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "judgements.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_one_judgement_per_line(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        '{"query": "one", "judgements": [[1, 3], [2, 1]]}\n'
        '{"query": "two", "judgements": [[7, 2]]}\n',
    )
    assert load(path) == [
        Judgement(query="one", grades={1: 3, 2: 1}),
        Judgement(query="two", grades={7: 2}),
    ]


def test_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "# why these labels exist\n"
        "\n"
        '{"query": "one", "judgements": [[1, 3]]}\n'
        "   \n"
        "  # an indented comment is still a comment\n",
    )
    assert load(path) == [Judgement(query="one", grades={1: 3})]


def test_an_empty_file_yields_nothing(tmp_path: Path) -> None:
    assert load(write(tmp_path, "")) == []


def test_a_query_may_have_no_judgements(tmp_path: Path) -> None:
    """A query nothing is relevant to is a real case, not a malformed line."""
    path = write(tmp_path, '{"query": "nothing matches", "judgements": []}\n')
    assert load(path) == [Judgement(query="nothing matches", grades={})]


def test_relevant_drops_grade_zero() -> None:
    judgement = Judgement(query="q", grades={1: 3, 2: 0, 3: 1})
    assert judgement.relevant == frozenset({1, 3})


@pytest.mark.parametrize(
    ("line", "fragment"),
    [
        ("{not json at all}", "not valid JSON"),
        ("[1, 2, 3]", "not a JSON object"),
        ('"a string"', "not a JSON object"),
        ('{"judgements": [[1, 3]]}', "'query' must be"),
        ('{"query": 7, "judgements": []}', "'query' must be"),
        ('{"query": "  ", "judgements": []}', "'query' must be"),
        ('{"query": "q"}', "'judgements' must be"),
        ('{"query": "q", "judgements": {}}', "'judgements' must be"),
        ('{"query": "q", "judgements": [1]}', "expected a [document, grade] pair"),
        ('{"query": "q", "judgements": [[1]]}', "expected a [document, grade] pair"),
        ('{"query": "q", "judgements": [[1, 2, 3]]}', "expected a [document, grade]"),
        ('{"query": "q", "judgements": [["a", 1]]}', "expected a [document, grade]"),
        ('{"query": "q", "judgements": [[1, "a"]]}', "expected a [document, grade]"),
        ('{"query": "q", "judgements": [[1, 1.5]]}', "expected a [document, grade]"),
        ('{"query": "q", "judgements": [[1, -1]]}', "non-negative"),
        # true decodes to a bool, which subclasses int, so a plain isinstance
        # check would accept this and silently store document 1.
        ('{"query": "q", "judgements": [[true, 2]]}', "expected a [document, grade]"),
        ('{"query": "q", "judgements": [[1, true]]}', "expected a [document, grade]"),
        ('{"query": "q", "judgements": [[1, 3], [1, 2]]}', "judged twice"),
    ],
)
def test_malformed_lines_raise(tmp_path: Path, line: str, fragment: str) -> None:
    with pytest.raises(JudgementFormatError, match=fragment.replace("[", r"\[")):
        load(write(tmp_path, line + "\n"))


def test_the_error_names_the_line_number(tmp_path: Path) -> None:
    """Correcting a hand-written file of hundreds of lines needs the number."""
    path = write(
        tmp_path,
        '# a comment\n{"query": "fine", "judgements": [[1, 1]]}\n\n{broken}\n',
    )
    with pytest.raises(JudgementFormatError, match="line 4"):
        load(path)


# Writes a file per generated example, so the timing measures the disk rather
# than the code. Hypothesis deadlines catch code that goes pathologically slow
# on some input, which is a real signal for a pure function and noise here.
@settings(deadline=None)
@given(
    st.lists(
        st.tuples(
            st.text(st.characters(min_codepoint=32, max_codepoint=126), min_size=1)
            .map(str.strip)
            .filter(bool),
            st.dictionaries(st.integers(0, 500), st.integers(0, 3), max_size=6),
        ),
        max_size=6,
    )
)
def test_anything_written_in_the_format_reads_back_unchanged(
    tmp_path_factory: pytest.TempPathFactory,
    rows: list[tuple[str, dict[int, int]]],
) -> None:
    """The falsifying direction: generated queries rather than chosen ones."""
    text = "".join(
        json.dumps({"query": query, "judgements": sorted(grades.items())}) + "\n"
        for query, grades in rows
    )
    path = tmp_path_factory.mktemp("rt") / "judgements.jsonl"
    path.write_text(text, encoding="utf-8")
    assert [(one.query, dict(one.grades)) for one in load(path)] == rows


class TestTheCommittedJudgements:
    """The judgement file is data, so it is checked against the corpus it names."""

    def test_every_judged_document_exists_in_the_corpus(self) -> None:
        """A judgement naming a deleted document quietly lowers every score."""
        corpus = {document.identifier for document in read(EVALUATION_CORPUS)}
        judged = {
            document_id
            for judgement in load(JUDGEMENTS)
            for document_id in judgement.grades
        }
        assert judged <= corpus

    def test_every_document_is_relevant_to_something(self) -> None:
        """A document no query reaches contributes nothing but noise."""
        corpus = {document.identifier for document in read(EVALUATION_CORPUS)}
        judged = {
            document_id
            for judgement in load(JUDGEMENTS)
            for document_id in judgement.grades
        }
        assert corpus <= judged

    def test_no_query_appears_twice(self) -> None:
        """A duplicated query is counted twice in every average."""
        queries = [judgement.query for judgement in load(JUDGEMENTS)]
        assert len(queries) == len(set(queries))

    def test_every_query_has_at_least_one_relevant_document(self) -> None:
        """A query with no relevant document scores zero and lowers every mean."""
        assert all(judgement.relevant for judgement in load(JUDGEMENTS))

    def test_grades_stay_on_the_documented_scale(self) -> None:
        """A grade outside 1 to 3 means the file and its header disagree."""
        assert {
            grade
            for judgement in load(JUDGEMENTS)
            for grade in judgement.grades.values()
        } <= {1, 2, 3}
