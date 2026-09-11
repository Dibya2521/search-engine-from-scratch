"""Tests for the command line interface and the packaging contract."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from search_engine import __version__
from search_engine.cli import (
    EXIT_BAD_INPUT,
    EXIT_NO_RESULTS,
    EXIT_OK,
    build_parser,
    is_segment,
    main,
)
from search_engine.segment import SEGMENT_HEADER
from tests.conftest import SAMPLE_CORPUS as FIXTURE

if TYPE_CHECKING:
    from pathlib import Path


def test_candidates_that_all_score_zero_report_no_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A term in every document has zero weight, so matching is not enough.

    The query finds candidates, then every one of them scores exactly zero, so
    there is genuinely nothing to rank. Distinct from finding no candidates at
    all, and it has to be handled separately.
    """
    corpus = tmp_path / "c.xml"
    corpus.write_text(
        "<page><id>1</id><text>alpha beta</text></page>"
        "<page><id>2</id><text>alpha gamma</text></page>",
        encoding="utf-8",
    )
    path = tmp_path / "c.index"
    assert main(["index", str(corpus), str(path)]) == EXIT_OK
    assert main(["search", str(path), "alpha"]) == EXIT_NO_RESULTS
    assert "no matching documents" in capsys.readouterr().out


def test_version_is_a_three_part_number() -> None:
    """The build reads the version from the package, so it must stay parseable."""
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_no_subcommand_prints_help_and_succeeds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == EXIT_OK
    output = capsys.readouterr().out
    assert "search-engine" in output
    assert "index" in output
    assert "search" in output


def test_version_flag_exits_zero_and_reports_the_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The version action exits the process, so SystemExit is expected."""
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_unknown_argument_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--no-such-flag"])
    assert caught.value.code != 0


def test_parser_program_name_matches_the_console_script() -> None:
    """A mismatch means the help text lies about how to invoke the tool."""
    assert build_parser().prog == "search-engine"


def test_indexing_reports_what_it_built(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "out.index"
    assert main(["index", str(FIXTURE), str(path)]) == EXIT_OK
    assert path.exists()
    assert "indexed 7 documents" in capsys.readouterr().out


def test_indexing_a_missing_corpus_reports_the_problem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["index", str(tmp_path / "nope.xml"), str(tmp_path / "o.index")])
    assert result == EXIT_BAD_INPUT
    assert "error:" in capsys.readouterr().err


def test_indexing_a_malformed_corpus_reports_the_problem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "bad.xml"
    corpus.write_text("<page><title>no id</title></page>", encoding="utf-8")
    assert main(["index", str(corpus), str(tmp_path / "o.index")]) == EXIT_BAD_INPUT
    assert "no <id>" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("query", "expected_document"),
    [
        ("computer", 2),
        ("gardening", 4),
        ('"computer science department"', 2),
        # Stemming has to work across the whole pipeline, not just in unit tests.
        ("connecting", 3),
    ],
)
def test_searching_ranks_the_right_document_first(
    built_index: Path,
    capsys: pytest.CaptureFixture[str],
    query: str,
    expected_document: int,
) -> None:
    assert main(["search", str(built_index), query]) == EXIT_OK
    first = capsys.readouterr().out.splitlines()[0]
    assert f"document {expected_document}" in first


def test_search_output_is_ranked_and_scored(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["search", str(built_index), "term frequency document"]) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    scores = [float(line.split()[1]) for line in lines]
    assert scores == sorted(scores, reverse=True)
    assert lines[0].startswith("  1.")


def test_the_limit_is_respected(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["search", str(built_index), "term index document", "--limit", "2"])
        == EXIT_OK
    )
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_a_query_matching_nothing_exits_one(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Distinct from success, so a shell pipeline can tell the difference."""
    assert main(["search", str(built_index), "nonexistentword"]) == EXIT_NO_RESULTS
    assert "no matching documents" in capsys.readouterr().out


def test_a_query_of_only_stopwords_exits_one(built_index: Path) -> None:
    assert main(["search", str(built_index), "the and of"]) == EXIT_NO_RESULTS


def test_searching_a_missing_index_reports_the_problem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["search", str(tmp_path / "nope.index"), "anything"])
    assert result == EXIT_BAD_INPUT
    assert "error:" in capsys.readouterr().err


def test_searching_a_corrupt_index_reports_the_problem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.index"
    path.write_text("not an index at all\n", encoding="utf-8")
    assert main(["search", str(path), "anything"]) == EXIT_BAD_INPUT
    assert "error:" in capsys.readouterr().err


def test_index_then_search_works_as_two_separate_processes(tmp_path: Path) -> None:
    """The real contract: build once, query later, in a fresh process.

    Also covers the ``__main__`` branch, which an in-process test cannot reach.
    """
    path = tmp_path / "e2e.index"
    build = subprocess.run(
        [sys.executable, "-m", "search_engine.cli", "index", str(FIXTURE), str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "indexed 7 documents" in build.stdout
    query = subprocess.run(
        [sys.executable, "-m", "search_engine.cli", "search", str(path), "computer"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "document 2" in query.stdout


@pytest.mark.parametrize("scorer", ["tfidf", "bm25"])
def test_either_scorer_finds_the_right_document(
    built_index: Path, capsys: pytest.CaptureFixture[str], scorer: str
) -> None:
    code = main(
        [
            "search",
            str(built_index),
            '"computer science department"',
            "--scorer",
            scorer,
        ]
    )
    assert code == EXIT_OK
    assert "document 2" in capsys.readouterr().out


def test_the_two_scorers_give_different_scores(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Otherwise the flag would be wired up but doing nothing."""
    outputs: list[str] = []
    for scorer in ("tfidf", "bm25"):
        assert main(["search", str(built_index), "computer", "--scorer", scorer]) == 0
        outputs.append(capsys.readouterr().out)
    assert outputs[0] != outputs[1]


def test_the_default_scorer_is_tf_idf(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Recorded as a test because the comparison did not justify changing it."""
    assert main(["search", str(built_index), "computer"]) == EXIT_OK
    without = capsys.readouterr().out
    assert main(["search", str(built_index), "computer", "--scorer", "tfidf"]) == 0
    assert capsys.readouterr().out == without


def test_an_unknown_scorer_is_rejected(built_index: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["search", str(built_index), "computer", "--scorer", "nonsense"])
    assert exit_info.value.code == 2


def test_indexing_on_disk_writes_a_segment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "on-disk.seg"
    assert main(["index", str(FIXTURE), str(output), "--on-disk"]) == EXIT_OK
    assert "indexed" in capsys.readouterr().out
    assert is_segment(output)


def test_indexing_without_the_flag_writes_the_whole_file_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "whole.index"
    assert main(["index", str(FIXTURE), str(output)]) == EXIT_OK
    capsys.readouterr()
    assert not is_segment(output)


@pytest.mark.parametrize("extra", [["--on-disk"], []], ids=["segment", "whole"])
@pytest.mark.parametrize("scorer", ["tfidf", "bm25"])
def test_both_formats_answer_the_same_query_the_same_way(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    scorer: str,
) -> None:
    """The point of the protocol: ranking does not know which format it has."""
    output = tmp_path / "index"
    assert main(["index", str(FIXTURE), str(output), *extra]) == EXIT_OK
    capsys.readouterr()

    assert main(["search", str(output), "web search", "--scorer", scorer]) == EXIT_OK
    assert "document" in capsys.readouterr().out


def test_a_corrupt_segment_reports_the_problem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "broken.seg"
    assert main(["index", str(FIXTURE), str(output), "--on-disk"]) == EXIT_OK
    capsys.readouterr()
    data = bytearray(output.read_bytes())
    data[len(SEGMENT_HEADER) + 20] ^= 0xFF
    output.write_bytes(bytes(data))

    assert main(["search", str(output), "web"]) == EXIT_BAD_INPUT
    assert "error:" in capsys.readouterr().err
    output.unlink()


def test_a_misspelled_query_suggests_a_correction(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The commonest cause of no results, and the engine already knows the terms.

    The suggestion goes to stderr so a caller piping results is not handed
    prose, and the exit code stays at "no results", which is still what
    happened.
    """
    assert main(["search", str(built_index), "serach"]) == EXIT_NO_RESULTS
    captured = capsys.readouterr()
    assert "no matching documents" in captured.out
    assert "did you mean" in captured.err
    assert "search" in captured.err


def test_a_query_with_nothing_near_it_suggests_nothing(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A suggestion nobody asked for is worse than none."""
    assert main(["search", str(built_index), "zzzzzzzzzzqqq"]) == EXIT_NO_RESULTS
    assert "did you mean" not in capsys.readouterr().err
