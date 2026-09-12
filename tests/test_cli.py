"""Tests for the command line interface and the packaging contract."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from search_engine import __version__
from search_engine.cli import (
    ANSI_BOLD,
    ANSI_RESET,
    DEFAULT_SCORER,
    ELLIPSIS,
    EXIT_BAD_INPUT,
    EXIT_NO_RESULTS,
    EXIT_OK,
    SCORERS,
    build_parser,
    is_segment,
    main,
)
from search_engine.segment import SEGMENT_HEADER
from tests.conftest import SAMPLE_CORPUS as FIXTURE

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TextIO


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
    assert main(["search", str(built_index), query, "--no-snippet"]) == EXIT_OK
    first = capsys.readouterr().out.splitlines()[0]
    assert f"document {expected_document}" in first


def test_search_output_is_ranked_and_scored(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["search", str(built_index), "term frequency document", "--no-snippet"])
        == EXIT_OK
    )
    lines = capsys.readouterr().out.splitlines()
    scores = [float(line.split()[1]) for line in lines]
    assert scores == sorted(scores, reverse=True)
    assert lines[0].startswith("  1.")


def test_the_limit_is_respected(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "search",
                str(built_index),
                "term index document",
                "--limit",
                "2",
                "--no-snippet",
            ]
        )
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
        [
            sys.executable,
            "-m",
            "search_engine.cli",
            "search",
            str(path),
            "computer",
            "--no-snippet",
        ],
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
            "--no-snippet",
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


def test_bm25f_is_selectable_as_a_scorer(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Registered, but not the default: a default changes on evidence."""
    assert main(["search", str(built_index), "search", "--scorer", "bm25f"]) == EXIT_OK
    assert capsys.readouterr().out.strip()


def test_the_default_scorer_is_still_tfidf() -> None:
    """Neither BM25, nor proximity, nor BM25F could be shown to beat it."""
    assert DEFAULT_SCORER == "tfidf"
    assert set(SCORERS) == {"tfidf", "bm25", "bm25f"}


# A term in every document has an inverse document frequency of zero, so a
# one-document corpus can never produce a result. Every corpus below carries a
# second page for that reason alone.
OTHER_PAGE = "<page><id>99</id><title>Other</title><text>unrelated filler</text></page>"


class Stream:
    """A stdout that reports whatever encoding a test needs it to report.

    ``sys.stdout.encoding`` is read only, so the stream has to be replaced
    rather than adjusted. Writes go on to the stream being captured.
    """

    def __init__(self, inner: TextIO, encoding: str | None, *, terminal: bool) -> None:
        self._inner = inner
        self.encoding = encoding
        self._terminal = terminal

    def write(self, text: str) -> int:
        """Pass the text on to the stream being captured."""
        return self._inner.write(text)

    def flush(self) -> None:
        """Flush the stream being captured."""
        self._inner.flush()

    def isatty(self) -> bool:
        """Report whatever this stand-in was told to report."""
        return self._terminal


def no_terminal_size(**_: object) -> os.terminal_size:
    """Stand in for a terminal that reports no width at all."""
    return os.terminal_size((0, 0))


def corpus_of(tmp_path: Path, pages: str) -> Path:
    path = tmp_path / "c.xml"
    path.write_text(pages + OTHER_PAGE, encoding="utf-8")
    return path


def indexed(tmp_path: Path, pages: str, *, on_disk: bool = False) -> Path:
    """Build an index from these page records and return its path.

    The build's own line of output is swallowed, so a test asserting on the
    first line of a search is looking at the search.
    """
    output = tmp_path / ("pages.seg" if on_disk else "pages.index")
    argv = ["index", str(corpus_of(tmp_path, pages)), str(output)]
    with contextlib.redirect_stdout(io.StringIO()):
        assert main([*argv, "--on-disk"] if on_disk else argv) == EXIT_OK
    return output


def test_a_result_shows_its_title_and_the_passage_that_matched(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["search", str(built_index), "inverted index", "--limit", "1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("  1. Inverted index")
    assert lines[0].endswith("0.6109")
    assert "An inverted index maps each term" in " ".join(lines[1:])


def test_the_segment_format_shows_titles_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both write paths store the text, so both can show it."""
    path = indexed(
        tmp_path,
        "<page><id>7</id><title>Skip pointers</title>"
        "<text>A skip pointer jumps over postings.</text></page>",
        on_disk=True,
    )
    assert main(["search", str(path), "skip"]) == EXIT_OK
    assert "Skip pointers" in capsys.readouterr().out


def test_a_document_with_no_title_falls_back_to_its_identifier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = indexed(tmp_path, "<page><id>9</id><text>lonely untitled page</text></page>")
    assert main(["search", str(path), "lonely"]) == EXIT_OK
    assert capsys.readouterr().out.startswith("  1. document 9")


def test_matched_words_are_bold_on_a_terminal(
    built_index: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdout", Stream(sys.stdout, "utf-8", terminal=True))
    assert main(["search", str(built_index), "gardening", "--limit", "1"]) == 0
    assert f"{ANSI_BOLD}Gardening{ANSI_RESET}" in capsys.readouterr().out


def test_a_pipe_gets_no_escape_codes(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """This output ends up in a file, and escape codes in a file are noise."""
    assert main(["search", str(built_index), "gardening", "--limit", "1"]) == 0
    assert "\x1b" not in capsys.readouterr().out


def test_json_output_parses_and_has_a_stable_key_set(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["search", str(built_index), "inverted index", "--json"]) == EXIT_OK
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows
    for row in rows:
        assert set(row) == {"identifier", "title", "score", "snippet", "highlights"}
        assert isinstance(row["identifier"], int)
        assert all(len(pair) == 2 for pair in row["highlights"])
    assert rows[0]["title"] == "Inverted index"


def test_json_output_stays_machine_readable_when_nothing_matched(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prose on stdout would break the caller that asked for JSON."""
    code = main(["search", str(built_index), "nonexistentword", "--json"])
    assert code == EXIT_NO_RESULTS
    assert capsys.readouterr().out == ""


def test_an_escape_sequence_in_a_title_is_neutralised(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corpus is untrusted input, and a terminal acts on what it is sent."""
    path = indexed(
        tmp_path,
        "<page><id>1</id><title>\x1b[31mred</title>"
        "<text>a page about roses</text></page>",
    )
    assert main(["search", str(path), "roses"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert "[31mred" in out


def test_a_newline_in_the_text_does_not_break_a_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = indexed(
        tmp_path,
        "<page><id>1</id><title>Wrapped</title><text>alpha\nbeta</text></page>",
    )
    assert main(["search", str(path), "alpha"]) == EXIT_OK
    assert capsys.readouterr().out.splitlines()[1] == "     alpha beta"


def test_a_long_title_is_truncated_to_the_terminal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "40")
    path = indexed(
        tmp_path,
        f"<page><id>1</id><title>{'verylong ' * 10}</title>"
        "<text>a page about roses</text></page>",
    )
    assert main(["search", str(path), "roses"]) == EXIT_OK
    heading = capsys.readouterr().out.splitlines()[0]
    assert len(heading) == 40
    assert heading[:-8].rstrip().endswith(ELLIPSIS)


def test_a_terminal_reporting_no_width_falls_back_to_eighty(
    built_index: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("search_engine.cli.shutil.get_terminal_size", no_terminal_size)
    assert main(["search", str(built_index), "inverted", "--limit", "1"]) == EXIT_OK
    assert all(len(line) <= 80 for line in capsys.readouterr().out.splitlines())


def test_a_character_the_console_cannot_encode_is_escaped_not_fatal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Windows console is not UTF-8, and a title it cannot write must not kill it."""
    path = indexed(
        tmp_path,
        "<page><id>1</id><title>café</title><text>a page about coffee</text></page>",
    )
    monkeypatch.setattr(sys, "stdout", Stream(sys.stdout, "ascii", terminal=False))
    assert main(["search", str(path), "coffee"]) == EXIT_OK
    assert "caf\\xe9" in capsys.readouterr().out


def test_a_stream_reporting_no_encoding_still_prints(
    built_index: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdout", Stream(sys.stdout, None, terminal=False))
    assert main(["search", str(built_index), "inverted", "--limit", "1"]) == EXIT_OK
    assert "Inverted index" in capsys.readouterr().out


def test_an_index_with_no_stored_text_still_searches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """What an index built before the text was stored beside it looks like."""
    path = indexed(
        tmp_path, "<page><id>3</id><title>Kept</title><text>alpha beta</text></page>"
    )
    (tmp_path / "pages.index.dat").unlink()
    assert main(["search", str(path), "alpha"]) == EXIT_OK
    captured = capsys.readouterr()
    assert captured.out.strip() == "1. 0.5774  document 3"
    assert "no stored text" in captured.err


def test_the_json_form_carries_the_snippet_and_its_ranges(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The shape an HTTP layer will return, settled here so it invents nothing."""
    assert (
        main(["search", str(built_index), "gardening", "--json", "--limit", "1"]) == 0
    )
    row = json.loads(capsys.readouterr().out)
    text = row["snippet"]
    # Both stem to `garden`, so both are part of why the document matched.
    assert [text[start:end] for start, end in row["highlights"]] == [
        "Gardening",
        "garden",
    ]


def test_a_passage_longer_than_the_terminal_is_wrapped(
    built_index: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "50")
    assert main(["search", str(built_index), "inverted index", "--limit", "1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) > 2
    assert all(len(line) <= 50 for line in lines)
    assert all(line.startswith("     ") for line in lines[1:])


def test_results_are_separated_by_a_blank_line(
    built_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two results run together are one wall of text, not a list."""
    assert main(["search", str(built_index), "index term", "--limit", "2"]) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("  1. ")
    blank = lines.index("")
    assert lines[blank + 1].startswith("  2. ")


def test_a_document_with_no_body_shows_its_title_and_nothing_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The title is indexed, so a document with an empty body is still findable."""
    path = indexed(
        tmp_path, "<page><id>4</id><title>Bodyless page</title><text></text></page>"
    )
    assert main(["search", str(path), "bodyless"]) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("  1. Bodyless page")
