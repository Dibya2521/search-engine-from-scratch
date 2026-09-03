"""Tests for the command line entry point and the packaging contract."""

from __future__ import annotations

import subprocess
import sys

import pytest

from search_engine import __version__
from search_engine.cli import build_parser, main


def test_version_is_a_three_part_number() -> None:
    """The build reads the version from the package, so it must stay parseable."""
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_main_with_no_arguments_prints_help_and_succeeds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main([])
    assert exit_code == 0
    assert "search-engine" in capsys.readouterr().out


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


def test_module_is_runnable_with_dash_m() -> None:
    """Covers the ``__main__`` branch, which an in-process test cannot reach."""
    completed = subprocess.run(
        [sys.executable, "-m", "search_engine.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert __version__ in completed.stdout
