"""Reading the relevance judgements that every offline evaluation depends on.

A judgement records that a person considered a document relevant to a query, and
how strongly. It cannot be derived from the corpus, cannot be computed, and is
the expensive half of retrieval evaluation.

The file holds one query per line as JSON, with blank lines and lines beginning
with ``#`` ignored so that the reasoning behind a set of labels can live beside
them:

```text
{"query": "inverted index", "judgements": [[1, 3], [20, 2]]}
```

Grades run 0 to 3, from not relevant to directly about the query. Binary metrics
treat anything above 0 as relevant; discounted cumulative gain uses the grades
themselves, which is the only reason to grade rather than mark yes or no.

Line-delimited JSON rather than one JSON document, because a judgement file is
edited by hand and appended to over years, and this way a malformed edit spoils
one line rather than the whole file. The alternative worth knowing is the format
used by the evaluation campaigns, which splits queries and judgements into two
files keyed by a query identifier. That normalises better and is harder to read,
and being able to see a query beside its labels is what makes labelling
reviewable.

Malformed input raises rather than being skipped. A judgement file silently
missing half its labels produces a plausible number that is wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeGuard, cast

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_COMMENT: Final = "#"
_PAIR_LENGTH: Final = 2


class JudgementFormatError(ValueError):
    """Raised when a judgement file cannot be understood.

    Carries the line number, because a file with hundreds of hand-written lines
    is otherwise painful to correct.
    """


@dataclass(frozen=True, slots=True)
class Judgement:
    """One query, and how relevant each judged document is to it."""

    query: str
    grades: Mapping[int, int]

    @property
    def relevant(self) -> frozenset[int]:
        """Documents graded above zero, for the metrics that ignore grades."""
        return frozenset(
            document_id for document_id, grade in self.grades.items() if grade > 0
        )


def load(path: Path) -> list[Judgement]:
    """Read every judgement in a file, in the order written.

    Raises:
        JudgementFormatError: If any line is not a well-formed judgement.
    """
    with path.open(encoding="utf-8") as handle:
        return [
            _parse(line, number)
            for number, line in enumerate(handle, start=1)
            if line.strip() and not line.lstrip().startswith(_COMMENT)
        ]


def _parse(line: str, number: int) -> Judgement:
    record = _decode(line, number)
    query = record.get("query")
    if not isinstance(query, str) or not query.strip():
        message = f"line {number}: 'query' must be a non-empty string"
        raise JudgementFormatError(message)
    entries = record.get("judgements")
    if not isinstance(entries, list):
        message = f"line {number}: 'judgements' must be a list of pairs"
        raise JudgementFormatError(message)
    return Judgement(query=query, grades=_grades(cast("list[object]", entries), number))


def _decode(line: str, number: int) -> Mapping[str, object]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        message = f"line {number} is not valid JSON: {error}"
        raise JudgementFormatError(message) from error
    if not isinstance(record, dict):
        message = f"line {number} is not a JSON object"
        raise JudgementFormatError(message)
    return cast("Mapping[str, object]", record)


def _grades(entries: list[object], number: int) -> dict[int, int]:
    grades: dict[int, int] = {}
    for entry in entries:
        document_id, grade = _pair(entry, number)
        if document_id in grades:
            message = f"line {number}: document {document_id} is judged twice"
            raise JudgementFormatError(message)
        grades[document_id] = grade
    return grades


def _pair(entry: object, number: int) -> tuple[int, int]:
    values = cast("list[object]", entry) if isinstance(entry, list) else []
    if len(values) == _PAIR_LENGTH:
        document_id, grade = values
        if _is_integer(document_id) and _is_integer(grade) and grade >= 0:
            return document_id, grade
    message = (
        f"line {number}: expected a [document, grade] pair with an integer "
        f"document and a non-negative integer grade, found {entry!r}"
    )
    raise JudgementFormatError(message)


def _is_integer(value: object) -> TypeGuard[int]:
    """Whether a decoded value is an integer, counting `true` as not one.

    JSON booleans decode to `bool`, which subclasses `int`, so a plain
    `isinstance` check accepts `[true, 3]` and silently stores document 1.
    """
    return isinstance(value, int) and not isinstance(value, bool)
