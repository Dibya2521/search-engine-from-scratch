"""The file that decides which segments exist.

A segment file that the manifest does not name does not exist, whatever is
sitting on the disk. That one rule is the whole of crash recovery: a process
that dies after writing a segment and before publishing a manifest leaves an
orphan that the next open ignores, and nothing is corrupt because nothing ever
pointed at it.

The manifest is never edited in place. A new one is written beside the old and
moved onto it, and `os.replace` makes that move atomic on both Unix and
Windows, so a reader sees either the whole of the old manifest or the whole of
the new one and never a half-written file.

It is JSON rather than the project's own binary codecs, deliberately. It is
tiny, it is read once per open, and being able to read it with `cat` during an
incident is worth more than the bytes it would save.

Every publication increases the generation. That number is what a cache entry
is keyed by, so an entry from an earlier generation can never be mistaken for a
current one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

from search_engine.analysis import fingerprint
from search_engine.segment import SegmentAnalyzerMismatchError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

MANIFEST_NAME: Final = "manifest.json"
TEMPORARY_NAME: Final = "manifest.json.tmp"
FORMAT_VERSION: Final = 1


class ManifestFormatError(ValueError):
    """Raised when a manifest is malformed or of an unknown format."""


@dataclass(frozen=True, slots=True)
class SegmentInfo:
    """One published segment: its file, its size, and what it should checksum to.

    The checksum repeats what the segment's own footer holds. It is here so that
    a substituted or truncated file is caught when the directory is opened
    rather than when a query happens to read the damaged part.
    """

    name: str
    documents: int
    checksum: int


@dataclass(frozen=True, slots=True)
class Manifest:
    """Everything a reader needs to know before it opens a single segment."""

    generation: int
    next_segment: int
    analyzer: str
    segments: tuple[SegmentInfo, ...]

    @property
    def document_count(self) -> int:
        """Return how many documents the published segments hold in total."""
        return sum(segment.documents for segment in self.segments)

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as the object written to disk."""
        return {
            "format": FORMAT_VERSION,
            "generation": self.generation,
            "next_segment": self.next_segment,
            "analyzer": self.analyzer,
            "segments": [
                {
                    "name": segment.name,
                    "documents": segment.documents,
                    "checksum": segment.checksum,
                }
                for segment in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        """Rebuild a manifest from the object read off disk.

        Raises:
            ManifestFormatError: If a field is missing, of the wrong type, or
                the format version is not one this build understands.
        """
        version = _field(data, "format", int)
        if version != FORMAT_VERSION:
            message = (
                f"manifest is format version {version}, "
                f"but this build writes version {FORMAT_VERSION}"
            )
            raise ManifestFormatError(message)
        return cls(
            generation=_field(data, "generation", int),
            next_segment=_field(data, "next_segment", int),
            analyzer=_field(data, "analyzer", str),
            segments=_segments(data),
        )

    def with_segments(
        self, segments: Sequence[SegmentInfo], next_segment: int
    ) -> Manifest:
        """Return the next generation of this manifest, naming these segments."""
        return Manifest(
            generation=self.generation + 1,
            next_segment=next_segment,
            analyzer=self.analyzer,
            segments=tuple(segments),
        )


def empty(analyzer: str | None = None) -> Manifest:
    """Return the manifest of a directory holding nothing."""
    return Manifest(
        generation=0,
        next_segment=0,
        analyzer=fingerprint() if analyzer is None else analyzer,
        segments=(),
    )


def read(directory: Path) -> Manifest:
    """Return the manifest of a directory, or an empty one if it has none.

    A directory with no manifest is a directory with no index, which is a
    normal state rather than an error.

    Raises:
        ManifestFormatError: If the manifest exists and is malformed.
        SegmentAnalyzerMismatchError: If it was written by a different analysis
            configuration, so its terms would not match anything produced now.
    """
    path = directory / MANIFEST_NAME
    if not path.is_file():
        return empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        message = f"manifest is not readable JSON: {error}"
        raise ManifestFormatError(message) from error
    if not isinstance(data, dict):
        message = f"manifest must be an object, found {type(data).__name__}"
        raise ManifestFormatError(message)
    manifest = Manifest.from_dict(cast("Mapping[str, Any]", data))
    current = fingerprint()
    if manifest.analyzer != current:
        raise SegmentAnalyzerMismatchError(manifest.analyzer, current)
    return manifest


def publish(directory: Path, manifest: Manifest) -> None:
    """Replace the manifest atomically, so a crash leaves the previous one whole.

    Writing to a temporary file and moving it into place is what makes this
    safe. Editing the manifest where it lies would leave a reader that arrives
    at the wrong moment with a file that is half one generation and half
    another.
    """
    temporary = directory / TEMPORARY_NAME
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=False)
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    # Path.replace is os.replace, and both are atomic on Unix and Windows
    # alike, unlike a rename over an existing file.
    temporary.replace(directory / MANIFEST_NAME)


def _field[T](data: Mapping[str, Any], name: str, kind: type[T]) -> T:
    if name not in data:
        message = f"manifest is missing the field {name!r}"
        raise ManifestFormatError(message)
    value = data[name]
    # bool is a subclass of int, and a manifest saying true where it means 1 is
    # malformed rather than merely surprising.
    if not isinstance(value, kind) or isinstance(value, bool) is not (kind is bool):
        found = type(value).__name__
        message = f"manifest field {name!r} must be {kind.__name__}, found {found}"
        raise ManifestFormatError(message)
    return value


def _segments(data: Mapping[str, Any]) -> tuple[SegmentInfo, ...]:
    raw = data.get("segments")
    if not isinstance(raw, list):
        found = type(raw).__name__
        message = f"manifest field 'segments' must be list, found {found}"
        raise ManifestFormatError(message)
    entries = cast("list[object]", raw)
    return tuple(_segment(entry) for entry in entries)


def _segment(entry: object) -> SegmentInfo:
    if not isinstance(entry, dict):
        message = f"manifest segment must be an object, found {type(entry).__name__}"
        raise ManifestFormatError(message)
    fields = cast("Mapping[str, Any]", entry)
    return SegmentInfo(
        name=_field(fields, "name", str),
        documents=_field(fields, "documents", int),
        checksum=_field(fields, "checksum", int),
    )
