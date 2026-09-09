"""Combining small segments into larger ones.

Without merging, a directory gains a segment on every flush. A query then opens
every one of them and pays a dictionary lookup in each, so read cost grows
without limit while write cost stays flat. Merging trades write work for read
speed, and that trade is the central idea of a log-structured merge tree.

**The policy is tiered.** A segment belongs to a size tier, and when a tier
holds MERGE_FACTOR segments they are combined into one, which usually lands in
the tier above and may in turn fill it.

The alternative is levelled merging, where each level holds one segment and a
new arrival is merged into it immediately. That gives better read amplification,
because there are fewer segments to search, and much worse write amplification,
because the same document is rewritten every time its level is touched. For an
index written in bulk and read constantly, tiered is the right side of that
trade, and it is what Lucene does.

**Deleted documents are dropped rather than carried across.** This is where the
space a tombstone marked comes back, and it is why deleting from a real search
engine frees nothing until a merge runs.

The ordering is what makes it safe. The output is written, then the manifest is
published naming the output and not the inputs, and only then are the inputs
deleted. A crash between the second and third steps leaves files that nothing
names, which the next open ignores. Deleting first would lose data outright.
"""

from __future__ import annotations

import heapq
from math import log
from typing import TYPE_CHECKING, Final

from search_engine import manifest as manifest_file
from search_engine.segment import (
    SegmentBuilder,
    SegmentReader,
    stored_checksum,
)
from search_engine.tombstones import Tombstones, tombstone_name

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from search_engine.manifest import Manifest, SegmentInfo

MERGE_FACTOR: Final = 4


def tier_of(documents: int) -> int:
    """Return the size tier a segment of this many live documents belongs to.

    Tiers are powers of MERGE_FACTOR, so merging MERGE_FACTOR segments of one
    tier produces one that sits in the next.
    """
    if documents < MERGE_FACTOR:
        return 0
    return int(log(documents) / log(MERGE_FACTOR))


def select_merge(segments: Sequence[SegmentInfo]) -> list[SegmentInfo]:
    """Return a group of segments worth merging, smallest tier first.

    Smallest first because merging the cheapest group buys the same reduction
    in segment count for the least work, and because a merge there may fill the
    tier above and be picked up on the next pass.
    """
    tiers: dict[int, list[SegmentInfo]] = {}
    for segment in segments:
        tiers.setdefault(tier_of(segment.live), []).append(segment)
    for tier in sorted(tiers):
        group = tiers[tier]
        if len(group) >= MERGE_FACTOR:
            return group
    return []


def merge_segments(
    directory: Path, inputs: Sequence[SegmentInfo], output_name: str
) -> SegmentInfo:
    """Combine several segments into one, leaving out what has been deleted.

    Terms are walked in sorted order across every input at once, so the output
    is produced in one pass and never holds more than one term's postings.
    """
    readers = _open_inputs(directory, inputs)
    path = directory / output_name
    try:
        surviving = sorted(document_id for _, live in readers for document_id in live)
        with path.open("wb") as sink:
            builder = SegmentBuilder(sink, surviving)
            for term, postings in _merged_terms(readers):
                builder.add(term, postings)
            builder.finish()
    finally:
        for reader, _ in readers:
            reader.close()
    return manifest_file.SegmentInfo(
        name=output_name,
        documents=len(surviving),
        checksum=stored_checksum(path.read_bytes()),
    )


def _open_inputs(
    directory: Path, inputs: Sequence[SegmentInfo]
) -> list[tuple[SegmentReader, frozenset[int]]]:
    """Open each input with the set of documents still visible in it.

    Caching is off because a merge reads every term exactly once, so a cache
    would hold every posting it has passed and buy nothing.
    """
    readers: list[tuple[SegmentReader, frozenset[int]]] = []
    try:
        for segment in inputs:
            reader = SegmentReader(directory / segment.name, verify=False, cache=False)
            deleted = Tombstones.read(
                directory / tombstone_name(segment.name), segment.documents
            )
            readers.append((reader, _live_ids(reader, deleted)))
    except Exception:
        for reader, _ in readers:
            reader.close()
        raise
    return readers


def _live_ids(reader: SegmentReader, deleted: Tombstones) -> frozenset[int]:
    return frozenset(
        document_id
        for ordinal, document_id in enumerate(reader.document_ids)
        if not deleted.is_deleted(ordinal)
    )


def _merged_terms(
    readers: Sequence[tuple[SegmentReader, frozenset[int]]],
) -> Iterator[tuple[str, Mapping[int, Sequence[int]]]]:
    """Yield every surviving term once, in sorted order, with its postings.

    `heapq.merge` over each input's sorted terms gives one ascending stream, and
    equal terms arrive together, which is what lets their postings be combined
    without holding the whole vocabulary.
    """
    streams = [reader.terms for reader, _ in readers]
    previous: str | None = None
    for term in heapq.merge(*streams):
        if term == previous:
            continue
        previous = term
        postings: dict[int, Sequence[int]] = {}
        for reader, live in readers:
            postings.update(
                {
                    document_id: positions
                    for document_id, positions in reader.postings(term).items()
                    if document_id in live
                }
            )
        # A term whose every document has been deleted has no postings left,
        # and a segment cannot hold a term that occurs nowhere.
        if postings:
            yield term, postings


def merge_once(directory: Path, manifest: Manifest) -> Manifest | None:
    """Merge one group of segments if any tier is full, and publish the result.

    Returns the new manifest, or None when nothing was worth merging.
    """
    group = select_merge(manifest.segments)
    if not group:
        return None
    output = f"segment-{manifest.next_segment:08d}.seg"
    info = merge_segments(directory, group, output)

    merged = {segment.name for segment in group}
    kept = [segment for segment in manifest.segments if segment.name not in merged]
    published = manifest.with_segments([*kept, info], manifest.next_segment + 1)
    manifest_file.publish(directory, published)

    # Only now, once nothing names them. A crash before this point leaves files
    # the next open ignores; deleting before publishing would lose documents.
    for segment in group:
        (directory / segment.name).unlink(missing_ok=True)
        (directory / tombstone_name(segment.name)).unlink(missing_ok=True)
    return published
