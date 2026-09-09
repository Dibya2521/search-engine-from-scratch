# ADR 0005: An index of many immutable segments, published by a manifest

- **Status:** accepted
- **Date:** 2026-09-09
- **Supersedes:** nothing
- **Superseded by:** nothing

## Context

An index could only be built in one shot. `add_document` refuses an identifier
it has already seen, and a ranker refuses an index that changed under it. Both
are deliberate and both mean a corpus that grows has to be built again from the
start, which at the measured 1.1 MB/s is about four and a half minutes per
document on a 300 MB corpus.

The on-disk format removed the cost of *reading* an index without loading it. It
did nothing about writing one, and it left several things impossible that
everything planned afterwards depends on: publishing new data safely, recovering
from a crash, reclaiming space, and caching anything at all without an
invalidation story.

Ingestion in any real system is also at-least-once, so the same document arrives
twice. `DuplicateDocumentError` is exactly the wrong response to that.

## Decision

**Keep the index as many immutable segments, and let a manifest decide which of
them exist.**

1. **A segment file is immutable.** Written once, never modified.
2. **The manifest is the truth.** A segment file it does not name does not
   exist, whatever is on the disk. This one rule is the whole of crash
   recovery.
3. **The manifest is replaced atomically**, written beside the old one and moved
   onto it with `Path.replace`, which is atomic on Unix and Windows alike.
4. **Deletes are tombstones**, a bitset over document ordinals beside each
   segment. Space returns at merge time.
5. **Adding is an upsert.** `DuplicateDocumentError` stays on the in-memory
   index, where a repeated identifier is a corpus bug; tolerance lives in the
   writer, the layer that can make an older copy invisible.
6. **Merging is tiered**, four segments to a tier, repeated until no tier is
   full, with tiers judged on live documents rather than stored ones.
7. **A write is durable when it is in the log**, an append-only file discarded
   only once its documents are named by the manifest.

## Consequences

**Immutability buys four things at once**: atomic publication, crash recovery by
construction, concurrent reads with no locking, and caches that cannot go stale.
The last is the one worth noticing, because cache invalidation is normally the
hard part and here it disappears as a side effect of a decision about file
layout.

**Ordering is the safety argument, everywhere.** Segment before manifest,
manifest before log truncation, merged output before deleting its inputs. Each
crash window leaves either an orphan nothing names or a replay that upsert makes
a no-op. Reversing any of them loses data.

**Recovery is simple only because upsert came first.** Replaying a document that
is already published has to be harmless, and that is a property of the write
path rather than a property of the log.

**Segments cost read speed.** Sixteen of them are 46 percent slower per term
than one and two and a half times slower to open, from
`uv run python benchmarks/segments.py`. That is a moderate cost at this scale
rather than the dramatic one the usual argument implies, and it is what the
merge policy exists to remove. The honest reason to merge anyway is that segment
count is otherwise unbounded and nothing here measures what a thousand would
cost.

**Durability costs a factor of 25** in write throughput, 75,407 documents per
second against 2,955. It is on by default because a silently lost document is
the worst failure a data system has.

**Merging streams.** A builder takes terms in sorted order and sends postings to
the file as they arrive, so a merge holds the vocabulary and the document list
but never the postings. Measured at less than a third of the peak memory of
building the merged index first, and an earlier version was 8.2x worse again
because the reader cached postings a merge reads exactly once.

**One writer per directory.** Two would each hold their own idea of the
generation and the last to publish would discard the other's work. Nothing
enforces it yet, and that is written down rather than assumed.

**Durability of the directory entry is not provided**, because there is no
portable way to `fsync` a directory on Windows. Stated rather than claimed away.

## Alternatives considered

| Rejected | Why |
| --- | --- |
| **Rebuild on every change** | The status quo. Correct and simple, at four and a half minutes per document on a 300 MB corpus. |
| **Mutable postings, edited in place** | Locking on every read, no atomic publication, no crash recovery by construction, no cache that cannot go stale. Immutability buys all four together. |
| **Levelled merging**, one segment per level | Better read amplification and much worse write amplification, since a document is rewritten whenever its level is touched. Wrong trade for an index written in bulk and read constantly. Tiered is what Lucene does, for the same reason. |
| **Deleting by rewriting the segment** | Constant-time delete becomes proportional to the segment, and immutability is given up to achieve it. |
| **A database for the manifest** | A runtime dependency, for a file read once per open and small enough to read with `cat`. |
| **Logging additions only** | Specified that way in the plan and changed during implementation: an unlogged delete is undone by the next recovery, which is data loss in the other direction. |
| **Making `DuplicateDocumentError` go away** | It is right where it is. Moving the tolerance up a layer and deleting the check look the same in a diff and are not. |
