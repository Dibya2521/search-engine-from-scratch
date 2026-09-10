# ADR 0006: Postings are walked in document order, in blocks

- **Status:** accepted
- **Date:** 2026-09-10
- **Supersedes:** part of the postings-shape decision recorded in
  [the inverted index](../03-inverted-index.md), under "Postings keyed by
  document, not held as a sorted list"
- **Superseded by:** nothing

## Context

The in-memory index maps each term to a dictionary keyed by document identifier.
That was chosen deliberately, and for a good reason: every query operation the
engine needed was a **set operation over document identifiers**. A free-text
query is a union of keys and a phrase query begins with an intersection of keys,
and a dictionary gives both directly where a sorted list needs a merge walk.

That reason has not stopped being true. What changed is that a new operation
arrived which the shape cannot serve.

Early termination walks several terms at once, always advancing whichever cursor
is furthest behind, and asks each of them "give me your first document at or
after this identifier". That is a **skip**, and it needs three things a
dictionary does not provide:

1. **An order.** The pivot rule depends on knowing that no cursor before the
   pivot has passed the pivot document yet. Without a defined order over
   identifiers, that sentence has no meaning.
2. **A way to jump.** Answering "first document at or after N" by examining
   every key is exactly the linear pass being avoided.
3. **A bound attached to a range of the list.** Per-block maxima only exist if
   the list has blocks, and a dictionary has no blocks to attach anything to.

The v1 segment format already wrote identifiers in sorted order and carried a
separate skip list of about `sqrt(n)` entries. That answered points 1 and 2 and
not point 3, because a skip entry carries a document identifier and a byte
offset and no term frequency at all.

## Decision

**A term's postings on disk are an ordered sequence of blocks, and the block is
the unit of both skipping and bounding.**

1. **Document-ordered.** Identifiers ascend within a term. This is what makes
   the pivot rule meaningful.
2. **Cut into blocks** of `POSTINGS_BLOCK_SIZE`, currently 128 documents.
3. **A table ahead of the blocks** carries, per block, its last document
   identifier, how many documents it holds, the highest term frequency inside
   it, and its length in bytes.
4. **Delta encoding restarts at every block boundary**, so a block can be
   decoded without decoding the blocks before it.
5. **The separate skip list is removed.** Block boundaries are the skip points.

This required the segment format to move to version 2. A version 1 file is
refused by name rather than misread.

**The in-memory index keeps its dictionary.** Nothing about this decision
applies to it. It has no blocks, reports one bound per term, and is searched
exactly as before.

## Consequences

**The skip list was better at skipping, and that is the price.** Measured on the
same 10,000 document corpus with
[`benchmarks/on_disk.py`](../../benchmarks/on_disk.py), against walking
identifiers from the start:

| Postings | v1 skip list | v2 block table |
| ---: | ---: | ---: |
| 10,000 | -93.2% | -84.0% |
| 26 | +4.9% | +26.3% |

Two causes, both known rather than guessed. `advance_to` now decodes a whole
block's identifiers and binary searches them, where the skip list walked gaps
and stopped at the first hit. And every term now carries a table, where v1 wrote
no skip list at all below 128 postings, so the shortest lists pay most.

**What the table bought is not on that page.** Per-block bounds cut documents
scored from 50,000 to 138 on a corpus where one document holds a term 500 times
and the rest hold it once. The query path does not call `advance_to`; it uses a
cursor that reads the table once and reuses each decoded block across advances.
See [early termination](../14-early-termination.md) for the full measurement.

**Blocks are not always worth their cost.** On a corpus whose document lengths
are drawn independently, a document with a high term frequency lands in any
block with equal probability, every block maximum ends up near the term's own,
and per-block bounds prune nothing that per-term bounds did not. They pay when
high frequencies are concentrated in few blocks, which, because postings are
ordered by identifier, means when identifier order correlates with term
frequency.

**Two shapes now coexist for the same data**, a dictionary in memory and an
ordered block list on disk, and they answer to different constraints. That was
already true before this decision; it is now true for a second reason.

**Every existing segment must be rebuilt.** While the major version is `0` a
minor release may change a format, and the changelog says so.

## What was rejected

**Keeping the skip list beside the block table.** Two structures answering the
same question drift apart, and the skip list could not have supplied the bound
because it stores no term frequency.

**Storing only a document count, a maximum frequency and a byte length per
block**, which was the format first sketched. A reader given those cannot tell
which block holds the document it wants without decoding blocks, which is the
entire cost the table exists to remove. The last-identifier column is what makes
the table a skip list rather than a length index.

**Reordering the in-memory index to match.** Its dictionary serves the set
operations that querying is built from, and nothing that runs against it needs
to skip.
