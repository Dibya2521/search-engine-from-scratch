# Segments, merges and recovery

## What it is

An index kept as many immutable files rather than one, with a manifest deciding
which of them exist, a log holding what has been accepted but not yet written,
and a background policy combining small files into larger ones.

```text
directory/
  manifest.json              the only file that decides what exists
  segment-00000001.seg       immutable, written once, never edited
  segment-00000001.del       tombstones for that segment
  segment-00000004.seg
  wal.log                    accepted, not yet in a segment
```

Five rules, and everything else follows from them:

1. **A segment file is immutable.** Once written it is never modified.
2. **The manifest is the truth.** A segment file the manifest does not name does
   not exist, whatever is on the disk.
3. **The manifest is replaced atomically**, never edited in place.
4. **Deletes are tombstones**, because a posting cannot be removed from an
   immutable file.
5. **A write is durable when it is in the log**, not when it is in a segment.

## Why it exists

Before this, adding one document meant rebuilding everything. `add_document`
refuses an identifier it has already seen and a ranker refuses an index that
changed under it, both on purpose, so there was no way to grow a corpus except
from the start. At the measured 1.1 MB/s that is about four and a half minutes
per document on a 300 MB corpus.

The deeper reason is that immutability buys four things at once, none of which
is available to a format that is loaded and rewritten whole:

- **Atomic publication.** A new segment becomes visible in one step.
- **Crash recovery by construction.** A half-written file is one nothing names.
- **Concurrent reads with no locking.** The files a query is reading cannot
  change under it.
- **Caches that cannot go stale.** An entry keyed by a segment or a generation
  describes something that will never be different.

The fourth is worth dwelling on. Cache invalidation is usually the hard part,
and here it disappears, not by cleverness but because a decision made about file
layout removed the problem three components away.

## How it works internally

### The write path

Documents accumulate in memory as text. When the buffer fills they are analysed
into an ordinary in-memory index and written out as a segment. Holding text
rather than an analysed index means replacing a document before it is written
costs nothing, and analysis happens once per document rather than once per
revision.

The order of a flush is the whole of its crash safety:

1. Append the document to the log, and do not return until it is on the disk.
2. Write the segment file.
3. Publish a manifest naming it.
4. Discard the log.

A crash between 2 and 3 leaves a file nothing names, which the next open
ignores, and the documents are still in the log. A crash between 3 and 4 replays
documents that are already indexed, which is harmless because adding is an
upsert. Publishing before writing would name a file that does not exist;
discarding the log first would lose everything in it.

### Upsert, and where tolerance belongs

Ingestion in any real system is at-least-once, so the same document arrives more
than once and indexing has to be idempotent. `DuplicateDocumentError` is exactly
the wrong behaviour there.

It is kept where it is right. On the in-memory index a repeated identifier in a
single-shot build is a bug in the corpus and still raises. Tolerating repeats
belongs in the writer, the layer that knows how to make an older copy invisible.
Making an error go away and putting the tolerance at the correct layer look
identical in a diff and are not the same thing.

### Tombstones, and why deleting frees nothing

A segment is immutable, so a delete records that a document is no longer
visible and queries filter it out. The space returns when a merge simply does
not carry the document across. **This is why deleting from a real search engine
does not free disk immediately.**

The bitset is over *ordinals*, meaning a document's position in the segment's
sorted list, not over identifiers. Identifiers are arbitrary and may be sparse,
so a bitset over them could be enormous for a segment holding three documents.
Ordinals are dense by construction, so it is always one bit per document.

The tombstone file is the one mutable thing in a design built on immutability.
That exception is worth naming rather than glossing: it never changes what a
posting means, only whether a document is visible, so nothing immutability buys
is lost. It is still replaced atomically rather than edited where it lies.

### The manifest

JSON, written to a temporary file and moved into place. `Path.replace` is atomic
on Unix and Windows alike, unlike a rename over an existing file, so a reader
sees one whole generation or the other and never half of each.

JSON rather than this project's own codecs, deliberately: it is tiny, it is read
once per open, and being able to read it during an incident is worth more than
the bytes.

Every publication advances a generation counter, and a reader keeps the
generation it opened at for its whole life.

### Merging

A segment belongs to a size tier, tiers being powers of four, and when a tier
holds four segments they are combined into one. That usually lands in the tier
above and may fill it in turn, so the policy is applied until it stops firing.

A tier is decided by *live* documents, so a segment whose contents are mostly
deleted counts as the small segment it effectively is and gets merged away
sooner, which is when its space comes back.

The merge walks every input's terms at once with `heapq.merge`, which is why the
dictionary is stored in sorted order and why a hash table on disk was rejected.
Postings go to the output as each term is finished, so a merge holds the
vocabulary and the document list but never the postings, which are almost all
the bytes.

Then: write the output, publish a manifest naming it and not the inputs, and
only then delete the inputs. A crash between the last two steps leaves orphans
that waste space and confuse nothing.

### The log

Append only, one record per accepted operation, each framed as a length, a
payload, and a checksum of the payload. The length comes first because the
reader needs it to know how much to read; the checksum comes last because it has
to cover everything just read.

A record whose length runs past the end of the file, or whose checksum does not
match, is the torn tail of a crash. Replay stops there rather than guessing,
because the bytes after it cannot even be located: the length that would locate
them is the part that is missing.

Deletions are logged as well as additions. A delete that was not recorded would
be undone by the next recovery, which is data loss in the other direction.

## What it cost, measured

### Segments make queries slower, and by how much

`uv run python benchmarks/segments.py`, 4,000 documents written at several
buffer sizes so that nothing differs except how many files they land in:

| Segments | Open s | Read 200 terms s | Per term ms | Against one segment |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0040 | 0.0125 | 0.062 | 1.00x |
| 2 | 0.0056 | 0.0129 | 0.065 | 1.04x |
| 4 | 0.0073 | 0.0145 | 0.073 | 1.17x |
| 8 | 0.0104 | 0.0152 | 0.076 | 1.22x |
| 16 | 0.0099 | 0.0182 | 0.091 | 1.46x |

**Sixteen segments cost 46 percent more per term than one, and two and a half
times as much to open.** That is the cost merging exists to remove, and it is
worth stating plainly that it is a moderate cost rather than the dramatic one
the usual argument implies at this scale. The growth is real and it is not
alarming, and a merge policy is still worth having because the segment count is
otherwise unbounded: nothing here says what 1,000 segments would cost, and the
curve does not look like it flattens.

### Durability costs a factor of 25

Forcing every accepted document to the disk before acknowledging it, over 2,000
documents:

| | Documents per second | Throughput |
| --- | ---: | ---: |
| Without `fsync` | 75,407 | 305.69 MB/s |
| With `fsync` | 2,955 | 11.98 MB/s |

**That is what durability actually costs**, and it is the largest single price
in this component. It is on by default, because a silently lost document is the
worst failure a data system has, and `sync=False` exists so the comparison stays
runnable rather than being argued about.

### Merging holds the vocabulary, not the postings

Building the merged index in memory before writing it was measured at more than
three times the peak of streaming it, over 240 documents of 200 tokens.

An earlier version of the merge was 8.2 times worse still, 6.9 MB against 842
kB, because the segment reader cached every posting it decoded. A merge reads
each term exactly once, so the cache held everything it had already passed and
bought nothing. Merging now opens its inputs with caching off.

## What this does not provide

**Durability of the directory entry.** `fsync` on a file does not guarantee the
directory entry naming it is durable on POSIX, and there is no portable way to
`fsync` a directory on Windows. A crash immediately after a create could in
principle leave a file the directory does not list. This is stated rather than
claimed away, because a reliability guarantee that has not been demonstrated is
worse than none.

**A vocabulary count that excludes deleted documents.** The count is of terms
the files hold, so a term left behind by a deleted document is counted until a
merge drops it. Establishing otherwise would mean reading its postings, which is
a high price for a number nothing scores with.

**Concurrent writers.** One writer per directory. Two would each hold their own
idea of the manifest generation and the last to publish would silently discard
the other's work. Nothing enforces this yet.

## Alternatives

| Rejected | Why |
| --- | --- |
| **Rebuild the index on every change** | What this project did until now. Correct, simple, and four and a half minutes per document at 300 MB. |
| **Mutable postings files, edited in place** | Needs locking on every read, has no atomic publication, no crash recovery by construction, and no cache that cannot go stale. Immutability buys all four at once. |
| **Levelled merging**, one segment per level | Better read amplification, much worse write amplification, because a document is rewritten every time its level is touched. Wrong side of the trade for an index written in bulk and read constantly. |
| **Deleting by rewriting the segment** | Turns a constant-time delete into one proportional to the segment, and gives up immutability to do it. |
| **A database for the manifest** | A runtime dependency, for a file read once per open and small enough to read with `cat`. |
| **Logging only additions** | Specified that way and changed. A delete that was not recorded would be undone by the next recovery. |

## How it connects to everything else

- **The on-disk format** is what a segment is. Merging depends on its dictionary
  being ordered, which is exactly why a hash table was rejected there.
- **Upsert is what makes recovery simple.** Replaying a published document is a
  no-op only because adding one twice already was.
- **Early termination** will read postings without materialising them, which the
  skip pointers in the segment format exist for.
- **Caching** becomes trivial here rather than hard: a key holding a segment
  name or a generation cannot describe something that has changed.
