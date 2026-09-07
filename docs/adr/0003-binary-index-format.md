# ADR 0003: A compressed binary index format, with a checksum and an analyzer stamp

- **Status:** accepted
- **Date:** 2026-09-07
- **Supersedes:** the format decision recorded in ADR 0001, in part
- **Superseded by:** nothing

## Context

The index format was plain text, one line per term, chosen for three stated
reasons: it is readable with `head`, it streams, and unlike `pickle` it cannot
execute code when loaded.

The recorded weakness was size. An index costs several times the text it was
built from, because a position is stored for every occurrence of every term, and
that is the direct price of phrase queries.

Both sequences the format stores are sorted, which is exactly the precondition
for delta encoding, and small numbers are exactly what variable-byte encoding
compresses well. The technique was therefore available and standard.

Two further problems were known and unaddressed:

- **A corrupt index is undetectable.** It returns wrong answers indefinitely.
- **A change to the analysis pipeline silently invalidates every index.** If the
  tokenizer, stopword list or stemmer changes, existing documents were analyzed
  one way and new queries another. Nothing raises. The engine returns nothing
  for queries that should match, and no test necessarily fails. This is the most
  common way a search system breaks quietly.

## Decision

**Add format version 2: binary, delta and variable-byte encoded, checksummed,
and stamped with a fingerprint of the analysis configuration. Keep reading
version 1.**

### Compression

Delta encoding then variable-byte encoding, in that order because the second
depends on the first. Byte aligned rather than bit aligned: Elias gamma and
Golomb compress better and decode several times slower, and decoding happens on
every query while bytes are saved once.

### A CRC32 checksum over the body

Accidental corruption is what this defends against, and it is not a defence
against a hostile file, since anyone who can alter the body can recompute the
checksum. Detecting accidental corruption is still worth having, because the
failure it prevents is silent and permanent.

### An analyzer fingerprint, and refusing a mismatch

A digest of the pipeline version, the token pattern, the stemmer variant and the
stopword list, which together determine every term produced. Loading an index
whose fingerprint differs raises `AnalyzerMismatchError`.

Refusing rather than warning. A warning on a silent-wrong-answer failure is a
warning nobody reads until the results have been wrong for a month.

### Version 1 still loads

`load` detects the format from the header. An index written by an older build
still opens, which is what any format change should do. Only `save_text` writes
version 1, and it is kept so the two formats stay comparable on the same index
rather than against a remembered number.

## Options considered

**Keep plain text and accept the size.** Rejected. Size is the recorded weakness
and the blocker for keeping the index on disk.

**Compress the whole file with gzip.** Rejected, and this is the interesting
rejection. It would be smaller and it would destroy the property that matters:
one term's postings could not be read without decompressing everything before
them. A format that must be read whole cannot be memory mapped, and that is the
next thing this format exists to enable.

**Keep the term dictionary readable and compress only the postings.** A genuine
option, and the mitigation named when this work was planned. Rejected for now
because it complicates the format for a benefit, `head`-readability, that the
header alone partly preserves. Worth revisiting when the term dictionary becomes
a separate structure, which the on-disk work will require anyway.

**Bit-aligned codes.** Rejected on the decode-speed argument above, without
measuring, which is worth admitting: the reasoning is standard and the
measurement was not made.

**Warn on an analyzer mismatch instead of refusing.** Rejected as above.

## Consequences

**What it bought,** measured with `uv run python benchmarks/index_format.py` on
10,000 documents and 1,000,000 stored occurrences:

| | v1 text | v2 binary | Change |
| --- | --- | --- | --- |
| File size | 6.629 MB | 2.722 MB | **-58.9%** |
| Bytes per occurrence | 6.63 | 2.72 | -58.9% |
| Save | 1.036 s | 1.084 s | +4.5% |
| Load | 1.231 s | 1.519 s | +23.4% |

**What it cost.**

- **`head` no longer works.** That was one of the three reasons the text format
  was chosen, and losing it is a real cost rather than an acceptable one. The
  header line stays ASCII so the file is still identifiable.
- **Load is slower**, by somewhere between 6 and 23 percent across repeated
  runs. Only the direction is reliable; the magnitude is inside the noise of
  this benchmark.
- **The body is built in memory before writing**, because the checksum covers it
  and cannot be known until it is complete.
- **Changing the stopword list, the tokenizer or the stemmer now invalidates
  every existing index.** That is the intended behaviour and it is still a cost.

**The finding worth carrying forward.** Compression is normally justified by
saying it saves time, because fewer bytes are read from disk. It did not save
time here. That argument holds when disk reading dominates, and a 6.6 MB file on
a warm page cache is not read from disk at all. **A performance claim carries
the conditions that made it true, and those conditions usually go unstated.**

**Two constant-factor fixes, both measured.** The first implementation was
+47.9% load and +136.6% save. Decoding sequences in one inline loop rather than
a function call per number measured 2.4 times faster for identical output, and a
single-byte fast path for gaps under 128 brought save back within noise. The
algorithm was right both times; the constant factor was not.

**What this enables.** An on-disk index, memory mapped, decoding only the
postings a query touches. At that point the load penalty measured here stops
being paid, because the whole file is no longer read. That is a prediction and
it will be measured when it is built.
