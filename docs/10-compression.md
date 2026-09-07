# Index compression

## What it is

Two techniques applied in order, which only pay off together.

**Delta encoding** replaces a sorted sequence with the gaps between consecutive
values. `[3, 8, 12, 40]` becomes `[3, 5, 4, 28]`. Nothing is saved by this step
alone, since a gap takes as much room as the value it replaced. What it does is
make the values small.

**Variable-byte encoding** stores an integer in as few bytes as it needs. Seven
bits of each byte carry data, and the top bit marks the last byte of a number:

```text
      5  ->  1000_0101                    one byte
    300  ->  0000_0010  1010_1100         two bytes
```

Applied to large numbers this saves nothing, which is why delta encoding comes
first. Applied to small ones it is close to optimal for a byte-aligned scheme.

## Why it exists

An inverted index is several times the size of the text it was built from,
because it stores a position for every occurrence of every term. That cost is
the direct price of being able to answer phrase queries.

The usual justification for compressing it is that a smaller file means fewer
bytes read from disk, and decompression is faster than the reading it saves.
**That justification did not hold here**, and the measurement below says why. It
is a claim with a condition attached, and the condition is that disk reading
dominates.

The reason compression is worth having in this project is different and more
specific: it is a prerequisite for keeping the index on disk and paging in only
what a query touches. A format you cannot fit is a format you have to load
entirely.

## Why the numbers get small

Both sequences an index stores are sorted, which is what makes delta encoding
work at all:

- **Positions within a document** are strictly increasing. In a 100-term
  document, every gap is between 1 and 100, so every gap fits in one byte.
- **Document identifiers within a postings list** are stored in sorted order.
  For a common term the gaps are tiny; for a rare term in a large corpus they
  are large, and the technique buys less.

The second case is worth stating plainly because it is where the technique
fails. There is a test asserting exactly that: values spaced further apart than
they are large compress no better than the values themselves. **Compression is a
property of the data, not of the code.**

## How it works internally

### Encoding one number

Split the value into 7-bit chunks, most significant first, and set the top bit
of the final chunk:

```text
300 = 0b100101100
    chunk 1: 0b0000010  ->  0x02
    chunk 2: 0b0101100  ->  0xAC, plus the terminator -> 0xAC | 0x80
```

Writing the most significant chunk first means a decoder can accumulate the
value as it goes and know it has finished when it sees a set top bit. No length
prefix is needed anywhere in the file, which is what makes the format readable
straight through.

### Decoding

Shift the accumulator left by 7 and add the low 7 bits of each byte. When a byte
has its top bit set, the number is complete.

**A number that never terminates is an error, not a smaller number.** Without
that check, a file truncated mid-number decodes to a plausible value and
produces a wrong index rather than a failure.

### Why byte alignment

Bit-level codes such as Elias gamma and Golomb compress better, because they do
not waste the padding that byte alignment implies. They are also several times
slower to decode, since every value requires bit shifting across byte
boundaries.

For an index the cost that matters is decoding, which happens on every query,
against bytes saved, which happens once. Byte alignment is the standard choice
in production systems for that reason.

## The file format

Version 2 of the index format is binary:

```text
header      the ASCII line "search-engine-index v2", newline terminated
analyzer    length-prefixed fingerprint of the analysis configuration
checksum    CRC32 of everything after it, four big-endian bytes
documents   count, then delta and variable-byte encoded identifiers
terms       count, then for each term in sorted order:
              length-prefixed ASCII term
              count of documents, then their delta-encoded identifiers
              for each document: count of positions, then their deltas
```

Three things it adds beyond size.

**A checksum.** A silently corrupt index returns wrong answers indefinitely,
which is worse than failing to open. CRC32 catches accidental corruption, which
is what this is for. It is not a defence against a hostile file, because anyone
who can alter the body can recompute the checksum.

**The analyzer fingerprint.** A digest of everything that determines which terms
get produced: the pipeline version, the token pattern, the stemmer variant and
the stopword list. Loading an index whose fingerprint differs from the current
build raises.

This is the guard against the most common way a search system fails quietly.
Change the tokenizer, and old documents are analyzed one way while new queries
are analyzed another. Nothing raises, no test necessarily fails, and the engine
returns nothing for queries that should have matched. Recording the
configuration turns a silent wrong answer into a loud error.

**Version 1 still loads.** `load` detects the format from the header, so an
index written by an older build still opens. Only `save_text` writes it, kept so
the two formats can be measured against each other rather than against a
remembered number.

## What it cost and what it bought

```bash
uv run python benchmarks/index_format.py
```

10,000 documents, 10.2 MB of source text, 8,000 distinct terms, 733,596 postings
entries and 1,000,000 stored occurrences:

| | v1 text | v2 binary | Change |
| --- | --- | --- | --- |
| File size | 6.629 MB | 2.722 MB | **-58.9%** |
| Against source text | 0.651x | 0.267x | -58.9% |
| Bytes per occurrence | 6.63 | 2.72 | -58.9% |
| Save | 1.036 s | 1.084 s | +4.5% |
| Load | 1.231 s | 1.519 s | **+23.4%** |

**The size figure is exact and repeatable.** It is a property of the bytes.

**The timing figures are noisy and must be read as a range.** Across repeated
runs of identical code, the load penalty was between +5.9% and +23.4%, and save
between +3.8% and +64.0%. Only the direction is reliable: version 2 is not
faster.

### The finding that contradicts the textbook

Compression is usually justified by saying it *saves* time, because fewer bytes
are read from disk. **Here it costs time.**

The reason is that the standard argument has a condition attached, and the
condition does not hold. It holds when disk reading dominates. A 6.6 MB file on
a warm page cache is not read from disk at all, so there is nothing to save, and
what remains is the decoding work, which text parsing does not have to do.

The general form of the lesson: **a performance claim carries the conditions
that made it true, and those conditions are usually not stated.** Compression
saves time when I/O dominates. At a scale where it does not, it is a
straightforward trade of time for space.

### Two optimisations that were measured rather than assumed

The first version was much worse, at +47.9% load and +136.6% save. Both were
fixed by changing how the loop is written, not what it computes.

**Bulk decoding.** Reading postings called a function per number. Rewriting the
loop inline for the sequence path measured 0.0167 s against 0.0404 s for 200,000
values, a factor of **2.4** for identical output. That took the load penalty
from +47.9% to roughly +20%.

**A single-byte fast path when encoding.** After delta encoding, most gaps are
below 128 and fit in one byte, so appending that byte directly avoids building a
list and a `bytes` object per value. Save went from +136.6% to within noise.

Both are cases where the algorithm was already right and the constant factor was
not.

## When to use this

- **When the index has to fit somewhere it currently does not**, in memory or on
  disk. This is the real reason.
- **When I/O genuinely dominates**, meaning a large index, a cold cache, or a
  network filesystem. Then the textbook justification applies and compression
  wins on both size and time.

Not worth it when the index is small and already cached, and load latency is
what you care about. At that point it is pure cost.

## Alternatives

| Scheme | Size | Decode speed | Notes |
| --- | --- | --- | --- |
| **Plain text (v1)** | largest | fastest here | readable with `head`, which is a real advantage |
| **Variable byte (v2)** | ~40% of text | fast | byte aligned, simple, the common choice |
| **Elias gamma / delta** | smaller | slower | bit aligned, no byte-boundary padding |
| **Golomb / Rice** | smallest for a known distribution | slowest | needs a parameter fitted to the gaps |
| **Frame of reference, PFOR** | small | fastest at scale | encodes a block against a common base, vectorises well |
| **Generic compression, gzip** | small | slow, and undirected | cannot decode one postings list without the rest |

The last row is the one worth understanding. A general-purpose compressor would
shrink the file further, and it would destroy the property that matters: you
could no longer read one term's postings without decompressing everything before
it. **A format that must be read whole cannot be memory mapped**, and that is the
next thing this format is for.

## How it connects to everything else

- **The inverted index** supplies the sorted sequences that make delta encoding
  work. Positions are increasing because a document is read in order; document
  identifiers are sorted at write time specifically so the same technique
  applies.
- **The analysis pipeline** is what the fingerprint identifies. Compression and
  analyzer versioning arrived together because both are properties of the file
  rather than of the search.
- **An on-disk, memory-mapped index** is what this format exists to enable. Once
  postings are addressable individually, only the ones a query touches need to
  be decoded, and the load penalty measured above stops being paid at all. That
  is a prediction, not a measurement, and it will be measured when it is built.
- **The stopword list is part of the fingerprint**, so changing it invalidates
  every existing index. That is correct: it changes which terms exist.
