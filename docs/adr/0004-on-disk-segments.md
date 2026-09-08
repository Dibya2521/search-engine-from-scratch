# ADR 0004: An index read where it lies, rather than loaded

- **Status:** accepted
- **Date:** 2026-09-08
- **Supersedes:** nothing
- **Superseded by:** nothing

## Context

Both index formats so far are read the same way: the whole file is decoded into
memory before a single query can be answered. The cost of a query is therefore
set by the size of the index rather than by the size of the question.

Measured over 20,000 documents with `uv run python benchmarks/at_scale.py`:
**20.0 seconds to load and 903 MB of memory**, neither of which depends on what
is being searched for.

Two further facts made this the right thing to fix next. Memory was already the
first thing measured to fail as a corpus grows, at a peak-to-source ratio of
9.9x and falling. And loading time was absent from the project's own list of
what breaks first, though it grows in step with the file at about 1.4 MB/s: a
1 GB index would take eleven minutes before answering anything.

An index that is loaded and rewritten whole also forecloses everything planned
after this. Segments that are written once and never modified are what make
atomic publication, crash recovery, merging, and caches that cannot go stale
possible at all.

## Decision

**Add a second format that is mapped and read in place, and keep the first.**

One segment is one immutable file. It carries a term dictionary that maps a term
to the bytes holding its postings, and a fixed-width footer at the end giving
the offset of every section.

- **The dictionary is front coded in blocks of 16.** Sorted terms share
  prefixes, so each entry stores only what changed. Blocks exist because front
  coding alone makes a term unreadable without decoding every term before it;
  each block starts afresh, so it can be decoded alone and the blocks can be
  binary searched.
- **The footer is last and fixed width**, so a reader seeks to a known distance
  from the end and learns where everything is. At the head it would require
  knowing every section's size before writing any of them.
- **Postings lists of at least 128 documents carry a skip list** of `isqrt(n)`
  entries, so finding a document does not mean decoding a list.
- **Two fields are stored that nothing yet reads**: each document's length in
  tokens, and each term's highest frequency in any one document. They cost a few
  bytes now and a format version later.
- **Both formats satisfy one protocol**, `ReadableIndex`. Querying and ranking
  depend on that and nothing else.
- **The checksum is verified on open by default**, and can be skipped.

## Consequences

**Opening is 156x faster and holds 189x less memory.** 1.3087 s and 207.4 MB
become 0.0084 s and 1.1 MB, from `uv run python benchmarks/on_disk.py`.

**Reading a term is about 1.4 ms slower**, because a binary search and a decode
cannot compete with a lookup in a dict that is already in memory. The two
approaches cross at roughly 950 distinct terms read from one open index. Below
that the segment wins, above it the loaded index does.

**This format is not faster.** It makes the cost proportional to the question
instead of to the index, which is what allows an index larger than memory to be
searched at all. Saying it is faster would be true only of the case that suits
it.

**Verifying the checksum reads every page**, which is what mapping the file was
meant to avoid. Measured at 2.8 ms for 2.79 MB, or 0.2 percent of what loading
costs, so it stays on by default and the cost is recorded rather than assumed.

**Skip lists cost 2.6 percent of the file**, 2.72 MB to 2.79 MB.

**One scorer had to change.** `BM25Ranker` recovered document lengths by walking
every posting of every term, which against a mapped file reads the whole thing
on the first query. Lengths are now stored in the segment and asked of the index
through the protocol, so the in-memory index computes them once on demand and
the segment reads them directly. This is the kind of coupling a protocol makes
visible: the scorer was reaching through the interface into the representation.

**Windows constrains the reader.** A mapped file cannot be deleted or replaced,
so a reader that outlives its usefulness blocks the next write, and every
failure during opening has to close the map before raising.

**Two formats now exist and both are maintained.** That is deliberate: the
comparison stays runnable, exactly as the plain-text writer was kept when the
binary format replaced it.

## Alternatives considered

| Rejected | Why |
| --- | --- |
| **A finite state transducer for the dictionary**, which is what Lucene uses | Much smaller, and prefix and wildcard queries come free. Also perhaps ten times the code, for a gain that is not the bottleneck here. Recorded as the upgrade path. |
| **A hash table on disk** | Constant-time lookup, but no ordered iteration, so no prefix queries, no range queries, and no merge join between segments. Merging is the next thing to be built and needs the order. |
| **Loading the dictionary into a dict at open** | Simplest, and it reintroduces the memory cost this format exists to remove. |
| **`pickle`** | Arbitrary code execution on load, no stability across versions, and it reads the whole file. Already rejected for the same reasons in ADR 0001. |
| **Replacing the whole-file format** | The comparison is the evidence. Keeping both costs a little code and preserves the ability to re-run the measurement. |
| **Interleaving positions with identifiers** | Would let a skip land on a document and read its term frequency immediately. It also breaks the property that identifiers are encoded exactly as the whole-file format encodes them, so the two could no longer be compared byte for byte. Worth revisiting when a scorer needs term frequencies without positions. |
| **Skipping the checksum by default** | It would make opening 38 percent faster and make a corrupt index undetectable. The measurement says the saving is 3 milliseconds. |
