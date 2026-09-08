# The on-disk index

## What it is

A segment: one immutable file holding an index, searched where it lies instead
of being read into memory. Opening one maps the file and reads a small table of
offsets. A query decodes the postings of the terms it asks for and nothing else.

```bash
uv run search-engine index corpus.xml index.seg --on-disk
uv run search-engine search index.seg "web search"
```

The same `search` command reads either format. It decides which by reading the
first bytes of the file, because the two headers differ.

## Why it exists

Loading an index decodes every posting before it can answer anything, so the
time and the memory a single query costs are set by the size of the index rather
than by the size of the question. Measured over a corpus of 20,000 documents:
**20.0 seconds to load and 903 MB of memory**, from
`uv run python benchmarks/at_scale.py`.

Neither number depends on what is being searched for. A one-word query over a
million-document index pays for the million documents.

The second reason is that this is what makes everything after it possible.
Segments that are written once and never modified can be published atomically,
recovered from a crash by ignoring what nothing names, cached without any risk
of the cache going stale, and merged in the background. None of that is
available to a format that has to be loaded and rewritten whole.

## How it works internally

### The file

```text
+------------------------------------------------------------+
| header       "search-engine-segment v1\n"                    |
| fingerprint  the analysis configuration that built it        |
+------------------------------------------------------------+
| postings     one block per term, in sorted term order        |
+------------------------------------------------------------+
| dictionary   the terms, front coded, in blocks of 16         |
+------------------------------------------------------------+
| block index  the first term of each block, and its offset    |
+------------------------------------------------------------+
| documents    the identifiers, and each one's length in tokens|
+------------------------------------------------------------+
| footer       eight 8-byte offsets, then a 4-byte checksum    |
+------------------------------------------------------------+
```

### Why the footer is last, and fixed width

A reader seeks to 68 bytes from the end, reads the footer, and from there knows
where every section is. Two bytes of arithmetic replace a scan.

At the head it could not be written without knowing the size of every section
first, which means either two passes over the index or building the whole file
in memory before writing any of it. Both give up the thing the format exists
for. Lucene puts its footer at the end for the same reason.

### Front coding

Terms are sorted, so neighbours share prefixes. Each entry records only what
changed:

```text
term          shared   suffix     stored
search             0   "search"   0, 6, "search"
searcher           6   "er"       6, 2, "er"
searching          6   "ing"      6, 3, "ing"
```

Shared prefixes are counted in **bytes, not characters**. A prefix measured in
characters can end in the middle of a multi-byte code point and produce a term
that is not valid text. There is a test that indexes terms differing inside a
three-byte character to pin it.

### Blocks, and why they exist

Front coding alone makes a term unreadable without decoding every term before
it, which is a scan of the whole dictionary for one lookup. So the dictionary is
cut into blocks of 16, and the first entry of each block stores its term in full
and an absolute offset. A block can therefore be decoded on its own.

A lookup then costs a binary search over one entry per block, then a scan of at
most 16 terms. For a vocabulary of 616,723 terms that is 38,545 block entries,
about 16 comparisons, and 16 short decodes.

### Reading through mmap

`mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)` maps the file into the
address space without reading it. The operating system faults in a page the
first time something touches it, so a query pays for the pages holding the terms
it asked about and nothing else. Slicing is done through a `memoryview`, so
reaching a term's postings copies nothing.

The file handle stays open for the life of the map. Closing it first works on
Unix and raises on Windows, and on Windows a mapped file cannot be deleted or
replaced at all, so a reader that is not closed blocks the next write. Every
failure during opening closes the map before it raises.

### Skip pointers

A postings list of at least 128 documents carries a skip list at the front of
its block: `isqrt(n)` entries, each a document identifier, the byte just past
its gap, and how many identifiers precede it. Finding the first document at or
after a target jumps to the last skip that cannot have passed it and walks from
there.

The square root is the classic choice. With `k` skips over `n` documents, a
search scans `k` skip entries and `n / k` identifiers, and `k = sqrt(n)`
minimises the sum, giving `O(sqrt(n))` for both halves.

The third field, how many identifiers a skip has consumed, is what tells the
walk after it when to stop. Without it the walk would run past the identifiers
and into the positions stored behind them, and report a document that does not
exist.

## What it cost and what it bought

10,000 documents, an 8,000 word vocabulary, best of five, from
`uv run python benchmarks/on_disk.py`:

| Reader | File MB | Open s | Memory after open MB | Cold read of 300 terms s | Memory after reads MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| whole file | 2.72 | 1.3087 | 207.4 | 0.0001 | 207.4 |
| segment, verified | 2.79 | 0.0084 | 1.1 | 0.4093 | 46.9 |
| segment, unverified | 2.79 | 0.0052 | 1.1 | 0.3806 | 46.9 |

**Opening is 156x faster and holds 189x less memory.** Reading a term is far
slower, because a binary search and a decode cannot compete with a lookup in a
dict that is already in memory.

### What got worse, stated plainly

**Per term, the segment is around 1.4 milliseconds slower.** Loading the index
makes every subsequent term free; mapping it makes every term cost something.

Which wins depends only on how many distinct terms one open index is asked for:

```text
whole file   1.3087 s + 0.0000 s per term
segment      0.0084 s + 0.0014 s per term
```

They cross at roughly **950 terms**. Below that the segment is ahead, and for a
single query it is ahead by a factor of about 150. Above it, an index that is
loaded once and queried forever is the better shape, which is exactly the
workload a long-lived server has.

The honest summary is that this format is not faster. It makes the cost
proportional to the question instead of to the index, and that is what allows an
index larger than memory to be searched at all.

## Checking the file is intact

The footer holds a CRC32 of everything before it. Verifying it reads every page
of the file, which is precisely what mapping the file was meant to avoid, so
this is a choice rather than a default that goes unexamined.

It was measured: **2.8 ms for 2.79 MB**, about 970 MB/s. That is 51 percent of
the open time and 0.2 percent of what loading the same index costs.

So verification is on by default. A silently corrupt index returns wrong answers
indefinitely, which is worse than failing, and at this price there is no
argument against paying. `SegmentReader(path, verify=False)` is available for a
caller who has measured their own case and wants the 3 milliseconds back.

## When this is the right choice

| Prefer | When |
| --- | --- |
| The segment | The index does not fit in memory, or a process opens it to answer a few queries and exits, or start-up time matters |
| The whole file | The index fits comfortably, the process is long lived, and it will answer thousands of queries from one open index |

Below about 950 distinct term reads per open, the segment wins. Above it, the
whole file does. That number is from one corpus on one machine and should be
re-measured rather than believed if the decision matters.

## What holds it together

Both formats satisfy the same protocol, `ReadableIndex`: postings, document
frequency, document length, membership, and four properties. Querying and
ranking depend on that and on nothing else, which is why neither of them changed
when this format was added. The command line picks a reader from the file's
header and hands it to code that cannot tell the difference.

`BM25Ranker` needed one change. It recovered each document's length by walking
every posting of every term, which against a mapped file would read the whole
thing on the first query. Document lengths are now stored in the segment and
asked of the index, so the in-memory index computes them once on demand and the
segment reads them.

## Alternatives

| Rejected | Why |
| --- | --- |
| A finite state transducer for the dictionary, as Lucene uses | Much smaller, and it gives prefix and wildcard queries for free. It is also perhaps ten times the code. Sorted blocks with binary search are the honest first version, and this is the upgrade path. |
| A hash table on disk | Constant-time lookup, but no ordered iteration, so no prefix queries, no range queries, and no merge join between two segments. Merging is the next thing to be built and it needs the order. |
| Loading the dictionary into a dict at open | The simplest thing that works, and it reintroduces exactly the memory cost this format exists to remove. |
| `pickle` | Arbitrary code execution on load, no stability across versions, and it reads the whole file. |
| Interleaving positions with identifiers | Would let a skip land on a document and read its term frequency immediately. It also breaks the property that identifiers are encoded exactly as the whole-file format encodes them, so the two could no longer be compared. Worth revisiting when a scorer needs term frequencies without positions. |

## How it connects to everything else

- **Compression** made the file small enough that mapping it is worthwhile, and
  the same codecs encode both formats.
- **The analyser fingerprint** is stored in the header, so a segment built by a
  different tokenizer, stopword list or stemmer is refused rather than silently
  mismatched.
- **Ranking** did not change, because both formats satisfy one protocol.
- **Skip pointers** have no caller yet beyond the benchmark. They exist because
  finding a document without decoding a list is what an intersection needs, and
  the measurement above is what justifies keeping them.
