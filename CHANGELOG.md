# Changelog

Everything notable that has changed in this project, newest first.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**While the major version is `0`, a minor bump may change a public interface.**
Anything that would be a breaking change after `1.0.0` is called out under
**Changed** or **Removed** with what to do instead.

Each entry says what moved and *why it moved*, because a list of features is a
worse record than a list of reasons. Where a change was made on the strength of
a measurement, the measurement is quoted and the command that produced it is
named, so the claim can be re-checked rather than believed.

## [Unreleased]

### Added

- **The source text is now kept beside each segment**, addressable by ordinal.
  The index holds nothing that could reconstruct a document, which is what makes
  it small and also why a result could name a document and score it but say
  nothing about what it says. Two files per segment: the documents as a
  length-prefixed title followed by a length-prefixed body, and a fixed-width
  offset file in which ordinal `n` sits at byte `8n`, so one seek reaches any
  document with no scan and no table held in memory.
- `IndexWriter.add` takes an optional `title`, kept for display rather than
  indexed. A caller that wants it searchable passes it inside the text as well,
  which is what indexing a corpus record already does.
- Merging carries the text across, renumbered into the merged segment's own
  ordinals. A document that survives a merge and loses its text has lost
  something no later pass can rebuild.
- **`search_engine.snippet`, the passage of a document that shows why it
  matched.** A window of 40 terms slides over the document and is scored on
  the number of distinct query terms first and their total occurrences second,
  so a passage showing all three query terms once beats one showing a single
  term five times. The best window is then widened to the sentence it sits in,
  because a passage that begins mid-sentence reads as damage rather than as an
  extract. Highlights come back as character ranges and not as marked-up text:
  a terminal wants ANSI codes, an HTTP response wants ranges in JSON, and a
  test wants neither.
- `tokenizer.tokenize_spans` and `analysis.analyze_spans`, returning each token
  or term with the range of text it came from. **The ranges index into the
  normalized text rather than into the original**, because normalization
  composes two code points into one and nothing maps an offset back across
  that. Both share their segmentation with the functions they mirror, so a
  highlighted word cannot drift away from a word that matched.
- **`search` prints the title of each result and the passage that matched**,
  with the matched words in bold when the output is a terminal and in plain
  text when it is not, because output that goes into a file should not carry
  escape codes.
- `--no-snippet`, printing one line per result with the document identifier on
  it, for a caller parsing the output rather than reading it.
- `--json`, printing one object per line carrying the identifier, title, score,
  snippet and highlight ranges. Nothing else goes to standard output in this
  mode, so a caller that asked for JSON is never handed prose.
- `index` writes the source text beside the index it builds, in both formats,
  which is what the passages are cut from.
- **`search_engine.metrics`: counters, gauges and histograms, exposed in the
  Prometheus text format.** Durations are counted into fixed buckets rather
  than averaged, because a mean hides the tail and the tail is what a person
  notices: a search averaging 20 ms can have a 99th percentile of four
  seconds. Bucket counts also add up across processes, and means do not.
- The engine records what it does: searches run, how long they took, how many
  raised, how many matched nothing, how many documents a scorer actually
  scored, how many segments are mapped, and the hit and miss counts of the
  query and postings caches. **Searches that match nothing are counted on
  their own**, because it is the cheapest proxy there is for relevance health.
- Every metric mutation takes a lock, because the serving layer will be
  concurrent and a count that is occasionally short is worse than no count: it
  is still believed.
- **`search_engine.logs`: one line of JSON per event, and one correlation
  identifier tying the lines of a request together.** Built on the standard
  library's `logging` rather than beside it, so an operator configures it the
  way they configure everything else. `json.dumps` escapes every newline,
  quote and backslash, so a query containing any of them cannot break the
  guarantee that one line is one record.
- `configure(stream, level, log_queries=False)` drops the query text from
  every record. **Logging a query is logging what a person searched for**,
  which in most places is regulated data with a retention policy attached. The
  switch is a filter on the handler, so the policy lives in the configuration
  and no code that logs has to know what it is.
- `correlated()` binds an identifier for a block, generating one when the
  caller has none. It is a `ContextVar` rather than a thread local, because
  that is correct under threads and coroutines alike.
- `search --log` writes those lines to stderr. Results stay on stdout, so a
  pipe gets one or the other and never both.
- **`search_engine.access`: a cost limit on queries, and a permit applied
  before the top k is chosen.** Both close defects rather than adding
  features.
- **Query cost is now bounded.** A phrase of fifty terms, or a query built from
  the commonest terms in the corpus, did work proportional to the whole index,
  which one client could use to occupy the engine indefinitely. Queries are
  judged by estimated cost rather than by shape, because a list of forbidden
  shapes is always one query behind whoever is writing them. **The estimate
  reads no postings**: document frequency already sits beside each term, put
  there so a scorer would not have to count, and it answers this too.
- `rank`, `rank_wand` and `search_wand` take a `permit`, deciding which
  documents a caller may see. **It is consulted before a document is scored**,
  not applied to the finished list.

### Changed

- **A directory written before this release cannot be merged.** Its segments
  have no stored text beside them, and a merge that quietly produced a segment
  without it would lose the text of everything it touched. The merge refuses and
  names the missing file. Rebuild the index.
- **The default output of `search` has changed shape.** It was one line per
  result carrying the document identifier; it is now the title, the score and
  the passage. `--no-snippet` prints exactly the old form, so anything parsing
  the output has one flag to add.
- **A title or a passage can no longer carry an escape sequence to a terminal.**
  A corpus is untrusted input, and every control character in it becomes a
  space before anything is printed. A character the console cannot encode is
  printed as its escape rather than ending the process.
- An index built before the text was stored beside it still searches. The
  command line says once, on stderr, that there are no snippets and falls back
  to printing identifiers. That is deliberately gentler than the merge path,
  where carrying on without the text would destroy it.
- `BaseRanker.rank` takes a `Collection` of candidates rather than an
  `Iterable`, so it can report how many it scored without consuming them
  twice. Every caller already passed a set.
- **A quantile read from a histogram is an upper bound, not the true value.**
  It is the bound of the bucket the quantile falls in, so it is never too
  small and is wrong by at most that bucket's width. An operator who does not
  know this will chase a regression that is a bucket boundary, so it is
  written in the docstring rather than left to be discovered.
- **Nothing is logged until logging is configured.** A library that writes to
  standard error whether or not it was asked to is one that has to be silenced
  before it can be used.
- **A result set is never logged**, only a line about one. It is unbounded, it
  is the largest thing in a request, and writing it is the fastest way to fill
  a disk.
- **Filtering results after selecting them is a data leak, and this is the
  release that stops it.** Ask for ten, get seven, and the three gaps have told
  you that documents you cannot see exist. With early termination it is worse:
  a forbidden document that reached the results sets the threshold, and pruning
  against a threshold set by documents you cannot see drops documents you can.
  A property test asserts that early termination and a full scan return exactly
  the same list under any permit.
- A query beyond the limits raises `QueryTooExpensiveError`, which the command
  line reports as bad input with exit code 2.

### Fixed

- **Publishing a manifest could fail outright on Windows.** `os.replace` is
  atomic there as it is on Unix, but being allowed to start it is not
  guaranteed: the move is refused while any other process holds the
  destination open, and a virus scanner or a file indexer holds a newly
  written file for a few milliseconds. A publish then raised `PermissionError`
  and an accepted document went unpublished.

  Observed once in six full test runs, on `manifest.json.tmp -> manifest.json`
  at generation 11 of a directory taking one document per segment. The move is
  now retried, backing off ten milliseconds at a time for at most ten
  attempts, and a destination still held after that is reported rather than
  hidden. Retrying does not weaken the atomicity, because a move either
  happened or did not.

### Measured

- Storing the text costs **4.16 times the size of the segment it sits beside**,
  and **0.26 percent more than the source text itself**, of which the offsets
  are exactly 8 bytes per document. Over 16,000 generated documents the whole
  on-disk footprint moves from 0.241 to 1.243 times the source, a factor of
  **5.16**. That cost is the entire argument against owning the text rather than
  seeking back into the corpus file, and it is why it is published rather than
  assumed. Reproduce with `uv run python benchmarks/document_store.py`.
- One metric update costs **186 ns for a counter and 341 ns for a histogram
  observation**, lock included, best of five runs of 200,000 through
  `timeit`, so both figures carry one Python call each. A search pays about
  900 ns of that fixed, against a per-candidate scoring cost of 1.7
  microseconds already measured. What the lock costs under real concurrency is
  a load-test question and is not answered by this.

## [0.6.0] - 2026-09-11

Text the engine could not read, and queries it could not understand.

Seven scripts produced no tokens at all before this release, so those documents
were unsearchable rather than merely ranked badly. Two spellings of the same
accented word produced different terms, so one could never match the other.
Both are fixed, and **every index built before this release is refused rather
than misread**: the terms are different, so an old file answers a different
question. To upgrade, rebuild it.

Alongside that, five attempts at understanding a query: spelling suggestions,
synonyms, entity segmentation, proximity and field weighting. **Two of the three
that could be measured could not be shown to help, and all of them are kept and
switched off**, which is now this project's answer three times running.

### Added

- **Unicode normalization before tokenizing.** The same word written two ways
  produced two different terms, so a document stored in one form could never
  match a query typed in the other. NFC is the default and NFKC is available as
  a parameter.
- **Text that is not ASCII is now indexed.** The token pattern produced zero
  tokens for Chinese, Japanese, Korean, Greek, Cyrillic, Hebrew and Arabic; those
  documents were unsearchable rather than degraded. A run of characters from a
  script written without spaces becomes overlapping character bigrams.
- `search_engine.spelling`, offering near spellings for a query that found
  nothing, through a BK-tree over edit distance. The command line prints them to
  stderr and keeps exit code 1.
- `search_engine.synonyms`, expanding equivalence groups at query time, and
  grouping adjacent words into one entity against a supplied phrase set.
- `search_engine.proximity`, scoring a document higher when its query terms
  appear close together. **Off by default.**
- `search_engine.bm25f`, BM25 over weighted fields, registered as
  `--scorer bm25f`. **Not the default.**
- **Fields.** A document can be indexed with its title recorded separately, as
  terms qualified `title:python` beside `python`. No format changed.
- `docs/16-unicode.md`, `docs/17-query-understanding.md` and ADR 0007.
- `benchmarks/unicode_forms.py`.

### Changed

- **Every index built before this release is refused rather than read.** The
  analyzer fingerprint changed, because normalization and the wider token
  pattern both alter the terms produced. An old index queried by a new build
  would return nothing for the affected documents with no error anywhere. There
  is no migration and there should not be: the terms are different, so the old
  file holds answers to a different question. **To upgrade, rebuild the index:**

  ```bash
  uv run search-engine index corpus.xml search.index
  ```

  A refused index names both fingerprints and says what to do:

  ```text
  index was built with analyzer 0da3e6c5c33c0e95, this build uses
  9190455cbe0ae89d; rebuild the index
  ```
- Stemming no longer runs on tokens outside ASCII. The stemmer implements
  English suffix rules and has no defined behaviour on other scripts. The
  stemmer itself is untouched.
- Early termination refuses a ranker built with the proximity boost, because it
  cannot reproduce a boost that depends on where terms sit and would otherwise
  disagree with a full scan silently.

### Measured

- **Three features could not be shown to help, and all three are kept and
  switched off.** The proximity boost: mean average precision +0.1 percent, sign
  test p **1.000**, nDCG 0.1 percent the wrong way. BM25F: mean average
  precision +0.3 percent, every metric up, sign test p **0.581**. The cause is
  the same both times and is not the features. Mean reciprocal rank on this
  collection is **0.9818** before either, so the first result is already
  relevant for nearly every query and a reordering has nowhere to show. **The
  default scorer is unchanged**, under the rule that a default changes on
  evidence and never on expectation.
- **The NFC against NFKC comparison could not be taken.** Across 24 sources
  holding 255,721 characters, only **21 lie outside ASCII** and no source
  changes under NFC at all. NFC is chosen on the conservative argument, that it
  cannot lose a distinction a reader can see, rather than on a measurement that
  did not happen.
- **Recording fields costs 10.0 percent of the vocabulary** and 9.4 percent of a
  whole-file index on the evaluation corpus, for 3.2 percent more postings. That
  is the price of the approach that changes no format.
- **BM25F reduces to BM25 to within 1.1e-16** when only the document field is
  weighted, which is one unit in the last place.

Reproduce with `uv run python benchmarks/ranking_quality.py` and
`uv run python benchmarks/unicode_forms.py`.

## [0.5.0] - 2026-09-10

An index that can grow, survive a crash, and answer a query without scoring
everything it holds.

The index becomes many immutable files published by a manifest, so a document
can be added without rebuilding the whole thing and a crash cannot lose one that
was accepted. Ranking learns to stop early, which is worth several fold on some
queries and costs on others; both are measured and both are stated. The segment
format changed to make the second of those possible, so **an index built by
0.4.0 must be rebuilt**.

### Added

- `search_engine.writer`, which buffers documents and flushes them as immutable
  segments, and `search_engine.directory`, which reads every published segment
  as one index. Both satisfy the same protocol as the in-memory index, so
  querying and ranking work against them unchanged.
- `search_engine.manifest`, the file that decides which segments exist. Replaced
  atomically, so a crash leaves the previous one whole. A segment it does not
  name is never opened, which is the whole of crash recovery.
- `search_engine.tombstones`, recording deleted documents beside a segment,
  since a posting cannot be removed from an immutable file.
- `search_engine.merge`, a tiered merge policy combining four segments of a tier
  into one, repeated until no tier is full. Deleted documents are dropped rather
  than carried across, which is where their space comes back.
- `search_engine.wal`, an append-only log holding accepted documents until they
  are durable in a published segment.
- `SegmentBuilder`, which writes a segment one term at a time, so a merge holds
  the vocabulary and the document list but never the postings.
- `docs/13-segments.md` and ADR 0005.
- `benchmarks/segments.py`, measuring what a query costs as segments accumulate.
- `search_engine.wand`, returning the best documents without scoring the rest.
  Each query term is walked by a cursor carrying an upper bound on what that
  term can contribute, and a document no combination of bounds can lift into the
  results is skipped rather than scored. Exact, not approximate: it returns what
  scoring every candidate returns, identifier for identifier and score for
  score, which is checked against generated indexes and queries.
- `BM25Ranker.upper_bound`, the most a term can contribute to any document,
  computed from its highest frequency and the shortest document in the corpus.
  Both are already stored, so the bound costs no postings read. It lives on the
  scorer so that it stays derived from the same `k1` and `b` as the score.
- `TopK` in `search_engine.ranking`, the same selection rule as `rank` for a
  caller producing scores one at a time and needing the current threshold before
  the last document has been seen.
- `search_engine.cache`, a bounded cache that evicts the least recently used
  entry and counts its hits and misses, used for parsed queries, decoded
  postings and ranked results. A miss returns a marker rather than `None`, since
  `None` is a value a caller may store and a cache that cannot tell the two
  apart recomputes it silently and forever.
- `CachedParser` in `search_engine.query`, which keeps the parses it has made.
  `parse` stays pure and the cache lives on an object, so nothing is held at
  module level and two callers can measure their own hit rates.
- `docs/14-early-termination.md`, `docs/15-caching.md` and ADR 0006.
- `benchmarks/early_termination.py`, comparing a full scan against both bound
  granularities and measuring what the postings cache is worth.

### Changed

- **Adding a document is now an upsert.** Ingestion is at-least-once, so the
  same document arrives twice and indexing has to be idempotent.
  `DuplicateDocumentError` stays on `InvertedIndex`, where a repeated identifier
  in a single-shot build is a bug in the corpus; the tolerance belongs in the
  writer, which is the layer that can make an older copy invisible.
- `SegmentReader` takes `cache=False`. A merge reads every term exactly once, so
  a cache holds everything it has already passed and buys nothing.
- **The segment format is now version 2, and version 1 files are refused.** A
  term's postings are an ordered sequence of blocks of 128 documents behind a
  table giving each block's last document identifier, size, highest term
  frequency and byte length, and delta encoding restarts at every block
  boundary. The separate skip list is removed: block boundaries are the skip
  points. **To upgrade, rebuild the index.** A version 1 file raises
  `SegmentVersionError` naming the version it holds, the version this build
  reads, and what to do, rather than being misread. The header is checked before
  the checksum, so an old file is reported as old rather than as corrupt. The
  reason for the change, and what it cost, are in ADR 0006.
- **The postings cache is a bounded least-recently-used cache**, replacing a
  dictionary that was emptied wholesale when it filled. `POSTINGS_CACHE_SIZE`
  moved to `search_engine.cache` and rose from 1,024 to 10,000. A term the
  segment does not hold now caches its empty result too, so a repeated query for
  an unknown word costs one dictionary lookup rather than one per repeat.

### Measured

- **Sixteen segments cost 46 percent more per term than one**, and two and a
  half times as much to open. That is a moderate cost at this scale rather than
  the dramatic one the usual argument implies, and it is stated that way. The
  reason to merge anyway is that segment count is otherwise unbounded.
- **Durability costs a factor of 25.5** in write throughput: 75,407 documents
  per second without `fsync` against 2,955 with it. It stays on by default,
  because a silently lost document is the worst failure a data system has.
- **Merging streams rather than materialising**, at less than a third of the
  peak memory of building the merged index first. An earlier version was 8.2x
  worse again, 6.9 MB against 842 kB, because the reader cached postings the
  merge reads once.

- **Early termination pays only under a condition, and the condition is
  stated.** Over 50,000 documents it runs 7.0x faster than a full scan when one
  query term is rare enough to drive the pivot, scoring 1,510 of 49,658
  candidates, and 14.1x faster when only the single best document is wanted. It
  runs **2.4x slower** when every query term is common, because nothing can be
  pruned and the cursor machinery is added on top of the same scoring work, and
  it loses on candidate sets of about a thousand. The code stays with the
  condition recorded, exactly as `save_text` and the naive phrase matcher do.
- **Per-block bounds cut documents scored from 50,000 to 138** on a corpus where
  one document holds a term 500 times and the rest hold it once, running 10.7x
  faster than a full scan. On a corpus whose document lengths are drawn
  independently they prune almost nothing more than a single bound per term,
  because a high frequency then lands in any block with equal probability and
  every block maximum ends up near the term's own. They pay when high
  frequencies are concentrated in few blocks.
- **The postings cache is load-bearing, not an optimisation.** Scoring asks the
  index for a term's postings once per document scored, so without the cache a
  reader decodes the whole list on every call and scoring is quadratic rather
  than linear in the candidates. Measured at 220x slower on 500 documents rising
  to 3,598x on 4,000: a ratio that grows with the corpus rather than a constant
  factor. Measuring it at 50,000 documents ran for 73.7 minutes at 94 percent of
  a core and produced nothing before it was stopped, which is the result rather
  than an obstacle to it.
- **Cache hit rates of 99.8 to 100 percent** on a mix that repeats one term.
  That characterises the benchmark and not production traffic, and it is
  published because a cache whose hit rate nobody has measured is memory spent
  on faith.

Reproduce with `uv run python benchmarks/segments.py` and
`uv run python benchmarks/early_termination.py`.

## [0.4.0] - 2026-09-08

The engine is measured at scale instead of extrapolated, and the index stops
having to be loaded before it can be read. Two of the projections this project
had been carrying were confirmed within 5 percent, one was corrected, and one
was found to be unanswerable from synthetic data at all.

### Added

- `benchmarks/at_scale.py`, which builds indexes at six corpus sizes from 500 to
  20,000 documents and records build time, memory, vocabulary, index size and
  load time at each. Written because every scale claim in this project had been
  extrapolated from a single 10,000 document run, and one point cannot
  distinguish a straight line from a curve.
- `docs/11-at-scale.md`, which publishes those measurements against the
  projections that preceded them and says which of the projections survived.
- `search_engine.segment`, an index written as one immutable file and searched
  where it lies. A term dictionary front coded in blocks of 16, binary searched
  through a block index, a fixed-width footer at the end giving every section's
  offset, postings decoded through `mmap` only when a term is asked for, and
  skip pointers on any postings list of at least 128 documents.
- `ReadableIndex`, the protocol both index formats satisfy. Querying and ranking
  depend on it and nothing more, which is why neither changed when the second
  format arrived.
- `InvertedIndex.document_length`, so a scorer with a length prior can ask the
  index rather than recover lengths by walking every posting.
- `search-engine index --on-disk`, writing a segment. `search-engine search`
  reads either format, deciding from the file's first bytes.
- `benchmarks/on_disk.py`, comparing the two formats and isolating what skip
  pointers are worth.
- `docs/12-on-disk-index.md` and ADR 0004, recording the format and the
  alternatives rejected.

### Changed

- **`BM25Ranker` asks the index for document lengths** instead of computing them
  from the postings. Against a mapped index the old path read the entire file on
  the first query, which would have undone the reason for mapping it.

### Measured

Reproduce the first group with `uv run python benchmarks/at_scale.py` and the
second with `uv run python benchmarks/on_disk.py`.

- **Build time is linear and the recorded throughput holds.** 0.8 to 0.9 seconds
  per source megabyte at every size, and 1.14, 1.18 and 1.16 MB/s at the three
  sizes whose timings are reliable. The 1.1 MB/s this project has carried since
  its first benchmark was measured once on a tenth of the corpus, and it lands
  inside that range.
- **Memory is not a fixed multiple of the source text.** The peak-memory to
  source ratio falls from 14.0x at 500 documents to 9.9x at 20,000 and is still
  falling, because the fixed cost of a term amortises over a longer postings
  list. The 9.4x figure previously used as a constant is closer to the limit the
  ratio approaches than to its value at any measured size.
- **The projection for a 100 MB corpus was right within 5 percent.** It predicted
  90 seconds of build time and 940 MB of memory; the measurement at 91.6 MB
  scales to 86 seconds and 986 MB.
- **Load time is 0.63 to 0.73 seconds per index megabyte**, a read throughput of
  about 1.4 MB/s, which grows in step with the file and was absent from the
  earlier list of what fails first.
- **The synthetic corpus cannot answer any question about vocabulary growth.**
  Fitting Heaps' law over the six sizes gives an exponent of 0.257 against a
  published English range of 0.4 to 0.6, with fit errors swinging from +20 to
  -15 percent. The generator draws from 100,000 distinct words and the corpus
  has found 99,992 of them by 20,000 documents, so vocabulary flattens because
  there is nothing left to discover. Every measurement here that does not depend
  on vocabulary stands; anything that does needs real text.
- **Opening a segment is 156x faster and holds 189x less memory** than loading
  the equivalent index: 0.0084 s and 1.1 MB against 1.3087 s and 207.4 MB, over
  10,000 documents.
- **Reading a term is about 1.4 ms slower**, because a binary search and a
  decode cannot beat a lookup in a dict that is already in memory. The two cross
  at roughly 950 distinct terms read from one open index. **This format is not
  faster; it makes the cost proportional to the question rather than to the
  index**, which is what allows an index larger than memory to be searched.
- **Skip pointers are worth 93.2 percent** on a 10,000 document postings list
  and 81.0 percent on one of 1,273, measured against walking the identifiers
  without them. On a list of 26, below the threshold at which one is written,
  they cost 4.9 percent. Most of the gain in finding a document comes from not
  decoding positions at all, which is 22x on the longest list and happens with
  or without a skip list; the figures above are what skipping adds on top.
- **Skip lists cost 2.6 percent of the file**, 2.72 MB to 2.79 MB.
- **Verifying the checksum costs 2.8 ms for 2.79 MB**, about 970 MB/s, which is
  0.2 percent of what loading the same index costs. It stays on by default,
  because a silently corrupt index returns wrong answers indefinitely.

## [0.3.0] - 2026-09-07

A compressed index format, and two guards against failures that were previously
silent. The measurement contradicted the textbook justification for the change,
which is recorded rather than smoothed over.

### Added

- `benchmarks/index_format.py`, measuring what an index costs on disk and how
  long it takes to read back. Added before the format changes, so the comparison
  afterwards is against a recorded number rather than a remembered one.

### Measured

- **The plain-text index format, as a baseline.** On a 10,000 document synthetic
  corpus of 10.2 MB with 8,000 distinct terms and 1,000,000 stored occurrences:
  file size 6.63 MB (0.65x the source text), 6.6 bytes per stored occurrence,
  1.902 s to save and 2.654 s to load, a load throughput of 2.5 MB/s. Reproduce
  with `uv run python benchmarks/index_format.py`.
- `search_engine.codecs`, variable-byte and delta encoding built from scratch.
- **Index format version 2**: binary, compressed, checksummed, and stamped with
  a fingerprint of the analysis configuration. `save` writes it; `load` reads
  version 1 as well, so an index written by an older build still opens.
- `search_engine.analysis.fingerprint`, a digest of everything that determines
  which terms get produced: the pipeline version, the token pattern, the stemmer
  variant and the stopword list.
- `save_text`, which writes the version 1 format, kept so the two stay
  comparable on the same index.
- `IndexCorruptError` and `AnalyzerMismatchError`.

### Changed

- **`save` now writes a binary file.** The index is no longer readable with
  `head`, which was one of three stated reasons for choosing a text format. The
  header line stays ASCII so the file is still identifiable. Use `save_text` for
  the old format.
- **An index built by a different tokenizer, stopword list or stemmer is now
  refused rather than loaded.** This is the guard against the most common silent
  failure in a search system: old documents analyzed one way, new queries
  analyzed another, nothing raising, and nothing matching. Changing the stopword
  list now invalidates every existing index, which is intended.

### Measured

- **The compressed format is 58.9 percent smaller**, 2.722 MB against 6.629 MB
  on 10,000 documents and 1,000,000 stored occurrences, or 2.72 bytes per
  occurrence against 6.63.
- **It is also slower to load**, by +23.4% on the recorded run and between +6%
  and +23% across repeated runs. Only the direction is reliable; the magnitude
  is inside this benchmark's noise. Save is within noise either way.
- **The textbook justification for index compression did not hold here.** It is
  usually said to save time because fewer bytes are read from disk. That holds
  when I/O dominates, and a 6.6 MB file on a warm page cache is not read from
  disk at all.
- **Two constant-factor fixes, both measured.** The first implementation was
  +47.9% load and +136.6% save. Decoding a sequence in one inline loop rather
  than a function call per number measured 0.0167 s against 0.0404 s for 200,000
  values, a factor of 2.4 for identical output. A single-byte fast path for gaps
  below 128 brought save back within noise.

## [0.2.0] - 2026-09-06

Relevance, measured rather than assumed. The headline is a null result: the new
ranking function did not beat the old one on this project's own test
collection, so the default did not change.

### Added

- This changelog. The engine reached a working state through thirteen commits
  with no running record of what changed between them, so the starting point
  below was reconstructed from the source and the commit history. Keeping it
  from here means the next change can always be compared against the one before
  it.
- A short contributing guide, stating the five rules the project runs on rather
  than a submission process, since there is one maintainer.
- A test collection: seventy documents in `tests/fixtures/evaluation_corpus.xml`
  and fifty-five queries with graded relevance judgements in
  `tests/fixtures/judgements.jsonl`. The existing seven-document fixture is
  large enough to assert postings by hand and far too small to tell two ranking
  functions apart, which is what this collection exists to do.
- `search_engine.judgements`, which reads a judgement file and refuses a
  malformed one. A file silently missing half its labels produces a plausible
  number that is wrong, which is worse than producing no number.
- Graded evaluation metrics: discounted cumulative gain and its normalized form,
  plus reciprocal rank and its mean. nDCG is the first metric here that uses the
  grade rather than a yes or no, which lets it separate two rankings that
  retrieved the same documents in a different order. Average precision scores
  both of those 1.0; nDCG scores them 1.0 and 0.80.
- `search_engine.bm25`, a BM25 scorer, and `--scorer` on the `search` command to
  choose between it and TF-IDF. BM25 saturates term frequency so repeated words
  stop helping, and corrects document length explicitly through a tunable
  parameter rather than implicitly through the cosine.
- `BaseRanker`, holding what does not depend on the scoring formula, so two
  scorers see the same candidates and the same selection rule and any difference
  in results is caused by the formula alone.
- `evaluation.sign_test`, an exact two-sided sign test, so a difference between
  two systems can be reported with whether it is distinguishable from chance.
- `benchmarks/ranking_quality.py`, which compares the two scorers over the
  judgement set and reports both a sign test and a paired permutation test.

### Changed

- Nothing about the default ranking. See below.

### Measured

- **BM25 did not beat TF-IDF on this collection, and the default did not
  change.** Mean average precision 0.7485 to 0.7586 (+1.3%), nDCG@10 0.8660 to
  0.8772 (+1.3%), mean reciprocal rank 0.9788 to 0.9818 (+0.3%), but
  precision@5 0.5921 to 0.5848 (-1.2%). Per query BM25 wins 14, loses 12, ties
  29. Sign test p = 0.845, paired permutation test p = 0.3275. The rule set down
  before the measurement was that BM25 would be adopted only if it proved
  better, so TF-IDF remains the default and BM25 ships as an option. Reproduce
  with `uv run python benchmarks/ranking_quality.py`.
- **The collection, not the scorer, is the limiting factor.** TF-IDF already
  achieves a mean reciprocal rank of 0.979, meaning the first result is relevant
  for nearly every query, so there is almost no headroom for a better scorer to
  occupy.
- **BM25 does fix the case that motivated it.** On the recorded example where a
  document is one query term repeated five times, its score falls from 72 percent
  of the winning document's to 52 percent.
- **The classic BM25 inverse document frequency would have produced negative
  weights on this corpus.** Three of 1,061 terms, worst case `document` at
  -0.704, present in 47 of 70 documents. A negative weight penalises a document
  for containing a query term, so the variant with an added constant is used
  instead.

## [0.1.0] - 2026-09-05

The first working engine: text goes in, ranked results come out, and every
mechanism in between was built rather than installed.

### Added

- **Tokenizer.** Lowercase, then keep maximal runs of `[a-z0-9]`. ASCII-only by
  design, which is a recall ceiling for non-Latin text and is documented as one.
  Measured at 21.1 MB/s.
- **Porter stemmer**, implemented from the published algorithm rather than
  imported, so that `connect` and `connecting` meet on one term. Verified
  against an independent implementation as a differential oracle: **zero
  disagreements across 34,814 words**. Measured at 128,688 words/s, 2.2 times
  the reference implementation.
- **Stopword filtering** over a published 25-word list, removing 41.9 percent of
  stored occurrences. Positions survive filtering on both sides, so phrase
  queries containing a stopword still match.
- **A single analysis pipeline** shared by indexing and querying. If those two
  paths ever diverge, queries ask for terms the index never stored and the
  engine silently returns nothing, which is the easiest way to break a search
  engine.
- **Inverted index with positional postings.** Each term maps to the documents
  containing it and the positions within them. Positions cost memory and buy
  phrase queries. Measured at 110,396 tokens/s, 1.1 MB/s, and 131 bytes per
  posting.
- **One-word, free-text and phrase queries.** A quoted query requires its terms
  adjacent and in order; an unquoted one matches any of them.
- **TF-IDF ranking with cosine similarity** over the vector space model, with
  top-k selection by heap. Measured at 1.7 microseconds per candidate, after a
  939-fold correction: scoring a document by building its full vector costs a
  pass over the whole vocabulary, which projected to 737 seconds per query
  before document vector lengths were precomputed once.
- **Index persistence** in a plain-text format, one line per term. Readable with
  `head`, streamable, and unable to execute code on load, which `pickle` is not.
- **Corpus reader** streaming `<page>` records a page at a time, written as a
  tolerant scanner rather than an XML parser so that an unwrapped dump still
  parses and entity-expansion attacks cannot apply.
- **Retrieval quality metrics**: precision@k, recall@k, average precision and
  mean average precision. These ask whether the results are *good*, which no
  amount of passing unit tests can answer.
- **Command line interface** with `index` and `search`, and exit codes that
  follow the shell convention so it composes in a pipeline.
- **Quality gates**: ruff format, ruff check, pylint, pyright strict and pytest
  with branch coverage, enforced by pre-commit and run in CI on Python 3.12,
  3.13 and 3.14.

### Notes on what this release deliberately does not do

Recorded here because a limitation you chose is worth more than a limitation you
did not notice.

- **No runtime dependencies.** `project.dependencies` is empty and stays that
  way. `numpy` was predicted as the likely first exception and measured to be
  unnecessary, because the document vectors are sparse and a dict of non-zero
  components beats a dense array.
- **Text that is not ASCII produces no tokens at all**, so such documents are
  unsearchable rather than merely degraded.
- **The same word in composed and decomposed Unicode form tokenizes
  differently**, so a document stored in one form and a query typed in the other
  will never match. A test pins this behaviour rather than hiding it.
- **Adding a document to a finished index is not possible.** The only way to add
  one is a full rebuild.

[Unreleased]: https://github.com/Dibya2521/search-engine-from-scratch/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/Dibya2521/search-engine-from-scratch/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Dibya2521/search-engine-from-scratch/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Dibya2521/search-engine-from-scratch/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Dibya2521/search-engine-from-scratch/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Dibya2521/search-engine-from-scratch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Dibya2521/search-engine-from-scratch/releases/tag/v0.1.0
