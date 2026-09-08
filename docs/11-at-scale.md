# What breaks first, at scale

## What it is

Six indexes, built at corpus sizes from 500 to 20,000 documents, with build
time, memory, vocabulary, file size and load time recorded at each. Reproduce
with:

```bash
uv run python benchmarks/at_scale.py
```

The point is not the largest number. It is the **shape**: which costs grow in
step with the corpus and which do not, because that is what decides what fails
first and at what size.

## Why it exists

Every claim this project had made about scale was extrapolated from a single
10,000 document run. An extrapolation is a belief about a curve you have seen
one point of, and there is no way to tell a straight line from a gentle bend
without a second point.

That mattered because a whole roadmap rested on it. The claim that memory is the
first thing to fail, and therefore that compression and an on-disk index come
before anything else, was derived from one ratio measured once.

**So the projections were checked. Most of them held.** That is worth saying
plainly, because a page that only reports what went wrong teaches the wrong
lesson about estimation.

## The measurement

| Documents | Source MB | Tokens | Vocabulary | Build s | Tokens/s | Peak MB | Index MB | Load s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 2.0 | 188,449 | 36,832 | 3.32 | 56,738 | 28 | 1.06 | 0.781 |
| 1,000 | 4.4 | 409,014 | 56,332 | 3.55 | 115,110 | 55 | 2.00 | 0.955 |
| 2,000 | 8.8 | 819,905 | 75,547 | 6.69 | 122,598 | 101 | 3.52 | 2.229 |
| 5,000 | 22.3 | 2,084,346 | 94,612 | 19.63 | 106,170 | 234 | 7.64 | 4.893 |
| 10,000 | 45.2 | 4,229,307 | 99,406 | 38.20 | 110,723 | 454 | 14.24 | 9.014 |
| 20,000 | 91.6 | 8,573,985 | 99,992 | 78.76 | 108,865 | 903 | 27.48 | 20.008 |

Peak memory is the `tracemalloc` peak during the build, measured in a second
pass with tracing on, because tracing costs time and a benchmark in this project
once inverted its own conclusion by having it on for one side and off for the
other. Build time is the best of three runs and load time the best of three
reads.

### Which of these numbers are reliable

Not all of them, and the difference is worth stating before anything is
concluded from them.

**The deterministic columns are exact.** Source bytes, token count, vocabulary,
peak memory and file size come from a fixed seed and are byte-identical across
runs. Two full runs produced the same values in every one of those cells.

**The timings are not, below about 5,000 documents.** The same two runs
disagreed like this:

| Documents | Build s, run 1 | Build s, run 2 | Disagreement |
| ---: | ---: | ---: | ---: |
| 500 | 2.09 | 3.32 | +59% |
| 1,000 | 4.66 | 3.55 | -24% |
| 2,000 | 7.53 | 6.69 | -11% |
| 5,000 | 19.02 | 19.63 | **+3%** |
| 10,000 | 36.47 | 38.20 | **+5%** |
| 20,000 | 77.33 | 78.76 | **+2%** |

A three second workload measures how busy the machine was as much as how fast
the code is, and taking the best of three runs does not fix it, as the 500
document row shows. **At and above 5,000 documents the two runs agree within 5
percent, so only those rows are used for any throughput claim below.** The
smaller rows are kept because their deterministic columns are exact and carry
the growth curve, which is what the page is for.

## What grows in step with the corpus, and what does not

| Documents | Peak memory / source | Index bytes / occurrence | Build s / source MB | Load s / index MB |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 14.0x | 5.62 | 1.6 | 0.737 |
| 1,000 | 12.7x | 4.89 | 0.8 | 0.477 |
| 2,000 | 11.5x | 4.29 | 0.8 | 0.633 |
| 5,000 | 10.5x | 3.66 | 0.9 | 0.641 |
| 10,000 | 10.0x | 3.37 | 0.8 | 0.633 |
| 20,000 | 9.9x | 3.20 | 0.9 | 0.728 |

**Build time is linear.** Seconds per source megabyte sit between 0.8 and 0.9
across every reliable row, and the throughput across the three of them is
**1.14, 1.18 and 1.16 MB/s**. The figure this project has carried since its
first benchmark is 1.1 MB/s, measured on a different corpus at a tenth of the
size, and it lands inside that range. Indexing one document costs the same
whether it is the hundredth or the twenty thousandth, which is what you would
expect from a dictionary of postings lists and is worth having confirmed rather
than assumed.

**Memory is not a fixed multiple of the source, and the roadmap assumed it
was.** The ratio falls from 14.0x to 9.9x and is still falling at the largest
size measured. The cause is amortisation: a term costs a dictionary entry, a
string, and a dictionary of postings lists whether it occurs twice or two
thousand times, and that fixed cost per term is a large share of a small index
and a small share of a large one.

So **9.4x was never the ratio, it is roughly the limit the ratio approaches.**
Used as a constant at small sizes it understates memory by up to 50 percent, and
used at large sizes it is close to right and slightly pessimistic. The
projections in the next section were built on it at large sizes, which is why
they survived.

**Index bytes per occurrence falls the same way and for the same reason**, from
5.62 to 3.20. The compressed format was separately measured at 2.72 bytes per
occurrence on a corpus with an eighth of the vocabulary and uniform document
lengths, and the gap between 2.72 and 3.20 is the cost of a larger vocabulary
and a wider length spread rather than a change in the format.

## What breaks first, in order

The original projections and how they fared. **Extrapolated from one point, and
they mostly held.**

| Corpus | What breaks | Projected | Measured or still projected |
| --- | --- | --- | --- |
| ~100 MB | Nothing yet | 90 s build, 940 MB RAM | **Measured at 91.6 MB: 78.8 s and 903 MB, which is 86 s and 986 MB at 100 MB. Both within 5 percent.** |
| ~1 GB | Memory | 9.4 GB for the index alone | Still projected. The ratio is below 9.9x and falling, so this is an upper bound rather than an estimate. |
| ~10 GB | Build time, single threaded | ~2.5 hours at 1.1 MB/s | Still projected, on throughput now confirmed at 1.16 MB/s over a 40x range. 10 GB is about 2.4 hours. |
| ~100 GB | One machine | ~940 GB | Still projected, and the least trustworthy, because nothing here is measured within three orders of magnitude of it. |
| Any size, high query rate | Query latency | 1.7 microseconds per candidate | Not measured here. This benchmark builds and reads indexes, it does not query them. |

**Memory still goes first**, and the reason it was worth confirming is that the
whole order of work after this depends on it: compression, then an on-disk
index, then segments, all of them before anything about distribution or
throughput.

Load time is worth one line of its own. It grows in step with the file, at
0.63 to 0.73 seconds per index megabyte, which is a read throughput of about
1.4 MB/s. Loading a 1 GB index would take roughly eleven minutes before the
first query could be answered. **That is not a memory problem and it is not on
the list above**, and it is the strongest argument for an index that is read
where it lies rather than loaded.

## Vocabulary, and why this corpus cannot answer the question

Vocabulary growth is normally described by **Heaps' law**: the number of
distinct terms grows as `K * N ** beta` in the token count `N`, with published
values for English of `K` between 10 and 100 and `beta` between 0.4 and 0.6. It
matters because index size is driven by vocabulary as much as by length, so any
projection of index size at a corpus you have not built rests on it.

Fitted over the six measurements by least squares on the logarithms:

```text
V = 1954.1 * N ** 0.257
```

| Documents | Vocabulary | Heaps predicts | Error |
| ---: | ---: | ---: | ---: |
| 500 | 36,832 | 44,273 | +20.2% |
| 1,000 | 56,332 | 54,025 | -4.1% |
| 2,000 | 75,547 | 64,593 | -14.5% |
| 5,000 | 94,612 | 82,089 | -13.2% |
| 10,000 | 99,406 | 98,453 | -1.0% |
| 20,000 | 99,992 | 118,053 | +18.1% |

**Both the exponent and the errors say the same thing: this is not a power law.**
`beta` at 0.257 is far below the published range, and the errors swing from
+20 percent through -15 percent and back to +18 percent, which is the signature
of a fit forced through a curve that is bending the wrong way rather than
scattering around a line.

The cause is visible in the vocabulary column. The generator draws from 100,000
distinct words, and by 10,000 documents the corpus has found 99,406 of them and
by 20,000 it has found 99,992. **It has run out of words.** Vocabulary cannot
keep growing because there is nothing left to discover, so the curve flattens by
construction and the fit reads that flattening as a small exponent.

**The conclusion, and it is the most useful sentence on this page: no
vocabulary-dependent projection can be made from this corpus.** Not index size
at an unbuilt scale, not dictionary size, not the memory a term dictionary would
need. The measurements above that do not depend on vocabulary, which is build
throughput and the memory and byte ratios, stand on their own. Anything that
does depend on it needs real text, and the honest answer until then is that it
is unknown.

This is a limitation of the generator rather than a defect in it. Its job is to
produce documents whose term frequencies follow a Zipf-like curve and whose
lengths span two orders of magnitude, which it does, and both of those are what
make the postings and compression measurements meaningful. Unbounded vocabulary
growth is a third property it was never built to have.

**What would settle it.** A real corpus, converted with
`scripts/make_corpus.py`, and the same benchmark run over increasing slices of
it. The comparison target from the source material this project started from is
41,141 documents, 301 MB of text and 616,723 distinct terms, which is a
vocabulary six times larger than this generator can produce from four times the
tokens.

## How it was measured

- **`varied_length_corpus`**, whose document lengths span 20 to 2,000 tokens.
  Uniform lengths would hide everything that depends on length, which includes
  the compression ratio and any scorer with a length prior.
- **A vocabulary of 100,000 words**, large enough that the Zipf draw keeps
  finding unseen terms for most of the range. A smaller one saturates sooner and
  measures the generator's ceiling even earlier than this one does.
- **Two passes per size**, one with `tracemalloc` off for timing and one with it
  on for memory, because the two cannot be measured in the same pass without the
  instrument changing the result.
- **The documents are generated before tracing starts**, so peak memory covers
  the index and the transient cost of analysis and nothing else.
- **Best of three** for build time and for load time. The fastest observed run is
  the one where the machine was doing least of anything else.

The range stops at 20,000 documents. 50,000 was measured first and dropped: at
the ratios above it needs about 2.8 GB of resident memory and around 29 minutes
for the passes, which is a benchmark nobody runs twice.

## How it connects to everything else

- **Compression** was justified by the memory ratio measured here, and the
  finding that the ratio falls means the justification is strongest at exactly
  the sizes where it matters.
- **An on-disk index** answers the load time in this table, which is the cost
  the "what breaks first" list did not contain.
- **Ranking** is untouched by any of this. Build and load costs do not move
  relevance, and nothing on this page says anything about whether results are
  good.
- **Any future claim about vocabulary or dictionary size** has to wait for real
  text. That is now written down so it is not quietly forgotten.
