# Operations

This page is written for the person reading it at three in the morning, when
something is wrong and the engine has to explain itself. Everything here is
about what the engine says while it runs, what those numbers do and do not
mean, and what to do when one of them moves.

**The runbooks at the end were written before the incident, not during it.**
That is the entire reason they exist.

## Turning it on

Nothing is emitted until it is asked for. A library that writes to standard
error whether or not it was asked to is one that has to be silenced before it
can be used.

```python
from search_engine.logs import configure, correlated
from search_engine.metrics import REGISTRY

configure(sys.stderr)  # one JSON object per line
configure(sys.stderr, log_queries=False)  # the same, without the query text
print(REGISTRY.render())  # the Prometheus text exposition format
```

From the command line, `search --log` writes the same lines to standard error.
Results stay on standard output, so a pipe gets one of them and never both
interleaved.

There is no HTTP endpoint here. `Registry.render` returns the exposition as a
string, and a serving layer is what puts it behind a URL.

## What the engine records

| Metric | Kind | What it is | What it means when it moves |
| --- | --- | --- | --- |
| `queries_total` | counter | searches run | Throughput. A fall with no deploy means traffic stopped arriving, which is a problem upstream of here. |
| `query_duration_seconds` | histogram | how long a search took, text to identifiers | The distribution, not the average. See below. |
| `query_errors_total` | counter | searches that raised instead of answering | The error rate. Any sustained non-zero rate is a defect or a caller sending something the engine refuses. |
| `query_zero_results_total` | counter | searches that matched no document | **The cheapest proxy there is for relevance health.** Nothing else in a running system notices that the engine has started answering nothing. |
| `documents_scored_total` | counter | candidates a scorer actually scored | Divided by `queries_total`, how much work a search costs. It is what early termination exists to reduce. |
| `segments_open` | gauge | segment files mapped and not yet closed | The merge backlog. It should oscillate around a small number and come back down. |
| `query_cache_hits_total`, `query_cache_misses_total` | counters | query parses reused against parses made | Whether the parse cache earns its memory. |
| `postings_cache_hits_total`, `postings_cache_misses_total` | counters | postings answered from a segment's cache against decoded from the file | The same question for the cache that keeps scoring linear. |

## Why a histogram and not a mean

A mean hides exactly what a person notices. Here is a real run of the
histogram in this repository, 1,000 requests:

```text
980 requests at 10 ms and 20 at 4 s
  mean  89.8 ms
  p50   0.01 s
  p95   0.01 s
  p99   5.0 s
  share above one second: 2 percent
```

**The mean says 90 ms and would pass most dashboards.** The distribution says
half of all requests are at 10 ms, and one in fifty waits four seconds. It is
that one in fifty who files the ticket, and no average will ever show them.

Bucket counts also add up across processes, which means the p99 of a fleet can
be computed from the counts each machine reports. Means cannot be combined that
way: averaging ten averages is only correct when all ten came from the same
number of observations, and they never do.

## What a quantile from buckets can and cannot tell you

**It is an upper bound, not the true value.** The answer is the upper bound of
the bucket the quantile falls in, so it is never too small and is wrong by at
most that bucket's width.

In the run above the true tail is 4.0 s and the reported p99 is **5.0 s**,
because 4.0 falls in the bucket that runs from 2.5 to 5.0. **The overstatement
there is 1.0 s, and the worst case anywhere in that bucket is 2.5 s.** The
widest bucket in the default set runs from 5.0 to 10.0, so the largest error
this can produce is 5.0 s.

The practical consequences:

- **Do not alert on a small change in a reported quantile.** Moving from 0.25
  to 0.5 can be one request crossing one boundary.
- **A quantile that reports `+Inf` means the observation was above every
  bucket**, which is 10 seconds here. That is a real signal, not a rendering
  problem.
- **An empty histogram answers 0.0** rather than raising. Nothing has happened
  yet, and a monitoring path that throws when nothing has happened is worse
  than one that says so.
- **Change the buckets rather than the reading** if the interesting range is
  one bucket wide. `Registry.histogram` takes bounds.

## What measuring costs

```bash
uv run python -c "import timeit; ..."   # 200,000 calls, best of five
```

```text
counter.increment     186 ns
histogram.observe     341 ns
```

Both figures include one Python call each, so both are upper bounds. A search
pays about **900 ns** of that fixed, against a per-candidate scoring cost of
**1.7 microseconds** measured with `uv run python benchmarks/index_build.py`.
Instrumenting a whole search costs less than scoring a single candidate.

Every mutation takes a lock, because a count that is occasionally short is
worse than no count: it is still believed.

## The log schema

One JSON object per line. `json.dumps` escapes every newline, quote and
backslash, so a query containing any of them cannot break the guarantee that a
line is a record.

| Field | On every record | What it is |
| --- | --- | --- |
| `time` | yes | ISO 8601, UTC |
| `level` | yes | `INFO`, `WARNING`, `ERROR` |
| `event` | yes | what happened: `query`, `search` |
| `correlation_id` | when one is bound | ties every line of one request together |
| `traceback` | when the record carries an exception | the formatted traceback |
| `query` | on a `query` event, unless turned off | the query as typed |
| `terms` | on a `query` event | how many terms it analysed to |
| `phrase` | on a `query` event | whether it was a quoted phrase |
| `candidates` | on a `query` event | how many documents matched |
| `duration_ms` | on a `query` event | how long the search took |
| `exit_code` | on a command line `search` event | what the process returned |

**A result set is never logged, only a line about one.** It is unbounded, it is
the largest thing in the request, and writing it is the fastest way to fill a
disk.

### The privacy note, which is not optional

**Logging a query is logging what a person searched for.** In most
jurisdictions that is regulated data with a retention policy attached, and in
all of them it is the most sensitive thing this system holds.

`configure(stream, log_queries=False)` drops the text from every record and
keeps the counts. It is a filter on the handler rather than a flag each call
site reads, so the policy lives where an operator sets it and no code that logs
can forget to honour it.

Decide this before the first line is written. Retrofitting it after somebody
asks what you keep is expensive; the switch itself is not.

## Query cost limits

An unbounded query is a denial of service that needs no exploit and no
privilege: a phrase of fifty terms, or a query built from the fifty commonest
terms in the corpus, does work proportional to the whole index.

| Limit | Default | What it bounds |
| --- | --- | --- |
| `MAX_QUERY_TERMS` | 32 | Terms in one query. Longer than any person's query, shorter than any machine's abuse of one. |
| `MAX_ESTIMATED_CANDIDATES` | 5,000,000 | Postings the query would touch. Roughly a second of scoring at 1.7 microseconds each. |

**Queries are judged on estimated cost, never on shape.** A list of forbidden
shapes is always one query behind whoever is writing them, and it refuses
legitimate queries that merely look alarming.

**The estimate reads no postings.** Document frequency already sits beside each
term in the dictionary, put there so a scorer would not have to count, and
summing it answers this question too. Checking admission therefore costs
nothing like the work it guards.

The cost is judged after synonym expansion, on the terms actually about to be
read. Judging the typed query would let a synonym table buy exactly the work
the limit exists to refuse.

To tune: raise `max_candidates` towards the number of postings you are willing
to spend a second on, and watch `query_duration_seconds`. If the limit is
refusing real queries, they will be visible as errors before they are visible
as complaints.

## Access filtering, and the ordering that matters

If different people may see different documents, **the filter must be applied
before the top k is chosen**, which is why `rank`, `rank_wand` and
`search_wand` take a `permit` rather than expecting the caller to filter the
list they return.

**Filtering afterwards leaks.** Ask for ten results, get seven, and the three
gaps have told you that documents you are not allowed to see exist. That is
information disclosure with no exploit required: it is the normal output.

**With early termination it is worse than a leak.** The search keeps a
threshold from the best results so far and prunes anything that cannot beat it.
A forbidden document that reaches the results sets that threshold, and pruning
against a threshold set by documents the caller cannot see drops documents they
can. **The answer is then wrong, not merely incomplete**, and wrong in a way
only the person denied the data could detect.

So the permit is consulted before a document is scored and before it can affect
anything. A property test asserts that early termination and a full scan return
exactly the same list under any permit and any limit.

A permit that raises propagates rather than being caught. An empty result looks
like a legitimate answer and would be acted on as one, which is the worst
available outcome for a function whose job is deciding who sees what.

A permit that is not deterministic has undefined results and defined safety:
which documents come back is not specified, and nothing is corrupted.

## Alerting: on symptoms, never on causes

**CPU is not a symptom. Neither is memory, nor a cache hit rate, nor a segment
count.** Nobody has ever been woken usefully by a cache hit rate. They are all
useful once you are awake, and none of them is worth waking for, because each
can move for a dozen reasons that nobody notices and each can stay still while
the service is unusable.

Alert on what a user would report:

| Alert on | Because |
| --- | --- |
| `query_duration_seconds` p99 above the objective | Someone is waiting. |
| `query_errors_total` rate above the objective | Someone got nothing. |
| `query_zero_results_total` share rising sharply | Everyone is getting nothing, and it looks like success. |

The third is the one usually missing. An engine answering every query with an
empty result has an error rate of zero and a latency profile that is better
than usual.

Suggested starting objectives, to be replaced by measured ones as soon as there
is traffic to measure: 99 percent of searches under 250 ms, error rate under
0.1 percent, zero-result share within a factor of two of its trailing week.

## Runbooks

### Latency is up

**What it means.** Searches are doing more work, or waiting on something.

**Confirm.** Compare `query_duration_seconds` p50 with p99. A p50 that moved
with it means every query is slower, which is usually the index growing or the
machine being shared. A p50 that did not move means a few queries are much
slower, which is usually one expensive query shape.

**Then.** Divide `documents_scored_total` by `queries_total` and compare with
yesterday. If it rose, queries are matching more documents: check
`postings_cache_hits_total` against misses, because a cache that stopped
working turns linear scoring into repeated decoding. If it did not rise, the
work per document went up, which points at the machine rather than the engine.

### The error rate is up

**What it means.** Searches are raising rather than answering.

**Confirm.** Read the logs for `level` of `ERROR` and group by `traceback`.
Every error record carries the correlation identifier, so one identifier gives
the whole request.

**Then.** `QueryTooExpensiveError` is the engine refusing work, not failing:
find the caller and either fix the query or raise the limit deliberately. A
storage error means a segment or its stored text is missing or damaged; the
message names the file. Anything else is a defect, and the traceback is the
report.

### The zero-result rate is up

**What it means.** The engine is answering nothing, successfully. **This is the
one that hides.**

**Confirm.** `query_zero_results_total` against `queries_total`. Then take a
handful of queries from the logs and run them by hand.

**Then.** If an index was rebuilt, check that it was rebuilt with the analysis
configuration in use: an index whose analyzer fingerprint differs is refused
outright, but one rebuilt from a truncated corpus is not. If the queries
changed rather than the index, a new client may be sending a form the parser
does not understand. If neither, spelling suggestions on standard error will
say whether the terms exist at all.

### The merge backlog is growing

**What it means.** `segments_open` climbs and does not come back down. Every
query opens every segment, so read cost grows without limit while write cost
stays flat.

**Confirm.** Count the segment files in the directory and compare with
`segments_open`. A gauge higher than the file count means readers are not being
closed, which is a leak rather than a backlog.

**Then.** A genuine backlog means writes are arriving faster than merges
complete. A leak means something holds a reader open: on Windows that also
blocks the file from being deleted or replaced, so it shows up as failures in
the write path before it shows up as slowness in the read path.

## How it connects to everything else

- **[Early termination](14-early-termination.md)** is what
  `documents_scored_total` measures. Watching that number is the only way to
  know it is working on real traffic rather than on a benchmark.
- **[Caching](15-caching.md)** publishes hit and miss counts here. The cache
  sizes are unmeasured defaults, and these counters are how they stop being
  guesses.
- **[Segments](13-segments.md)** explains what a merge is and why the backlog
  matters.
- **[Querying](04-querying.md)** covers what a result looks like and where the
  text in a snippet comes from.
