# Reading a corpus, and saving the index

The two file-facing parts of the engine. Neither is algorithmically
interesting, and both are where a project like this usually acquires its worst
bugs, because file formats fail quietly.

Implemented at [`src/search_engine/corpus.py`](../src/search_engine/corpus.py)
and [`src/search_engine/persistence.py`](../src/search_engine/persistence.py).

## Reading the corpus

### The format

A sequence of page records:

```text
<page><id>42</id><title>Some title</title><text>Body...</text></page>
```

### Why this is a scanner and not an XML parser

The obvious implementation is `xml.etree.ElementTree.iterparse`. It was
rejected for two reasons, and the second is the stronger one.

**Real dumps are not always well-formed as a whole.** A strict parser needs a
single root element. Corpus files are frequently concatenations of records, or
truncated partway through a download, and a parser rejects the entire file over
a problem in its last hundred bytes.

**An XML parser is an attack surface, and a scanner is not.** The billion
laughs attack defines a few nested entities that expand exponentially, and a
few kilobytes of input consumes all available memory. Every general XML parser
must at least decide what to do about it. A scanner that matches `<page>` and
`</page>` and never expands an entity has nothing to exploit. The one piece of
XML semantics actually needed, unescaping named entities such as `&amp;`, is
done explicitly with `html.unescape`.

The cost is real and worth stating: no namespaces, no attribute parsing, no
validation, no nested-structure awareness. Anything beyond these three fields
needs a proper parser. What is bought is that a 300 MB partially-corrupt file
still yields every complete record in it.

### Streaming, and how that is proved

A corpus is hundreds of megabytes, and the index built from it already costs
several times the source in memory. The reader must add nothing to that, so it
holds one page at a time: read a chunk, extract every complete record in the
buffer, discard what was consumed, repeat.

Buffer growth is bounded by the largest single record. One subtlety: if a file
contains no records at all, the buffer would grow forever, so once no `<page>`
remains in it the buffer is discarded except for the last few characters, which
might be a partial opening tag.

**How the streaming is verified.** The same fixture is read at chunk sizes 1, 2,
7, 64 and 1 MB, and the results must be identical. At chunk size 1 every record
is split across a thousand reads, so if the buffer handling were wrong at any
boundary that test fails. The property test generates corpora and an arbitrary
chunk size together for the same reason.

### Decisions inside it

**The title is indexed with the body.** A title is the most concentrated
description a document has, so ignoring it would discard the best signal in the
file. Joined with a space, so the last title word and the first body word
cannot merge into one token.

**The first `<id>` inside a page wins.** Real dumps nest further identifiers
inside `<revision>` and `<contributor>`. Searching only within one page record
and taking the first match gets the page's own.

**A record with no identifier is refused, not skipped.** Skipping it silently
would leave the index quietly missing a document, and nothing downstream could
detect that. A truncated final record is dropped, because half a document has
incomplete text and is worse than none.

**Unicode round-trips exactly.** Verified on accented Latin, CJK, Greek, and
astral-plane characters, including a control character. The reader does not
normalise, so what the file says is what the index sees. Combined with the
tokenizer's normalisation sensitivity, that means the corpus file's Unicode form
matters, which is documented under [Tokenization](01-tokenization.md).

## Saving the index

### The format

```text
# search-engine-index v1
# documents: 1,2,3,5
algorithm|2:14;3:5
comput|1:1,6;2:3;4:0
```

A header, a document identifier list, then one line per term: the term, a pipe,
then `docID:pos,pos` groups separated by semicolons.

### Why plain text

**It is readable.** A wrong index can be diagnosed with `head`, which is worth
a great deal the first time an index looks wrong.

**It streams.** Neither writing nor reading needs the whole structure in
memory as a serialised blob.

**It cannot execute code.** `pickle` would be shorter and faster and would also
mean that loading an index file runs whatever that file says to run. The moment
an index arrives from anywhere but your own machine, that is a remote code
execution vulnerability. Text has no such property.

**No escaping is needed anywhere**, and that is not luck. Terms are maximal runs
of `[a-z0-9]`, so none of `|`, `:`, `;`, `,` or a newline can occur inside one.
The format is safe because of a guarantee established three stages earlier in
the pipeline.

### Why the document list is in the header

A document containing no indexable terms, pure punctuation or nothing but
stopwords, appears in **no** postings list. Its identifier is therefore
unrecoverable from the postings alone.

Losing it would reduce the document count, which is the `N` in `log(N/df)`, so
every inverse document frequency and therefore every score would change. The
header carries the identifiers explicitly so that cannot happen.

This was not foreseen. The first implementation tried to recover identifiers
from the postings and raised when the counts disagreed, which made saving such
an index impossible. The fix was for the index to expose its identifiers
directly.

### Why the version number, and why it is checked

Storing TF-IDF weights instead of raw positions is a natural future change, and
it would alter this format. A v2 file loaded by v1 code would parse without
error and produce silently wrong results, which is the worst failure mode
available. So the header is verified rather than assumed, and a mismatch raises.

### Determinism

Terms and document identifiers are written in sorted order, so indexing the
same corpus twice produces byte-identical files. That makes the output diffable
and lets the tests assert exact content rather than approximate structure.

### What is validated on load

Every structural assumption, each with its own error message: the header, the
document line, the presence of the pipe, the presence of the colon, and that
every identifier and position is an integer. Ten malformed files are asserted to
raise rather than load.

The reasoning is uniform: a malformed index that loads gives wrong answers
without any signal, so every one of these is worth an explicit check.

## The bug that only a test found

An empty index wrote `# documents: ` with a trailing space. Reading it back
called `.strip()` on the line, which removed that space, so the line no longer
matched the prefix it had just been written with, and an empty index could not
be loaded.

Nothing about that is visible by reading the code. It needed a test that saved
an empty index and loaded it again, and it is the reason the round-trip tests
cover the degenerate cases rather than only the interesting ones.

## Alternatives

| Approach | Wins when | Loses when |
| --- | --- | --- |
| **Plain text (this)** | debuggable, safe, streamable, diffable | largest on disk, slowest to parse |
| **JSON** | ubiquitous tooling | no streaming without a special parser, and larger for this shape |
| **pickle** | fastest, least code | **executes code on load**, and is version-fragile |
| **SQLite** | random access to one term without reading the file | a dependency on file layout, and overkill for a whole-index load |
| **Compressed binary with delta and varbyte encoding** | what production systems use; several times smaller | unreadable, and needs its own test suite |

The last row is the real next step, and the reason is in the
[index measurements](03-inverted-index.md): positional postings cost 9.4 times
the source text. Delta-encoding positions, so `0,2,5` becomes `0,2,3`, and then
varbyte-encoding the small numbers that produces, is where the large saving is.
It should be measured against this format rather than assumed to be better.

## How it connects to everything else

- **The tokenizer's character class** is what makes the on-disk format need no
  escaping.
- **Stopword removal** creates the gaps in stored positions, which the format
  carries verbatim.
- **The index** gained `from_postings` for the load path, which deliberately
  bypasses analysis: stored terms are already analyzed, and re-analyzing them
  would change them, since stemming is not idempotent.
- **Ranking** depends on the document count surviving the round trip, which is
  why the header exists.
- **The CLI** is the only caller of either module, and it turns them into the
  two phases of using the engine: index once, query many times.
