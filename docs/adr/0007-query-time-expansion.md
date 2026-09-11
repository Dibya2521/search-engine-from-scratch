# ADR 0007: Expansion at query time, and fields as term prefixes

- **Status:** accepted
- **Date:** 2026-09-11
- **Supersedes:** nothing
- **Superseded by:** nothing

## Context

Two decisions in the same release both come down to the same question: **what
gets baked into the index, and what stays changeable.**

The first is synonyms. `laptop` should find a document that only says
`notebook`, and the equivalence between them can be applied when the document is
indexed or when the query is run.

The second is fields. A term in a title means more than one in the body, and
`Document.indexable_text` concatenated title and body so the distinction was
destroyed before the index saw it. Recording which field an occurrence came from
can be done three ways, and they cost very different amounts.

## Decision

**Expand synonyms at query time. Record fields as prefixed terms in the one
dictionary.**

### Synonyms, at query time

Expansion rewrites the query before it reaches the postings. Nothing about a
synonym is stored.

**Rejected: index-time expansion**, storing every member of a group in the
postings of every other.

Two reasons, and the second is the one that is easy to miss.

1. **The synonym list is the thing that changes most often.** Index-time
   expansion means no synonym can be added without reindexing every document. On
   a large index that is the difference between a one-line edit and a day, and
   the edit is one somebody wants to make weekly.
2. **It corrupts inverse document frequency.** Storing `notebook` in every
   document that says `laptop` inflates the document frequency of both, which
   lowers the weight of both, for every query, including queries that never
   mention either. **Adding one synonym would silently change the ranking of
   unrelated searches**, and nothing would report it.

Query-time expansion changes one query and nothing else.

### Fields, as prefixed terms

`title:python` sits in the same dictionary beside `python`. A document is
indexed once unqualified over its whole text, which is what every existing query
searches, and once qualified per field.

| Approach | What it changes |
| --- | --- |
| Per-field postings maps inside the index | The in-memory structure, the whole-file format, the segment format and every reader |
| A field bitmask beside each position | The postings encoding, so the segment format bumps again |
| **Field-prefixed terms** | **Nothing** |

**Rejected: the first two.** The segment format had already been bumped once in
the previous release, which cost every existing index a rebuild. Spending that
again for a feature that changes no answer by itself is a poor trade.

The dictionary is already sorted, so all of one field's terms are contiguous,
which makes iterating a field cheap without any extra structure.

## Consequences

**The field cost is real and was measured**, over the 70-document evaluation
corpus:

```text
vocabulary +106 terms, +10.0%
postings   +145 entries, +3.2%
index file +1,880 bytes, +9.4%
segment    +2,213 bytes, +7.9%
```

Vocabulary grows faster than postings, 10.0 against 3.2 percent, because a title
contributes a distinct term for every distinct word it holds while contributing
far fewer occurrences than a body does. On a corpus with longer bodies the
vocabulary cost would stay and the postings cost would shrink.

**A qualified term is excluded from every statistic describing a document's own
content**: its length, its TF-IDF vector and that vector's norm. A qualified
term is a second copy of content already counted, not new content. Counting it
would make a document with a long title look longer than it is and score lower
for every query, and would inflate every vector norm. Both failures are silent.

**The separator needs no escaping.** A colon is not a word character, so the
tokenizer can never emit a token containing one, and a body holding the literal
text `title:python` produces the two ordinary terms `titl` and `python`. Forging
a qualified term from document text is impossible by construction rather than
prevented by validation. A test asserts it, since a later change to the token
pattern could quietly remove the guarantee.

**The fields overlap, which textbook BM25F does not assume.** The unqualified
postings cover the whole document including the title, so a title occurrence is
counted twice by a field-weighted scorer. That makes the title weight a boost
rather than a partition, and it is documented in
[query understanding](../17-query-understanding.md) rather than left to be
inferred.

**A synonym member must be a single index term.** Matching a multi-word synonym
means matching a phrase, and expansion produces terms. A multi-word member is
refused at load time, naming the line, rather than being accepted and expanded
to its words separately, which would make `portable computer` match any document
containing `computer`.

**Query-time expansion shifts scores** and there is no down-weighting yet, since
the ranker takes one weight per term and offers no way to scale one of them.
That is stated in the module rather than approximated.

## What was rejected and why it looks arbitrary until it is not

Both rejections look like premature caution until the first time somebody needs
to add one synonym to a large index, or bump a format for the second time in two
releases. Neither cost is visible in the code that would have been written; both
are visible in what it would have taken to change it afterwards.
