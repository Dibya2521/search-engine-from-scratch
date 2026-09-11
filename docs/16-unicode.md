# Unicode

What happens to text that is not plain ASCII, and what it cost to stop ignoring
it.

## What it is

Two changes, both in the tokenizer, both correctness fixes rather than features.

1. Text is **normalized** to a canonical form before anything else touches it,
   so two ways of writing the same word produce the same term.
2. The token pattern matches **every script**, and a run of characters from a
   script written without spaces is cut into overlapping pairs.

Implemented at [`src/search_engine/tokenizer.py`](../src/search_engine/tokenizer.py).

## The defect that was fixed first

One word can be written as more than one sequence of code points, and the two
render identically on screen.

```text
"café" composed    c a f é          4 code points
"café" decomposed  c a f e  ́       5 code points
```

Before normalization those produced different terms. Composed, the accented
letter was a single non-ASCII code point and the whole letter was dropped;
decomposed, the base letter was ASCII and only the combining mark was dropped.

```text
composed    ->  ["caf"]
decomposed  ->  ["cafe"]
```

**A document stored in one form could never match a query typed in the other.**
Nothing anywhere would report it: the query simply returned less than it should.
Both now give `["café"]`.

Three tests pinned the broken behaviour so it could not be fixed silently. They
now assert that both spellings agree, and a property test checks it over
generated text.

## NFC or NFKC, and why the answer rests on an argument

Two normalization forms were available and they differ in what they merge.

- **NFC** composes a letter and its accent into the single code point they
  render as. It only ever merges spellings that are already indistinguishable on
  screen.
- **NFKC** also folds compatibility forms. It recovers tokens that the pattern
  would otherwise handle differently, and it destroys distinctions.

Every case below is built from explicit code points rather than written as a
literal, because these turn entirely on which code points are present and a
literal leaves that to whatever an editor wrote to disk. Produced by
[`benchmarks/unicode_forms.py`](../benchmarks/unicode_forms.py):

| Case | Code points | NFC | NFKC |
| --- | --- | --- | --- |
| composed accent | U+0063 U+0061 U+0066 U+00E9 | `['café']` | `['café']` |
| combining accent | U+0063 U+0061 U+0066 U+0065 U+0301 | `['café']` | `['café']` |
| fi ligature | U+FB01 then `refighting` | `['ﬁrefighting']` | `['firefighting']` |
| superscript two | U+0078 U+00B2 | `['x²']` | `['x2']` |
| fullwidth letters | U+FF21 U+FF22 | `['ａｂ']` | `['ab']` |
| roman numeral four | U+2163 | `['ⅳ']` | `['iv']` |
| mathematical bold A | U+1D400 | `['𝐀']` | `['a']` |

**NFKC recovers four of those and ruins one.** A superscript two becoming an
ordinary two is wrong wherever the difference carries meaning, which is
chemistry, mathematics, and any citation with a footnote marker.

### The measurement that could not be taken

The choice was supposed to be settled against real text. It was not, and saying
so is the honest outcome:

```text
sources measured                 : 24
sources whose tokens change, NFC : 0
sources changing under NFKC only : 1
characters in all sources        : 255,721
characters outside ASCII         : 21
```

**Zero sources change under NFC because nothing here exercises NFC**, not
because NFC does nothing. Twenty-one non-ASCII characters in a quarter of a
million is not a corpus, and the coverage line is printed beside the counts so
the null result cannot be misread as a positive one.

**NFC is therefore the default on the conservative argument**: it cannot lose a
distinction a reader can see, and NFKC can. NFKC is available as a parameter and
is not chosen. Running the same benchmark against a real multilingual dump is
what would settle it, and nothing in the benchmark needs to change to do so.

## Every index built before this is refused

Normalization changes the terms of any document containing a decomposed letter,
and widening the pattern changes the terms of every document containing anything
outside ASCII. An old index queried by a new build would return nothing for
those documents, **with no error anywhere**. That is the commonest way a search
system breaks quietly, and the analyzer fingerprint is the guard built for it.

The fingerprint is a digest of everything that decides the terms produced: the
pipeline version, the token pattern, the stemmer version, the normalization form
and the stopword list. Two configurations share it only if they agree on all
five.

```text
index was built with analyzer 0da3e6c5c33c0e95, this build uses
9190455cbe0ae89d; rebuild the index
```

**To upgrade, rebuild the index.** There is no migration, and there should not
be: the terms are different, so the old file holds answers to a different
question.

The pipeline version was bumped twice in this release even though the pattern
change alone would have altered the digest. Two of the changes, the bigrams and
the stemming guard, are invisible in the pattern, and depending on a
coincidental pattern change in the same release would leave the next invisible
change unprotected.

## The ceiling that was removed

`[a-z0-9]+` produced **zero tokens** for Chinese, Japanese, Korean, Greek,
Cyrillic, Hebrew and Arabic. Those documents were not degraded, they were
unsearchable, and **no amount of better ranking fixes a ceiling set three stages
upstream.**

The pattern is now `[^\W_]+`: a word character that is not an underscore. Python
matches `\w` against Unicode by default for `str` patterns, so this covers every
alphabet and every set of digits.

| Script | Text | Tokens before | Tokens now |
| --- | --- | ---: | ---: |
| Greek | Ελλάδα | 0 | 1 |
| Cyrillic | Привет | 0 | 1 |
| Arabic | مرحبا | 0 | 1 |
| Hebrew | שלום | 0 | 1 |
| Japanese | 日本語 | 0 | 2 |
| Korean | 한국어 | 0 | 2 |

**Not `\w+`**, which is the obvious choice and includes the underscore. That
would have made `snake_case_name` a single token and silently reversed the
documented rule that every non-alphanumeric character separates. The two were
compared side by side on all six scripts before choosing and differ only there.

Both are linear with no backtracking, which matters because query text is
untrusted and a backtracking pattern is a way to hang the engine. A test
tokenizes 100,000 characters of near-matching input inside a time budget, and
the source says any replacement must keep that property.

## CJK bigrams, and their three costs

Chinese and Japanese put no spaces between words, so a run of them arrives as
one enormous token. Real segmentation needs a dictionary and a model, which is a
different project. A run is cut into overlapping character pairs instead, which
is what Lucene's CJK analyser does.

```text
東京大学  ->  [東京, 京大, 大学]
```

Four characters give three tokens; one character stays one token. They overlap
rather than partition, because a partition would miss any word straddling a pair
boundary. A run holding both kinds is cut at the change of script, so a product
name with Latin and Japanese in it gives whole words for one half and pairs for
the other.

**Three costs, all real:**

1. **Roughly twice the postings** for text in those scripts, since a run of `n`
   characters produces `n - 1` tokens where one whole-word token would do.
2. **False matches across word boundaries.** A bigram spanning the end of one
   word and the start of the next matches text that means something else, so
   precision falls.
3. **Recall goes from zero to imperfect.** That is the whole of the value here,
   and it is worth saying plainly rather than dressing up.

## The stemmer does not run on non-ASCII

The Porter stemmer implements English suffix rules. Its behaviour on Cyrillic or
Greek is not wrong so much as undefined: it would strip letter sequences that
spell nothing in those scripts.

Leaving a token whole costs recall on inflected forms. Mangling it costs
correctness, which is worse, so tokens outside ASCII are passed through
unchanged.

**The stemmer itself was not touched.** Its agreement with an independent
implementation over 34,814 words with zero disagreements is the strongest
correctness evidence in this project, and the guard lives in the caller so that
evidence still applies exactly as it did.

## The separator that needs no escaping

Fields are recorded by qualifying a term with the field it came from, so
`title:python` sits beside `python` in the same dictionary. A body containing
the literal text `title:python` might be expected to forge one.

It cannot. **A colon is not a word character**, so the tokenizer can never emit
a token containing one:

```text
tokenize("title:python")  ->  ["title", "python"]
```

Forging a qualified term from document text is impossible by construction rather
than prevented by validation, which is the stronger of the two and needs no
code. A test asserts the property directly, because a later change to the token
pattern could quietly remove the guarantee.

## Connections

- **[Tokenization](01-tokenization.md)** is where all of this happens, and holds
  the full rule.
- **[Stemming](02-stemming.md)** is the stage guarded to ASCII, and the
  equivalence evidence that guard protects.
- **[The inverted index](03-inverted-index.md)** stores the qualified terms and
  excludes them from every statistic about a document's own content.
- **[Query understanding](17-query-understanding.md)** covers what happens to a
  query after this, including the field weighting these terms make possible.
