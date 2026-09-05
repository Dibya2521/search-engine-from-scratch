# Tokenization

The first stage of the pipeline, and the one that silently decides what the
engine is capable of finding at all. Every later stage, from stopword removal to
TF-IDF ranking, operates on whatever this stage decided a word is.

## What it is

Tokenization turns an unstructured string into an ordered sequence of discrete
units called tokens.

```text
"The Quick, Brown Fox!"  ->  ["the", "quick", "brown", "fox"]
```

Stripped of jargon: it decides where words begin and end, and normalises them so
that two spellings of the same word become the same string.

The rule used here:

1. Lowercase the text.
2. A token is a maximal run of `[a-z0-9]`.
3. Every other character is a separator and is discarded.

Implemented at [`src/search_engine/tokenizer.py`](../src/search_engine/tokenizer.py).

## Why it exists

The index is a map from a term to the documents containing it. A map needs
discrete, comparable keys, and raw text has none: it is one long sequence of
characters with no inherent word boundaries. Something must manufacture those
keys.

The deeper reason is **matching**. A user types `search` and a document contains
`Search`. Those are different strings, so a naive index would not match them.
Tokenization collapses irrelevant surface variation so that a query and a
document can meet on common ground. Hence the critical consequence, and the
source of most bugs in a search engine:

> **The query must be tokenized by exactly the same function as the documents.**

If the two ever diverge, the query produces keys the index does not contain and
the engine silently returns nothing. This is why tokenization is one pure
function shared by both paths rather than two similar-looking pieces of code.

### Why whitespace splitting is not enough

| Input | `text.split()` | What the index needs |
| --- | --- | --- |
| `"Hello, World!"` | `["Hello,", "World!"]` | `["hello", "world"]` |
| `"The THE the"` | `["The", "THE", "the"]` | `["the", "the", "the"]` |
| `"state-of-the-art"` | `["state-of-the-art"]` | `["state", "of", "the", "art"]` |

Punctuation sticks to words, so `world` and `world!` become different terms, and
case is not folded, so `The` and `the` become different terms. Both inflate the
vocabulary with duplicates and cause query misses. Character-class tokenization
fixes both with one rule.

## How it works, the internals

```python
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [match.group() for match in _TOKEN_PATTERN.finditer(text.lower())]
```

Three mechanisms, in order.

### `text.lower()`

Unicode case folding, not a 32-offset on ASCII. Python applies the Unicode
simple lowercase mapping to every code point, which has two properties that
matter here and that both contradict what the code looks like it does.

**It is not length-preserving.** `U+0130`, Turkish dotted capital I, lowercases
to *two* code points: `i` plus `U+0307` combining dot above.

```text
len("İSTANBUL")          == 8
len("İSTANBUL".lower())  == 9
```

Consequence, and it is load-bearing: **a character offset taken before
lowercasing does not survive it.** That is the concrete reason token positions
in the index are indices into the token sequence rather than character offsets
into the source text.

**It can pull characters *into* the ASCII set.** `U+212A`, the Kelvin sign,
lowercases to a plain ASCII `k`, so `tokenize("K")` yields `["k"]`.

So the ASCII filter cannot be hoisted before lowercasing as an optimisation.
The order is mandatory: lowercase first, filter second. Reversing them drops
tokens.

### The pattern `[a-z0-9]+`

`[a-z0-9]` matches one code point in either of two ranges. `+` makes it
one-or-more, and the engine is greedy, so at each start position it consumes the
longest available run. That greediness is what produces *maximal* runs, and why
`a1b2c3` is one token rather than six.

The pattern has no alternation, no backreferences and no nested quantifiers, so
there is nothing to backtrack over. Each character is examined at most twice,
giving **O(n) time in the length of the text** with no pathological input. That
matters for a component that will process attacker-supplied query strings: a
carelessly written equivalent such as `([a-z]+)+` is exponential on the same
input and would be a denial-of-service risk.

Using `re` is not a violation of the no-dependency principle. That principle
targets *retrieval* mechanisms, the ones worth understanding here: the stemmer,
the index, the intersection algorithms, TF-IDF. A character-class scan is
general string processing whose mechanism is fully described above, and
hand-writing the equivalent character loop would teach nothing about
information retrieval while running slower.

### `finditer` rather than `findall`

Both produce identical tokens. `findall` is one C-level call returning a list of
strings and is marginally faster; `finditer` returns a lazy iterator of `Match`
objects.

`finditer` was chosen because it is precisely typed: `Match[str].group()` returns
`str`, whereas `findall` is typed `list[Any]`, which would place an untyped
value inside the one function every other stage depends on. Under strict type
checking that is a real cost, and the speed difference does not register against
the 21 MB/s measured below.

## The design decisions inside it

**It is a pure function.** Same input, same output, no state, no I/O. That is
what makes the property tests possible, and what lets the query path and the
indexing path share it with no risk of one mutating something the other relies
on.

**It does not assign positions.** Phrase queries need to know that `computer`
appeared at position 2 in document 5. The tokenizer could emit `(token,
position)` pairs, but instead it emits an ordered sequence and the indexer calls
`enumerate` on it. The reasoning is single responsibility: the tokenizer answers
"what are the words", the index answers "where are they", and each is testable
without the other. It also keeps position numbering the index's business, so if
positions later need to account for removed stopwords, only the indexer changes.

## When this rule is the right choice

Conditions under which `[a-z0-9]+` on lowercased text is genuinely correct:

- The corpus is English, or close enough that losing accented forms is
  tolerable.
- Recall on non-Latin scripts is not required. Under this rule, CJK, Greek,
  Cyrillic, Hebrew and Arabic text produce **zero tokens** and are completely
  unsearchable, not merely degraded.
- Punctuation carries no meaning. This is false for source code (`C++` becomes
  `c`), identifiers (`snake_case_name` splits into three), version numbers
  (`Python3.14` becomes `python3` and `14`) and addresses (`192.168.0.1` becomes
  four tokens).
- Splitting contractions is acceptable: `don't` becomes `don` and `t`, and `t`
  then pollutes the vocabulary as a meaningless single-character term.

Where it is the wrong choice: multilingual corpora, code search, legal or
medical text where hyphenated compounds are terms of art, and anything where a
product identifier must match exactly.

**The conditions here** are an English corpus and a design whose measured term
counts should stay comparable to the published figures for the same approach.
Under those conditions this rule is the right one.

## The measured cost

Each row is real output, produced before any assertion was written.

| Input | Tokens | Why |
| --- | --- | --- |
| `café` composed | `["caf"]` | `é` is one code point, filtered out entirely |
| `café` decomposed | `["cafe"]` | `e` is ASCII, only the combining mark is filtered |
| `naïve` composed | `["na", "ve"]` | word split in two |
| `Müller` composed | `["m", "ller"]` | word split in two |
| `Straße` | `["stra", "e"]` | sharp s filtered |
| `ﬁre` (fi ligature) | `["re"]` | the ligature is a single non-ASCII code point |
| `co­operate` (soft hyphen) | `["co", "operate"]` | an *invisible* character splits a word |
| `x²` | `["x"]` | superscript digits are not in `[0-9]` |
| `ＡＢ` fullwidth | `[]` | fullwidth Latin is outside the ASCII range |
| `日本語` | `[]` | entirely unsearchable |

### The real defect: normalisation form changes the answer

The first two rows are the same word, rendered identically on screen,
tokenizing differently:

```text
"café"  composed   (c a f é)     ->  ["caf"]
"café"  decomposed (c a f e ́)   ->  ["cafe"]
```

In composed form the accented letter is a single non-ASCII code point and
vanishes. In decomposed form it is ASCII `e` followed by a combining mark, so
the `e` survives. **A document stored in one form and a query typed in the other
will not match**, even though both look identical to the user.

This is a defect in the rule, not in its implementation. The fix is a single
`unicodedata.normalize` call before lowercasing. It is deliberately left unfixed
for now so the choice can be made against a measurement on a real corpus rather
than on intuition, and the behaviour is pinned by a test so it cannot change
silently.

### Throughput

```text
input      : 0.72 MB, 720,000 chars
tokens     : 120,000
elapsed    : 34.1 ms
throughput : 21.1 MB/s
```

Extrapolated to a 300 MB corpus: roughly **14 seconds** of pure tokenization.
This is the baseline against which the cost of stemming gets measured.

## Alternatives

| Approach | How it works | Wins when | Loses when |
| --- | --- | --- | --- |
| **Whitespace split** | `text.split()` | never, for indexing | punctuation and case both break matching |
| **Character class (this one)** | maximal `[a-z0-9]` runs | English, punctuation-insensitive | non-ASCII, code, identifiers |
| **Unicode-aware class** | `\w+` plus normalisation | multilingual corpora | still no word boundaries for CJK, which has no spaces |
| **Dictionary segmentation** | longest match against a lexicon | CJK, Thai and other unspaced scripts | needs a per-language lexicon, fails on unknown words |
| **Character n-grams** | every window of length n | any script, robust to typos, no lexicon | index grows several times larger, precision drops |
| **BPE / WordPiece / SentencePiece** | statistical merges learned from a corpus, splitting rare words into subwords | neural models needing a fixed finite vocabulary | unusable for an inverted index |

The last row is worth dwelling on, because it is the most familiar tokenizer to
anyone coming from machine learning. **Retrieval tokenization and neural
tokenization solve opposite problems.** BPE exists to bound vocabulary size for
an embedding matrix, and it happily produces `token` plus `##ization` because
the model can recombine meaning from subword vectors. An inverted index cannot:
a postings list for `##ization` is not something a user can query, and the term
statistics attached to it are meaningless. Retrieval wants tokens that are
*words*. Neural models want tokens that are *frequent byte sequences*.

## How it connects to everything else

- **Stopword filtering** consumes this output, and can only remove tokens this
  stage produced. The list is therefore defined against a specific tokenizer,
  and every entry is asserted to survive tokenization unchanged.
- **Stemming** also consumes it, and only makes sense on word-like tokens. The
  `t` left over from `don't` will be stemmed to `t`, wasting a vocabulary slot.
- **The inverted index** calls `enumerate` on this sequence, which is where
  positions come from. Every position is an index into this list, which is why
  the length-preservation problem above does not bite.
- **Phrase queries** depend on positions, and on the gaps in them. Stopword
  filtering removes terms but keeps their positions, so `"king of england"` and
  `"king england"` stay distinguishable. See [Stopwords](05-stopwords.md).
- **[Ranking](06-ranking.md)** counts these tokens. Splitting `naïve` into `na` and `ve` creates
  two junk terms, each with its own document frequency, which shifts the IDF of
  every other term by changing the vocabulary size.
- **The query path** must call this exact function, for the reason given at the
  top.

## Verification

[`tests/test_tokenizer.py`](../tests/test_tokenizer.py) holds 33 example-based
cases, every expected value read from a real run first, plus four property-based
tests over arbitrary generated text. The properties are what can actually
falsify the design rather than agree with it:

1. every token matches `[a-z0-9]+`
2. `tokenize(" ".join(tokenize(t))) == tokenize(t)`, covering order,
   completeness and separator collapsing in a single invariant
3. `tokenize(a + " " + b) == tokenize(a) + tokenize(b)`, the property that makes
   the corpus parser's title-plus-body concatenation safe
4. text containing no ASCII alphanumerics yields no tokens
