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

1. Normalize the text to a canonical Unicode form.
2. Lowercase it.
3. A token is a maximal run of word characters, excluding the underscore.
4. Every other character is a separator and is discarded.
5. A run of characters from a script written without spaces is then cut into
   overlapping pairs.

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
TOKEN_PATTERN = re.compile(r"[^\W_]+")


def tokenize(text: str, form: NormalForm = DEFAULT_FORM) -> list[str]:
    normalized = unicodedata.normalize(form, text).lower()
    return [
        token
        for match in TOKEN_PATTERN.finditer(normalized)
        for token in _segment(match.group())
    ]
```

Four mechanisms, in order.

### Normalization

One word can be written as more than one sequence of code points. `café` with a
composed acute accent is a single code point; the same word with a combining
accent is an ASCII `e` followed by a mark. They render identically and used to
produce different tokens, so a document stored in one form could never match a
query typed in the other.

`NFC` composes them, and is the default. `NFKC` also folds compatibility forms,
recovering tokens that would otherwise be dropped and destroying distinctions in
the same breath: it turns a ligature into two letters, and a superscript two
into an ordinary two. NFC is the default because it only ever merges spellings
that render identically, so it cannot lose a distinction a reader can see.

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

### The pattern `[^\W_]+`

`\W` is any character that is not a word character, so `[^\W_]` is a word
character that is not an underscore. Python matches `\w` against Unicode by
default for `str` patterns, so this covers every alphabet and every set of
digits, not just Latin ones. The underscore is excluded because it is a
separator here like every other punctuation mark, which is why
`snake_case_name` is three tokens.

`+` makes it one-or-more, and the engine is greedy, so at each start position it
consumes the longest available run. That greediness is what produces *maximal*
runs, and why `a1b2c3` is one token rather than six.

**The pattern was `[a-z0-9]+` until version 0.6.0**, which produced *zero*
tokens for Chinese, Japanese, Korean, Greek, Cyrillic, Hebrew and Arabic. Those
documents were not degraded, they were unsearchable, and no amount of better
ranking fixes a ceiling set this far upstream.

The pattern still has no alternation, no backreferences and no nested
quantifiers, so there is nothing to backtrack over. **Any replacement must keep
that property**, and a comment in the source says so. Each character is examined at most twice,
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

### Bigrams for scripts written without spaces

Chinese and Japanese put no spaces between words, so a run of them arrives as a
single enormous match. Splitting it properly needs a dictionary and a
segmentation model, which is a different project. Overlapping character bigrams
are the standard cheap answer and what Lucene's CJK analyser does:

```text
東京大学  ->  [東京, 京大, 大学]
```

Four characters give three tokens; one character stays one token. They overlap
rather than partition, because partitioning would miss any word straddling a
pair boundary.

Three costs, all real:

1. Roughly twice the postings for text in those scripts.
2. False matches across word boundaries, so precision falls.
3. Recall goes from zero to imperfect. **That is the whole of the value here**,
   and it is worth saying plainly rather than dressing up.

A run holding both kinds is cut at the change of script, so a product name with
Latin and Japanese in it gives whole words for one half and bigrams for the
other.

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

### Conditions under which this rule is genuinely correct

The original ASCII-only rule was correct under these conditions, and the list is
kept because it is what the widened pattern had to escape:

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

| Input | Before 0.6.0 | Now | Why it changed |
| --- | --- | --- | --- |
| `café` composed | `["caf"]` | `["café"]` | the accented letter is a word character |
| `café` decomposed | `["cafe"]` | `["café"]` | composed first, so both spellings agree |
| `naïve` | `["na", "ve"]` | `["naïve"]` | no longer split at the accent |
| `Müller` | `["m", "ller"]` | `["müller"]` | no longer split at the accent |
| `Straße` | `["stra", "e"]` | `["straße"]` | sharp s is a word character |
| `ﬁre` (fi ligature) | `["re"]` | `["ﬁre"]` | kept whole under NFC, `["fire"]` under NFKC |
| `co­operate` (soft hyphen) | `["co", "operate"]` | `["co", "operate"]` | an *invisible* character still splits a word |
| `x²` | `["x"]` | `["x²"]` | superscript two is a digit, `["x2"]` under NFKC |
| `ＡＢ` fullwidth | `[]` | `["ａｂ"]` | fullwidth Latin are letters, `["ab"]` under NFKC |
| `日本語` | `[]` | `["日本", "本語"]` | bigrams, where there were no tokens at all |
| `Ελλάδα` | `[]` | `["ελλάδα"]` | one word, where there were no tokens at all |

### The defect that is now fixed: normalisation form changed the answer

The first two rows were the same word, rendered identically on screen,
tokenizing differently:

```text
"café"  composed   (c a f é)     ->  ["caf"]
"café"  decomposed (c a f e ́)   ->  ["cafe"]
```

In composed form the accented letter was a single non-ASCII code point and
vanished. In decomposed form it was ASCII `e` followed by a combining mark, so
the `e` survived. **A document stored in one form and a query typed in the other
did not match**, though both looked identical to the user.

That was a defect in the rule, not in its implementation, and the fix is a
single `unicodedata.normalize` call before lowercasing. Three tests pinned the
broken behaviour so it could not be fixed silently; they now assert that both
spellings agree.

**Every index built before this change is refused rather than read**, because
the analyzer fingerprint changed. An old index queried by a new build would
return nothing for the affected documents with no error anywhere, so the
mismatch is raised instead, naming both fingerprints and saying to rebuild.

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

1. every token matches `[^\W_]+`
2. `tokenize(" ".join(tokenize(t))) == tokenize(t)`, covering order,
   completeness and separator collapsing in a single invariant
3. `tokenize(a + " " + b) == tokenize(a) + tokenize(b)`, the property that makes
   the corpus parser's title-plus-body concatenation safe
4. text containing no ASCII alphanumerics yields no tokens
