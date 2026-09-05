# Stemming

The stage that decides which *different* words the engine treats as the same
word. Tokenization decides where words begin and end; stemming decides which of
them are the same concept.

## What it is

Stemming maps inflected and derived forms of a word to one shared form.

```text
connect, connected, connecting, connection, connections  ->  connect
```

The shared form need not be a real word, and usually is not. `relational`
becomes `relat`, `triplicate` becomes `triplic`. That is not a defect. The stem
is an internal key, and since documents and queries both pass through the same
function they meet on the same non-word.

Implemented at [`src/search_engine/stemmer.py`](../src/search_engine/stemmer.py).

## Why it exists

Two reasons, and the first matters far more.

**Recall.** Somebody searching for `retrieve` should find a document that says
`retrieval`. Without stemming those are unrelated keys in the index and the
document is invisible. Every inflected form the user did not happen to type is
a document they cannot reach.

**A smaller vocabulary.** Each distinct term costs a dictionary entry and a
postings list. Collapsing five surface forms into one key removes four entries
and merges their postings.

The cost is precision. Collapsing forms that are *not* the same concept
introduces false matches, and the measured section below shows this happening.
Stemming is a deliberate trade of precision for recall, which is usually right
for general search and usually wrong when an identifier must match exactly.

## How it works, the internals

Not a dictionary. No word list, no language model, no lookup table of
exceptions. Just suffix rewriting under a length condition, which is why the
whole thing is a few hundred lines and needs no data files.

### The measure

Every word is viewed as the shape `[C](VC)^m[V]`, where `C` is a run of
consonants and `V` a run of vowels. The **measure** `m` is how many `VC`
repetitions there are. It is the algorithm's proxy for "how much word is
there", and every rule is gated on it.

| Word | Shape | m |
| --- | --- | --- |
| `tr`, `ee`, `tree`, `y`, `by` | no complete VC pair | 0 |
| `trouble`, `oats`, `trees`, `ivy` | one | 1 |
| `troubles`, `private`, `oaten`, `orrery` | two | 2 |

This is what stops the rules destroying short words. The rule that strips `ate`
requires `m > 1` on what would remain, so `activate` becomes `activ` while
`rate` stays `rate`: there is not enough word underneath.

### The awkwardness of y

A consonant is any letter other than `a e i o u`, except that `y` depends on
its neighbour: it is a consonant at the start of a word or after a vowel, and a
vowel after a consonant.

```text
toy     -> consonant, vowel, consonant
syzygy  -> the three y letters are all vowels
```

Because each `y` depends on the letter before it, the classification is
computed left to right in one pass. A recursive definition reads more directly
but recurses once per letter, so a long run of `y` would overflow the stack.

### The five steps

Applied in order, each one at most once.

| Step | Job | Example |
| --- | --- | --- |
| 1a | plurals | `caresses` to `caress`, `ponies` to `poni`, `cats` to `cat` |
| 1b | past and progressive, then repair the damage | `plastered` to `plaster`, `hopping` to `hop`, `filing` to `file` |
| 1c | terminal `y` to `i`, so later rules can match it | `happy` to `happi` |
| 2, 3 | derivational suffixes | `relational` to `relate`, `electrical` to `electric` |
| 4 | strip the residue when enough word remains | `revival` to `reviv`, `adoption` to `adopt` |
| 5 | tidy a final `e` and a final double `l` | `probate` to `probat`, `controll` to `control` |

Step 1b is the subtle one. Removing `ed` or `ing` can leave something
malformed, so the step repairs its own output: put the vowel back
(`conflat` to `conflate`), collapse a doubled consonant (`hopp` to `hop`), or
restore an `e` on a short stem (`fil` to `file`). This is why `filing` becomes
`file` but `failing` becomes `fail`.

### Longest match, and one turn only

Within a step, the rule with the **longest** matching suffix fires. If that
rule's condition fails, the step makes no change and a shorter rule does not
get a turn. `rational` shows why this matters: it ends in `ational`, whose rule
requires `m > 0` on the remaining `r`, which is `m = 0`. So step 2 does nothing
at all, rather than falling back to the shorter `tional` rule.

## Two deliberate departures from the original 1980 rules

Both are the algorithm author's own later revisions, and both are adopted here.

**`bli -> ble` instead of `abli -> able`.** The narrower rule only handles
words where the letter before `bli` is `a`. The general form also catches
`possibli`, which the original leaves untouched.

**`logi -> log` added.** This is the one with a measurable retrieval effect.
Without it, `technology` stems to `technologi` while `technological` stems to
`technolog`, so the two forms of one word never meet. With it, both become
`technolog`.

**A third departure is this implementation's own:** words shorter than three
characters are returned untouched. The published rules do not exempt them, and
the reference implementation consequently turns the single word `s` into an
**empty string**. An empty term cannot be stored or queried, so the guard is a
correctness requirement here rather than a preference.

## Verification

Correctness was not argued, it was tested against an independent implementation.

Tests written from the same assumptions as the code can only agree with it. A
second implementation of the same specification, written by someone else, is
the only thing that can genuinely falsify these rules. So the suite runs both
over generated words and asserts they agree, at
[`tests/test_stemmer.py`](../tests/test_stemmer.py).

Result over 34,814 distinct words, mostly randomly generated:

| Compared against | Disagreements | On words longer than 2 characters |
| --- | --- | --- |
| The author's revised algorithm | **0** | 0 |
| The original 1980 rules | 39 | **0** |

The second row is the useful one. Every single disagreement with the original
rules is a word of two characters or fewer, which is exactly the deliberate
guard described above. There is no other difference, anywhere, over 34,814
words. That is a much stronger statement than any hand-written test could make.

This process also corrected a planned test. Idempotence,
`stem(stem(w)) == stem(w)`, looks like an obvious property to assert, and it is
**false**: it fails for 110 of those words, because stripping one suffix can
expose another. The reference implementation fails it too, so this is the
algorithm's nature rather than a defect. Callers must stem exactly once, on
documents and queries alike. Had the property been assumed rather than
measured, it would have become a wrong test enforcing wrong behaviour.

## What it actually achieves, measured

Word families that collapse to a single term, which is the whole objective:

| Family | Stem |
| --- | --- |
| `connect`, `connected`, `connecting`, `connection`, `connections` | `connect` |
| `compute`, `computer`, `computing`, `computed`, `computation` | `comput` |
| `technology`, `technologies`, `technological`, `technologically` | `technolog` |
| `analogy`, `analogous`, `analogical` | `analog` |

Families that do **not** collapse, which is just as important to know:

| Words | Stems | Why |
| --- | --- | --- |
| `biology`, `biological` | `biologi`, `biolog` | the stem under `logi` is `bio`, whose measure is 0, so the rule cannot fire |
| `relativity`, `relate` | `rel`, `relat` | over-stemming: `iviti` takes more than the shared root |
| `argument`, `argue` | `argument`, `argu` | under-stemming: too little word beneath `ment` |
| `apologetic`, `apology` | `apologet`, `apolog` | different suffixes, no shared rule |

`biology` is the sharpest illustration of the whole design. The measure
condition exists to protect short words, and here it protects one that should
have been stemmed. The algorithm has no notion of meaning, so it cannot tell
the difference. These four cases are pinned by tests, so if a future change
alters them it will be visible rather than silent.

### Throughput

```text
ours      : 128,688 words/s
reference :  58,240 words/s   (2.21x slower)
```

Compared against the tokenizer's 3.5 million tokens per second, stemming is
roughly **27 times more expensive per token** than tokenizing. That is worth
knowing before the index is built, and it makes stemming the obvious first
suspect if indexing turns out slow. What fraction of total index build time it
accounts for is a different question, since that also includes reading the
corpus and writing postings, and it can only be answered once the index exists.

## When stemming is the right choice, and the alternatives

| Approach | How it works | Wins when | Loses when |
| --- | --- | --- | --- |
| **No stemming** | exact terms only | identifiers, code, product codes, legal citations | ordinary prose, where recall collapses |
| **Porter (this one)** | suffix rules under a measure condition | English prose, no data files, predictable and fast | aggressive on some families, timid on others |
| **Snowball / Porter2** | the same author's later framework, more rules, many languages | multilingual corpora, slightly better accuracy | more complexity for a modest gain |
| **Lancaster** | iterative, strips until nothing matches | maximum recall | severe over-stemming, stems often unreadable |
| **Lemmatization** | dictionary plus part-of-speech to find the real base word | `better` to `good`, `ran` to `run`, and human-readable output | needs a dictionary, a tagger, and far more time per word |

Lemmatization is what stemming approximates. It is more accurate and returns
real words, and it costs a data file, a part-of-speech tagger and orders of
magnitude more time per token. For an English corpus where the stem is never
shown to the user, that accuracy buys very little.

**The conditions here** are English text, no runtime data files, and a stem
that is only ever an internal key. Porter is the right choice under exactly
those conditions.

One note for anyone arriving from machine learning: modern neural retrieval
does not stem at all. Subword tokenizers already split `connecting` into pieces
that share representation with `connect`, and the embedding space handles the
rest. Stemming is a lexical-matching technique, and it exists because an
inverted index compares strings rather than meanings.

## How it connects to everything else

- **Tokenization** runs first and feeds this. Its ASCII-only rule leaves
  debris, and stemming propagates it: the `t` from `don't` stems to `t` and
  occupies a vocabulary slot forever. Garbage in, garbage indexed.
- **[Stopword filtering](05-stopwords.md)** interacts by order. Stemming first turns `having`
  into `have`, which changes whether a stopword list matches, so the list is
  defined against a particular position in the pipeline rather than in the
  abstract.
- **The inverted index** stores stems as its keys, so this function determines
  the vocabulary and therefore the index size.
- **Phrase queries** are unaffected in structure, since stemming is
  position-preserving: it rewrites tokens without adding or removing any.
- **[Ranking](06-ranking.md)** is affected substantially. Merging forms merges their document
  frequencies, so every stemmed term has a different, lower IDF than its
  surface forms would have had, and the vocabulary size that normalises
  everything shrinks.
- **The query path** must apply this function exactly once, exactly as the
  indexing path does. Given that stemming is not idempotent, applying it twice
  on one side and once on the other would silently break matching.
