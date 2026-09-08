"""A synthetic corpus shared by every benchmark, so they describe one index.

Real text is the right thing to measure on, and a dump is not always to hand.
This generates text with the two properties that decide what an index costs.

**Term frequencies follow a Zipf-like curve.** A handful of terms appear in
almost every document and most appear in very few, which is what gives postings
lists their real length spread. Drawing terms uniformly instead would make every
postings list the same length and every measurement wrong.

**Document lengths vary by roughly two orders of magnitude.** Uniform lengths
hide anything that depends on length, which includes compression ratios and any
scorer with a length prior.

Only the timings and sizes measured against this are claimed. Anything about
real text needs real text, which `scripts/make_corpus.py` produces from a dump.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

SEED = 20260903

STEMS = (
    "connect",
    "compute",
    "retrieve",
    "document",
    "position",
    "invert",
    "rank",
    "score",
    "weight",
    "frequent",
    "index",
    "search",
    "query",
    "term",
    "vector",
    "normal",
    "similar",
    "relate",
    "general",
    "national",
)
SUFFIXES = ("", "s", "ed", "ing", "ion", "ions", "al", "ally", "ive", "ness")

# Roughly the shape of real article lengths: most are short, a few are long.
SHORTEST = 20
LONGEST = 2_000


def make_vocabulary(size: int) -> list[str]:
    """Build distinct inflected words, so stemming has real work to do."""
    words: list[str] = []
    for number in range(size):
        base = STEMS[number % len(STEMS)]
        suffix = SUFFIXES[(number // len(STEMS)) % len(SUFFIXES)]
        words.append(f"{base}{suffix}{number // (len(STEMS) * len(SUFFIXES))}")
    return words


def zipf_weights(size: int) -> list[float]:
    """Return weights falling as one over the rank."""
    return [1.0 / (rank + 1) for rank in range(size)]


def fixed_length_corpus(
    documents: int, tokens: int, vocabulary: list[str], seed: int = SEED
) -> list[str]:
    """Generate documents of one length, for comparing against earlier runs.

    Kept because the build and format benchmarks were first measured this way,
    and changing the generator would make those numbers incomparable.
    """
    rng = random.Random(seed)
    weights = zipf_weights(len(vocabulary))
    return [
        " ".join(rng.choices(vocabulary, weights=weights, k=tokens))
        for _ in range(documents)
    ]


def varied_length_corpus(
    documents: int, vocabulary: list[str], seed: int = SEED
) -> Iterator[str]:
    """Yield documents whose lengths span two orders of magnitude.

    Lengths are drawn from a curve that puts most documents near the short end
    and a few far out, which is how article lengths actually distribute.
    Yielded one at a time so a corpus larger than memory can be written.
    """
    rng = random.Random(seed)
    weights = zipf_weights(len(vocabulary))
    for _ in range(documents):
        # Uniform in the exponent gives a length distribution that is uniform
        # per order of magnitude rather than per token.
        length = int(SHORTEST * (LONGEST / SHORTEST) ** rng.random())
        yield " ".join(rng.choices(vocabulary, weights=weights, k=length))
