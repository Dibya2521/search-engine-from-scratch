"""Compare TF-IDF and BM25 on the committed judgement set.

Run with ``uv run python benchmarks/ranking_quality.py``.

Both scorers are given the same index, the same candidate set from the same
matcher, and the same result limit, so any difference in the numbers comes from
the scoring formula alone.

The judgements were written and committed before this comparison existed, which
the commit history shows. Judgements made after seeing results follow the engine
instead of judging it.

Reports mean average precision, mean nDCG at 10 and mean reciprocal rank. Mean
average precision is the headline number because it accounts for both what was
returned and the order it came back in. nDCG is reported alongside because it is
the only one of the three that uses the relevance grades.

A difference between two means is not by itself evidence that one system is
better, so two significance tests are reported with it. The sign test uses only
which system won each query. The permutation test also uses the size of each
difference, by repeatedly reassigning the differences to either system at random
and counting how often chance alone produces a gap this large.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from search_engine import bm25, bm25f, judgements, ranking
from search_engine.corpus import read
from search_engine.evaluation import (
    average_precision,
    normalized_discounted_cumulative_gain,
    precision_at_k,
    reciprocal_rank,
    sign_test,
)
from search_engine.index import InvertedIndex
from search_engine.query import search

if TYPE_CHECKING:
    from search_engine.judgements import Judgement
    from search_engine.ranking import BaseRanker

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
CORPUS = FIXTURES / "evaluation_corpus.xml"
JUDGEMENTS = FIXTURES / "judgements.jsonl"

LIMIT = 10
PRECISION_AT = 5
PERMUTATION_TRIALS = 100_000
SEED = 20260906


@dataclass(frozen=True, slots=True)
class Result:
    """The aggregate scores one scorer achieved over the whole query set."""

    name: str
    mean_average_precision: float
    mean_ndcg: float
    mean_reciprocal_rank: float
    mean_precision_at_k: float
    per_query_average_precision: dict[str, float]


def build_index() -> InvertedIndex:
    """Index the evaluation corpus."""
    index = InvertedIndex()
    for document in read(CORPUS):
        index.add_document(document.identifier, document.indexable_text)
    return index


def build_index_with_fields() -> InvertedIndex:
    """Index the evaluation corpus, recording each document's title as a field."""
    index = InvertedIndex()
    for document in read(CORPUS):
        index.add_document(
            document.identifier, document.indexable_text, document.fields
        )
    return index


def evaluate(
    name: str, ranker: BaseRanker, index: InvertedIndex, labelled: list[Judgement]
) -> Result:
    """Score every judged query with one ranker and aggregate the metrics."""
    precisions: dict[str, float] = {}
    ndcgs: list[float] = []
    reciprocals: list[float] = []
    at_k: list[float] = []
    for judgement in labelled:
        candidates = search(index, judgement.query)
        retrieved = [
            document_id
            for document_id, _ in ranker.rank(judgement.query, candidates, limit=LIMIT)
        ]
        precisions[judgement.query] = average_precision(retrieved, judgement.relevant)
        ndcgs.append(
            normalized_discounted_cumulative_gain(retrieved, judgement.grades, LIMIT)
        )
        reciprocals.append(reciprocal_rank(retrieved, judgement.relevant))
        at_k.append(precision_at_k(retrieved, judgement.relevant, PRECISION_AT))
    count = len(labelled)
    return Result(
        name=name,
        mean_average_precision=sum(precisions.values()) / count,
        mean_ndcg=sum(ndcgs) / count,
        mean_reciprocal_rank=sum(reciprocals) / count,
        mean_precision_at_k=sum(at_k) / count,
        per_query_average_precision=precisions,
    )


def permutation_test(differences: list[float]) -> float:
    """Return the two-sided p-value of a paired permutation test.

    If the two scorers were equally good, the sign of each per-query difference
    would be arbitrary. Flipping each sign at random many times and counting how
    often the resulting mean difference is at least as large as the observed one
    gives the probability of seeing this result by chance.

    Unlike the sign test this uses the size of each difference, so a few large
    swings count for more than many small ones.
    """
    generator = random.Random(SEED)
    observed = abs(sum(differences) / len(differences))
    at_least_as_extreme = sum(
        1
        for _ in range(PERMUTATION_TRIALS)
        if abs(sum(generator.choice((one, -one)) for one in differences))
        >= observed * len(differences)
    )
    # Add one to each side so a p-value of exactly zero is never reported: the
    # observed arrangement is itself one of the arrangements being counted.
    return (at_least_as_extreme + 1) / (PERMUTATION_TRIALS + 1)


def print_aggregates(baseline: Result, challenger: Result) -> None:
    """Print the four aggregate metrics for both scorers, side by side."""
    header = f"{'metric':<24}{baseline.name:>12}{challenger.name:>12}{'change':>12}"
    print(header)
    print("-" * len(header))
    rows = (
        (
            "mean average precision",
            baseline.mean_average_precision,
            challenger.mean_average_precision,
        ),
        ("mean nDCG@10", baseline.mean_ndcg, challenger.mean_ndcg),
        (
            "mean reciprocal rank",
            baseline.mean_reciprocal_rank,
            challenger.mean_reciprocal_rank,
        ),
        (
            f"mean precision@{PRECISION_AT}",
            baseline.mean_precision_at_k,
            challenger.mean_precision_at_k,
        ),
    )
    for label, left, right in rows:
        change = f"{(right - left) / left:+.1%}" if left else "n/a"
        print(f"{label:<24}{left:>12.4f}{right:>12.4f}{change:>12}")


def print_per_query(baseline: Result, challenger: Result) -> None:
    """Print the win and loss record, its significance, and every difference."""
    queries = list(baseline.per_query_average_precision)
    paired = [
        challenger.per_query_average_precision[query]
        - baseline.per_query_average_precision[query]
        for query in queries
    ]
    wins = sum(1 for one in paired if one > 0)
    losses = sum(1 for one in paired if one < 0)
    print(
        f"\nper-query average precision: {challenger.name} wins {wins}, "
        f"loses {losses}, ties {len(paired) - wins - losses}"
    )
    print(f"sign test p-value         {sign_test(paired):>9.3f}")
    print(f"permutation test p-value  {permutation_test(paired):>9.4f}")
    print("above 0.05 means the difference is not distinguishable from chance")
    print()

    changes = sorted(
        (change, query)
        for change, query in zip(paired, queries, strict=True)
        if change != 0
    )
    for change, query in changes:
        left = baseline.per_query_average_precision[query]
        right = challenger.per_query_average_precision[query]
        print(f"  {change:+.4f}  {left:.4f} -> {right:.4f}  {query}")


if __name__ == "__main__":
    built = build_index()
    all_judgements = judgements.load(JUDGEMENTS)
    tfidf = evaluate("tf-idf", ranking.Ranker(built), built, all_judgements)
    okapi = evaluate("bm25", bm25.BM25Ranker(built), built, all_judgements)
    near = evaluate(
        "bm25+proximity",
        bm25.BM25Ranker(built, proximity=True),
        built,
        all_judgements,
    )

    print(f"corpus    {built.document_count} documents")
    print(f"queries   {len(all_judgements)}")
    print(f"limit     top {LIMIT}")
    print()
    print_aggregates(tfidf, okapi)
    print_per_query(tfidf, okapi)
    print()
    print("=" * 60)
    print("proximity boost against the same scorer without it")
    print("=" * 60)
    print()
    print_aggregates(okapi, near)
    print_per_query(okapi, near)

    fielded = build_index_with_fields()
    weighted = evaluate("bm25f", bm25f.BM25FRanker(fielded), fielded, all_judgements)
    plain_on_fielded = evaluate(
        "bm25", bm25.BM25Ranker(fielded), fielded, all_judgements
    )
    print()
    print("=" * 60)
    print("bm25f against bm25, both over the same field-aware index")
    print("=" * 60)
    print()
    print_aggregates(plain_on_fielded, weighted)
    print_per_query(plain_on_fielded, weighted)
