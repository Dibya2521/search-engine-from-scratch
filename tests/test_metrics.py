"""Tests for the counters, gauges and histograms, and for what records into them.

The engine records into one process-wide registry, so the wiring tests read a
metric before and after and assert the difference. Asserting an absolute would
make a test depend on every test that ran before it.
"""

from __future__ import annotations

import math
import re
import threading
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from search_engine.bm25 import BM25Ranker
from search_engine.index import InvertedIndex
from search_engine.metrics import (
    DEFAULT_BUCKETS,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    MetricError,
    Registry,
)
from search_engine.query import CachedParser, search
from search_engine.ranking import Ranker
from search_engine.segment import SegmentReader, write_segment
from search_engine.wand import search_wand
from tests.conftest import CORPUS, build_index

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

SAMPLE = "sample_seconds"

_SAMPLE_LINE = re.compile(
    r"[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? -?([0-9.eE+]+|\+Inf|NaN)\Z"
)


class BrokenIndex(InvertedIndex):
    """An index whose postings cannot be read, the way a damaged file is."""

    def postings(self, term: str) -> Mapping[int, Sequence[int]]:
        """Fail rather than answer, so the error path can be exercised."""
        message = f"postings unreadable: {term}"
        raise OSError(message)


def histogram(*bounds: float) -> Histogram:
    """Return a fresh histogram over these bounds."""
    return Histogram(SAMPLE, "For a test.", bounds or DEFAULT_BUCKETS)


def test_a_counter_starts_at_zero_and_only_goes_up() -> None:
    counter = Counter("things_total", "Things.")
    assert counter.value == 0
    counter.increment()
    counter.increment(4)
    assert counter.value == 5


def test_a_counter_refuses_to_decrease() -> None:
    """A scraper reads a fall as a restart, so it must not be possible."""
    counter = Counter("things_total", "Things.")
    with pytest.raises(MetricError, match="cannot decrease"):
        counter.increment(-1)


def test_a_gauge_goes_both_ways() -> None:
    gauge = Gauge("things_open", "Things.")
    gauge.increment(3)
    gauge.decrement()
    assert gauge.value == 2


def test_a_bucket_holds_the_value_on_its_own_bound() -> None:
    """At or below, so an observation of exactly 0.01 is in the 0.01 bucket."""
    subject = histogram(0.01, 0.1)
    subject.observe(0.01)
    assert subject.quantile(1.0) == 0.01
    subject.observe(0.010001)
    assert subject.quantile(1.0) == 0.1


def test_the_quantile_of_nothing_is_zero_rather_than_an_error() -> None:
    assert histogram().quantile(0.99) == 0.0


def test_an_observation_above_every_bound_lands_in_the_overflow() -> None:
    subject = histogram(0.01, 0.1)
    subject.observe(50.0)
    assert subject.count == 1
    assert subject.total == 50.0
    assert subject.quantile(1.0) == math.inf


def test_a_quantile_is_the_bound_of_the_bucket_it_falls_in() -> None:
    """An upper bound, so it is never too small, and never the exact value."""
    subject = histogram(1.0, 2.0, 3.0)
    for value in (0.5, 1.5, 2.5, 2.9):
        subject.observe(value)
    assert subject.quantile(0.0) == 1.0
    assert subject.quantile(0.5) == 2.0
    assert subject.quantile(1.0) == 3.0


def test_an_empty_leading_bucket_cannot_answer_below_every_observation() -> None:
    """The adversarial case for the rank arithmetic, and it was wrong once."""
    subject = histogram(1.0, 2.0, 3.0)
    subject.observe(2.5)
    assert subject.quantile(0.0) == 3.0


def test_bounds_that_do_not_ascend_are_refused() -> None:
    with pytest.raises(MetricError, match="must ascend"):
        histogram(1.0, 1.0)
    with pytest.raises(MetricError, match="must ascend"):
        Histogram(SAMPLE, "For a test.", ())


@pytest.mark.parametrize("value", [-0.001, math.inf, -math.inf, math.nan])
def test_an_observation_that_is_not_a_measurement_is_refused(value: float) -> None:
    """A NaN would poison the sum silently and an infinity would do it forever."""
    with pytest.raises(MetricError, match="finite and not negative"):
        histogram().observe(value)


@pytest.mark.parametrize("q", [-0.1, 1.1])
def test_a_quantile_outside_zero_to_one_is_refused(q: float) -> None:
    with pytest.raises(MetricError, match="between 0 and 1"):
        histogram().quantile(q)


def test_twenty_threads_observing_leave_an_exact_count() -> None:
    """An int increment is not atomic, and a metric quietly short is believed."""
    subject = histogram()
    per_thread = 500

    def observe_many() -> None:
        for _ in range(per_thread):
            subject.observe(0.002)

    threads = [threading.Thread(target=observe_many) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert subject.count == 20 * per_thread
    assert subject.total == pytest.approx(20 * per_thread * 0.002)


def test_twenty_threads_incrementing_leave_an_exact_count() -> None:
    counter = Counter("things_total", "Things.")

    def increment_many() -> None:
        for _ in range(500):
            counter.increment()

    threads = [threading.Thread(target=increment_many) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert counter.value == 20 * 500


@given(observations=st.lists(st.floats(0.0, 100.0), max_size=60))
def test_a_higher_quantile_is_never_a_lower_value(observations: list[float]) -> None:
    subject = histogram()
    for value in observations:
        subject.observe(value)
    assert subject.quantile(1.0) >= subject.quantile(0.5) >= subject.quantile(0.0)


def test_registering_the_same_name_returns_the_same_metric() -> None:
    """A module asks at import and a test asks again to read it."""
    registry = Registry()
    first = registry.counter("things_total", "Things.")
    first.increment()
    assert registry.counter("things_total", "Things.") is first
    assert registry.counter("things_total", "Things.").value == 1


def test_a_name_already_taken_by_another_kind_is_refused() -> None:
    registry = Registry()
    registry.counter("things_total", "Things.")
    with pytest.raises(MetricError, match="already registered as a counter"):
        registry.gauge("things_total", "Things.")
    with pytest.raises(MetricError, match="already registered as a counter"):
        registry.histogram("things_total", "Things.")


@pytest.mark.parametrize("name", ["1things", "things-total", "things total", ""])
def test_a_name_a_scraper_would_reject_is_refused(name: str) -> None:
    """An exposition holding one bad name is rejected whole, costing all of them."""
    with pytest.raises(MetricError, match="not a valid metric name"):
        Registry().counter(name, "Things.")


def test_every_rendered_line_is_valid_exposition() -> None:
    registry = Registry()
    registry.counter("things_total", "Things counted.").increment(3)
    registry.gauge("things_open", "Things open.").increment(2)
    subject = registry.histogram("wait_seconds", "Waits.", (0.1, 1.0))
    subject.observe(0.05)
    subject.observe(20.0)

    rendered = registry.render()
    for line in rendered.splitlines():
        if line.startswith("#"):
            assert re.fullmatch(r"# (HELP|TYPE) [a-zA-Z_:][a-zA-Z0-9_:]* .+", line)
        else:
            assert _SAMPLE_LINE.fullmatch(line), line

    assert "things_total 3\n" in rendered
    assert "things_open 2\n" in rendered
    assert 'wait_seconds_bucket{le="1.0"} 1\n' in rendered
    assert 'wait_seconds_bucket{le="+Inf"} 2\n' in rendered
    assert "wait_seconds_count 2\n" in rendered


def test_metrics_are_rendered_in_a_stable_order() -> None:
    """Two renderings of one state are the same bytes, so a diff means something."""
    registry = Registry()
    registry.gauge("zulu", "Last.")
    registry.counter("alpha_total", "First.")
    assert registry.names == ("alpha_total", "zulu")
    assert registry.render().index("alpha_total") < registry.render().index("zulu")


def test_a_search_is_counted_timed_and_its_emptiness_recorded() -> None:
    index = build_index(CORPUS)
    queries = REGISTRY.counter("queries_total", "Searches run.")
    empty = REGISTRY.counter("query_zero_results_total", "Nothing matched.")
    duration = REGISTRY.histogram("query_duration_seconds", "How long.")
    before = (queries.value, empty.value, duration.count)

    assert search(index, "index")
    assert not search(index, "nonexistentword")

    assert queries.value == before[0] + 2
    assert empty.value == before[1] + 1
    assert duration.count == before[2] + 2


def test_a_search_that_raises_is_counted_as_an_error() -> None:
    errors = REGISTRY.counter("query_errors_total", "Searches that raised.")
    before = errors.value
    with pytest.raises(OSError, match="postings unreadable"):
        search(BrokenIndex(), "alpha")
    assert errors.value == before + 1


def test_a_cached_parse_is_counted_as_a_hit_and_a_miss() -> None:
    hits = REGISTRY.counter("query_cache_hits_total", "Parses reused.")
    misses = REGISTRY.counter("query_cache_misses_total", "Parses made.")
    before = (hits.value, misses.value)

    parser = CachedParser()
    parser.parse("alpha beta")
    parser.parse("alpha beta")

    assert misses.value == before[1] + 1
    assert hits.value == before[0] + 1


def test_scoring_counts_every_candidate_it_was_given() -> None:
    index = build_index(CORPUS)
    scored = REGISTRY.counter("documents_scored_total", "Documents scored.")
    before = scored.value
    Ranker(index).rank("index", {1, 2, 3})
    assert scored.value == before + 3


def test_an_open_segment_is_counted_until_it_is_closed(tmp_path: Path) -> None:
    """The gauge is the merge backlog, so it has to come back down."""
    path = tmp_path / "one.seg"
    write_segment(build_index(CORPUS), path)
    open_segments = REGISTRY.gauge("segments_open", "Segments mapped.")
    before = open_segments.value

    with SegmentReader(path) as reader:
        assert open_segments.value == before + 1
        hits = REGISTRY.counter("postings_cache_hits_total", "Cache hits.")
        misses = REGISTRY.counter("postings_cache_misses_total", "Cache misses.")
        marks = (hits.value, misses.value)
        reader.postings("index")
        reader.postings("index")
        assert misses.value == marks[1] + 1
        assert hits.value == marks[0] + 1

    assert open_segments.value == before


def test_a_segment_that_fails_to_open_leaves_the_gauge_where_it_was(
    tmp_path: Path,
) -> None:
    """The constructor closes what it opened, and the gauge has to follow."""
    path = tmp_path / "broken.seg"
    write_segment(build_index(CORPUS), path)
    path.write_bytes(path.read_bytes()[:-1])
    open_segments = REGISTRY.gauge("segments_open", "Segments mapped.")
    before = open_segments.value

    with pytest.raises(ValueError, match="segment"):
        SegmentReader(path)
    assert open_segments.value == before


def test_early_termination_reports_what_it_actually_scored() -> None:
    """The number this path exists to make smaller, so it has to be the one shown."""
    index = build_index(CORPUS)
    scored = REGISTRY.counter("documents_scored_total", "Documents scored.")
    before = scored.value
    result = search_wand(index, BM25Ranker(index), ["index"], 1)
    assert scored.value == before + result.scored
