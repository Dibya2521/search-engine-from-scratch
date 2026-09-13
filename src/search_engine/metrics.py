"""What the engine can be watched by.

Nothing here changes an answer. It changes whether anyone can tell what the
engine is doing while it does it.

**A mean hides everything worth knowing.** A search averaging 20 ms can have a
99th percentile of four seconds, and the four seconds is what a person notices
and remembers. So durations are counted into fixed buckets rather than averaged:
buckets keep enough of the distribution to answer for the tail, they cost a
fixed amount of memory however many observations arrive, and counts from several
processes can be added together while means cannot.

**A quantile read from buckets is an upper bound, not the true value.** That is
stated in `Histogram` and in `docs/18-operations.md` because an operator who
does not know it will chase a regression that is a bucket boundary.

Every mutation takes a lock. The serving layer is concurrent, an `int += 1` is
not atomic once the interpreter stops guaranteeing it, and a metric that is
occasionally wrong is worse than no metric because it is believed. What the lock
costs is measured under load.

The exposition format is Prometheus text, because it is the format every
scraper already reads and it is small enough to produce by hand. There is no
HTTP endpoint here; the serving layer adds one over `Registry.render`.
"""

from __future__ import annotations

import math
import re
import threading
from bisect import bisect_left
from itertools import pairwise
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Powers of roughly ten with a midpoint, from a millisecond to ten seconds.
# Wide enough that nothing lands in the overflow bucket in normal running, and
# fine enough that the interesting range is not one bucket.
DEFAULT_BUCKETS: Final = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

_NAME_PATTERN: Final = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*\Z")


class MetricError(ValueError):
    """Raised when a metric is named, registered or observed wrongly."""


class Metric:
    """One named measurement, and the kind a scraper should be told it is."""

    __slots__ = ("_lock", "help_text", "kind", "name")

    def __init__(self, name: str, help_text: str, kind: str) -> None:
        """Take the exposed name, its one-line description, and its kind."""
        self.name = name
        self.help_text = help_text
        self.kind = kind
        self._lock = threading.Lock()

    def expose(self) -> str:
        """Return this metric's lines in the text exposition format."""
        raise NotImplementedError

    def _header(self) -> str:
        return f"# HELP {self.name} {self.help_text}\n# TYPE {self.name} {self.kind}\n"


class Counter(Metric):
    """A count that only ever goes up."""

    __slots__ = ("_value",)

    def __init__(self, name: str, help_text: str) -> None:
        """Start a counter at zero."""
        super().__init__(name, help_text, "counter")
        self._value = 0

    def increment(self, amount: int = 1) -> None:
        """Add to the count.

        Raises:
            MetricError: If the amount is negative. A scraper computes a rate
                from the difference between two readings, so a counter going
                down is read as a process restart rather than as a smaller
                number.
        """
        if amount < 0:
            message = f"a counter cannot decrease, got {amount}"
            raise MetricError(message)
        with self._lock:
            self._value += amount

    @property
    def value(self) -> int:
        """Return the count so far."""
        with self._lock:
            return self._value

    def expose(self) -> str:
        """Return the counter as one sample line under its header."""
        return f"{self._header()}{self.name} {self.value}\n"


class Gauge(Metric):
    """A count of things that currently exist, which can go either way."""

    __slots__ = ("_value",)

    def __init__(self, name: str, help_text: str) -> None:
        """Start a gauge at zero."""
        super().__init__(name, help_text, "gauge")
        self._value = 0

    def increment(self, amount: int = 1) -> None:
        """Add to the gauge."""
        with self._lock:
            self._value += amount

    def decrement(self, amount: int = 1) -> None:
        """Subtract from the gauge."""
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> int:
        """Return the gauge as it now stands."""
        with self._lock:
            return self._value

    def expose(self) -> str:
        """Return the gauge as one sample line under its header."""
        return f"{self._header()}{self.name} {self.value}\n"


class Histogram(Metric):
    """Observations counted into fixed buckets, so the tail survives.

    A bucket counts every observation at or below its bound, which is what
    lets a quantile be computed from the counts alone and lets counts from
    several processes be added together.

    **A quantile read from these is an upper bound and not the true value.** It
    is the bound of the bucket the quantile falls in, so it is never too small
    and is wrong by at most that bucket's width. With the default buckets a
    99th percentile in the tail is reported as 5.0 when the truth is anywhere
    from 2.5 to 5.0.
    """

    __slots__ = ("_bounds", "_counts", "_total")

    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> None:
        """Take the bucket bounds, which must ascend and must not be empty.

        Raises:
            MetricError: If the bounds are empty or not strictly ascending.
                Equal or unordered bounds would make one bucket unreachable
                and every quantile read from them wrong.
        """
        super().__init__(name, help_text, "histogram")
        if not buckets or any(low >= high for low, high in pairwise(buckets)):
            message = f"bucket bounds must ascend and cannot be empty, got {buckets!r}"
            raise MetricError(message)
        self._bounds = tuple(buckets)
        # One more count than bounds: the last is everything above them all.
        self._counts = [0] * (len(self._bounds) + 1)
        self._total = 0.0

    def observe(self, value: float) -> None:
        """Record one observation.

        Raises:
            MetricError: If the value is negative or is not finite. Every
                observation here is a duration or a size, so neither is data.
                A NaN would silently poison the sum and every mean taken from
                it, and an infinity would do the same permanently.
        """
        if not math.isfinite(value) or value < 0:
            message = f"an observation must be finite and not negative, got {value!r}"
            raise MetricError(message)
        with self._lock:
            self._total += value
            self._counts[bisect_left(self._bounds, value)] += 1

    def quantile(self, q: float) -> float:
        """Return an upper bound on the value at this quantile.

        An empty histogram answers 0.0 rather than raising, so a monitoring
        path does not have to guard every read.

        Infinity is returned when the quantile falls above the largest bucket,
        because the bucket has no upper bound to report.

        Raises:
            MetricError: If the quantile is outside 0 to 1.
        """
        if not 0.0 <= q <= 1.0:
            message = f"a quantile must be between 0 and 1, got {q!r}"
            raise MetricError(message)
        with self._lock:
            counts = list(self._counts)
            total = sum(counts)
        if not total:
            return 0.0
        # The rank of the observation being asked for, never below the first,
        # so an empty leading bucket cannot answer below every observation.
        rank = max(math.ceil(q * total), 1)
        seen = 0
        for bound, count in zip(self._bounds, counts, strict=False):
            seen += count
            if seen >= rank:
                return bound
        return math.inf

    @property
    def count(self) -> int:
        """Return how many observations have been recorded."""
        with self._lock:
            return sum(self._counts)

    @property
    def total(self) -> float:
        """Return the sum of every observation."""
        with self._lock:
            return self._total

    def expose(self) -> str:
        """Return the cumulative buckets, the sum and the count."""
        with self._lock:
            counts = list(self._counts)
            total = self._total
        lines = [self._header()]
        running = 0
        for bound, count in zip(self._bounds, counts, strict=False):
            running += count
            lines.append(f'{self.name}_bucket{{le="{bound}"}} {running}\n')
        running += counts[-1]
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {running}\n')
        lines.append(f"{self.name}_sum {total}\n")
        lines.append(f"{self.name}_count {running}\n")
        return "".join(lines)


class Registry:
    """Every metric in one place, and the exposition of all of them."""

    __slots__ = ("_lock", "_metrics")

    def __init__(self) -> None:
        """Start with nothing registered."""
        self._metrics: dict[str, Metric] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help_text: str) -> Counter:
        """Return the counter with this name, creating it the first time."""
        return self._register(name, "counter", lambda: Counter(_valid(name), help_text))

    def gauge(self, name: str, help_text: str) -> Gauge:
        """Return the gauge with this name, creating it the first time."""
        return self._register(name, "gauge", lambda: Gauge(_valid(name), help_text))

    def histogram(
        self,
        name: str,
        help_text: str,
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> Histogram:
        """Return the histogram with this name, creating it the first time."""
        return self._register(
            name, "histogram", lambda: Histogram(_valid(name), help_text, buckets)
        )

    def render(self) -> str:
        """Return every metric in the Prometheus text exposition format.

        Sorted by name, so two renderings of the same state are the same
        bytes and a diff between them means something changed.
        """
        with self._lock:
            metrics = [self._metrics[name] for name in sorted(self._metrics)]
        return "".join(metric.expose() for metric in metrics)

    @property
    def names(self) -> tuple[str, ...]:
        """Return every registered name, sorted."""
        with self._lock:
            return tuple(sorted(self._metrics))

    def _register[M: Metric](self, name: str, kind: str, build: Callable[[], M]) -> M:
        """Return the metric with this name, building it if it is new.

        Registration is idempotent so that a module can ask for its metrics at
        import time and a test can ask for the same one back to read it.

        Raises:
            MetricError: If the name is already registered as another kind,
                which would make the exposed output contradict itself.
        """
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                if existing.kind != kind:
                    message = (
                        f"{name} is already registered as a {existing.kind}, "
                        f"not as a {kind}"
                    )
                    raise MetricError(message)
                return cast("M", existing)
            created = build()
            self._metrics[name] = created
            return created


def _valid(name: str) -> str:
    """Return a metric name, having checked a scraper will accept it.

    Raises:
        MetricError: If the name is not one Prometheus allows. An exposition
            containing one is rejected whole, so a bad name costs every other
            metric as well.
    """
    if not _NAME_PATTERN.fullmatch(name):
        message = f"not a valid metric name: {name!r}"
        raise MetricError(message)
    return name


# The one registry the engine records into. Metrics are process wide by nature:
# the alternative is threading a registry through every signature between the
# entry point and the event, which couples every layer to an observability
# concern. A test constructs its own `Registry` rather than clearing this one.
REGISTRY: Final = Registry()
