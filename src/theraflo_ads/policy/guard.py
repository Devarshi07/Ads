"""Circuit breakers and latency SLOs for policy levels (design doc: Circuit Breakers, Timeouts).

A guarded level that keeps failing, or keeps exceeding its latency budget, opens
its circuit: it is skipped (the fallback chain serves) until a cooldown passes,
then one probe request is let through.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from theraflo_ads.domain import Ad, AdRequest
from theraflo_ads.policy.policies import ActionDistribution, Policy


class CircuitOpen(RuntimeError):
    pass


class LatencyBudgetExceeded(RuntimeError):
    pass


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    reset_after_s: float = 30.0
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            if self.clock() - self._opened_at >= self.reset_after_s:
                return "half_open"
            return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold or self._opened_at is not None:
                self._opened_at = self.clock()


class GuardedPolicy:
    """Wraps a policy level with a circuit breaker and a latency budget.

    Scoring runs in-process, so a slow call cannot be interrupted; it is counted as a
    failure after the fact, and repeated breaches open the circuit.
    """

    def __init__(
        self,
        policy: Policy,
        *,
        breaker: CircuitBreaker | None = None,
        latency_budget_ms: float = 40.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._policy = policy
        self.breaker = breaker or CircuitBreaker()
        self._budget = latency_budget_ms / 1000.0
        self._clock = clock
        self.served_by = policy.served_by
        self.version = policy.version
        self.latencies_ms: list[float] = []

    def distribution(self, request: AdRequest, eligible: Sequence[Ad]) -> ActionDistribution:
        if not self.breaker.allow():
            raise CircuitOpen(self.version)
        start = self._clock()
        try:
            dist = self._policy.distribution(request, eligible)
        except Exception:
            self.breaker.record_failure()
            raise
        elapsed = self._clock() - start
        self.latencies_ms.append(elapsed * 1000)
        if elapsed > self._budget:
            self.breaker.record_failure()
            raise LatencyBudgetExceeded(f"{elapsed * 1000:.1f}ms > {self._budget * 1000:.0f}ms")
        self.breaker.record_success()
        return dist
