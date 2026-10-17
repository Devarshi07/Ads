"""Rate limiting (design doc: Security & Threat Model → abuse, API rate limits).

GCRA (generic cell rate algorithm): each key may make `burst` requests at once
and then `rate` per second. The in-process limiter serves one API process; the
Redis limiter shares the budget across replicas with one atomic script.

A broken limiter fails open (and is counted): rate limiting protects capacity,
while safety-critical state already fails closed elsewhere.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimit:
    rate: float  # sustained requests per second
    burst: int  # requests allowed at once

    @property
    def interval_ms(self) -> float:
        return 1000.0 / self.rate


DEFAULT_LIMITS: Mapping[str, RateLimit] = {
    "rank": RateLimit(10.0, 60),  # per user
    "events": RateLimit(30.0, 120),  # per user
    "click": RateLimit(5.0, 30),  # per client address (unauthenticated)
    "postback": RateLimit(100.0, 500),  # per advertiser
    "creative": RateLimit(0.5, 20),  # per advertiser
    "report": RateLimit(0.5, 10),  # per advertiser
    "why": RateLimit(1.0, 10),  # per user
    "admin": RateLimit(5.0, 30),  # per admin
}


@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after_s: float = 0.0


def limit_key(scope: str, principal: str) -> str:
    """Principals are hashed: rate-limit keys outlive requests and must not hold user IDs."""
    digest = hashlib.sha256(principal.encode()).hexdigest()[:24]
    return f"rl:{scope}:{digest}"


class InProcessRateLimiter:
    def __init__(
        self,
        limits: Mapping[str, RateLimit] = DEFAULT_LIMITS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limits = dict(limits)
        self._clock = clock
        self._tat: dict[str, float] = {}
        self._lock = threading.Lock()

    def check(self, scope: str, principal: str) -> Decision:
        limit = self._limits.get(scope)
        if limit is None:
            return Decision(True)
        key = limit_key(scope, principal)
        now = self._clock() * 1000.0
        with self._lock:
            tat = max(self._tat.get(key, now), now)
            new_tat = tat + limit.interval_ms
            allow_at = new_tat - limit.burst * limit.interval_ms
            if allow_at > now:
                return Decision(False, (allow_at - now) / 1000.0)
            self._tat[key] = new_tat
            if len(self._tat) > 100_000:  # drop keys whose budget has fully recovered
                self._tat = {k: v for k, v in self._tat.items() if v > now}
        return Decision(True)


_GCRA = """
local now = tonumber(ARGV[1])
local interval = tonumber(ARGV[2])
local tolerance = tonumber(ARGV[3])
local tat = tonumber(redis.call('GET', KEYS[1]) or ARGV[1])
if tat < now then tat = now end
local new_tat = tat + interval
local allow_at = new_tat - tolerance
if allow_at > now then
  return {0, tostring(allow_at - now)}
end
redis.call('SET', KEYS[1], tostring(new_tat), 'PX', math.ceil(new_tat - now) + 1000)
return {1, '0'}
"""


class RedisRateLimiter:
    def __init__(
        self,
        client,
        limits: Mapping[str, RateLimit] = DEFAULT_LIMITS,
        on_error: Callable[[], None] | None = None,
    ) -> None:
        self._redis = client
        self._limits = dict(limits)
        self._script = client.register_script(_GCRA)
        self._on_error = on_error

    def check(self, scope: str, principal: str) -> Decision:
        import redis

        limit = self._limits.get(scope)
        if limit is None:
            return Decision(True)
        try:
            # Server time, so replicas with skewed clocks share one budget correctly.
            seconds, micros = self._redis.time()
            now = seconds * 1000.0 + micros / 1000.0
            allowed, wait = self._script(
                keys=[limit_key(scope, principal)],
                args=[now, limit.interval_ms, limit.burst * limit.interval_ms],
            )
        except redis.RedisError:
            if self._on_error is not None:
                self._on_error()
            return Decision(True)
        return Decision(bool(allowed), float(wait) / 1000.0)


class NoRateLimit:
    def check(self, scope: str, principal: str) -> Decision:
        return Decision(True)


def retry_after_header(decision: Decision) -> str:
    return str(max(1, math.ceil(decision.retry_after_s)))
