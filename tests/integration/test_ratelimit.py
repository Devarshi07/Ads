"""The Redis rate limiter shares one budget across API replicas and fails open."""

from __future__ import annotations

import pytest
import redis

from tests.integration.conftest import REDIS_URL
from theraflo_ads.observability.ratelimit import RateLimit, RedisRateLimiter

pytestmark = pytest.mark.integration


def test_replicas_share_one_budget(settings):
    limits = {"rank": RateLimit(rate=0.01, burst=3)}
    a = RedisRateLimiter(redis.Redis.from_url(REDIS_URL), limits)
    b = RedisRateLimiter(redis.Redis.from_url(REDIS_URL), limits)
    results = [limiter.check("rank", "user-1").allowed for limiter in (a, b, a, b)]
    assert results == [True, True, True, False]
    denied = a.check("rank", "user-1")
    assert not denied.allowed and denied.retry_after_s > 0
    assert b.check("rank", "user-2").allowed


def test_limiter_outage_fails_open_and_is_counted(settings):
    errors = []
    down = redis.Redis(port=1, socket_connect_timeout=0.05)
    limiter = RedisRateLimiter(down, {"rank": RateLimit(1.0, 1)}, on_error=lambda: errors.append(1))
    assert limiter.check("rank", "user-1").allowed and errors == [1]
