"""Circuit breakers (spec §12): closed -> open (N failures in a window) -> half-open probe.

State is exported as the `atlas_circuit_state` gauge (0 closed / 1 half_open / 2 open).
A failure is ANY exception raised by the wrapped call — retries live inside the wrapped
callables (adapters), so by the time the breaker sees an exception the call is truly dead.
"""

import collections
import time
from collections.abc import Callable
from typing import Any

from atlas.observability.metrics import CIRCUIT_STATE

CLOSED = "closed"
HALF_OPEN = "half_open"
OPEN = "open"

_STATE_CODE = {CLOSED: 0, HALF_OPEN: 1, OPEN: 2}


class CircuitOpenError(Exception):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"circuit open: {name}")


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        window_s: float = 60.0,
        reset_timeout_s: float = 30.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.window_s = window_s
        self.reset_timeout_s = reset_timeout_s
        self._failures: collections.deque[float] = collections.deque()
        self._state = CLOSED
        self._opened_at = 0.0
        self._probe_in_flight = False
        self._publish()

    @property
    def state(self) -> str:
        if self._state == OPEN and time.monotonic() - self._opened_at >= self.reset_timeout_s:
            self._state = HALF_OPEN
            self._probe_in_flight = False
            self._publish()
        return self._state

    def _publish(self) -> None:
        CIRCUIT_STATE.labels(dependency=self.name).set(_STATE_CODE[self._state])

    def _record_success(self) -> None:
        self._failures.clear()
        self._state = CLOSED
        self._probe_in_flight = False
        self._publish()

    def _record_failure(self) -> None:
        now = time.monotonic()
        while self._failures and self._failures[0] <= now - self.window_s:
            self._failures.popleft()
        self._failures.append(now)
        if self._state == HALF_OPEN or len(self._failures) >= self.failure_threshold:
            self._state = OPEN
            self._opened_at = now
            self._probe_in_flight = False
        self._publish()

    async def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        state = self.state
        if state == OPEN:
            raise CircuitOpenError(self.name)
        if state == HALF_OPEN:
            if self._probe_in_flight:
                raise CircuitOpenError(self.name)
            self._probe_in_flight = True
            try:
                result = await fn(*args, **kwargs)
            except Exception:
                self._probe_in_flight = False
                self._record_failure()
                raise
            self._probe_in_flight = False
            self._record_success()
            return result
        try:
            result = await fn(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        return result


class Breakers:
    """One breaker per dependency (spec §12): llm, qdrant, redis, rerank."""

    def __init__(
        self,
        llm: CircuitBreaker | None = None,
        qdrant: CircuitBreaker | None = None,
        redis: CircuitBreaker | None = None,
        rerank: CircuitBreaker | None = None,
    ) -> None:
        self.llm = llm or CircuitBreaker("llm")
        self.qdrant = qdrant or CircuitBreaker("qdrant")
        self.redis = redis or CircuitBreaker("redis")
        self.rerank = rerank or CircuitBreaker("rerank")

    @classmethod
    def build(cls) -> "Breakers":
        return cls()
