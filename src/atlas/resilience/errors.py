"""Typed error taxonomy mapped to RFC 9457 application/problem+json (spec §12).

Stable `code`, `title`, `status`; `request_id` attached at construction; no internals
leak into `detail`. `Retry-After` is emitted for 429/503 families.
"""

import uuid
from typing import Any


class AtlasError(Exception):
    status: int = 500
    code: str = "internal"
    title: str = "Internal error"
    retry_after: int | None = None

    def __init__(
        self,
        detail: str = "",
        request_id: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        self.detail = detail
        self.request_id = request_id or uuid.uuid4().hex
        if retry_after is not None:
            self.retry_after = retry_after
        super().__init__(detail)

    def problem(self) -> dict[str, Any]:
        return {
            "type": f"https://atlas-rag.dev/errors/{self.code}",
            "title": self.title,
            "status": self.status,
            "code": self.code,
            "detail": self.detail,
            "request_id": self.request_id,
        }


class Unauthorized(AtlasError):
    status = 401
    code = "unauthorized"
    title = "Invalid or missing credentials"


class GuardBlocked(AtlasError):
    status = 403
    code = "guard_blocked"
    title = "Request blocked by the input guard"


class RateLimited(AtlasError):
    status = 429
    code = "rate_limited"
    title = "Rate limit exceeded"
    retry_after = 60


class BudgetExceeded(AtlasError):
    status = 429
    code = "budget_exceeded"
    title = "Budget exhausted"
    retry_after = 3600


class UpstreamUnavailable(AtlasError):
    status = 503
    code = "upstream_unavailable"
    title = "Upstream unavailable"
    retry_after = 30


class UpstreamTimeout(AtlasError):
    status = 504
    code = "upstream_timeout"
    title = "Upstream timeout"
    retry_after = 10


class ValidationFailed(AtlasError):
    status = 422
    code = "validation_failed"
    title = "Validation failed"


class Internal(AtlasError):
    status = 500
    code = "internal"
    title = "Internal error"
