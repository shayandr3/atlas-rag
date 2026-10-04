"""API key authentication (spec §11.1).

Keys exist only as sha256(pepper + raw) hashes in API_KEYS_JSON (env var — survives
Redis restarts). Verification is constant-time per candidate. Three token realms:
/v1/* (API keys), /metrics (METRICS_BEARER_TOKEN), /admin/* (ADMIN_TOKEN).
"""

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass

from atlas.config import Settings

logger = logging.getLogger("atlas.security")


@dataclass(frozen=True)
class ApiKeyRecord:
    id: str
    sha256_hash: str
    rpm: int = 60
    daily_budget_usd: float = 0.25
    scopes: tuple[str, ...] = ("ask",)


def hash_key(raw_key: str, pepper: str) -> str:
    return hashlib.sha256((pepper + raw_key).encode("utf-8")).hexdigest()


def load_keys(api_keys_json: str) -> list[ApiKeyRecord]:
    if not api_keys_json.strip():
        return []
    try:
        rows = json.loads(api_keys_json)
    except json.JSONDecodeError:
        logger.error("API_KEYS_JSON is not valid JSON — no keys loaded")
        return []
    records: list[ApiKeyRecord] = []
    for row in rows if isinstance(rows, list) else []:
        try:
            records.append(
                ApiKeyRecord(
                    id=str(row["id"]),
                    sha256_hash=str(row["sha256_hash"]),
                    rpm=int(row.get("rpm", 60)),
                    daily_budget_usd=float(row.get("daily_budget_usd", 0.25)),
                    scopes=tuple(row.get("scopes", ["ask"])),
                )
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("skipping malformed API key record: %r", row)
    return records


def verify_key(raw_key: str, records: list[ApiKeyRecord], pepper: str) -> ApiKeyRecord | None:
    """Constant-time comparison against every configured hash."""
    if not records or not raw_key:
        return None
    digest = hash_key(raw_key, pepper)
    for record in records:
        if hmac.compare_digest(digest, record.sha256_hash):
            return record
    return None


@dataclass(frozen=True)
class Principal:
    """Authenticated caller; anonymous in dev-only open mode."""

    id: str
    rpm: int
    daily_budget_usd: float
    scopes: tuple[str, ...] = ("ask",)
    anonymous: bool = False


def resolve_principal(
    bearer: str | None, settings: Settings, records: list[ApiKeyRecord] | None = None
) -> Principal | None:
    """None → invalid/missing credentials (401). Anonymous principal only in dev open mode."""
    records = load_keys(settings.api_keys_json) if records is None else records
    if bearer is not None:
        record = verify_key(bearer, records, settings.api_key_pepper)
        if record is None:
            return None
        return Principal(record.id, record.rpm, record.daily_budget_usd, record.scopes)
    if not records and settings.app_env == "dev":
        # dev open mode: no keys configured — local iteration stays frictionless
        logger.warning("dev open mode: API_KEYS_JSON empty; anonymous principal granted")
        return Principal(
            "anonymous",
            rpm=60,
            daily_budget_usd=settings.default_key_daily_budget_usd,
            anonymous=True,
        )
    return None


def check_token(provided: str | None, expected: str) -> bool:
    """Bearer-token realm check (metrics/admin). Empty expected token denies non-dev."""
    if not expected or not provided:
        return False
    return hmac.compare_digest(provided, expected)
