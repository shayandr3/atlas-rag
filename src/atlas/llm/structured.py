"""Structured-JSON output helper (spec §5): parse, validate, one repair retry."""

import json
import re
from typing import Any, Protocol

_JSON_START = re.compile(r"[\[{]")


class StructuredOutputError(Exception):
    pass


class _JsonLLM(Protocol):
    async def complete(self, *, system: str, user: str, model: str, max_tokens: int) -> Any: ...


def parse_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    match = _JSON_START.search(cleaned)
    if match is None:
        raise ValueError("no JSON found")
    candidate = cleaned[match.start() :]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        for end_char in ("]", "}"):
            idx = candidate.rfind(end_char)
            if idx != -1:
                return json.loads(candidate[: idx + 1])
        raise


async def complete_json(
    llm: _JsonLLM,
    *,
    system: str,
    user: str,
    model: str,
    max_tokens: int = 800,
    attempts: int = 2,
) -> Any:
    """Strict-JSON completion with one repair retry; raises StructuredOutputError."""
    last_err: Exception | None = None
    for attempt in range(attempts):
        suffix = (
            ""
            if attempt == 0
            else "\n\nYour previous reply was not valid JSON. Output ONLY the JSON."
        )
        resp = await llm.complete(
            system=system, user=user + suffix, model=model, max_tokens=max_tokens
        )
        try:
            return parse_json(resp.text)
        except (ValueError, json.JSONDecodeError) as exc:
            last_err = exc
    raise StructuredOutputError(f"model output not parseable as JSON: {last_err}")
