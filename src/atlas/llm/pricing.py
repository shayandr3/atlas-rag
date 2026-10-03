"""Token pricing loaded from pricing.yaml.

Per the spec's working agreement, cost accounting must fail loudly when a price is
missing — a wrong $0.00 is worse than an exception (spec §0.3, §10.4).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_TOKENS_PER_1M = 1_000_000


class PricingError(Exception):
    pass


class PricingNotConfiguredError(PricingError):
    pass


class MissingPriceError(PricingError):
    pass


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m_usd: float | None = None
    cached_input_per_1m_usd: float | None = None
    output_per_1m_usd: float | None = None


class PricingTable:
    def __init__(self, models: dict[str, ModelPrice]) -> None:
        self._models = models

    @classmethod
    def load(cls, path: str | Path = "pricing.yaml") -> "PricingTable":
        file = Path(path)
        if not file.exists():
            raise PricingNotConfiguredError(f"pricing file not found: {file}")
        raw: dict[str, Any] = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        models_raw: dict[str, Any] = raw.get("models") or {}
        return cls({name: ModelPrice(**(spec or {})) for name, spec in models_raw.items()})

    def cost_usd(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        price = self._models.get(model)
        if price is None:
            raise MissingPriceError(f"no price configured for model {model!r} in pricing.yaml")
        input_price = price.input_per_1m_usd
        if input_price is None:
            raise MissingPriceError(f"input price not configured for model {model!r}")
        output_price = price.output_per_1m_usd
        if output_price is None:
            raise MissingPriceError(f"output price not configured for model {model!r}")
        billed_input = max(input_tokens - cached_tokens, 0)
        cost = billed_input * input_price / _TOKENS_PER_1M
        cost += output_tokens * output_price / _TOKENS_PER_1M
        if cached_tokens:
            cached_price = price.cached_input_per_1m_usd
            if cached_price is None:
                raise MissingPriceError(f"cached-input price not configured for model {model!r}")
            cost += cached_tokens * cached_price / _TOKENS_PER_1M
        return round(cost, 6)
