import pytest

from atlas.llm.pricing import (
    MissingPriceError,
    ModelPrice,
    PricingNotConfiguredError,
    PricingTable,
)


def test_cost_math() -> None:
    table = PricingTable(
        {"m": ModelPrice(input_per_1m_usd=3.0, cached_input_per_1m_usd=0.3, output_per_1m_usd=15.0)}
    )

    full = table.cost_usd(model="m", input_tokens=1_000_000, output_tokens=100_000)
    cached = table.cost_usd(
        model="m", input_tokens=1_000_000, output_tokens=0, cached_tokens=500_000
    )

    assert full == pytest.approx(3.0 + 1.5)
    assert cached == pytest.approx(0.5 * 3.0 + 0.5 * 0.3)


def test_missing_model_fails_loudly() -> None:
    table = PricingTable({})

    with pytest.raises(MissingPriceError):
        table.cost_usd(model="unknown-model", input_tokens=10, output_tokens=10)


def test_missing_output_price_fails_loudly() -> None:
    table = PricingTable({"m": ModelPrice(input_per_1m_usd=1.0)})

    with pytest.raises(MissingPriceError):
        table.cost_usd(model="m", input_tokens=10, output_tokens=10)


def test_missing_cached_price_only_needed_when_cache_used() -> None:
    table = PricingTable({"m": ModelPrice(input_per_1m_usd=1.0, output_per_1m_usd=2.0)})

    assert table.cost_usd(model="m", input_tokens=100, output_tokens=100) == pytest.approx(0.0003)

    with pytest.raises(MissingPriceError):
        table.cost_usd(model="m", input_tokens=100, output_tokens=100, cached_tokens=10)


def test_load_missing_file_raises(tmp_path) -> None:
    with pytest.raises(PricingNotConfiguredError):
        PricingTable.load(tmp_path / "nope.yaml")
