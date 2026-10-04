# ADR 0002: LLM provider — DeepSeek via the OpenAI-compatible API

- Date: 2026-10-04
- Status: accepted

## Context

The spec (§2.2 #1) assumes Anthropic or OpenAI. The operator could not obtain a working
Anthropic key from their region (provider-side availability). DeepSeek offers an
OpenAI-compatible chat API with prompt caching and published peak/off-peak pricing.

## Decision

Keep the thin-adapter design (spec §5) and add `OpenAICompatLLM` over the official `openai`
SDK with a configurable `OPENAI_BASE_URL`. Provider selection is config-driven
(`LLM_PROVIDER=openai`); the Anthropic adapter remains and both implement the same `LLM`
protocol. Model ids: `deepseek-flash` for both cheap and strong roles initially.

## Alternatives

- Waiting on Anthropic access: blocks every LLM-dependent milestone.
- A third-party relay for Anthropic: an extra moving part with ToS and privacy risk.

## Consequences

- `pricing.yaml` carries DeepSeek's officially published prices; peak rates are used
  conservatively (off-peak is half of peak, so spend is never undercounted).
- DeepSeek's cache-hit billing maps cleanly onto the PricingTable (miss = input price,
  hit = cached_input price).
- Switching providers later is a config change, not a code change — the deviation from the
  spec's "Anthropic or OpenAI" is intentional and recorded here.
