# ADR 0003: Free-tier LLM provider — Z.AI GLM-4.7-Flash

- Date: 2026-10-04
- Status: accepted
- Supersedes (operationally) ADR 0002, which remains the paid-fallback record

## Context

The operator's paid options collapsed in practice: Anthropic is not obtainable from their
region, and the DeepSeek key was rejected by DeepSeek itself (401). Endpoint reachability
was probed from the operator's machine: OpenRouter, Groq, Gemini and Together are
CDN-blocked (403); Mistral, Z.AI and DashScope International answer (401 = reachable,
auth required). The project needs LLM calls for ingestion, evals and the live demo with
zero budget.

## Decision

Use Z.AI's free `glm-4.7-flash` through the existing `OpenAICompatLLM` adapter
(`OPENAI_BASE_URL=https://api.z.ai/api/paas/v4`, officially OpenAI-SDK-compatible).
`LLM_PROVIDER=openai` is unchanged — this is a config + pricing-table change only.
pricing.yaml records the true unit cost (0.0) and keeps DeepSeek's paid prices as the
funded fallback.

## Alternatives

- Mistral free Experiment tier (reachable; requires phone verification and training-data
  consent).
- DashScope International trial (reachable; 1M free tokens per model for 90 days, then
  expires — good backup, not durable).
- OpenRouter / Groq / Gemini free tiers: best-known free catalogs, but CDN-blocked from
  the operator's network.
- Local models via Ollama for offline evals; impossible for the 512 MB Render demo.

## Consequences

- Per-query USD cost is genuinely 0.0; the binding constraint becomes free-tier rate
  limits, which the rate limiter and pipeline caps must respect (M4).
- Free-tier models/limits change often: the smoke test runs before eval jobs, and
  pricing.yaml's zero prices are honest accounting, not a promise of unlimited capacity.
- Switching providers remains a `.env` + pricing.yaml change (ADR 0002's design holds).
