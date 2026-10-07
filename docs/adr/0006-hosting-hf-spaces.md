# ADR 0006: Live hosting on Hugging Face Spaces (Render free tier unusable)

- Date: 2026-10-04
- Status: accepted
- Supersedes (operationally) the Render deployment path from spec §16.2 / render.yaml

## Context

Render's free tier demanded card/identity verification the operator cannot provide (no
access to internationally accepted payment cards). The spec's rule 0.1 (human owns
accounts and spending) and rule 8 (surface infeasibility, propose a change) apply.
Hugging Face Spaces supports Docker deployments on a free CPU tier without a card, and
huggingface.co is reachable from the operator's network (verified during provider
probing for ADR 0003).

## Decision

Deploy the same Docker image as a public HF Space:
- README carries the Spaces metadata header (`sdk: docker`, `app_port: 10000`).
- Secrets go into Space **variables and secrets** — never the repo.
- No managed Redis on the platform: the M2 caches fail open and the M4 rate limiter
  uses its in-process fallback, which is sound for the single-worker serving process
  (documented limitation in docs/security.md).
- `RERANK_BACKEND=none` stays (ADR 0004); the free Space has more memory than Render's
  512 MB, so enabling the local reranker is a measured experiment for M8, not a default.

## Alternatives considered

- Koyeb / Fly.io / Railway / Cloud Run: all require card or billing verification.
- Oracle Cloud free tier: card required.
- PythonAnywhere / Vercel: no Docker runtime; would force an application rewrite.

## Consequences

- `render.yaml` is retained as the spec-conformant artifact but is not the live path.
- Free Spaces sleep after ~48 h without traffic; the Grafana Cloud 1-minute scrape
  (ADR 0005) keeps the Space awake and touches Qdrant via `/readyz`.
- The Space's git remote is an additional push target (`git push space main`); deploys
  are git-push-driven, same as Render's auto-deploy.
- Public repo: secrets live only in Space settings; nothing sensitive is committed
  (gitleaks in CI guards this).
