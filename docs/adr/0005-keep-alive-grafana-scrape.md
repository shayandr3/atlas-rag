# ADR 0005: Keep-alive strategy — 1-minute Grafana Cloud scrape keeps the free service awake

- Date: 2026-10-04
- Status: accepted (final numbers to re-verify against current Render/Grafana docs at deploy)

## Context

Render free web services spin down after 15 minutes idle, costing a ~1-minute cold start on
the next request. Qdrant Cloud free clusters suspend after 1 week of inactivity. Grafana
Cloud's Metrics Endpoint integration counts as inbound traffic, so a 1-minute scrape of
`/metrics` both keeps the Render service warm and (via `/readyz` scraping Qdrant through the
app) touches the vector DB.

## Decision

Use the Grafana Cloud 1-minute scrape as the keep-alive. The cost is ~744 of the workspace's
750 free instance-hours per month, which forbids running a second always-on free service in
the same workspace — accepted, because the flagship is the only service.

## Alternatives

- No keep-alive: accept ~1-minute cold starts on the demo (bad for the 60-second recruiter
  pitch) and manual Qdrant revival.
- External pinger (UptimeRobot): a second third party for something Grafana Cloud already
  does as a side effect of the observability requirement.

## Consequences

- Cold starts effectively disappear from the demo; `AtlasHighLatency` alerts should be read
  with this in mind (sustained, not one-sample spikes).
- The free-hours allowance is consumed — do not deploy a second always-on Render free
  service in this workspace (Project B must wait or use a different workspace).
- `/readyz` hits Qdrant on every scrape, keeping the vector cluster warm; verify Grafana
  Cloud's current scrape-auth behavior (bearer on `/metrics`) before relying on it.
