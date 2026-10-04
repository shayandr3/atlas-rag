# Security (spec §11)

Defense in depth: every control has a test in `tests/security/` and a row below.
Measured rates come from the committed red-team suite (`eval/redteam/redteam.jsonl`,
43 prompts) and the reviewed golden set, run 2026-10-04 (commit a888893+).

## Measured rates

| metric | value | source |
|---|---|---|
| red-team block rate (heuristic layer only) | **93% (40/43)** | `pytest tests/security/test_redteam.py -s` |
| false-positive rate on benign golden questions | **0% (0/60)** | same run |
| red-team misses | 3 indirect-injection prompts (subtle, no lexical markers) | manual review |

The 3 misses are exactly why the guard is layered: at runtime the ambiguous band
(risk score 1) is additionally judged by the cheap-LLM classifier before retrieval,
and the corpus itself is allowlisted (arXiv) with payload provenance.

## Controls

| control | implementation | test |
|---|---|---|
| API-key auth | `Authorization: Bearer` on `/v1/*`; keys stored only as sha256(pepper+raw) in `API_KEYS_JSON`; constant-time compare; per-key rpm + daily USD budget + scopes | `tests/security/test_auth.py` |
| dev open mode | anonymous principal ONLY when `APP_ENV=dev` and no keys configured | `test_resolve_principal_anonymous_only_in_dev` |
| metrics/admin realms | `/metrics` requires `METRICS_BEARER_TOKEN`; `/admin/*` requires `ADMIN_TOKEN` | `tests/security/test_api_security.py` |
| rate limiting | Redis sliding-window (ZSET) per key and per IP; in-process fallback when Redis is down (fail SAFE for abuse protection) | `tests/security/test_ratelimit_budget.py` |
| input guard | length cap → heuristic patterns → ambiguous-band LLM classifier → PII/secret redaction before logging and before the LLM | `tests/security/test_guards.py` |
| output guard | canary-leak block, unknown-`[S#]` citation stripping, PII/secret redaction of answers | `test_sanitize_output_*` |
| indirect injection | sources wrapped in `<sources>` + instruction-hierarchy system prompt + delimiter-smuggling patterns + corpus allowlist; generator has no tools/agency | guard patterns + prompt (spec §8.6) |
| budgets | per-key daily, global daily and monthly USD counters (Redis INCRBYFLOAT + local fail-safe); exhaustion → HTTP 429; ledger survives via env-stored key hashes | `test_ratelimit_budget.py` |
| request hygiene | body-size ceiling (413), strict CORS allowlist, `X-Content-Type-Options`, `Referrer-Policy`, CSP, HSTS in prod, no provider error bodies to clients | `test_security_headers_present`, `test_body_too_large_rejected` |
| privacy & logging | structured logs carry rules/lengths, not raw queries; raw queries never leave the process unredacted | guard unit tests |
| supply chain | pinned deps, `pip-audit` + `gitleaks` + Trivy (CI), non-root container (M7) | CI workflows |

## OWASP Top 10 for LLM Applications mapping

| OWASP risk | control here |
|---|---|
| LLM01 prompt injection (direct + indirect) | layered input guard (93% heuristic block rate), instruction-hierarchy prompt, `<sources>` delimiting, no tool/agency for the generator |
| LLM02 sensitive information disclosure | PII/secret redaction in+out, canary leak detection, no raw queries in logs, keys only as peppered hashes |
| LLM03 supply chain | pinned versions, pip-audit + gitleaks + Trivy in CI, minimal deps in serving image |
| LLM04 data & model poisoning | ingestion corpus allowlist (arXiv only), payload provenance (`doc_id`, `source_url`), access-controlled Qdrant key |
| LLM05 improper output handling | output guard (citation validation, canary, redaction), safe rendering in the demo UI (no innerHTML) |
| LLM06 excessive agency | the generator has zero tools/agency; retrieval and generation are read-only steps |
| LLM07 system prompt leakage | canary token in the system prompt; any response containing it is withheld |
| LLM08 vector/embedding weaknesses | Qdrant Cloud API key, collection-scoped access, payload provenance, no user-written content enters the index |
| LLM09 misinformation | inline citations validated against retrieved chunks, CRAG grading, abstention on insufficient evidence, sampled faithfulness judge |
| LLM10 unbounded consumption | per-key/global daily+monthly USD budgets, per-key + IP rate limits, per-request token ceilings, hop/loop caps, concurrency limits |

## Limitations (honest)

- The heuristic layer is lexical; paraphrased attacks without marker words pass it and rely on
  the runtime classifier and the untrusted-data prompt discipline. Measured: 3/43 misses.
- Rate limiting with Redis down falls back to in-process windows — correct for the single
  free-tier worker, insufficient for multi-worker deployments.
- The budget ledger's in-process fallback is per-process; on Render free there is exactly
  one worker, so this is sound for the target deployment.
