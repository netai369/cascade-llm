# Dynamic Upstreams & Node Management

Real-time backend management without gateway restarts. All admin endpoints are
guarded by the `x-cascade-admin-key` header (`CASCADE_ADMIN_KEY` env; if unset,
the admin API is disabled).

## Roles

| Role        | Purpose                                            | Static fallback (env)      |
|-------------|----------------------------------------------------|----------------------------|
| `main`      | Default chat & reasoning                           | `LARGE_TEXT_URL`           |
| `auxiliary` | Compression / summarization / fast text            | `SMALL_MLLM_URL`           |
| `ocr`       | Vision / document parsing                          | `OCR_URL`                  |
| `image`     | Image generation (SDXL/Flux/zimage)                | `IMAGE_GENERATION_URL`     |
| `rag_worker`| Heavy LightRAG graph-extraction jobs               | none (503 when absent)     |

Legacy alias: role path `inference` maps to `main` (backward compatibility with
existing provisioners).

## REST API

### Register / update a node

```
PUT /web/api/upstreams/{role}          # or /api/upstreams/{role}
x-cascade-admin-key: <ADMIN_KEY>

{
  "endpoint_url": "http://10.0.0.45:8080/v1/chat/completions",
  "bearer_token": "optional-node-auth",
  "weight": 1,
  "max_context_length": 131072,
  "id": "optional-idempotency-id",
  "label": "Vast.ai RTX 4070",
  "provider": "vast.ai",            // cloud badge in UI
  "cost_per_hour": 0.076,
  "health_url": "http://10.0.0.45:8080/health",   // optional override
  "model": "openai"                 // optional: force model name upstream
}
```

Legacy field names `url` / `bearer` are accepted. Re-PUT with the same URL (or
same `id`) updates the existing node instead of duplicating it. Returns
`{"status":"ok","node":{...}}`; the response never contains the bearer token.

### Deregister

```
DELETE /web/api/upstreams/{role}       # clears the whole role pool
DELETE /web/api/upstreams/{role}?id=…  # removes a single node
```

### Activate / deactivate (pool toggle)

```
POST /api/v1/inference-nodes/{id}/activate
POST /api/v1/inference-nodes/{id}/deactivate
```

Deactivated nodes stay registered but are skipped by routing (used after
provisioning or for maintenance windows).

### Pool status & telemetry

```
GET  /api/v1/admin/upstreams            # nodes, health, latency EMA, VRAM-ish
                                       # badges, active roles, strategy
POST /api/v1/admin/upstreams/probe     # trigger an immediate health sweep
GET  /web/api/upstreams                # legacy-compatible list view
```

## Routing behaviour

* Selection per request: weighted **round-robin** (default) or
  **least-latency** over active+healthy nodes (`CASCADE_ROUTING_STRATEGY`
  or `routing_strategy:` in the YAML file).
* Chat requests resolve their role from the registry; static config URLs are
  used only when no active node is registered → zero-downtime hot swapping.
* Responses carry `x-cascade-route` (backend class) and `x-cascade-node`
  (registry node id) headers for routing transparency.
* OCR payloads (file attachments, document flags/markers) route to an `ocr`
  node only while one is registered and healthy; otherwise the multimodal
  main/auxiliary backends handle scans natively (graceful fallback).
* `POST /v1/rag/extract` forwards bodies untouched to an active `rag_worker`;
  without one it returns a structured `503 rag_worker_unavailable`.

## Health checks

A background prober hits `<origin>/health` on every registered node every
`HEALTH_CHECK_INTERVAL_SECS` (default 30). Any HTTP response below 500 counts
as alive (minimal workers without a `/health` route — 404/405 — stay in the
pool); 5xx and connection failures count as failed. After
`HEALTH_CHECK_FAILURE_THRESHOLD` consecutive failures (default 2) a node is
demoted (unhealthy) and stops receiving traffic; a later successful probe
promotes it again automatically. Latency-to-first-byte of successful probes
and of real proxied requests (EMA) is tracked per node.

## Routing strategies

* `round_robin` (default): weighted fair distribution.
* `least_latency`: lowest request-latency EMA wins. Freshly registered nodes
  have no EMA yet — they are served first via round-robin until measured,
  so they can never starve behind an established node.

## Hosted APIs: Pollinations.ai

Register `provider: "pollinations"` nodes for zero-cost cloud inference:

```bash
# Image generation (free, anonymous tier):
curl -X PUT localhost:3000/web/api/upstreams/image -H "x-cascade-admin-key: …" \
  -d '{"endpoint_url":"https://image.pollinations.ai","provider":"pollinations",
       "model":"flux","label":"Pollinations flux"}'

# Text chat (anonymous tier rate-limited; token recommended):
curl -X PUT localhost:3000/web/api/upstreams/auxiliary -H "x-cascade-admin-key: …" \
  -d '{"endpoint_url":"https://text.pollinations.ai/openai","provider":"pollinations",
       "model":"openai","bearer_token":"<POLLINATIONS_TOKEN>"}'
```

Adapter behaviour:

* **Image**: `POST /v1/images/generations` bodies are translated to the
  Pollinations GET API (`/prompt/{prompt}?width=&height=&model=&seed=`,
  `nologo=true`) and returned as OpenAI-shaped JSON (`b64_json`, or
  `response_format:"url"`). Client `size` maps to width/height (clamped to
  64–2048). The node's `model` (default `flux`) applies unless overridden
  per request.
* **Text**: requests are forwarded to the OpenAI-compatible endpoint as-is;
  the client's `model` field is replaced by the node's registered `model`
  (`openai`, `mistral`, …) since Pollinations rejects unknown ids. The
  bearer token is used as the Pollinations auth-tier token.

The NetAI GPU provisioner manages both node types via
`POST /api/v1/inference-nodes/pollinions` on its own API and re-registers
them automatically after gateway restarts (reconcile self-heal).

## Web UI

The dashboard (served at `/`, `/web/`, and via Caddy at `/cascade/`) has a
Node Control Center with active toggles, probe-now, register form and
cost/provider badges. All admin endpoints are mirrored under both
`/api/v1/…` and `/web/api/v1/…`; `/settings` redirects to `/web/settings`
so relative links work from every mount point.

## Declarative seed file (LiteLLM-inspired)

Optional `CASCADE_CONFIG_FILE=/path/cascade_config.yaml`:

```yaml
admin_key: my-secret            # only used if CASCADE_ADMIN_KEY is unset
routing_strategy: least_latency # round_robin | least_latency
health:
  interval_secs: 30
  failure_threshold: 2
  timeout_secs: 5
upstreams:
  - role: main
    endpoint_url: http://inference-server:8080/v1/chat/completions
    weight: 3
    max_context_length: 131072
  - role: rag_worker
    endpoint_url: http://gpu-node:8000/jobs
    provider: runpod
    cost_per_hour: 0.44
```

Env vars always win over file values. Seeded pools can be managed at runtime
via the API afterwards.
