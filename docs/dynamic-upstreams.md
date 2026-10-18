# Dynamic Upstreams & Node Management

Real-time backend management without gateway restarts. All admin endpoints are
guarded by the `x-cascade-admin-key` header (`CASCADE_ADMIN_KEY` env; if unset,
the admin API is disabled).

## Roles

Every endpoint that serves traffic routes to a registered node of its role
first. The static config URL is the fallback used only while no active,
healthy node is registered for that role.

| Role            | Endpoint(s) served                          | Upstream path on the node         | Static fallback (env)                          | Probe            |
|-----------------|---------------------------------------------|-----------------------------------|------------------------------------------------|------------------|
| `main`          | `/v1/chat/completions`, `/v1/messages`, LLM | `/chat/completions`               | `LARGE_TEXT_URL`                               | `/health`        |
| `auxiliary`     | compression / fast text, chat aux routing   | `/chat/completions`               | `SMALL_MLLM_URL`                               | `/health`        |
| `ocr`           | document / vision parsing                   | `/chat/completions`               | `OCR_URL`                                      | `/health`        |
| `extraction`    | `/extraction/v1/chat/completions`           | `/chat/completions`               | configured extraction backends (`/extraction/v1/backends`) | `/health`   |
| `rag_worker`    | `/v1/rag/extract`                           | `/chat/completions`               | none (503 when absent)                         | `/health`        |
| `embeddings`    | `/v1/embeddings`                            | `/embeddings`                     | `EMBEDDINGS_URL`                               | `/v1/models`     |
| `rerank`        | `/v1/rerank`                                | `/rerank`                         | `RERANK_URL`                                   | `/v1/models`     |
| `transcription` | `/v1/audio/transcriptions` (multipart)      | `/audio/transcriptions`           | `STT_URL` (`/v1/audio/diarize` stays on it)    | `/v1/models`     |
| `speech`        | `/v1/audio/speech` (streamed audio)         | `/audio/speech`                   | `TTS_URL`                                      | `/v1/models`     |
| `image`         | `/v1/images/generations`                    | `/images/generations`             | `IMAGE_GENERATION_URL` (+ Pollinations fallback) | `/v1/models`   |

Aliases accepted on the admin API: `inference` → `main`, `aux` → `auxiliary`,
`vision` → `ocr`, `stt` → `transcription`, `tts` → `speech`, `reranker` →
`rerank`, `embedding` → `embeddings`, `img` → `image`, `rag` → `rag_worker`.
The role catalog (path and model kind per role) is served on the upstream
listings as `role_catalog`.

Node selection, health and metering are the same for every role: weighted
round-robin or least-latency over active + healthy nodes, a per-node `model`
alias rewrites the outgoing request (JSON `model`, or the multipart `model`
field for transcription), a per-node `bearer_token` authenticates it, and each
successful call records exactly one usage event (`chat` for extraction,
`embeddings`, `rerank`, `audio_transcription` by duration, `audio_speech` by
input characters, `image` by image count).

### Migration of the saved TTS / STT URLs

The console's "Chat upstream overrides" no longer edit chat URLs as the only
override path. Saved TTS and STT URLs (`settings.audio.tts_url` /
`stt_url`) are converted at startup into registry nodes:

* `tts_url` → node of role `speech`, id and label `manual-speech`, endpoint
  `<base>/v1/audio/speech`
* `stt_url` → node of role `transcription`, id and label `manual-transcription`,
  endpoint `<base>/v1/audio/transcriptions`

Both base forms (`http://host:port` and `http://host:port/v1/audio`) are
accepted. The migration is idempotent (fixed node ids), runs on every start and
also when settings are saved, where an empty value removes the node again. The
`manual-main` / `manual-auxiliary` chat nodes are unchanged. The static
`TTS_URL` / `STT_URL` remain the fallback whenever no node is active.

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

A background prober checks every registered node every
`HEALTH_CHECK_INTERVAL_SECS` (default 30). The probe depends on the role:

* Chat-style roles (`main`, `auxiliary`, `ocr`, `extraction`, `rag_worker`)
  GET `<origin>/health`.
* Embeddings, rerank, transcription, speech and image nodes GET
  `<origin>/v1/models`. They never receive a chat-style probe, and a failing
  chat endpoint cannot demote them.
* A 404 or 405 on the primary probe falls back to the other one. Any 2xx or
  3xx response counts as alive; everything else, and connection errors, count as
  failed. An explicit `health_url` on a node overrides the primary probe.

After `HEALTH_CHECK_FAILURE_THRESHOLD` consecutive failures (default 2) a node is
demoted and stops receiving traffic; a later successful probe promotes it again.
Latency-to-first-byte of probes and of proxied requests (EMA) is tracked per
node.

## Endpoint request counters (overview)

The overview's "Requests by endpoint" lists the route groups (`inference (main
LLM)`, `auxiliary (aux LLM)`, `ocr`, `image`, `rag_worker`, `extraction`,
`session_affinity`) plus one group per endpoint, keyed by its full path:
`/v1/messages`, `/v1/realtime`, `/v1/embeddings`, `/v1/rerank`,
`/v1/audio/transcriptions`, `/v1/audio/speech`, `/v1/audio/diarize`,
`/v1/video/generations`, `/v1/models`. Idle endpoints show 0. Path counters are
endpoint hits and are not part of the `requests_total` KPI, because a
`/v1/messages` call is also counted through its route group.

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

The console is one document served at `/`, `/web/` and (through Caddy's
`/cascade/* → /web*` rewrite) at `/cascade/`. Sections:

| Section | What it manages |
|---|---|
| **Overview** | one-shot header (health, registry, providers, fleet), traffic per backend, fallbacks per reason, and a routing self-test that reports the `x-cascade-route` a real request got |
| **Backends** | the registry: per-node activate/deactivate, inline weight, edit, remove, probe-all, refresh-model-names, selection strategy, add-node form, static defaults, image-role probe |
| **GPU fleet** | the provisioner: status, instances (flagging anything still billing), inference nodes with activate/deactivate/terminate/logs, budget, cost history, cloud providers, admission rules, hosted-API (Pollinations) registration |
| **Providers** | external APIs: add / edit / enable / disable / test / remove, key state (never the key), and the registry node each one occupies |
| **Models** | `/v1/models` discovery plus a live probe of what each role is actually serving |
| **Routing** | router thresholds, default route, hybrid routing, marker mode, chat upstream overrides, TTS/STT endpoints |
| **Diagnostics** | the documented direct bindings, extraction backends, parsed Prometheus counters, raw endpoint links |

Admin endpoints are mirrored under both `/api/…` and `/web/api/…`;
`/web/settings` serves the same document so existing bookmarks keep working.
Root-level paths the console needs (`/health`, `/extraction/v1/backends`) also
have `/web/api/…` aliases, because the Caddy rewrite puts a request for them
under `/web` where they would otherwise 404.

`GET/POST/PUT/DELETE /web/api/provisioner/*` is an **allowlist**, not a proxy: only
the documented read and lifecycle paths are forwarded to
`$GPU_PROVISIONER_URL`, behind the admin key, with the provisioner's own API key
attached server-side and never echoed.

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
