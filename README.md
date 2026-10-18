# Cascade LLM

<p align="center">
  <img src="docs/assets/hero.svg" alt="Cascade LLM: clients send OpenAI- and Anthropic-compatible requests to the gateway, which routes them to local engines and hosted APIs" width="100%">
</p>

**Cascade** is a self-hosted, OpenAI-compatible (and Anthropic-compatible) LLM gateway written in Rust.
Point your existing SDKs at one endpoint, and Cascade routes each call to the right backend: a
local vLLM, OpenVINO or llama.cpp server, a hosted API, or a free hosted tier. It ships as a single
binary with a built-in SQLite store and a web console, and it meters every billable call.

## Who is this for

- **Any developer building an AI stack** who runs models locally and wants one stable URL for
  chat, embeddings, audio, images and realtime voice, whatever engine serves them underneath.
- **Teams mixing local and external providers** who want one place to register, weight, health-check,
  enable and disable backends, and to see which node served which request.
- **Platform and SaaS builders** who need per-tenant usage metering and a fair-share concurrency
  limit so one noisy client cannot starve the others.
- **Anyone who wants the gateway to stay on-premise.** Cascade has no required cloud dependency:
  the only outbound calls are to the upstreams you configure.

Cascade is a general-purpose gateway. It is also the central LLM entry point of the NetAI Stack SE
self-hosted platform, but it runs standalone and has no NetAI-specific requirements.

## Features

### API compatibility
- `POST /v1/chat/completions` (OpenAI Chat Completions, streaming via SSE)
- `POST /v1/messages` and `POST /v1/messages/count_tokens` (Anthropic Messages API, including the SSE event protocol)
- `GET /v1/models` (the models the gateway currently serves)
- `POST /v1/embeddings`, `POST /v1/rerank` (pass-through to the embeddings service)
- `POST /v1/audio/speech` (TTS), `POST /v1/audio/transcriptions` (STT), `POST /v1/audio/diarize`
- `POST /v1/images/generations`, `POST /v1/video/generations`
- `GET /v1/realtime` (OpenAI-Realtime-compatible WebSocket proxy to the STT realtime upstream)

### Routing and upstreams
- **Role-based dispatch**: `main`, `auxiliary`, `ocr`, `image` and `rag_worker` pools, each with its own
  nodes and static fallback URLs from the environment.
- **Dynamic upstream registry**: register, update, toggle and remove nodes at runtime through the admin
  API, with no gateway restart. Re-registering the same URL or `id` updates the node in place.
- **Selection strategies**: weighted round-robin (default) or least-latency, over active and healthy nodes.
- **Health prober**: periodic health sweeps with configurable interval, timeout and failure threshold;
  an immediate sweep can be triggered from the API.
- **Circuit breaker** settings (`CIRCUIT_BREAKER_THRESHOLD`, `CIRCUIT_BREAKER_RESET_SECS`).
- **Hosted adapters**: Pollinations nodes for image generation and text models; a free image fallback
  selected by `CASCADE_IMAGE_MODE`.
- **Routing transparency**: responses carry `x-cascade-route` (backend class) and `x-cascade-node`
  (registry node id) headers.
- Optional **intent and complexity routing** for hybrid auxiliary/main selection
  (`CASCADE_HYBRID_AUTO`), and a configurable default route (`CASCADE_DEFAULT_ROUTE`).

### Metering and billing hooks
- **One usage event per billable call**: chat, completions, embeddings, audio, images, video and realtime
  voice sessions are recorded in the local SQLite ledger.
- Attribution from request headers: `X-Billing-Tenant`, `X-Billing-User-Id`, `X-Billing-Source`,
  `X-Billing-Metered`.
- Events can be shipped to a webhook (`CASCADE_USAGE_WEBHOOK_URL`, optional `CASCADE_USAGE_WEBHOOK_KEY`);
  metering can be turned off with `CASCADE_USAGE_ENABLED=false`, and retention is set with
  `CASCADE_USAGE_RETENTION_DAYS`.
- Admin usage report at `/api/v1/admin/usage`.

### Multi-tenant fairness
- **Per-tenant fair-share concurrency gate** on `/v1/chat/completions` and `/v1/messages`.
  Off by default. Set `CASCADE_TENANT_MAX_CONCURRENCY`, optionally per plan with
  `CASCADE_TENANT_CONCURRENCY_BY_PLAN`. Requests over the limit wait up to
  `CASCADE_TENANT_QUEUE_WAIT_SECS`, then receive HTTP 429 with `Retry-After`.
- Requests without a tenant header are not limited.

### Media, audio and realtime
- TTS, STT (with diarization), image and video generation, and embeddings/rerank proxies.
- Realtime voice WebSocket proxy at `/v1/realtime`, so a voice client can switch between the STT service
  and Cascade by changing only the URL.
- Document extraction for RAG workers at `/v1/rag/extract`, routed to the `rag_worker` pool.

### Admin console and admin API
- Web console at `/web/`: **Overview**, **Backends** (pool and health prober), **GPU fleet**,
  **Providers** (with **Register an API**), **Models**, **Routing**, **Diagnostics**, and a settings page
  with the reference docs at `/web/settings`.
- Admin API for upstreams, nodes, providers, routing strategy and settings, guarded by the
  `x-cascade-admin-key` header (`CASCADE_ADMIN_KEY`). When the key is unset, the admin routes are disabled.
- Endpoint and API documentation: [`docs/dynamic-upstreams.md`](docs/dynamic-upstreams.md),
  [`docs/anthropic-api.md`](docs/anthropic-api.md), [`docs/routing-settings.md`](docs/routing-settings.md).

### Observability and health
- `GET /health` (also `/web/api/health`) for liveness and upstream state.
- Prometheus metrics at `/web/metrics`.
- Dashboard and overview snapshots at `/web/api/dashboard` and `/web/api/overview`.
- Structured logs through `tracing`.

## Architecture

```mermaid
flowchart LR
    subgraph Clients
        OAI[OpenAI SDK]
        ANT[Anthropic SDK]
        RT[Voice / realtime]
        AG[Agents and apps]
    end

    subgraph Cascade["Cascade gateway (single binary)"]
        API[Axum HTTP API]
        RTR[Router<br/>role dispatch]
        REG[Upstream registry<br/>+ health prober]
        FS[Fair-share gate]
        MTR[Usage meter]
        DB[(SQLite)]
        API --> FS --> RTR
        RTR --> REG
        API --> MTR --> DB
    end

    subgraph Upstreams
        LOC[Local engines<br/>vLLM / OVMS / llama.cpp]
        HOST[Hosted APIs]
        POLL[Pollinations]
        SVC[Audio, embeddings,<br/>rerank, RAG workers]
    end

    OAI --> API
    ANT --> API
    RT --> API
    AG --> API
    REG --> LOC
    REG --> HOST
    REG --> POLL
    RTR --> SVC
    MTR -. optional webhook .-> BILL[(Billing / ledger)]
```

## Quick start

### Build and run from source

Requires a recent stable Rust toolchain.

```bash
cargo build --release
./target/release/cascade-llm
```

The gateway listens on `0.0.0.0:3000`. The console is at `http://localhost:3000/web/`.

### Docker

The repository's `Dockerfile` builds a release binary and runs it on a distroless base:

```bash
docker build -t cascade-llm .
docker run -p 3000:3000 \
  -e LARGE_TEXT_URL=http://host.docker.internal:8080/v1/chat/completions \
  -e SMALL_MLLM_URL=http://host.docker.internal:8082/v1/chat/completions \
  -v cascade-data:/data \
  cascade-llm
```

The state database defaults to `/data/cascade.db` (`CASCADE_DB`), so mount a volume there to keep usage
and registry data across restarts. The CI workflow in `.github/workflows/` publishes the image on pushes
to `main` and on `v*` tags.

### Minimal configuration

Every upstream has a default, and the defaults are Docker service names (for example
`http://inference-server:8080/v1/chat/completions`). Set the ones that match your engines:

```bash
export LARGE_TEXT_URL=http://localhost:8080/v1/chat/completions   # main chat backend
export SMALL_MLLM_URL=http://localhost:8082/v1/chat/completions   # auxiliary backend
export CASCADE_ADMIN_KEY=change-me                                # enables the admin API
./target/release/cascade-llm
```

For a larger setup, put upstream pools in a YAML file and point `CASCADE_CONFIG_FILE` at it. The file
seeds the registry and settings at boot, and environment variables take precedence over file values.
Nodes can also be added at runtime through the console or the admin API.

### First request

```bash
# OpenAI-compatible
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"cascade-hybrid-v1","messages":[{"role":"user","content":"Hello"}]}'

# Anthropic-compatible
curl http://localhost:3000/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"cascade-hybrid-v1","max_tokens":256,"messages":[{"role":"user","content":"Hello"}]}'
```

The model name `cascade-hybrid-v1` is the default value of `CASCADE_MODEL_NAME`. Set it to any name you like.

## API overview

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI Chat Completions (streaming supported) |
| POST | `/v1/messages` | Anthropic Messages API |
| POST | `/v1/messages/count_tokens` | Anthropic token counting |
| GET | `/v1/models` | Models served by the gateway |
| POST | `/v1/embeddings` | Embeddings pass-through |
| POST | `/v1/rerank` | Reranking pass-through |
| POST | `/v1/audio/speech` | Text-to-speech |
| POST | `/v1/audio/transcriptions` | Speech-to-text |
| POST | `/v1/audio/diarize` | Speaker diarization |
| GET | `/v1/realtime` | Realtime voice WebSocket |
| POST | `/v1/images/generations` | Image generation |
| POST | `/v1/video/generations` | Video generation |
| POST | `/v1/rag/extract` | Document extraction (`rag_worker` pool) |
| GET | `/health` | Health check |
| GET | `/web/` | Admin console |
| PUT / DELETE | `/web/api/upstreams/{role}` | Register or remove upstream nodes (admin) |
| GET | `/api/v1/admin/upstreams` | Pool status and telemetry |
| POST | `/api/v1/admin/upstreams/probe` | Trigger an immediate health sweep |
| POST | `/api/v1/inference-nodes/{id}/activate` · `/deactivate` | Toggle a node |
| GET | `/api/v1/admin/usage` | Usage report (admin) |
| GET / PUT | `/web/api/providers`, `/web/api/settings`, `/web/api/routing-strategy` | Providers, settings, routing strategy (admin for writes) |
| GET | `/web/metrics` | Prometheus metrics |

Each route is also served under `/api/...` and `/web/api/...` for backward compatibility. Admin
routes take the `x-cascade-admin-key` header. Extraction, GPU provisioner and dashboard routes are
listed in `src/router.rs`.

## Configuration reference

The most relevant variables. Full defaults are in `src/config.rs`.

### Upstreams

| Variable | Default | Purpose |
|---|---|---|
| `LARGE_TEXT_URL` | `http://inference-server:8080/v1/chat/completions` | Main text backend |
| `LARGE_MLLM_URL` | `http://inference-server:8080/v1/chat/completions` | Main multimodal backend |
| `SMALL_MLLM_URL` | `http://auxiliary-server:8080/v1/chat/completions` | Auxiliary backend |
| `OCR_URL` | `http://ocr-server:8082/v1/chat/completions` | OCR backend |
| `INFERENCE_URL` | `http://netai-inference:8080` | Inference base URL |
| `IMAGE_GENERATION_URL` | `http://localhost:8080/v1/images/generations` | Image fallback |
| `VIDEO_GENERATION_URL` | `http://localhost:8080/v1/video/generations` | Video backend |
| `TTS_URL` | `http://speech-tts:8800/v1/audio` | Text-to-speech |
| `STT_URL` | `http://speech-stt:5092/v1/audio` | Speech-to-text |
| `STT_REALTIME_URL` | `ws://speech-stt:5093/v1/realtime` | Realtime STT (WebSocket) |
| `EMBEDDINGS_URL` / `RERANK_URL` | `http://embeddings:8801` | Embeddings and rerank |
| `LLAMA_API_KEY` | none | Bearer token for static local upstreams |
| `MODEL_NAME` labels | see `src/config.rs` | `MAIN_MODEL_NAME`, `SMALL_MODEL_NAME`, `OCR_MODEL_NAME` |
| `CASCADE_MODEL_NAME` | `cascade-hybrid-v1` | Model name the gateway advertises |

### Routing

| Variable | Default | Purpose |
|---|---|---|
| `CASCADE_ROUTING_STRATEGY` | `round_robin` | `round_robin` (weighted) or `least_latency` |
| `CASCADE_DEFAULT_ROUTE` | none | Deterministic route: `inference`, `auxiliary` or `ocr` |
| `CASCADE_HYBRID_AUTO` | `false` | Complexity-based auxiliary/main selection |
| `CASCADE_HYBRID_AUX_MAX_CHARS` | `600` | Prompt size cut-off for the auxiliary route |
| `CASCADE_IMAGE_MODE` | `local_free` | `local`, `local_free` or `free` |
| `CASCADE_EXTRACTION_STRATEGY` | see `src/config.rs` | Extraction backends: `priority` or `round_robin` |
| `CIRCUIT_BREAKER_THRESHOLD` / `CIRCUIT_BREAKER_RESET_SECS` | `5` / `60` | Static-upstream circuit breaker |
| `HEALTH_CHECK_INTERVAL_SECS` / `_TIMEOUT_SECS` / `_FAILURE_THRESHOLD` | `30` / `5` / `2` | Health prober (interval min 5 s) |

### Metering and fairness

| Variable | Default | Purpose |
|---|---|---|
| `CASCADE_DB` | `/data/cascade.db` | SQLite state file |
| `CASCADE_USAGE_ENABLED` | `true` | Record usage events |
| `CASCADE_USAGE_WEBHOOK_URL` / `CASCADE_USAGE_WEBHOOK_KEY` | none | Ship usage events to a billing service |
| `CASCADE_USAGE_FLUSH_SECS` | `5` | Webhook flush interval |
| `CASCADE_USAGE_RETENTION_DAYS` | `30` | Local ledger retention |
| `CASCADE_TENANT_MAX_CONCURRENCY` | `0` (off) | Per-tenant concurrency limit |
| `CASCADE_TENANT_CONCURRENCY_BY_PLAN` | none | Per-plan limits |
| `CASCADE_TENANT_QUEUE_WAIT_SECS` | `30` | Queue wait before HTTP 429 |

### Admin and configuration file

| Variable | Default | Purpose |
|---|---|---|
| `CASCADE_ADMIN_KEY` | unset (admin API disabled) | Key for `x-cascade-admin-key` |
| `CASCADE_CONFIG_FILE` | none | Optional YAML file seeding upstreams and settings |
| `CASCADE_HOST_GATEWAY` | `false` | Opt-in host-gateway handling for node URLs (see `src/registry.rs`) |
| `GPU_PROVISIONER_URL` / `GPU_PROVISIONER_API_KEY` | see `src/config.rs` | GPU provisioner passthrough (`/web/api/provisioner/*`) |
| `EXTRACT_CLOUD_URL` / `EXTRACT_CLOUD_API_KEY` / `EXTRACT_CLOUD_MODEL` | none | Cloud extraction backend |

The listening port is fixed at 3000 in `src/main.rs`.

## Integration with NetAI Stack SE

In the NetAI Stack SE deployment, Cascade is the single LLM entry point: LibreChat, the agent, voice
chat, dictation and LightRAG all send their model calls through it, and it meters them for the billing
service. Nothing in Cascade depends on NetAI, so the same binary works as a standalone gateway in any
other stack.

## Development and testing

```bash
cargo build                 # debug build
cargo test                  # unit tests and integration tests in tests/
cargo clippy --all-targets  # lints
cargo fmt --check           # formatting
```

Integration tests in `tests/` exercise the router, the upstream registry, the Anthropic layer, the
console API, audio URL handling and fallback behaviour without a live model backend.

## License

Cascade LLM is released under the MIT License. See [`LICENSE`](LICENSE).
