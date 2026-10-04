# AGENTS.md — cascade-llm

Rust axum gateway (single binary, multithreaded tokio) — central LLM entrypoint
for NetAI-Stack-SE. Branch: `feature/cascade_central`. Build `cargo build`,
test `cargo test`, lint `cargo clippy --all-targets`.

## Architecture

```
client ──► /v1/chat/completions ──► inspect_route (3-condition dispatcher)
                                     ├─ hints/headers/metadata (deterministic)
                                     ├─ B: agent-compression markers ──► auxiliary
                                     ├─ A(dynamic): doc/OCR payload + active
                                     │   ocr node registered ────────────► ocr
                                     └─ C: default ──────────────────────► main
            /v1/images/generations ──► image role (fallback IMAGE_GENERATION_URL)
            /v1/rag/extract        ──► rag_worker pool (503 if none)
            /extraction/v1/*       ──► extraction backend registry
            /v1/embeddings         ──► EMBEDDINGS_URL passthrough (:8801)
            /v1/rerank             ──► RERANK_URL passthrough (:8801)
            /v1/audio/*, /v1/video  ─► pass-through proxies
```

Modules (`src/lib.rs` exposes all; binary is a thin bootstrap in `main.rs`,
router assembly in `router.rs`):

- `registry.rs` — dynamic UpstreamRegistry: per-role node pools
  (main|auxiliary|ocr|image|rag_worker), weighted round-robin /
  least-latency selection, health prober task, YAML seed support.
- `state.rs` — GatewayState: routing engine, circuit breaker, session cache,
  SSE normalization, complexity routing ("auto" mode).
- `handlers.rs` — HTTP handlers incl. admin API (`x-cascade-admin-key`).
- `config.rs` — env config (+ optional CASCADE_CONFIG_FILE yaml seeds).
- `embeddings.rs` — pass-through proxies for `/v1/embeddings` + `/v1/rerank`
  (OpenAI-compatible embeddings/rerank backends, `EMBEDDINGS_URL`/`RERANK_URL`).
- `media.rs`, `audio.rs` — media/audio proxies; `db.rs` — SQLite persistence;
  `cascade_features.rs` — Prometheus metrics.
- `provisioner.rs` — admin-guarded, allowlisted passthrough to the stack's GPU
  provisioner, so the fleet is manageable from the gateway console.
- `web/app.html` — the console: one document, seven sections (Overview,
  Backends, GPU fleet, Providers, Models, Routing, Diagnostics), embedded via
  `include_str!`. Analysis of the previous two-page UI, and of every defect the
  rework fixed, is in `docs/gui-rework-analysis.md`.

## Dynamic upstream management (v0.6.0)

Full API reference: docs/dynamic-upstreams.md. Short form:

- `PUT/DELETE /web/api/upstreams/{role}` — register/deregister nodes at runtime
  (accepts legacy `url`/`bearer` fields; idempotent by URL or id).
- `POST /api/v1/inference-nodes/{id}/activate|deactivate` — pool toggles.
- `GET /api/v1/admin/upstreams`, `POST .../probe` — status + on-demand sweep.
- Background health prober demotes failing nodes (threshold) and auto-recovers.
- Routing picks registry nodes first; static URLs are fallback → hot swaps
  without dropping sessions. Responses tagged with `x-cascade-route` +
  `x-cascade-node`.
- LiteLLM-inspired: least_latency strategy (with unmeasured-node warm-up),
  declarative cascade_config.yaml, probe endpoint. Deliberately NOT embedding
  LiteLLM (Python sidecar would duplicate the gateway).
- Hosted APIs: `provider: pollinations` nodes get an image adapter
  (OpenAI images → Pollinations GET) and model normalization for text;
  optional per-node `model` field forces the upstream model id.
- Health prober treats any HTTP <500 as alive (404-only workers stay in pool).
- Console: one page at `/`, `/web/` and `/cascade/`; admin API mirrored under
  `/api/…` and `/web/api/…`. **Every request goes through one prefix helper**
  (`gw()` in `web/app.html`) — a relative fetch resolves one level too deep under
  Caddy's `/cascade/* → /web*` rewrite, which is how the image test used to 404
  in production while working on the raw port. No CDN assets (charts are inline
  SVG) and no `innerHTML` with API data.
- Provider ⇄ registry node is one function (`handlers::sync_provider_node`),
  called by create, edit *and* the boot-time restore. A disabled provider that is
  re-registered at startup looks persisted right up until the next deploy.

## Conventions

- **★ After ANY source change, rebuild the deployed image — the running
  container does NOT pick up code by itself.** The stack builds this repo
  locally (`build.context: /home/edgar/Dev/cascade-llm`, image
  `ghcr.io/netai369/cascade-llm:main`) and runs that binary; editing `src/`
  leaves the live gateway on the old code until you rebuild + recreate:
  `scripts/compose-up.sh build llm-gateway && scripts/compose-up.sh up -d
  --no-deps llm-gateway` (from NetAI-Stack-SE). Symptom of forgetting: new
  routes silently `404` (e.g. `/v1/embeddings`, `/v1/rerank` were WIP in git
  but absent live because the image predated them) or old routing behaviour.
  Always curl the endpoint after deploying.
- No comments unless non-obvious; clean, readable code.
- Tests: unit tests live next to code (`registry.rs`); black-box API tests in
  `tests/upstream_registry.rs` boot the real router on an ephemeral port.
- Metrics labels for new backends must be pre-seeded in
  `cascade_features::MetricsRegistry::init`.
- Bearer tokens/API keys never serialized to clients — always mask.
