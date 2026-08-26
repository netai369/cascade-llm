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
            /v1/audio/*, /v1/video  ──► pass-through proxies
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
- `media.rs`, `audio.rs` — media/audio proxies; `db.rs` — SQLite persistence;
  `cascade_features.rs` — Prometheus metrics; `web/` — embedded dashboard UI.

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
- LiteLLM-inspired: least_latency strategy, declarative cascade_config.yaml,
  probe endpoint. Deliberately NOT embedding LiteLLM (Python sidecar would
  duplicate the gateway).

## Conventions

- No comments unless non-obvious; clean, readable code.
- Tests: unit tests live next to code (`registry.rs`); black-box API tests in
  `tests/upstream_registry.rs` boot the real router on an ephemeral port.
- Metrics labels for new backends must be pre-seeded in
  `cascade_features::MetricsRegistry::init`.
- Bearer tokens/API keys never serialized to clients — always mask.
