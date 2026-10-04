# Native macOS inference behind cascade-llm

On Apple Silicon, run the model engine directly on macOS so it can use Metal / Unified Memory. Run cascade-llm either natively too (recommended for the simplest networking) or in Docker with `host.docker.internal`.

## Native gateway start

```bash
cd cascade-llm
export CASCADE_ADMIN_KEY="$(openssl rand -hex 32)"
scripts/start_mac_native.sh
```

The launcher starts only cascade-llm. Start your chosen engine separately and expose its OpenAI-compatible API. Defaults are MLX/vmlx on `localhost:8080`, Inferencer or llama.cpp on `localhost:8081`, and sd.cpp on `localhost:8082`; override with environment variables where needed. vmlx, Inferencer and llama.cpp are not hard dependencies: they are ordinary OpenAI-compatible upstreams.

## Register native backends dynamically

```bash
BASE=http://127.0.0.1:3000
KEY="$CASCADE_ADMIN_KEY"

# Main reasoning engine. A /v1 base URL is expanded to /v1/chat/completions.
curl -fsS -X PUT "$BASE/web/api/upstreams/main" \
  -H "x-cascade-admin-key: $KEY" -H 'content-type: application/json' \
  -d '{"endpoint_url":"http://localhost:8080/v1","engine":"vmlx","label":"MLX main","model":"mlx-model"}'

# Fast/vision engine. `vision` is an alias for the gateway's OCR/vision role.
curl -fsS -X PUT "$BASE/web/api/upstreams/auxiliary" \
  -H "x-cascade-admin-key: $KEY" -H 'content-type: application/json' \
  -d '{"endpoint_url":"http://localhost:8081/v1","engine":"inferencer","label":"Inferencer fast/vision","model":"vision-model"}'

# Explicit vision registration uses the same canonical OCR role.
curl -fsS -X PUT "$BASE/web/api/upstreams/vision" \
  -H "x-cascade-admin-key: $KEY" -H 'content-type: application/json' \
  -d '{"endpoint_url":"http://localhost:8081/v1","engine":"llamacpp","label":"llama.cpp vision","model":"llava-model"}'

# Image generation.
curl -fsS -X PUT "$BASE/web/api/upstreams/image" \
  -H "x-cascade-admin-key: $KEY" -H 'content-type: application/json' \
  -d '{"endpoint_url":"http://localhost:8082/v1","engine":"sd.cpp","label":"Stable Diffusion.cpp"}'

# Embeddings and rerank share the registered embeddings role; the gateway
# appends /v1/embeddings or /v1/rerank according to the incoming route.
curl -fsS -X PUT "$BASE/web/api/upstreams/embeddings" \
  -H "x-cascade-admin-key: $KEY" -H 'content-type: application/json' \
  -d '{"endpoint_url":"http://localhost:8081/v1","engine":"inferencer","label":"Local embeddings"}'

# Health/latency/context/roles; bearer tokens are masked in responses.
curl -fsS "$BASE/api/v1/admin/upstreams" -H "x-cascade-admin-key: $KEY"
curl -fsS -X POST "$BASE/api/v1/admin/upstreams/probe" -H "x-cascade-admin-key: $KEY"
```

Remove a single node by its returned id, or a whole role pool:

```bash
curl -fsS -X DELETE "$BASE/web/api/upstreams/main?id=main-<node-id>" -H "x-cascade-admin-key: $KEY"
curl -fsS -X DELETE "$BASE/web/api/upstreams/image" -H "x-cascade-admin-key: $KEY"
```

`engine` is diagnostic metadata; routing is based on role and URL, not the engine label. Standard chat and image payloads pass through without vendor-specific rewriting. OpenAI multimodal `image_url` content is preserved by the existing chat request model. Unsupported vendor-specific schemas should be adapted by the backend or a future explicit adapter rather than silently mutated in the gateway.

## Gateway in Docker, inference on the host

The stack Compose service adds `host.docker.internal:host-gateway` and sets `CASCADE_HOST_GATEWAY=true`. Registering `localhost:<port>` or `127.0.0.1:<port>` then stores `host.docker.internal:<port>`; service DNS and remote URLs are left untouched. When running cascade natively, leave this flag false/unset so localhost stays local.

## Failure and health behavior

The registry is SQLite-persisted (including inactive state and credentials); health and latency observations are fresh after restart. The background prober uses `/health`, then `/v1/models` when `/health` is absent. Unreachable chat/embedding routes return OpenAI-shaped 503 JSON. Start or restart the named native engine and probe again; a recovered endpoint is automatically returned to service after a successful health sweep.
