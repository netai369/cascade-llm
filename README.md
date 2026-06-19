# Cascade LLM

A high-performance, self-hosted LLM router written in Rust. Routes requests between multiple local models based on complexity scoring, tool awareness, and vision capabilities — all in a single binary with ~50 MB RAM idle.

## Features

- **Complexity-based routing** — character count + keyword analysis routes simple queries to cheap models, complex ones to powerful models
- **Tool-aware routing** — detects `tools`/`functions` in requests and routes to models that support function calling
- **Vision→text pipeline** — downloads images, describes them with a vision model, then routes text + tools to the large model
- **Confidence-based rerouting** — uses logprobs to evaluate small model responses and reroutes to large model when confidence is low
- **Streaming support** — proxies SSE streams without buffering
- **Configurable via env vars** — no config files needed
- **Single static binary** — no Redis, no Node.js, no Python venv

## Quick Start

### Docker

```bash
docker run -p 3000:3000 \
  -e SMALL_MLLM_URL=http://localhost:8082/v1/chat/completions \
  -e LARGE_MLLM_URL=http://localhost:8080/v1/chat/completions \
  -e LARGE_TEXT_URL=http://localhost:8080/v1/chat/completions \
  -e ROUTER_THRESHOLD=0.5 \
  -e CONFIDENCE_THRESHOLD=0.7 \
  -e LARGE_MODEL_MULTIMODAL=true \
  -e ROUTE_TOOLS_TO_LARGE=true \
  ghcr.io/YOUR_ORG/cascade-llm:latest
```

### Build from Source

```bash
cargo build --release
./target/release/llm_gateway
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SMALL_MLLM_URL` | `http://localhost:8082/v1/chat/completions` | URL for the small/vision model |
| `LARGE_MLLM_URL` | `http://localhost:8080/v1/chat/completions` | URL for the large multimodal model |
| `LARGE_TEXT_URL` | `http://localhost:8080/v1/chat/completions` | URL for the large text-only model |
| `ROUTER_THRESHOLD` | `0.5` | Complexity threshold for routing (0.0–1.0) |
| `CONFIDENCE_THRESHOLD` | `0.7` | Minimum logprob confidence to keep small model response |
| `LARGE_MODEL_MULTIMODAL` | `true` | Whether the large model supports images |
| `ROUTE_TOOLS_TO_LARGE` | `true` | Route tool calls to the large model |

## Architecture

```
┌─────────────────────────────────────────────┐
│              cascade-llm                    │
│         Axum HTTP API + Auth + MCP          │
├──────────┬──────────┬───────────────────────┤
│ router   │confidence│    vision pipeline    │
│ scoring  │ rerouting│  download→base64→desc │
├──────────┴──────────┴───────────────────────┤
│              Request Router                 │
│  tools?  → large model (if configured)      │
│  image?  → describe with small → text+tools │
│  simple  → small model → confidence check   │
│  complex → large model                      │
└─────────────────────────────────────────────┘
```

## License

AGPL-3.0
