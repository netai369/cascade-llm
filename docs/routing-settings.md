# Routing & Reasoning Settings Reference

The runtime knobs on the **Settings** page (`/web/settings.html`) and their
environment equivalents. Live values are visible in `GET /web/api/settings` and
the dashboard's `GET /web/api/dashboard`.

## Route decision

A chat completion is routed top-to-bottom by `inspect_route`:

1. **Default route pin** – `CASCADE_DEFAULT_ROUTE` (`default_route`) forces
   `inference` / `auxiliary` for everything. Empty = content detection.
2. **Explicit hint** – header (`x-cascade-meta`) or body `route`/`metadata`
   reproduce the enum: `inference`, `auxiliary`, (`ocr`), `auto`.
   Regex `route_hint` is a top-level body key; `metadata: {"route": "auto"}`
   is the idiomatic form.
3. **Marker conditions** – compression/sub-agent markers → auxiliary; document
  /file payload → OCR (only when an `ocr` node is registered).
4. **Default** – inference.

`route:auto` skips the markers and runs the **adaptive router** (`route_auto`).

## Adaptive router (`route:auto`)

* **Intent detection** (`intent_routing.rs`) classifies the conversation:
  complex/planning/math → **large** (`LARGE_TEXT_URL`/test env), simple/trivial →
  **small** (`SMALL_MLLM_URL`).
* **Session affinity** – the winning backend is cached per `session_key`; a
  cached small route is **invalidated** when the *history* carries tool calls
  (`assistant` role with `tool_calls`, or a `tool` role message). Tools always
  win over a cached small target, so tool-bearing follow-ups hit the large
  model (`ROUTE_TOOLS_TO_LARGE`).
* **`ROUTE_TOOLS_TO_LARGE`** (settings checkbox, env `ROUTE_TOOLS_TO_LARGE`)
  also routes *fresh* payloads that declare `tools` straight to the large model
  — tool *selection* needs the big context window.
* **Confidence gate** (non-streaming small path) – the small response's mean
  token logprob is compared against `CONFIDENCE_THRESHOLD` (0.7). Below it the
  ORIGINAL request is re-sent to the large model; the small answer is dropped
  and `x-cascade-route: auto-large` is returned.
* **Fallback marker** – the small model receives a system instruction to emit
  `<CASCADE_FALLBACK>` when it cannot do the task. In non-streaming mode the
  tag re-routes to the large model. In streaming mode the marker (and any
  below-threshold logprob chunk) is stripped mid-stream — the request is not
  re-routable once started, so intent pre-bypass is the primary mechanism.
  Streaming URL: `/v1/chat/completions` with `stream:true` plus
  `metadata:{"route":"auto"}` when you want adaptive mode explicitly.
* `ROUTER_THRESHOLD` (0.5) is the legacy complexity score boundary used only by
  non-intent heuristics; the intent classifier supersedes it.

Response headers tell you what happened:

| `x-cascade-route` | Meaning |
|---|---|
| `inference-server` | large/main (`LARGE_TEXT_URL`) |
| `auxiliary-server` | small/aux (`SMALL_MLLM_URL`) |
| `auto-small` | adaptive router → small |
| `auto-large` | adaptive router → large (incl. confidence/fallback reroute) |
| `session-affinity` | reused cached backend |
| `ocr` / rag-worker / node ids | registry roles |

## Reasoning passthrough

CoT from llama.cpp arrives in `delta.reasoning_content` (streaming) or
`message.reasoning_content` (non-streaming). Cascade **never folds reasoning
into `content`** — it is forwarded in its own field so consumers that bind it
to a separate UI surface (Hermes `show_reasoning:false`, LibreChat) can ignore
it cleanly. `tool_calls`/`function_call` deltas are forwarded verbatim
(model-id rewrite skipped) so clients can call back with the exact ids.

## Settings-page ↔ env map

| Settings field | Env / config | Effect |
|---|---|---|
| Router Threshold | `ROUTER_THRESHOLD` | legacy complexity boundary |
| Confidence Threshold | `CONFIDENCE_THRESHOLD` | small→large non-stream reroute |
| Route Tools to Large | `ROUTE_TOOLS_TO_LARGE` | tool-bearing payloads → large |
| Default Route | `CASCADE_DEFAULT_ROUTE` | deterministic pin |
| Marker Mode | `CASCADE_MARKER_MODE` (`substring`/`prefix`) | doc/compression trigger match |
| Inference URL | `LARGE_TEXT_URL` / `LARGE_MLLM_URL` | main backend override (e.g. rented GPU) |
| Auxiliary URL | `SMALL_MLLM_URL` | compression/fast backend override |
| TTS / STT URL | `TTS_URL` / `STT_URL` | realtime audio passthroughs |

All settings apply live and persist via the settings file DB; env vars provide
the boot-time defaults.

## Model identity

`CASCADE_MODEL_NAME` (default `cascade-hybrid-v1`) is the single id advertised
by `/v1/models`, `/model` and `GET /web/api/model`, and stamped into every
response's `model` field. Set it to keep multi-backend routing invisible to
clients (LibreChat/Hermes just see one model). The real per-backend ids stay
visible in the `/models` bodies but are rewritten on egress.