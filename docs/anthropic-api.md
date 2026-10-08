# Anthropic-compatible Messages API

The gateway exposes an Anthropic Messages API next to its OpenAI-compatible
API. Anthropic clients (Claude Code, the Anthropic SDKs, etc.) can point at the
gateway and their requests are translated into the internal OpenAI chat
shape, routed by the same three-condition dispatcher, and translated back.

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/v1/messages` | Messages API (buffered + streaming) |
| POST | `/v1/messages/count_tokens` | Approximate input-token count |

## Example

```bash
curl -s http://localhost:3000/v1/messages \
  -H 'content-type: application/json' \
  -H 'x-api-key: anything' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Streaming (`"stream": true`) emits the standard Anthropic SSE event protocol:
`message_start`, `content_block_start`, `content_block_delta`,
`content_block_stop`, `message_delta`, `message_stop`.

## Translation

Inbound (Anthropic → internal OpenAI):

- `system` (string or text blocks) → a `system` message.
- `messages[].content` strings or blocks:
  - `text` → text content
  - `image` (`base64` or `url` source) → OpenAI `image_url`
  - `tool_use` (assistant) → OpenAI `tool_calls`
  - `tool_result` (user) → a `tool` role message with `tool_call_id`
- `tools[].input_schema` → `function.parameters`; `tool_choice` types
  `auto`/`any`/`tool` map to `auto`/`required`/named function.
- `stop_sequences` → `stop`; `max_tokens` → `max_tokens` (default 4096).
- `metadata.user_id` → `user`.

Outbound (internal OpenAI → Anthropic):

- `message.content` → a `text` block; `tool_calls` → `tool_use` blocks with the
  parsed JSON arguments as `input`.
- `finish_reason`: `stop`/`content_filter` → `end_turn`, `length` →
  `max_tokens`, `tool_calls`/`function_call` → `tool_use`.
- `usage.prompt_tokens`/`completion_tokens` → `input_tokens`/`output_tokens`.

The requested `model` is echoed back unchanged; routing ignores it (the
gateway picks the `main`/`auxiliary`/`ocr` upstream itself). `x-cascade-route`,
`x-cascade-node` and `x-model-name` response headers are preserved on both the
buffered and streaming responses.

## Notes

- `count_tokens` is an estimate (~4 characters per token) because the gateway
  does not run the model tokenizer; it is suitable for client-side budgeting,
  not exact accounting.
- `reasoning_content`/thinking deltas are not surfaced as Anthropic `thinking`
  blocks (they lack the signature the API requires).
