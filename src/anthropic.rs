//! Anthropic Messages API compatibility layer (`/v1/messages`).
//!
//! Inbound Anthropic requests are translated into the gateway's internal
//! OpenAI chat-completions shape, routed through the existing three-condition
//! dispatcher, and the upstream OpenAI response is translated back into the
//! Anthropic Messages response shape (including the SSE event protocol).

use crate::state::GatewayState;
use crate::types::*;
use axum::{
    body::{Body, Bytes},
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::Response,
};
use futures_util::{stream, Stream, StreamExt};
use http_body_util::BodyExt;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ============================================================================
// Request types (tolerant: content blocks stay as raw JSON)
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<Value>,
    #[serde(default)]
    pub system: Option<Value>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

// ============================================================================
// Anthropic -> OpenAI (internal) translation
// ============================================================================

fn message(role: &str, content: Option<MessageContent>) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

fn text_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| {
                if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                    b.get("text").and_then(|t| t.as_str()).map(str::to_string)
                } else {
                    b.as_str().map(str::to_string)
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn system_text(system: &Value) -> String {
    text_of(system)
}

fn parts_to_content(parts: Vec<MessageContentPart>) -> Option<MessageContent> {
    if parts.is_empty() {
        return None;
    }
    if parts.len() == 1 {
        if let MessageContentPart::Text { text } = &parts[0] {
            return Some(MessageContent::Text(text.clone()));
        }
    }
    Some(MessageContent::Parts(parts))
}

fn image_part(block: &Value) -> Option<MessageContentPart> {
    let source = block.get("source")?;
    let url = match source
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("base64")
    {
        "base64" => {
            let media = source
                .get("media_type")
                .and_then(|m| m.as_str())
                .unwrap_or("image/png");
            let data = source.get("data").and_then(|d| d.as_str())?;
            format!("data:{};base64,{}", media, data)
        }
        "url" => source.get("url").and_then(|u| u.as_str())?.to_string(),
        _ => return None,
    };
    Some(MessageContentPart::ImageUrl {
        image_url: ImageUrlTarget { url },
    })
}

fn tool_use_entry(block: &Value) -> Option<Value> {
    let name = block.get("name").and_then(|n| n.as_str())?;
    let id = block
        .get("id")
        .and_then(|i| i.as_str())
        .unwrap_or("toolu_cascade");
    let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
    let arguments = if input.is_string() {
        input.as_str().unwrap_or("{}").to_string()
    } else {
        serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string())
    };
    Some(json!({
        "id": id,
        "type": "function",
        "function": { "name": name, "arguments": arguments }
    }))
}

fn convert_blocks(role: &str, blocks: &[Value], out: &mut Vec<ChatMessage>) {
    if role == "assistant" {
        let mut parts: Vec<MessageContentPart> = Vec::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        for block in blocks {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        parts.push(MessageContentPart::Text {
                            text: text.to_string(),
                        });
                    }
                }
                Some("tool_use") => {
                    if let Some(entry) = tool_use_entry(block) {
                        tool_calls.push(entry);
                    }
                }
                Some("image") => {
                    if let Some(part) = image_part(block) {
                        parts.push(part);
                    }
                }
                _ => {}
            }
        }
        let content = parts_to_content(parts);
        if tool_calls.is_empty() {
            if let Some(content) = content {
                out.push(message("assistant", Some(content)));
            }
        } else {
            let mut msg = message("assistant", content);
            msg.tool_calls = Some(Value::Array(tool_calls));
            out.push(msg);
        }
        return;
    }

    // user (and any other role): tool_result blocks become OpenAI `tool`
    // messages; text/image blocks become a single user message.
    let mut parts: Vec<MessageContentPart> = Vec::new();
    for block in blocks {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("tool_result") => {
                let tool_use_id = block
                    .get("tool_use_id")
                    .and_then(|i| i.as_str())
                    .unwrap_or_default();
                let content = block.get("content").map(text_of).unwrap_or_default();
                let mut msg = message("tool", Some(MessageContent::Text(content)));
                msg.tool_call_id = Some(tool_use_id.to_string());
                out.push(msg);
            }
            Some("text") => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    parts.push(MessageContentPart::Text {
                        text: text.to_string(),
                    });
                }
            }
            Some("image") => {
                if let Some(part) = image_part(block) {
                    parts.push(part);
                }
            }
            _ => {}
        }
    }
    if let Some(content) = parts_to_content(parts) {
        out.push(message(role, Some(content)));
    }
}

fn convert_tools(tools: &[Value]) -> Value {
    Value::Array(
        tools
            .iter()
            .filter_map(|t| {
                let name = t.get("name").and_then(|n| n.as_str())?;
                let description = t.get("description").and_then(|d| d.as_str()).unwrap_or("");
                let parameters = t
                    .get("input_schema")
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
                Some(json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters
                    }
                }))
            })
            .collect(),
    )
}

fn convert_tool_choice(choice: &Value) -> Value {
    match choice.get("type").and_then(|t| t.as_str()) {
        Some("auto") => json!("auto"),
        Some("any") => json!("required"),
        Some("none") => json!("none"),
        Some("tool") => {
            let name = choice.get("name").and_then(|n| n.as_str()).unwrap_or("");
            json!({"type": "function", "function": { "name": name }})
        }
        _ => json!("auto"),
    }
}

/// Translate an Anthropic Messages request into the gateway's internal
/// OpenAI chat-completions request.
pub fn to_openai_request(req: &AnthropicMessagesRequest) -> ChatCompletionRequest {
    let mut messages: Vec<ChatMessage> = Vec::new();

    if let Some(system) = &req.system {
        let text = system_text(system);
        if !text.is_empty() {
            messages.push(message("system", Some(MessageContent::Text(text))));
        }
    }

    for msg in &req.messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        match msg.get("content") {
            Some(Value::String(s)) => {
                messages.push(message(role, Some(MessageContent::Text(s.clone()))));
            }
            Some(Value::Array(blocks)) => convert_blocks(role, blocks, &mut messages),
            _ => {}
        }
    }

    let stop = req
        .stop_sequences
        .as_ref()
        .filter(|s| !s.is_empty())
        .map(|s| serde_json::to_value(s).unwrap_or(Value::Null));

    let user = req
        .metadata
        .as_ref()
        .and_then(|m| m.get("user_id"))
        .and_then(|u| u.as_str())
        .map(str::to_string);

    ChatCompletionRequest {
        model: req.model.clone(),
        messages,
        stream: req.stream,
        temperature: req.temperature,
        max_tokens: req.max_tokens.or(Some(4096)),
        logprobs: None,
        top_logprobs: None,
        tools: req
            .tools
            .as_deref()
            .filter(|t| !t.is_empty())
            .map(convert_tools),
        tool_choice: req.tool_choice.as_ref().map(convert_tool_choice),
        functions: None,
        function_call: None,
        user,
        stop,
        response_format: None,
        metadata: None,
        route_hint: None,
        // The Anthropic stream converter reads the terminal usage chunk for
        // `message_delta.usage`, so it always opts in upstream (the client here
        // never sees OpenAI frames, only the translated Anthropic events).
        stream_options: if req.stream.unwrap_or(false) {
            Some(serde_json::json!({"include_usage": true}))
        } else {
            None
        },
    }
}

// ============================================================================
// OpenAI -> Anthropic response translation
// ============================================================================

fn map_finish(finish: Option<&str>) -> &'static str {
    match finish {
        Some("length") => "max_tokens",
        Some("tool_calls") | Some("function_call") => "tool_use",
        Some("content_filter") => "end_turn",
        _ => "end_turn",
    }
}

/// Translate a normalized OpenAI chat-completion response into the Anthropic
/// Messages response shape.
pub fn to_anthropic_response(openai: &Value, requested_model: &str, message_id: &str) -> Value {
    let choice = openai
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first());
    let message = choice.and_then(|c| c.get("message"));

    let mut content: Vec<Value> = Vec::new();
    let text = message
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    if !text.is_empty() {
        content.push(json!({"type": "text", "text": text}));
    }

    if let Some(tool_calls) = message
        .and_then(|m| m.get("tool_calls"))
        .and_then(|t| t.as_array())
    {
        for tc in tool_calls {
            let id = tc
                .get("id")
                .and_then(|i| i.as_str())
                .unwrap_or("toolu_cascade");
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let args = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .unwrap_or("{}");
            let input = serde_json::from_str::<Value>(args).unwrap_or_else(|_| json!({}));
            content.push(json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input
            }));
        }
    }

    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }

    let finish = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|f| f.as_str());
    let usage = openai.get("usage");
    let input_tokens = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output_tokens = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    json!({
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": map_finish(finish),
        "stop_sequence": Value::Null,
        "usage": { "input_tokens": input_tokens, "output_tokens": output_tokens }
    })
}

// ============================================================================
// Streaming: OpenAI SSE -> Anthropic SSE event protocol
// ============================================================================

#[derive(PartialEq, Eq, Clone, Copy)]
enum BlockKind {
    None,
    Text,
    Tool,
}

struct AnthropicStream<S> {
    inner: Pin<Box<S>>,
    buffer: Vec<u8>,
    model: String,
    message_id: String,
    block_open: bool,
    block_index: i64,
    block_kind: BlockKind,
    tool_blocks: HashMap<i64, i64>,
    stop_reason: Option<String>,
    output_tokens: u64,
    input_tokens: u64,
    done: bool,
    pending: VecDeque<Bytes>,
}

impl<S> AnthropicStream<S>
where
    S: Stream<Item = Result<Bytes, axum::Error>> + Send + 'static,
{
    fn new(inner: S, model: String, message_id: String) -> Self {
        let mut state = Self {
            inner: Box::pin(inner),
            buffer: Vec::new(),
            model,
            message_id,
            block_open: false,
            block_index: -1,
            block_kind: BlockKind::None,
            tool_blocks: HashMap::new(),
            stop_reason: None,
            output_tokens: 0,
            input_tokens: 0,
            done: false,
            pending: VecDeque::new(),
        };
        let start = json!({
            "type": "message_start",
            "message": {
                "id": state.message_id,
                "type": "message",
                "role": "assistant",
                "model": state.model,
                "content": [],
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": { "input_tokens": 0, "output_tokens": 0 }
            }
        });
        state.emit(start);
        state
    }

    fn emit(&mut self, event: Value) {
        let name = event
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("message");
        let data = serde_json::to_string(&event).unwrap_or_default();
        self.pending
            .push_back(Bytes::from(format!("event: {}\ndata: {}\n\n", name, data)));
    }

    fn close_block(&mut self) {
        if self.block_open {
            self.emit(json!({
                "type": "content_block_stop",
                "index": self.block_index
            }));
            self.block_open = false;
            self.block_kind = BlockKind::None;
        }
    }

    fn open_text_block(&mut self) {
        if self.block_open && self.block_kind == BlockKind::Text {
            return;
        }
        self.close_block();
        self.block_index += 1;
        self.block_open = true;
        self.block_kind = BlockKind::Text;
        self.emit(json!({
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": { "type": "text", "text": "" }
        }));
    }

    fn process(&mut self, chunk: &[u8]) {
        self.buffer.extend_from_slice(chunk);
        while let Some(pos) = find_subslice(&self.buffer, b"\n\n") {
            let frame: Vec<u8> = self.buffer.drain(..pos + 2).collect();
            let text = String::from_utf8_lossy(&frame[..pos]).to_string();
            self.handle_frame(&text);
        }
    }

    fn handle_frame(&mut self, frame: &str) {
        let mut data: Option<&str> = None;
        for line in frame.lines() {
            if let Some(rest) = line.strip_prefix("data:") {
                data = Some(rest.trim());
            }
        }
        let Some(data) = data else { return };
        if data.is_empty() {
            return;
        }
        if data == "[DONE]" {
            self.finish();
            return;
        }
        if let Ok(chunk) = serde_json::from_str::<Value>(data) {
            self.handle_chunk(&chunk);
        }
    }

    fn handle_chunk(&mut self, chunk: &Value) {
        if let Some(usage) = chunk.get("usage") {
            if let Some(v) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
                self.input_tokens = v;
            }
            if let Some(v) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
                self.output_tokens = v;
            }
        }

        let Some(choice) = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first())
        else {
            return;
        };

        if let Some(fr) = choice.get("finish_reason").and_then(|f| f.as_str()) {
            self.stop_reason = Some(map_finish(Some(fr)).to_string());
        }

        let Some(delta) = choice.get("delta") else {
            return;
        };

        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
            if !content.is_empty() {
                self.open_text_block();
                let index = self.block_index;
                self.emit(json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": { "type": "text_delta", "text": content }
                }));
            }
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tool_calls {
                let idx = tc.get("index").and_then(|i| i.as_i64()).unwrap_or(0);
                let args = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|a| a.as_str())
                    .unwrap_or("");

                if !self.tool_blocks.contains_key(&idx) {
                    self.close_block();
                    self.block_index += 1;
                    self.block_open = true;
                    self.block_kind = BlockKind::Tool;
                    self.tool_blocks.insert(idx, self.block_index);
                    let id = tc
                        .get("id")
                        .and_then(|i| i.as_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("toolu_{}", self.block_index));
                    let name = tc
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("");
                    let index = self.block_index;
                    self.emit(json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": {}
                        }
                    }));
                }

                if !args.is_empty() {
                    let index = self.tool_blocks[&idx];
                    self.emit(json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": { "type": "input_json_delta", "partial_json": args }
                    }));
                }
            }
        }
    }

    fn finish(&mut self) {
        if self.done {
            return;
        }
        if !self.buffer.is_empty() {
            let leftover = String::from_utf8_lossy(&self.buffer).to_string();
            self.buffer.clear();
            self.handle_frame(&leftover);
        }
        self.close_block();
        let stop_reason = self
            .stop_reason
            .clone()
            .unwrap_or_else(|| "end_turn".to_string());
        self.emit(json!({
            "type": "message_delta",
            "delta": { "stop_reason": stop_reason, "stop_sequence": Value::Null },
            "usage": { "output_tokens": self.output_tokens }
        }));
        self.emit(json!({ "type": "message_stop" }));
        self.done = true;
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn stream_openai_to_anthropic<S>(
    inner: S,
    model: String,
    message_id: String,
) -> impl Stream<Item = Result<Bytes, axum::Error>>
where
    S: Stream<Item = Result<Bytes, axum::Error>> + Send + 'static,
{
    let state = AnthropicStream::new(inner, model, message_id);
    stream::unfold(state, |mut st| async move {
        loop {
            if let Some(out) = st.pending.pop_front() {
                return Some((Ok(out), st));
            }
            if st.done {
                return None;
            }
            match st.inner.next().await {
                Some(Ok(chunk)) => st.process(&chunk),
                Some(Err(e)) => {
                    st.done = true;
                    return Some((Err(e), st));
                }
                None => st.finish(),
            }
        }
    })
}

// ============================================================================
// Handlers
// ============================================================================

static MESSAGE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn new_message_id() -> String {
    let n = MESSAGE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = chrono::Utc::now().timestamp_millis();
    format!("msg_{:x}{:04x}", ts, (n & 0xffff) as u16)
}

fn json_body(value: Value, status: StatusCode) -> Response {
    let mut resp = Response::new(Body::from(value.to_string()));
    *resp.status_mut() = status;
    resp.headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    resp
}

fn anthropic_error(status: StatusCode, err_type: &str, message: &str) -> Response {
    json_body(
        json!({"type": "error", "error": { "type": err_type, "message": message }}),
        status,
    )
}

fn copy_cascade_headers(dst: &mut HeaderMap, src: &HeaderMap) {
    for (k, v) in src.iter() {
        let name = k.as_str();
        if name.starts_with("x-cascade") || name == "x-model-name" || name == "x-confidence" {
            dst.insert(k.clone(), v.clone());
        }
    }
}

async fn parse_request(
    req: Request<Body>,
) -> Result<(AnthropicMessagesRequest, HeaderMap), Response> {
    let headers = req.headers().clone();
    let bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => {
            return Err(anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "Invalid request body",
            ))
        }
    };
    match serde_json::from_slice::<AnthropicMessagesRequest>(&bytes) {
        Ok(r) => Ok((r, headers)),
        Err(e) => Err(anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &format!("Invalid JSON: {}", e),
        )),
    }
}

/// POST /v1/messages — Anthropic Messages API.
pub async fn messages(State(state): State<Arc<GatewayState>>, req: Request<Body>) -> Response {
    let (areq, headers) = match parse_request(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let origin = headers
        .get("x-request-origin")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("anthropic")
        .to_string();
    state.metrics.record_request_origin(&origin);
    let tier = headers
        .get("x-tier")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("standard");

    let permit = match state.tenant_limiter.acquire(&headers).await {
        Ok(permit) => permit,
        Err(rej) => return crate::tenant_limit::rejection_response(rej),
    };

    let is_streaming = areq.stream.unwrap_or(false);
    let requested_model = areq.model.clone();
    let message_id = new_message_id();
    let mut openai_req = to_openai_request(&areq);
    // Anthropic clients always send Claude model ids, which the local /
    // registered backends do not know. Route with the gateway's unified model
    // id; the requested id is still echoed back in the response. Registry nodes
    // with a per-node `model` override it again downstream.
    if !state.config.cascade_model_name.is_empty() {
        openai_req.model = state.config.cascade_model_name.clone();
    }

    match state
        .route_request_with_fallback(openai_req, is_streaming, tier, &headers, &origin)
        .await
    {
        Ok((hdrs, body)) => {
            let body = crate::tenant_limit::guard_response_body(body, permit);
            if is_streaming {
                let mut response_headers = HeaderMap::new();
                response_headers.insert(
                    "content-type",
                    HeaderValue::from_static("text/event-stream"),
                );
                response_headers.insert("cache-control", HeaderValue::from_static("no-cache"));
                response_headers.insert("connection", HeaderValue::from_static("keep-alive"));
                copy_cascade_headers(&mut response_headers, &hdrs);
                let stream = stream_openai_to_anthropic(
                    body.into_data_stream(),
                    requested_model,
                    message_id,
                );
                let mut resp = Response::new(Body::from_stream(stream));
                *resp.status_mut() = StatusCode::OK;
                *resp.headers_mut() = response_headers;
                resp
            } else {
                let bytes = match body.collect().await {
                    Ok(b) => b.to_bytes(),
                    Err(_) => {
                        return anthropic_error(
                            StatusCode::BAD_GATEWAY,
                            "api_error",
                            "failed to read backend response",
                        )
                    }
                };
                let openai: Value = serde_json::from_slice(&bytes).unwrap_or_else(|_| json!({}));
                let value = to_anthropic_response(&openai, &requested_model, &message_id);
                let mut resp = json_body(value, StatusCode::OK);
                copy_cascade_headers(resp.headers_mut(), &hdrs);
                resp
            }
        }
        Err(e) => {
            let status = StatusCode::from_u16(e.status).unwrap_or(StatusCode::BAD_GATEWAY);
            anthropic_error(
                status,
                "api_error",
                &format!("{}: {}", e.backend, e.context),
            )
        }
    }
}

/// POST /v1/messages/count_tokens — approximate token counting.
///
/// The gateway does not run the model tokenizer, so this returns a documented
/// estimate (~4 characters per token) over the translated prompt.
pub async fn count_tokens(State(_state): State<Arc<GatewayState>>, req: Request<Body>) -> Response {
    let (areq, _headers) = match parse_request(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let openai_req = to_openai_request(&areq);
    let text = crate::language::extract_text(&openai_req.messages);
    let tokens = (text.chars().count() / 4).max(1) as u64;
    json_body(json!({ "input_tokens": tokens }), StatusCode::OK)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(value: Value) -> AnthropicMessagesRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn maps_system_and_plain_messages() {
        let req = parse(json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        }));
        let openai = to_openai_request(&req);
        assert_eq!(openai.model, "claude-3-5-sonnet");
        assert_eq!(openai.max_tokens, Some(100));
        assert_eq!(openai.messages.len(), 2);
        assert_eq!(openai.messages[0].role, "system");
        assert_eq!(openai.messages[1].role, "user");
    }

    #[test]
    fn maps_tools_and_tool_choice() {
        let req = parse(json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }],
            "tool_choice": {"type": "tool", "name": "get_weather"}
        }));
        let openai = to_openai_request(&req);
        let tools = openai.tools.unwrap();
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(
            openai.tool_choice.unwrap()["function"]["name"],
            "get_weather"
        );
    }

    #[test]
    fn splits_tool_results_into_tool_messages() {
        let req = parse(json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}
                ]}
            ]
        }));
        let openai = to_openai_request(&req);
        assert_eq!(openai.messages[0].role, "assistant");
        assert!(openai.messages[0].tool_calls.is_some());
        assert_eq!(openai.messages[1].role, "tool");
        assert_eq!(openai.messages[1].tool_call_id.as_deref(), Some("toolu_1"));
    }

    #[test]
    fn maps_response_with_tool_use() {
        let openai = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Checking",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4}
        });
        let out = to_anthropic_response(&openai, "claude-3-5-sonnet", "msg_1");
        assert_eq!(out["type"], "message");
        assert_eq!(out["stop_reason"], "tool_use");
        assert_eq!(out["content"][0]["type"], "text");
        assert_eq!(out["content"][1]["type"], "tool_use");
        assert_eq!(out["content"][1]["input"]["city"], "Paris");
        assert_eq!(out["usage"]["input_tokens"], 12);
    }

    #[test]
    fn maps_length_finish_to_max_tokens() {
        let openai = json!({
            "choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
            "usage": {}
        });
        let out = to_anthropic_response(&openai, "m", "msg_2");
        assert_eq!(out["stop_reason"], "max_tokens");
    }

    async fn run_stream(chunks: Vec<&str>) -> String {
        let items: Vec<Result<Bytes, axum::Error>> = chunks
            .iter()
            .map(|c| Ok(Bytes::from(c.to_string())))
            .collect();
        let inner = stream::iter(items);
        let out = stream_openai_to_anthropic(inner, "m".to_string(), "msg_3".to_string());
        futures_util::pin_mut!(out);
        let mut collected = String::new();
        while let Some(item) = out.next().await {
            collected.push_str(&String::from_utf8_lossy(&item.unwrap()));
        }
        collected
    }

    #[tokio::test]
    async fn streams_text_events() {
        let out = run_stream(vec![
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ])
        .await;
        assert!(out.contains("event: message_start"));
        assert!(out.contains("\"type\":\"content_block_start\""));
        assert!(out.contains("\"type\":\"text_delta\""));
        assert!(out.contains("\"text\":\"Hello\""));
        assert!(out.contains("\"text\":\" world\""));
        assert!(out.contains("event: message_delta"));
        assert!(out.contains("\"stop_reason\":\"end_turn\""));
        assert!(out.contains("event: message_stop"));
        assert_eq!(out.matches("\"type\":\"content_block_start\"").count(), 1);
    }

    #[tokio::test]
    async fn streams_tool_use_events() {
        let out = run_stream(vec![
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Paris\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        ]).await;
        assert!(out.contains("\"type\":\"tool_use\""));
        assert!(out.contains("\"type\":\"input_json_delta\""));
        assert!(out.contains("\"stop_reason\":\"tool_use\""));
        assert!(out.contains("event: message_stop"));
    }

    #[tokio::test]
    async fn handles_frames_split_across_chunks() {
        let out = run_stream(vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"He",
            "llo\"}}]}\n\ndata: [DONE]\n\n",
        ])
        .await;
        assert!(out.contains("\"text\":\"Hello\""));
        assert!(out.contains("event: message_stop"));
    }
}
