use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Json, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    routing::post,
    Router,
};
use futures_util::stream::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod language;

async fn fetch_large_model_multimodal_async(inference_url: &str) -> bool {
    let models_url = format!("{}/models", inference_url.trim_end_matches('/'));
    match reqwest::Client::new()
        .get(&models_url)
        .send()
        .await
    {
        Ok(resp) => match resp.json::<Value>().await {
            Ok(json) => {
                let empty_vec = Vec::new();
                let models = json.get("models").and_then(|m| m.as_array()).unwrap_or(&empty_vec);
                for model in models {
                    let empty_caps = Vec::new();
                    let caps = model.get("capabilities").and_then(|c| c.as_array()).unwrap_or(&empty_caps);
                    if caps.iter().any(|c| c.as_str() == Some("multimodal")) {
                        info!("Auto-detected: large model supports multimodal");
                        return true;
                    }
                }
                info!("Auto-detected: large model is text-only");
                false
            }
            Err(e) => {
                warn!("Failed to parse /models response JSON: {}, defaulting to false", e);
                false
            }
        },
        Err(e) => {
            warn!("Failed to fetch /models endpoint (inference server not ready?): {}", e);
            false
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrlTarget },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ImageUrlTarget {
    pub url: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<MessageContentPart>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<Value>,
}

#[derive(Debug, Clone)]
struct CircuitBreaker {
    failures: Arc<RwLock<HashMap<String, Vec<Instant>>>>,
    threshold: u32,
    reset_duration: Duration,
}

impl CircuitBreaker {
    fn new(threshold: u32, reset_duration_secs: u64) -> Self {
        Self {
            failures: Arc::new(RwLock::new(HashMap::new())),
            threshold,
            reset_duration: Duration::from_secs(reset_duration_secs),
        }
    }

    async fn record_failure(&self, url: &str) {
        let mut failures = self.failures.write().await;
        let now = Instant::now();
        let entry = failures.entry(url.to_string()).or_default();
        entry.push(now);
        entry.retain(|t| now.duration_since(*t) < self.reset_duration);
        warn!(
            "Circuit breaker: {} failures for {} in last {}s",
            entry.len(),
            url,
            self.reset_duration.as_secs()
        );
    }

    async fn is_open(&self, url: &str) -> bool {
        let failures = self.failures.read().await;
        if let Some(times) = failures.get(url) {
            let now = Instant::now();
            let recent: Vec<_> = times
                .iter()
                .filter(|t| now.duration_since(**t) < self.reset_duration)
                .collect();
            recent.len() as u32 >= self.threshold
        } else {
            false
        }
    }

    async fn record_success(&self, url: &str) {
        let mut failures = self.failures.write().await;
        if failures.remove(url).is_some() {
            info!("Circuit breaker reset for {}", url);
        }
    }
}

#[derive(Debug, Clone)]
struct LoadTracker {
    request_count: Arc<AtomicU64>,
    total_complexity: Arc<AtomicU64>,
}

impl Default for LoadTracker {
    fn default() -> Self {
        Self {
            request_count: Arc::new(AtomicU64::new(0)),
            total_complexity: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl LoadTracker {
    fn record(&self, complexity: f64) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        self.total_complexity
            .fetch_add((complexity * 100.0) as u64, Ordering::Relaxed);
    }

    fn avg_complexity(&self) -> f64 {
        let count = self.request_count.load(Ordering::Relaxed);
        if count == 0 {
            0.0
        } else {
            let total = self.total_complexity.load(Ordering::Relaxed);
            total as f64 / count as f64 / 100.0
        }
    }
}

#[derive(Debug, Clone)]
struct GatewayState {
    http_client: Arc<reqwest::Client>,
    small_mllm_url: String,
    large_mllm_url: String,
    large_text_url: String,
    router_threshold: f64,
    confidence_threshold: f64,
    large_model_multimodal: bool,
    route_tools_to_large: bool,
    circuit_breaker: CircuitBreaker,
    load_tracker: LoadTracker,
}

impl GatewayState {
    fn new(large_model_multimodal: bool) -> Self {
        let small_mllm_url = std::env::var("SMALL_MLLM_URL").unwrap_or_else(|_| "http://localhost:8082/v1/chat/completions".to_string());
        let large_mllm_url = std::env::var("LARGE_MLLM_URL").unwrap_or_else(|_| "http://localhost:8080/v1/chat/completions".to_string());
        let large_text_url = std::env::var("LARGE_TEXT_URL").unwrap_or_else(|_| "http://localhost:8080/v1/chat/completions".to_string());
        let router_threshold = std::env::var("ROUTER_THRESHOLD")
            .unwrap_or_else(|_| "0.5".to_string())
            .parse::<f64>()
            .unwrap_or(0.5);
        let confidence_threshold = std::env::var("CONFIDENCE_THRESHOLD")
            .unwrap_or_else(|_| "0.7".to_string())
            .parse::<f64>()
            .unwrap_or(0.7);
        let route_tools_to_large = std::env::var("ROUTE_TOOLS_TO_LARGE")
            .unwrap_or_else(|_| "true".to_string())
            .eq_ignore_ascii_case("true");

        let http_client = Arc::new(
            reqwest::Client::builder()
                .pool_max_idle_per_host(0)
                .pool_idle_timeout(std::time::Duration::from_secs(90))
                .build()
                .expect("Failed to build reqwest client"),
        );

        let cb_threshold = std::env::var("CIRCUIT_BREAKER_THRESHOLD")
            .unwrap_or_else(|_| "5".to_string())
            .parse::<u32>()
            .unwrap_or(5);
        let cb_reset = std::env::var("CIRCUIT_BREAKER_RESET_SECS")
            .unwrap_or_else(|_| "60".to_string())
            .parse::<u64>()
            .unwrap_or(60);

        Self {
            http_client,
            small_mllm_url,
            large_mllm_url,
            large_text_url,
            router_threshold,
            confidence_threshold,
            large_model_multimodal,
            route_tools_to_large,
            circuit_breaker: CircuitBreaker::new(cb_threshold, cb_reset),
            load_tracker: LoadTracker::default(),
        }
    }

    fn detect_language(&self, messages: &[ChatMessage]) -> &'static str {
        language::detect_language(messages)
    }

    fn evaluate_complexity(&self, messages: &[ChatMessage]) -> f64 {
        let mut total_chars = 0;
        let mut keyword_score = 0.0;
        let keywords = ["analyze deeply", "write code", "expert", "reasoning", "logic", "complex"];

        for msg in messages {
            match &msg.content {
                MessageContent::Text(text) => {
                    total_chars += text.len();
                    for keyword in &keywords {
                        if text.to_lowercase().contains(keyword) {
                            keyword_score += 0.2;
                        }
                    }
                }
                MessageContent::Parts(parts) => {
                    for part in parts {
                        match part {
                            MessageContentPart::Text { text } => {
                                total_chars += text.len();
                                for keyword in &keywords {
                                    if text.to_lowercase().contains(keyword) {
                                        keyword_score += 0.2;
                                    }
                                }
                            }
                            MessageContentPart::ImageUrl { .. } => {
                                total_chars += 100;
                            }
                        }
                    }
                }
            }
        }

        let char_score = (total_chars as f64 / 1000.0).min(1.0);
        let mut score = 0.5 * char_score + 0.5 * keyword_score;
        score = score.min(1.0).max(0.0);
        score
    }

    fn extract_confidence(&self, body: &[u8]) -> Option<f64> {
        let value: Value = serde_json::from_slice(body).ok()?;
        let choices = value.get("choices")?.as_array()?;
        let logprobs = choices.first()?.get("logprobs")?;
        let content = logprobs.get("content")?.as_array()?;

        if content.is_empty() {
            return None;
        }

        let sum: f64 = content
            .iter()
            .filter_map(|t| t.get("logprob")?.as_f64())
            .sum();

        let mean = sum / content.len() as f64;
        Some(mean.exp())
    }

    fn pick_model(&self, has_image: bool, complexity: f64, tier: &str) -> (bool, &str) {
        // Premium tier always routes to large model
        if tier == "premium" {
            info!("PREMIUM TIER: routing to large model");
            return (false, &self.large_text_url);
        }

        if has_image {
            if self.large_model_multimodal && complexity > self.router_threshold {
                info!("MODEL SELECTION: image present, complexity {:.2} > threshold {}, routing to large multimodal model",
                    complexity, self.router_threshold);
                (false, &self.large_mllm_url)
            } else if self.large_model_multimodal {
                info!("MODEL SELECTION: image present but complexity {:.2} <= threshold {}, routing to small multimodal model (cost optimization)",
                    complexity, self.router_threshold);
                (true, &self.small_mllm_url)
            } else {
                info!("MODEL SELECTION: image present but large model is text-only, routing to small multimodal model");
                (true, &self.small_mllm_url)
            }
        } else if complexity > self.router_threshold {
            info!("MODEL SELECTION: text-only, complexity {:.2} > threshold {}, routing to large text model",
                complexity, self.router_threshold);
            (false, &self.large_text_url)
        } else {
            info!("MODEL SELECTION: text-only, complexity {:.2} <= threshold {}, routing to small model (cost optimization)",
                complexity, self.router_threshold);
            (true, &self.small_mllm_url)
        }
    }

    async fn proxy_to_backend(&self, payload: &ChatCompletionRequest, url: &str, is_streaming: bool) -> Result<(HeaderMap, Body), StatusCode> {
        let backend_response = self
            .http_client
            .post(url)
            .json(payload)
            .send()
            .await
            .map_err(|_| StatusCode::BAD_GATEWAY)?;

        let mut headers = HeaderMap::new();
        if is_streaming {
            headers.insert("content-type", HeaderValue::from_static("text/event-stream"));
            headers.insert("cache-control", HeaderValue::from_static("no-cache"));
            headers.insert("connection", HeaderValue::from_static("keep-alive"));
        } else {
            headers.insert("content-type", HeaderValue::from_static("application/json"));
        }

        let stream = backend_response
            .bytes_stream()
            .map(|item| item.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)));

        let body = Body::from_stream(stream);
        Ok((headers, body))
    }

    /// Download image from URL and convert to base64 data URL.
    async fn download_image_as_base64(&self, url: &str) -> Option<String> {
        if url.starts_with("data:") {
            return Some(url.to_string());
        }
        let resp = self.http_client.get(url).send().await.ok()?;
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("image/png")
            .to_string();
        let bytes = resp.bytes().await.ok()?;
        let encoded = base64_encode(&bytes);
        Some(format!("data:{};base64,{}", content_type, encoded))
    }

    /// Describe an image using the small vision model.
    /// Downloads the image in the gateway and sends as base64 data URL
    /// to avoid llama.cpp's external URL download issues.
    async fn describe_image(&self, image_url: &str, language: &str) -> Option<String> {
        info!("Downloading image for description: {}", image_url);
        let data_url = self.download_image_as_base64(image_url).await?;
        info!("Image downloaded, size: {} bytes", data_url.len());

        let prompt_text = language::get_image_prompt(language);

        let desc_payload = ChatCompletionRequest {
            model: "vision".to_string(),
            messages: vec![
                ChatMessage {
                    role: "user".to_string(),
                    content: MessageContent::Parts(vec![
                        MessageContentPart::Text { text: prompt_text.to_string() },
                        MessageContentPart::ImageUrl { image_url: ImageUrlTarget { url: data_url } },
                    ]),
                },
            ],
            stream: Some(false),
            temperature: Some(0.2),
            max_tokens: Some(512),
            logprobs: None,
            top_logprobs: None,
            tools: None,
            tool_choice: None,
            functions: None,
            function_call: None,
        };

        let resp = self.http_client
            .post(&self.small_mllm_url)
            .json(&desc_payload)
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            info!("Image description failed: HTTP {}", resp.status());
            return None;
        }

        let body: Value = resp.json().await.ok()?;
        body.get("choices")?
            .as_array()?
            .first()?
            .get("message")?
            .get("content")?
            .as_str()
            .map(|s| s.to_string())
    }

    /// Replace image_url parts with text descriptions in the payload.
    fn replace_images_with_text(&self, payload: &mut ChatCompletionRequest, descriptions: &[String]) {
        let mut desc_idx = 0;
        for msg in payload.messages.iter_mut() {
            if let MessageContent::Parts(parts) = &mut msg.content {
                let mut new_parts = Vec::new();
                for part in parts.drain(..) {
                    match part {
                        MessageContentPart::ImageUrl { .. } => {
                            if desc_idx < descriptions.len() {
                                new_parts.push(MessageContentPart::Text {
                                    text: format!("[Image: {}]", descriptions[desc_idx]),
                                });
                                desc_idx += 1;
                            }
                        }
                        _ => new_parts.push(part),
                    }
                }
                if new_parts.len() == 1 {
                    if let MessageContentPart::Text { text } = new_parts.remove(0) {
                        msg.content = MessageContent::Text(text);
                    }
                } else {
                    msg.content = MessageContent::Parts(new_parts);
                }
            }
        }
    }

    /// Extract all image URLs from messages.
    fn extract_image_urls(&self, messages: &[ChatMessage]) -> Vec<String> {
        let mut urls = Vec::new();
        for msg in messages {
            if let MessageContent::Parts(parts) = &msg.content {
                for part in parts {
                    if let MessageContentPart::ImageUrl { image_url } = part {
                        urls.push(image_url.url.clone());
                    }
                }
            }
        }
        urls
    }

    /// Check if a backend URL is available (circuit breaker closed).
    async fn is_backend_available(&self, url: &str, fallback: &str) -> String {
        if self.circuit_breaker.is_open(url).await {
            warn!("Circuit breaker OPEN for {}, using fallback {}", url, fallback);
            return fallback.to_string();
        }
        url.to_string()
    }

    async fn route_request(&self, payload: ChatCompletionRequest, is_streaming: bool, tier: &str) -> Result<(HeaderMap, Body), StatusCode> {
        let has_image = self.detect_image(&payload.messages);
        let has_tools = payload.tools.is_some() || payload.functions.is_some();
        let complexity_score = self.evaluate_complexity(&payload.messages);
        let language = self.detect_language(&payload.messages);

        // Record load metrics
        self.load_tracker.record(complexity_score);

        // Check circuit breaker for backends
        let small_url = self.is_backend_available(&self.small_mllm_url, &self.large_text_url).await;
        let large_url = self.is_backend_available(&self.large_text_url, &self.large_mllm_url).await;

        // If both image AND tools: describe image with small vision model,
        // then route text description + tools to large text model
        if has_image && has_tools {
            info!("IMAGE + TOOLS: describing image with small vision model first");

            let image_urls = self.extract_image_urls(&payload.messages);

            // Describe each image
            let mut descriptions = Vec::new();
            for url in &image_urls {
                if let Some(desc) = self.describe_image(url, language).await {
                    info!("Image description: {}", desc.chars().take(100).collect::<String>());
                    descriptions.push(desc);
                } else {
                    descriptions.push("[Image could not be described]".to_string());
                }
            }

            // Replace images with descriptions and route to large model
            let mut modified_payload = payload.clone();
            self.replace_images_with_text(&mut modified_payload, &descriptions);

            info!("IMAGE + TOOLS: routing text+tools to large text model");
            let result = self.proxy_to_backend(&modified_payload, &large_url, is_streaming).await;
            // Track success/failure
            match &result {
                Ok(_) => { self.circuit_breaker.record_success(&large_url).await; }
                Err(_) => { self.circuit_breaker.record_failure(&large_url).await; }
            }
            return result;
        }

        // If tools are present and route_tools_to_large is enabled, route to large model
        if has_tools && self.route_tools_to_large {
            let target = if has_image { &large_url } else { &large_url };
            info!("TOOLS DETECTED + route_tools_to_large=true: routing to {}", if has_image { "large multimodal model" } else { "large text model" });
            let result = self.proxy_to_backend(&payload, target, is_streaming).await;
            match &result {
                Ok(_) => { self.circuit_breaker.record_success(target).await; }
                Err(_) => { self.circuit_breaker.record_failure(target).await; }
            }
            return result;
        }

        info!(
            "Routing decision: has_image={}, complexity_score={:.2}, threshold={}, large_multimodal={}",
            has_image, complexity_score, self.router_threshold, self.large_model_multimodal
        );

        let (use_small, target_url) = self.pick_model(has_image, complexity_score, tier);
        let target_url = target_url.to_owned();
        info!("SELECTED_URL: {}", target_url);

        if !use_small {
            let result = self.proxy_to_backend(&payload, &target_url, is_streaming).await;
            match &result {
                Ok(_) => { self.circuit_breaker.record_success(&target_url).await; }
                Err(_) => { self.circuit_breaker.record_failure(&target_url).await; }
            }
            return result;
        }

        // === Small model path ===
        let small_payload = language::inject_language_prompt(language, payload.clone());

        // Streaming: proxy directly (no confidence check possible on a stream)
        if is_streaming {
            let result = self.proxy_to_backend(&small_payload, &target_url, true).await;
            match &result {
                Ok(_) => { self.circuit_breaker.record_success(&target_url).await; }
                Err(_) => { self.circuit_breaker.record_failure(&target_url).await; }
            }
            return result;
        }

        // Non-streaming: try small model with logprobs for confidence-based rerouting
        let mut small_payload = small_payload;
        small_payload.logprobs = Some(true);
        small_payload.top_logprobs = Some(0);

        let backend_response = self
            .http_client
            .post(&target_url)
            .json(&small_payload)
            .send()
            .await
            .map_err(|_| StatusCode::BAD_GATEWAY)?;

        let status = backend_response.status();
        let body_bytes = backend_response
            .bytes()
            .await
            .map_err(|_| StatusCode::BAD_GATEWAY)?;

        // Small model error: fall through to large model
        if !status.is_success() {
            info!("Small model returned HTTP {}, rerouting original request to large model", status);
            self.circuit_breaker.record_failure(&target_url).await;
            let result = self.proxy_to_backend(&payload, &large_url, false).await;
            if result.is_ok() {
                self.circuit_breaker.record_success(&large_url).await;
            }
            return result;
        }

        self.circuit_breaker.record_success(&target_url).await;

        // Extract confidence from logprobs
        let confidence = self.extract_confidence(&body_bytes);
        let keep_small = match confidence {
            Some(c) if c >= self.confidence_threshold => {
                info!("SMALL MODEL CONFIDENCE: {:.4} >= threshold {:.4}, keeping response", c, self.confidence_threshold);
                true
            }
            Some(c) => {
                info!("SMALL MODEL CONFIDENCE: {:.4} < threshold {:.4}, rerouting to large model", c, self.confidence_threshold);
                false
            }
            None => {
                info!("No logprobs in small model response (model may not support it), keeping response");
                true
            }
        };

        if keep_small {
            let mut headers = HeaderMap::new();
            headers.insert("content-type", HeaderValue::from_static("application/json"));
            if let Some(c) = confidence {
                let val = format!("{:.4}", c);
                if let Ok(hv) = HeaderValue::from_str(&val) {
                    headers.insert("x-confidence", hv);
                }
            }
            return Ok((headers, Body::from(body_bytes)));
        }

        // Reroute to large model with original payload (no German injection)
        info!("Rerouting original request to large text model");
        let result = self.proxy_to_backend(&payload, &large_url, false).await;
        if result.is_ok() {
            self.circuit_breaker.record_success(&large_url).await;
        }
        result
    }

    fn detect_image(&self, messages: &[ChatMessage]) -> bool {
        for msg in messages {
            match &msg.content {
                MessageContent::Text(_) => {}
                MessageContent::Parts(parts) => {
                    for part in parts {
                        if matches!(part, MessageContentPart::ImageUrl { .. }) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

/// Simple base64 encoder (no external dependency needed).
fn base64_encode(input: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity((input.len() + 2) / 3 * 4);
    let chunks = input.chunks(3);
    for chunk in chunks {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        output.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        output.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            output.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            output.push('=');
        }
        if chunk.len() > 2 {
            output.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            output.push('=');
        }
    }
    output
}

async fn handler(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    Json(payload): Json<ChatCompletionRequest>,
) -> Result<(HeaderMap, Body), StatusCode> {
    let is_streaming = payload.stream.unwrap_or(false);
    let tier = headers
        .get("x-tier")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("standard");
    state.route_request(payload, is_streaming, tier).await
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "info".to_string()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("cascade-llm v{}", env!("CARGO_PKG_VERSION"));

    // Auto-detect LARGE_MODEL_MULTIMODAL from inference server if not set in env
    let inference_url = std::env::var("INFERENCE_URL")
        .unwrap_or_else(|_| "http://netai-inference:8080".to_string());
    let large_model_multimodal = match std::env::var("LARGE_MODEL_MULTIMODAL") {
        Ok(v) => v.eq_ignore_ascii_case("true"),
        Err(_) => {
            info!("LARGE_MODEL_MULTIMODAL not set, auto-detecting from inference server...");
            match tokio::time::timeout(
                std::time::Duration::from_secs(5),
                fetch_large_model_multimodal_async(&inference_url)
            ).await {
                Ok(result) => result,
                Err(_) => {
                    warn!("Timeout fetching multimodal capability, defaulting to true");
                    true
                }
            }
        }
    };

    let state = GatewayState::new(large_model_multimodal);
    let app = Router::new()
        .route("/v1/chat/completions", post(handler))
        .layer(DefaultBodyLimit::disable())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    info!("Listening on {}", addr);
    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}