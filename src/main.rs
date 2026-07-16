use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Json, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::Response,
    routing::{get, post},
    Router,
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use futures_util::stream::StreamExt;
use moka::future::Cache;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::net::TcpListener;
use tokio::sync::{RwLock, Semaphore};
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
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

    #[allow(dead_code)]
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
    main_model_name: String,
    small_model_name: String,
    router_threshold: f64,
    confidence_threshold: f64,
    large_model_multimodal: bool,
    route_tools_to_large: bool,
    circuit_breaker: CircuitBreaker,
    load_tracker: LoadTracker,
    session_cache: Cache<String, String>,
    image_semaphore: Arc<Semaphore>,
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

        let main_model_name = std::env::var("MAIN_MODEL_NAME")
            .unwrap_or_else(|_| "gpt-3.5-turbo".to_string());
        let small_model_name = std::env::var("SMALL_MODEL_NAME")
            .unwrap_or_else(|_| "gpt-4o-mini".to_string());

        let max_concurrent_images = std::env::var("MAX_CONCURRENT_IMAGES")
            .unwrap_or_else(|_| "4".to_string())
            .parse::<usize>()
            .unwrap_or(4);

        Self {
            http_client,
            small_mllm_url,
            large_mllm_url,
            large_text_url,
            main_model_name,
            small_model_name,
            router_threshold,
            confidence_threshold,
            large_model_multimodal,
            route_tools_to_large,
            circuit_breaker: CircuitBreaker::new(cb_threshold, cb_reset),
            load_tracker: LoadTracker::default(),
            session_cache: Cache::builder()
                .max_capacity(10_000)
                .time_to_live(Duration::from_secs(3600))
                .build(),
            image_semaphore: Arc::new(Semaphore::new(max_concurrent_images)),
        }
    }

    fn detect_language(&self, messages: &[ChatMessage]) -> &'static str {
        language::detect_language(messages)
    }

    fn evaluate_complexity(&self, messages: &[ChatMessage]) -> f64 {
        let detected_lang = language::detect_language(messages);
        let mut total_chars = 0;
        let mut keyword_score = 0.0;

        let keywords: Vec<&str> = match detected_lang {
            "de" => vec!["analysiere", "schreibe", "experte", "logik", "komplex"],
            "fr" => vec!["analyser", "écrire", "expert", "logique", "complexe"],
            "es" => vec!["analizar", "escribir", "experto", "lógica", "complejo"],
            "it" => vec!["analizza", "scrivi", "esperto", "logica", "complesso"],
            "pl" => vec!["analizuj", "napisz", "ekspert", "logika", "skomplikowany"],
            "hu" => vec!["elemezd", "írj", "szakértő", "logika", "komplex"],
            _ => vec!["analyze deeply", "write code", "expert", "reasoning", "logic", "complex"],
        };

        let complex_indicators: Vec<&str> = match detected_lang {
            "de" => vec![
                "analysiere tief", "schreibe code", "experte", "logik", "komplex",
                "schritt 1", "schritt 2", "erstens", "zweitens", "dritten",
                "architektur", "infrastruktur", "debuggen", "optimieren", "refaktorisieren",
                "theorem", "beweis", "berechnen", "gleichung", "ableitung",
            ],
            "fr" => vec![
                "analyser en détail", "écrire du code", "expert", "logique", "complexe",
                "étape 1", "étape 2", "premièrement", "deuxièmement", "troisièmement",
                "architecture", "infrastructure", "déboguer", "optimiser", "refactoriser",
                "théorème", "preuve", "calculer", "équation", "dérivée",
            ],
            "es" => vec![
                "analizar profundamente", "escribir código", "experto", "lógica", "complejo",
                "paso 1", "paso 2", "primero", "segundo", "tercero",
                "arquitectura", "infraestructura", "depurar", "optimizar", "refactorizar",
                "teorema", "prueba", "calcular", "ecuación", "derivada",
            ],
            "it" => vec![
                "analizza approfonditamente", "scrivi codice", "esperto", "logica", "complesso",
                "passo 1", "passo 2", "prima", "seconda", "terza",
                "architettura", "infrastruttura", "debuggare", "ottimizzare", "refattorizzare",
                "teorema", "prova", "calcolare", "equazione", "derivata",
            ],
            "pl" => vec![
                "analizuj głęboko", "napisz kod", "ekspert", "logika", "skomplikowany",
                "krok 1", "krok 2", "pierwszy", "drugi", "trzeci",
                "architektura", "infrastruktura", "debugować", "zoptymalizować", "refaktoryzować",
                "twierdzenie", "dowód", "obliczyć", "równanie", "pochodna",
            ],
            "hu" => vec![
                "elemezd mélyen", "írj kódot", "szakértő", "logika", "komplex",
                "1. lépés", "2. lépés", "elsőként", "másodikként", "harmadik",
                "architektúra", "infrastruktúra", "hibakeresés", "optimalizálás", "refaktorálás",
                "tétel", "bizonyíték", "számít", "egyenlet", "derivál",
            ],
            _ => vec![
                "analyze deeply", "write code", "expert", "reasoning", "logic", "complex",
                "step 1", "step 2", "first,", "second,", "third,", "four", "five",
                "architecture", "infrastructure", "debug", "optimize", "refactor",
                "theorem", "proof", "calculate", "compute", "equation", "derivative",
            ],
};
        for msg in messages {
            if let Some(ref content) = msg.content {
                match content {
                    MessageContent::Text(text) => {
                        total_chars += text.len();
                        for keyword in &keywords {
                            if text.to_lowercase().contains(keyword) {
                                keyword_score += 0.2;
                            }
                        }
                        for indicator in &complex_indicators {
                            if text.to_lowercase().contains(indicator) {
                                keyword_score += 0.15;
                            }
                        }
                        // Count code blocks (```)
                        let code_block_count = text.matches("```").count() / 2;
                        keyword_score += code_block_count as f64 * 0.25;
                        // Count numbered lists (multilingual)
                        let list_patterns = ["\n1.", "\n2.", "\n3.", "\n1)", "\na)", "\na."];
                        let list_count = list_patterns.iter().map(|p| text.matches(p).count()).sum::<usize>();
                        keyword_score += list_count as f64 * 0.1;
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
                                    for indicator in &complex_indicators {
                                        if text.to_lowercase().contains(indicator) {
                                            keyword_score += 0.15;
                                        }
                                    }
                                    // Count code blocks (```) in parts
                                    let code_block_count = text.matches("```").count() / 2;
                                    keyword_score += code_block_count as f64 * 0.25;
                                    // Count numbered lists in parts
                                    let list_patterns = ["\n1.", "\n2.", "\n3.", "\n1)", "\na)", "\na."];
                                    let list_count = list_patterns.iter().map(|p| text.matches(p).count()).sum::<usize>();
                                    keyword_score += list_count as f64 * 0.1;
                                }
                                MessageContentPart::ImageUrl { .. } => {
                                    total_chars += 100;
                                }
                            }
                        }
                    }
                }
            }
        }

        let char_score = (total_chars as f64 / 1000.0).min(1.0);
        let mut score = 0.5 * char_score + 0.5 * keyword_score.min(1.0);
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

        let status = backend_response.status();
        if !status.is_success() {
            let err_body = backend_response
                .text()
                .await
                .unwrap_or_default();
            warn!("Backend error HTTP {} from {}: {}", status, url, err_body);
            let error_code = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            return Err(error_code);
        }

        let mut headers = HeaderMap::new();
        if is_streaming {
            headers.insert("content-type", HeaderValue::from_static("text/event-stream"));
            headers.insert("cache-control", HeaderValue::from_static("no-cache"));
            headers.insert("connection", HeaderValue::from_static("keep-alive"));
        } else {
            headers.insert("content-type", HeaderValue::from_static("application/json"));
        }

        if is_streaming {
            let on_chunk = move |bytes: &mut Vec<u8>| {
                let s = match std::str::from_utf8(bytes) {
                    Ok(s) => s,
                    Err(_) => return,
                };
                if !s.starts_with("data: ") && !s.starts_with("data:{") {
                    return;
                }

                let trimmed = s.trim_start_matches("data: ").trim();
                if trimmed == "[DONE]" { return; }

                if let Ok(mut event) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    let mut modified = false;
                    if let Some(delta) = event.get_mut("delta") {
                        // Lib. The order must be think chunks first, then text chunks.
                        if let Some(reasoning) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                            delta["content"] = serde_json::json!([{"type":"think","think":reasoning}]);
                            delta.as_object_mut().unwrap().remove("reasoning_content");
                            modified = true;
                        } else if let Some(reasoning) = delta.get("reasoning").and_then(|v| v.as_str()) {
                            delta["content"] = serde_json::json!([{"type":"think","think":reasoning}]);
                            delta.as_object_mut().unwrap().remove("reasoning");
                            modified = true;
                        } else if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
                            if !text.is_empty() {
                                delta["content"] = serde_json::json!([{"type":"text","text":text}]);
                                modified = true;
                            }
                        }
                    }
                    if modified {
                        if let Ok(new_s) = serde_json::to_string(&event) {
                            *bytes = if s.starts_with("data: ") {
                                format!("data: {}\n\n", new_s).into_bytes()
                            } else {
                                format!("{}\n\n", new_s).into_bytes()
                            };
                        }
                    }
                }
            };

            let stream = backend_response
                .bytes_stream()
                .map(move |item| {
                    let mut chunk = match item {
                        Ok(c) => Ok::<_, std::io::Error>(c),
                        Err(e) => Err(std::io::Error::new(std::io::ErrorKind::Other, e)),
                    };
                    if let Ok(ref mut c) = chunk {
                        let mut buf = c.to_vec();
                        on_chunk(&mut buf);
                        *c = axum::body::Bytes::from(buf);
                    }
                    chunk
                });
            let body = Body::from_stream(stream);
            return Ok((headers, body));
        }

        let body_bytes = backend_response
            .bytes()
            .await
            .map_err(|_| StatusCode::BAD_GATEWAY)?;

        // Transform chunk to LibreChat-compatible delta format
        if let Ok(mut event) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
            if let Some(choices) = event.get_mut("choices").and_then(|c| c.as_array_mut()) {
                if let Some(choice) = choices.get_mut(0) {
                    if let Some(delta) = choice.get_mut("delta") {
                        if let Some(reasoning) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                            delta["content"] = serde_json::json!([{"type":"think","think":reasoning}]);
                            delta.as_object_mut().unwrap().remove("reasoning_content");
                        } else if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
                            if !text.is_empty() {
                                delta["content"] = serde_json::json!([{"type":"text","text":text}]);
                            }
                        }
                    }
                }
            }
            let new_body = serde_json::to_vec(&event).unwrap_or(body_bytes.to_vec());
            return Ok((headers, Body::from(new_body)));
        }

        Ok((headers, Body::from(body_bytes)))
    }

    /// Download image from URL and convert to base64 data URL.
    /// Uses semaphore to limit concurrent downloads.
    async fn download_image_as_base64(&self, url: &str) -> Option<String> {
        if url.starts_with("data:") {
            return Some(url.to_string());
        }
        // Acquire semaphore permit to limit concurrent downloads
        let _permit = self.image_semaphore.acquire().await.ok()?;
        
        let resp = self.http_client.get(url).send().await.ok()?;
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("image/png")
            .to_string();
        let bytes = resp.bytes().await.ok()?;
        let encoded = BASE64_STANDARD.encode(&bytes);
        Some(format!("data:{};base64,{}", content_type, encoded))
    }

    /// Describe an image using the small vision model.
    /// Downloads the image in the gateway and sends as base64 data URL
    /// to avoid llama.cpp's external URL download issues.
    async fn describe_image(&self, image_url: &str, language: &str) -> Option<String> {
        let url_preview = if image_url.starts_with("data:") {
            format!("data:...<{} bytes>", image_url.len())
        } else {
            image_url.to_string()
        };
        info!("Downloading image for description: {}", url_preview);
        
        let data_url = self.download_image_as_base64(image_url).await?;
        info!("Image downloaded, size: {} bytes", data_url.len());

        let prompt_text = language::get_image_prompt(language);
        info!("IMAGE_DESCRIPTION_LANGUAGE: {}, PROMPT_LENGTH: {}", language, prompt_text.len());

        let desc_payload = ChatCompletionRequest {
            model: "vision".to_string(),
            messages: vec![
                ChatMessage {
                    role: "user".to_string(),
                    content: Some(MessageContent::Parts(vec![
                        MessageContentPart::Text { text: prompt_text.to_string() },
                        MessageContentPart::ImageUrl { image_url: ImageUrlTarget { url: data_url } },
                    ])),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
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
            user: None,
            stop: None,
            response_format: None,
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
        info!("Vision raw response: {}", body.to_string().chars().take(200).collect::<String>());

        let msg = body.get("choices")?
            .as_array()?
            .first()?
            .get("message")?;
        
        let content = msg.get("content").and_then(|v| v.as_str()).map(|s| s.to_string());
        let reasoning = msg.get("reasoning_content").and_then(|v| v.as_str()).map(|s| s.to_string());

        let description = match (content, reasoning) {
            (Some(ref c), _) if !c.is_empty() => Some(c.clone()),
            (_, Some(ref r)) if !r.is_empty() => Some(r.clone()),
            _ => None,
        };

        // Clean out special Unicode tags (e.g. \ue202turn0description1) produced by certain vision models
        description.map(|desc| {
            let re = Regex::new(r"\\?u[eE][0-9a-fA-F]{3}[^\s:]*[:\s]*").unwrap();
            re.replace_all(&desc, "").to_string()
        })
    }

    /// Converts all image URLs in the payload to base64 data URIs
    /// to ensure llama.cpp can read them. Uses concurrent downloads with semaphore.
    async fn inline_all_images(&self, payload: &mut ChatCompletionRequest) {
        for msg in payload.messages.iter_mut() {
            if let Some(ref mut content) = msg.content {
                if let MessageContent::Parts(parts) = content {
                    // Collect image URLs that need to be downloaded
                    let urls_to_download: Vec<_> = parts
                        .iter()
                        .filter_map(|p| {
                            if let MessageContentPart::ImageUrl { image_url } = p {
                                if !image_url.url.starts_with("data:") {
                                    Some(image_url.url.clone())
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .collect();

                    // Download all images concurrently with semaphore-limited concurrency
                    let download_futures = urls_to_download.into_iter().map(|url| async move {
                        self.download_image_as_base64(&url).await
                    });
                    let downloaded_urls = futures_util::future::join_all(download_futures).await;

                    // Replace image URLs with downloaded data URLs
                    let mut url_idx = 0;
                    for part in parts.iter_mut() {
                        if let MessageContentPart::ImageUrl { image_url } = part {
                            if !image_url.url.starts_with("data:") && url_idx < downloaded_urls.len() {
                                if let Some(base64_data) = &downloaded_urls[url_idx] {
                                    image_url.url = base64_data.clone();
                                }
                                url_idx += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Replace image_url parts with text descriptions in the payload.
    fn replace_images_with_text(&self, payload: &mut ChatCompletionRequest, descriptions: &[String]) {
        let mut desc_idx = 0;
        for msg in payload.messages.iter_mut() {
            if let Some(ref mut content) = msg.content {
                if let MessageContent::Parts(parts) = content {
                    let mut new_parts = Vec::new();
                    for part in parts.drain(..) {
                        match part {
                            MessageContentPart::ImageUrl { .. } => {
                                if desc_idx < descriptions.len() {
                                    new_parts.push(MessageContentPart::Text {
                                        text: format!("[System - Image Description: {}]", descriptions[desc_idx]),
                                    });
                                    desc_idx += 1;
                                }
                            }
                            _ => new_parts.push(part),
                        }
                    }
                    if new_parts.len() == 1 {
                        if let MessageContentPart::Text { text } = new_parts.remove(0) {
                            msg.content = Some(MessageContent::Text(text));
                        }
                    } else {
                        msg.content = Some(MessageContent::Parts(new_parts));
                    }
                }
            }
        }
    }

    /// Extract all image URLs from messages.
    fn extract_image_urls(&self, messages: &[ChatMessage]) -> Vec<String> {
        let mut urls = Vec::new();
        for msg in messages {
            if let Some(ref content) = msg.content {
                if let MessageContent::Parts(parts) = content {
                    for part in parts {
                        if let MessageContentPart::ImageUrl { image_url } = part {
                            urls.push(image_url.url.clone());
                        }
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

    async fn route_request(&self, payload: ChatCompletionRequest, is_streaming: bool, tier: &str, headers: &HeaderMap) -> Result<(HeaderMap, Body), StatusCode> {
        let has_image = self.detect_image(&payload.messages);
        let has_tools = payload.tools.is_some() || payload.functions.is_some();
        let complexity_score = self.evaluate_complexity(&payload.messages);
        let language = self.detect_language(&payload.messages);

        // Check if history contains any tool role or tool_calls
        let history_has_tools = payload.messages.iter().any(|m| {
            m.role == "tool" || m.tool_calls.is_some()
        });

        // Determine session key
        let session_key = headers
            .get("x-conversation-id")
            .or_else(|| headers.get("x-librechat-conversation-id"))
            .and_then(|v| v.to_str().ok().map(|s| s.to_string()));

        let has_reliable_session_key = session_key.is_some();

        // Fallback key uses user ID + content hash to distinguish concurrent chats from same user
        let aggregated_text = language::extract_text(&payload.messages);
        let session_key = session_key.or_else(|| {
            payload.user.as_ref().map(|u| {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                aggregated_text.hash(&mut hasher);
                format!("user:{}:conv_{}", u, hasher.finish())
            })
        });

        let cached_route = if let Some(ref key) = session_key {
            self.session_cache.get(key).await
        } else {
            None
        };

         info!("[REQ_START] has_image={}, has_tools={}, history_has_tools={}, session_key={:?}, cached_route={:?}, tier={}, has_reliable_session_key={}",
             has_image, has_tools, history_has_tools, session_key, cached_route, tier, has_reliable_session_key);
         info!("[REQ] user_text={:?}", language::extract_text(&payload.messages));

         info!("DETECTED_LANGUAGE: {}", language);

         info!("REQUEST_MESSAGES: count={}, aggregated_text_length={}, aggregated_text_preview={}",
            payload.messages.len(), aggregated_text.len(), aggregated_text.chars().take(200).collect::<String>());
        for (i, msg) in payload.messages.iter().enumerate() {
            if let Some(ref content) = msg.content {
                match content {
                    MessageContent::Text(t) => info!("MSG_{}: role={}, type=Text, length={}, text={:?}", i, msg.role, t.len(), t),
                    MessageContent::Parts(parts) => {
                        info!("MSG_{}: role={}, type=Parts, part_count={}", i, msg.role, parts.len());
                        for (j, part) in parts.iter().enumerate() {
                            match part {
                                MessageContentPart::Text { text } => info!("  PART_{}_TEXT: length={}, text={:?}", j, text.len(), text),
                                MessageContentPart::ImageUrl { .. } => info!("  PART_{}: ImageUrl", j),
                            }
                        }
                    }
                }
            } else {
                info!("MSG_{}: role={}, type=None", i, msg.role);
            }
        }

        let mut injected_payload = language::inject_language_prompt(language, payload);
        if has_image {
            self.inline_all_images(&mut injected_payload).await;
        }

        let _on_chunk = |chunk_bytes: &mut Vec<u8>, _streaming: bool| {
            if let Ok(s) = std::str::from_utf8(chunk_bytes) {
                let finish_reason_pos = s.find("\"finish_reason\":");
                let has_reasoning = s.find("\"reasoning_content\":\"") != None;
                let has_delta_content = !has_reasoning && s.find("\"delta\":{") != None;

                if !has_reasoning && !has_delta_content {
                    return;
                }

                if let Some(p) = finish_reason_pos {
                    let after_key = &s[p + "\"finish_reason\":".len()..];
                    let val_start = after_key.strip_prefix(' ')
                        .or_else(|| after_key.strip_prefix('\"'))
                        .map(|s| s.as_ptr());
                    if let Some(start) = val_start {
                        let mut end = start;
                        unsafe { while *end != b',' && *end != b'}' && *end != b'\n' { end = end.add(1); } }
                        let val = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(start, end.offset_from(start) as usize)) };
                        let trimmed = val.strip_prefix('\"').unwrap_or(val).strip_suffix('\"').unwrap_or(val);
                        let fr_val = format!("\"finish_reason\":\"{trimmed}\"");
                        let old_fr = &s[p..p + "\"finish_reason\":".len() + val.len() + 2];
                        *chunk_bytes = s.replace(old_fr, &fr_val).into_bytes();
                        return;
                    }
                }

                if let Some(pos) = s.find("\"reasoning_content\":\"") {
                    let rest = &s[pos + "\"reasoning_content\":\"".len()..];
                    let mut end = 0;
                    let mut bs_count = 0;
                    while end < rest.len() {
                        let c = rest.as_bytes()[end];
                        if c == b'\\' { bs_count += 1; }
                        else if c == b'"' && bs_count % 2 == 0 { break; }
                        else { bs_count = 0; }
                        end += 1;
                    }
                    let _raw = &rest[..end];
                    let val = rest[..end].replace('"', "\\\"");
                    let replacement = format!("[{{\"type\":\"think\",\"think\":\"{val}\"}}]");
                    let old_slice = &s[pos..pos + "\"reasoning_content\":\"".len() + end];
                    *chunk_bytes = s.replacen(old_slice, &replacement, 1).into_bytes();
                    return;
                }

                if has_delta_content {
                    if let Some(pos) = s.find("\"content\":\"") {
                        let rest = &s[pos + "\"content\":\"".len()..];
                        let mut end = 0;
                        let mut bs_count = 0;
                        while end < rest.len() {
                            let c = rest.as_bytes()[end];
                            if c == b'\\' { bs_count += 1; }
                            else if c == b'"' && bs_count % 2 == 0 { break; }
                            else { bs_count = 0; }
                            end += 1;
                        }
                        let _raw = &rest[..end];
                        let val = rest[..end].replace('"', "\\\"");
                        let replacement = format!("[{{\"type\":\"text\",\"text\":\"{val}\"}}]");
                        let old_slice = &s[pos..pos + "\"content\":\"".len() + end];
                        *chunk_bytes = s.replacen(old_slice, &replacement, 1).into_bytes();
                    }
                }
            }
        };

        // Record load metrics
        self.load_tracker.record(complexity_score);

        // Check circuit breaker for backends
        let _small_url = self.is_backend_available(&self.small_mllm_url, &self.large_text_url).await;
        let large_url = self.is_backend_available(&self.large_text_url, &self.large_mllm_url).await;

        // Sticky Session Affinity Logic
        // Determine routing target based on cache or tool history
        let mut target_override = if let Some(url) = cached_route {
            info!("SESSION AFFINITY: candidate from cache target_url={}", url);
            Some(url)
        } else if history_has_tools {
            info!("SESSION AFFINITY: History has tools but no cached route. Forcing large model target_url={}", large_url);
            if let Some(ref key) = session_key {
                self.session_cache.insert(key.clone(), large_url.clone()).await;
            }
            Some(large_url.clone())
        } else {
            None
        };

        if let Some(ref _url) = target_override {
            if has_image && !self.large_model_multimodal {
                warn!("SESSION AFFINITY INVALIDATED: request has images but large model is text-only; falling back to image routing via small vision model");
                target_override = None;
                if let Some(ref key) = session_key {
                    self.session_cache.remove(key).await;
                }
            }
        }

        if let Some(target) = target_override {
            info!("SESSION AFFINITY ROUTE: Proxying directly to target={}", target);
            let result = self.proxy_to_backend(&injected_payload, &target, is_streaming).await;
            match &result {
                Ok(_) => { self.circuit_breaker.record_success(&target).await; }
                Err(_) => { self.circuit_breaker.record_failure(&target).await; }
            }
            return result;
        }

        // If both image AND tools: describe image with small vision model,
        // then route text description + tools to large text model
        if has_image && has_tools {
            info!("IMAGE + TOOLS: describing images with small vision model first");

            let image_urls = self.extract_image_urls(&injected_payload.messages);

            // Describe images concurrently with semaphore-limited downloads
            let description_futures = image_urls.iter().map(|url| async {
                match self.describe_image(url, language).await {
                    Some(desc) => {
                        info!("Image description: {}", desc.chars().take(100).collect::<String>());
                        desc
                    }
                    None => "[Image could not be described]".to_string()
                }
            });
            let descriptions = futures_util::future::join_all(description_futures).await;

            // Replace images with descriptions and route to large text model as requested
            let mut modified_payload = injected_payload.clone();
            self.replace_images_with_text(&mut modified_payload, &descriptions);

            info!("IMAGE + TOOLS: routing text+descriptions+tools to large text model ({})", large_url);
            let result = self.proxy_to_backend(&modified_payload, &large_url, is_streaming).await;
            match &result {
                Ok(_) => {
                    self.circuit_breaker.record_success(&large_url).await;
                    if let Some(ref key) = session_key {
                        self.session_cache.insert(key.clone(), large_url.clone()).await;
                    }
                }
                Err(_) => { self.circuit_breaker.record_failure(&large_url).await; }
            }
            return result;
        }

// If tools are present and route_tools_to_large is enabled, route to large model
        if has_tools && self.route_tools_to_large {
            let target = &large_url;
            info!("TOOLS DETECTED + route_tools_to_large=true: routing to large text model");
            let result = self.proxy_to_backend(&injected_payload, target, is_streaming).await;
            match &result {
                Ok(_) => {
                    self.circuit_breaker.record_success(target).await;
                    if let Some(ref key) = session_key {
                        self.session_cache.insert(key.clone(), target.clone()).await;
                    }
                }
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
            let result = self.proxy_to_backend(&injected_payload, &target_url, is_streaming).await;
            match &result {
                Ok(_) => { 
                    self.circuit_breaker.record_success(&target_url).await;
                    if let Some(ref key) = session_key {
                        self.session_cache.insert(key.clone(), target_url.clone()).await;
                    }
                }
                Err(_) => { self.circuit_breaker.record_failure(&target_url).await; }
            }
            return result;
        }

        // === Small model path ===
        info!("SMALL_MODEL_PATH: target_url={}, use_small=true", target_url);
        let mut small_payload = injected_payload.clone();
        info!("SMALL_MODEL_PAYLOAD_SENT: messages={}", small_payload.messages.len());

        if small_payload.max_tokens.is_none() {
            small_payload.max_tokens = Some(4096);
        }

        if is_streaming {
            let result = self.proxy_to_backend(&small_payload, &target_url, true).await;
            match &result {
                Ok(_) => { 
                    self.circuit_breaker.record_success(&target_url).await;
                    if let Some(ref key) = session_key {
                        self.session_cache.insert(key.clone(), target_url.clone()).await;
                    }
                }
                Err(_) => { self.circuit_breaker.record_failure(&target_url).await; }
            }
            return result;
        }

        // Non-streaming: try small model with logprobs for confidence-based rerouting
        let mut small_payload = small_payload;
        small_payload.logprobs = Some(true);
        small_payload.top_logprobs = Some(0);

        if let Some(max_tokens) = injected_payload.max_tokens {
            small_payload.max_tokens = Some(max_tokens);
        }

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
            let result = self.proxy_to_backend(&injected_payload, &large_url, false).await;
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
            // Since we routed to and decided to keep the small model, let's cache it for session affinity
            if let Some(ref key) = session_key {
                self.session_cache.insert(key.clone(), target_url.clone()).await;
            }
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

        // Reroute to large model with language-injected payload
        info!("Rerouting original request to large text model");
        let result = self.proxy_to_backend(&injected_payload, &large_url, false).await;
        if result.is_ok() {
            self.circuit_breaker.record_success(&large_url).await;
            if let Some(ref key) = session_key {
                self.session_cache.insert(key.clone(), large_url.clone()).await;
            }
        }
        result
    }

    fn detect_image(&self, messages: &[ChatMessage]) -> bool {
        for msg in messages {
            if let Some(ref content) = msg.content {
                match content {
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
        }
        false
    }
}

// ── OpenAI-compatible model response types ──

#[derive(Debug, Serialize)]
struct ModelPermission {
    id: String,
    object: String,
    created: u64,
    allow_create_engine: bool,
    allow_sampling: bool,
    allow_logprobs: bool,
    allow_search_indices: bool,
    allow_view: bool,
    allow_fine_tuning: bool,
    organization: String,
    group: Option<String>,
    is_blocking: bool,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
    permission: Vec<ModelPermission>,
    root: String,
    parent: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

fn build_model_info(id: &str) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        object: "model".to_string(),
        created: 1740000000,
        owned_by: "netai-stack".to_string(),
        permission: vec![ModelPermission {
            id: format!("modelperm-{}", id),
            object: "model_permission".to_string(),
            created: 1740000000,
            allow_create_engine: false,
            allow_sampling: true,
            allow_logprobs: true,
            allow_search_indices: false,
            allow_view: true,
            allow_fine_tuning: false,
            organization: "*".to_string(),
            group: None,
            is_blocking: false,
        }],
        root: id.to_string(),
        parent: None,
    }
}

async fn fetch_models_handler(State(state): State<GatewayState>) -> Response {
    let models = ModelList {
        object: "list".to_string(),
        data: vec![
            build_model_info(&state.main_model_name),
            build_model_info(&state.small_model_name),
        ],
    };
    let body = Body::from(serde_json::to_string(&models).unwrap_or_default());
    let mut response = Response::new(body);
    response.headers_mut().insert("content-type", HeaderValue::from_static("application/json"));
    response
}

async fn model_handler(State(state): State<GatewayState>) -> Response {
    let model = build_model_info(&state.main_model_name);
    let body = Body::from(serde_json::to_string(&model).unwrap_or_default());
    let mut response = Response::new(body);
    response.headers_mut().insert("content-type", HeaderValue::from_static("application/json"));
    response
}

async fn health_handler(State(state): State<GatewayState>) -> Response {
    let body = Body::from(
        serde_json::json!({
            "status": "ok",
            "large_model_multimodal": state.large_model_multimodal,
            "router_threshold": state.router_threshold,
            "confidence_threshold": state.confidence_threshold,
            "session_cache_entries": state.session_cache.entry_count() as u64
        }).to_string()
    );
    let mut response = Response::new(body);
    response.headers_mut().insert("content-type", HeaderValue::from_static("application/json"));
    response
}

async fn metrics_handler(State(state): State<GatewayState>) -> Response {
    let body = Body::from(
        serde_json::json!({
            "request_count": state.load_tracker.request_count.load(Ordering::Relaxed),
            "avg_complexity": state.load_tracker.avg_complexity(),
            "session_cache_entries": state.session_cache.entry_count() as u64
        }).to_string()
    );
    let mut response = Response::new(body);
    response.headers_mut().insert("content-type", HeaderValue::from_static("application/json"));
    response
}

async fn handler(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    let is_streaming = payload.stream.unwrap_or(false);
    let tier = headers
        .get("x-tier")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("standard");

    match state.route_request(payload, is_streaming, tier, &headers).await {
        Ok((hdrs, body)) => {
            let mut response = Response::new(body);
            *response.headers_mut() = hdrs;
            response
        }
        Err(status) => {
            let error_json = Body::from(
                serde_json::json!({
                    "error": {
                        "message": format!("HTTP {}", status.as_u16()),
                        "type": "cascade_proxy_error",
                        "param": Value::Null,
                        "code": Value::Number(status.as_u16().into())
                    }
                })
                .to_string(),
            );
            let mut response = Response::new(error_json);
            response.headers_mut().insert("content-type", HeaderValue::from_static("application/json"));
            *response.status_mut() = status;
            response
        }
    }
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
        .route("/v1/models", get(fetch_models_handler))
        .route("/model", get(model_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::disable())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    info!("Listening on {}", addr);
    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}