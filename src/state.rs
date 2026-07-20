use crate::config::AppConfig;
use crate::db::Db;
use crate::language;
use crate::types::*;

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use futures_util::stream::StreamExt;
use moka::future::Cache;
use regex::Regex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing::{info, warn};

pub async fn fetch_large_model_multimodal_async(inference_url: &str) -> bool {
    let models_url = format!("{}/models", inference_url.trim_end_matches('/'));
    match reqwest::Client::new()
        .get(&models_url)
        .send()
        .await
    {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(json) => {
                let empty_vec = Vec::new();
                let models = json
                    .get("models")
                    .and_then(|m| m.as_array())
                    .unwrap_or(&empty_vec);
                for model in models {
                    let empty_caps = Vec::new();
                    let caps = model
                        .get("capabilities")
                        .and_then(|c| c.as_array())
                        .unwrap_or(&empty_caps);
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

#[derive(Debug, Clone)]
struct CircuitBreaker {
    failures: Arc<tokio::sync::RwLock<HashMap<String, Vec<Instant>>>>,
    threshold: u32,
    reset_duration: Duration,
}

impl CircuitBreaker {
    fn new(threshold: u32, reset_duration_secs: u64) -> Self {
        Self {
            failures: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
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
}

#[derive(Debug)]
pub struct GatewayState {
    pub config: AppConfig,
    pub http_client: Arc<reqwest::Client>,
    circuit_breaker: CircuitBreaker,
    load_tracker: LoadTracker,
    pub session_cache: Cache<String, String>,
    image_semaphore: Arc<Semaphore>,
    pub metrics: Arc<crate::cascade_features::MetricsRegistry>,
    pub db: Arc<Db>,
    pub start_time: Instant,
}

impl GatewayState {
    pub fn new(
        config: AppConfig,
        metrics: Arc<crate::cascade_features::MetricsRegistry>,
        db: Arc<Db>,
    ) -> Self {
        let http_client = Arc::new(
            reqwest::Client::builder()
                .pool_max_idle_per_host(0)
                .pool_idle_timeout(Duration::from_secs(90))
                .build()
                .expect("Failed to build reqwest client"),
        );

        Self {
            circuit_breaker: CircuitBreaker::new(config.cb_threshold, config.cb_reset_secs),
            load_tracker: LoadTracker::default(),
            session_cache: Cache::builder()
                .max_capacity(10_000)
                .time_to_live(Duration::from_secs(3600))
                .build(),
            image_semaphore: Arc::new(Semaphore::new(config.max_concurrent_images)),
            config,
            http_client,
            metrics,
            db,
            start_time: Instant::now(),
        }
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
                        let lower = text.to_lowercase();
                        for keyword in &keywords {
                            if lower.contains(keyword) {
                                keyword_score += 0.2;
                            }
                        }
                        for indicator in &complex_indicators {
                            if lower.contains(indicator) {
                                keyword_score += 0.15;
                            }
                        }
                        let code_block_count = text.matches("```").count() / 2;
                        keyword_score += code_block_count as f64 * 0.25;
                        let list_patterns = ["\n1.", "\n2.", "\n3.", "\n1)", "\na)", "\na."];
                        let list_count = list_patterns.iter().map(|p| text.matches(p).count()).sum::<usize>();
                        keyword_score += list_count as f64 * 0.1;
                    }
                    MessageContent::Parts(parts) => {
                        for part in parts {
                            match part {
                                MessageContentPart::Text { text } => {
                                    total_chars += text.len();
                                    let lower = text.to_lowercase();
                                    for keyword in &keywords {
                                        if lower.contains(keyword) {
                                            keyword_score += 0.2;
                                        }
                                    }
                                    for indicator in &complex_indicators {
                                        if lower.contains(indicator) {
                                            keyword_score += 0.15;
                                        }
                                    }
                                    let code_block_count = text.matches("```").count() / 2;
                                    keyword_score += code_block_count as f64 * 0.25;
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
        let value: serde_json::Value = serde_json::from_slice(body).ok()?;
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
        if tier == "premium" {
            info!("PREMIUM TIER: routing to large model");
            return (false, &self.config.large_text_url);
        }

        if has_image {
            if self.config.large_model_multimodal && complexity > self.config.router_threshold {
                info!(
                    "MODEL SELECTION: image present, complexity {:.2} > threshold {}, routing to large multimodal model",
                    complexity, self.config.router_threshold
                );
                (false, &self.config.large_mllm_url)
            } else if self.config.large_model_multimodal {
                info!(
                    "MODEL SELECTION: image present but complexity {:.2} <= threshold {}, routing to small multimodal model",
                    complexity, self.config.router_threshold
                );
                (true, &self.config.small_mllm_url)
            } else {
                info!("MODEL SELECTION: image present but large model is text-only, routing to small multimodal model");
                (true, &self.config.small_mllm_url)
            }
        } else if complexity > self.config.router_threshold {
            info!(
                "MODEL SELECTION: text-only, complexity {:.2} > threshold {}, routing to large text model",
                complexity, self.config.router_threshold
            );
            (false, &self.config.large_text_url)
        } else {
            info!(
                "MODEL SELECTION: text-only, complexity {:.2} <= threshold {}, routing to small model",
                complexity, self.config.router_threshold
            );
            (true, &self.config.small_mllm_url)
        }
    }

    async fn proxy_to_backend(
        &self,
        payload: &ChatCompletionRequest,
        url: &str,
        is_streaming: bool,
    ) -> Result<(HeaderMap, Body), StatusCode> {
        let backend_response = self
            .http_client
            .post(url)
            .json(payload)
            .send()
            .await
            .map_err(|_| StatusCode::BAD_GATEWAY)?;

        let status = backend_response.status();
        if !status.is_success() {
            let err_body = backend_response.text().await.unwrap_or_default();
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
                if trimmed == "[DONE]" {
                    return;
                }
                if let Ok(mut event) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    let mut modified = false;
                    if let Some(delta) = event.get_mut("delta") {
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

            let stream = backend_response.bytes_stream().map(move |item| {
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

    async fn download_image_as_base64(&self, url: &str) -> Option<String> {
        if url.starts_with("data:") {
            return Some(url.to_string());
        }
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
        info!(
            "IMAGE_DESCRIPTION_LANGUAGE: {}, PROMPT_LENGTH: {}",
            language,
            prompt_text.len()
        );

        let desc_payload = ChatCompletionRequest {
            model: "vision".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Parts(vec![
                    MessageContentPart::Text {
                        text: prompt_text.to_string(),
                    },
                    MessageContentPart::ImageUrl {
                        image_url: ImageUrlTarget { url: data_url },
                    },
                ])),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
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

        let resp = self
            .http_client
            .post(&self.config.small_mllm_url)
            .json(&desc_payload)
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            info!("Image description failed: HTTP {}", resp.status());
            return None;
        }

        let body: serde_json::Value = resp.json().await.ok()?;
        info!(
            "Vision raw response: {}",
            body.to_string().chars().take(200).collect::<String>()
        );

        let msg = body.get("choices")?.as_array()?.first()?.get("message")?;

        let content = msg
            .get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let reasoning = msg
            .get("reasoning_content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let description = match (content, reasoning) {
            (Some(ref c), _) if !c.is_empty() => Some(c.clone()),
            (_, Some(ref r)) if !r.is_empty() => Some(r.clone()),
            _ => None,
        };

        description.map(|desc| {
            let re = Regex::new(r"\\?u[eE][0-9a-fA-F]{3}[^\s:]*[:\s]*").unwrap();
            re.replace_all(&desc, "").to_string()
        })
    }

    async fn inline_all_images(&self, payload: &mut ChatCompletionRequest) {
        for msg in payload.messages.iter_mut() {
            if let Some(ref mut content) = msg.content {
                if let MessageContent::Parts(parts) = content {
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

                    let download_futures = urls_to_download
                        .into_iter()
                        .map(|url| async move { self.download_image_as_base64(&url).await });
                    let downloaded_urls = futures_util::future::join_all(download_futures).await;

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
                                        text: format!(
                                            "[System - Image Description: {}]",
                                            descriptions[desc_idx]
                                        ),
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

    async fn is_backend_available(&self, url: &str, fallback: &str) -> String {
        if self.circuit_breaker.is_open(url).await {
            warn!("Circuit breaker OPEN for {}, using fallback {}", url, fallback);
            return fallback.to_string();
        }
        url.to_string()
    }

    pub async fn route_request_with_fallback(
        &self,
        payload: ChatCompletionRequest,
        is_streaming: bool,
        tier: &str,
        headers: &HeaderMap,
    ) -> Result<(HeaderMap, Body), StatusCode> {
        let has_image = self.detect_image(&payload.messages);
        let has_tools = payload.tools.is_some() || payload.functions.is_some();
        let complexity_score = self.evaluate_complexity(&payload.messages);
        let language = language::detect_language(&payload.messages);

        let history_has_tools = payload
            .messages
            .iter()
            .any(|m| m.role == "tool" || m.tool_calls.is_some());

        let session_key = headers
            .get("x-conversation-id")
            .or_else(|| headers.get("x-librechat-conversation-id"))
            .and_then(|v| v.to_str().ok().map(|s| s.to_string()));

        let has_reliable_session_key = session_key.is_some();

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

        info!(
            "[REQ_START] has_image={}, has_tools={}, history_has_tools={}, session_key={:?}, cached_route={:?}, tier={}, has_reliable_session_key={}",
            has_image, has_tools, history_has_tools, session_key, cached_route, tier, has_reliable_session_key
        );
        info!(
            "[REQ] user_text={:?}",
            language::extract_text(&payload.messages)
        );
        info!("DETECTED_LANGUAGE: {}", language);
        info!(
            "REQUEST_MESSAGES: count={}, aggregated_text_length={}, aggregated_text_preview={}",
            payload.messages.len(),
            aggregated_text.len(),
            aggregated_text.chars().take(200).collect::<String>()
        );
        for (i, msg) in payload.messages.iter().enumerate() {
            if let Some(ref content) = msg.content {
                match content {
                    MessageContent::Text(t) => info!(
                        "MSG_{}: role={}, type=Text, length={}, text={:?}",
                        i, msg.role, t.len(), t
                    ),
                    MessageContent::Parts(parts) => {
                        info!("MSG_{}: role={}, type=Parts, part_count={}", i, msg.role, parts.len());
                        for (j, part) in parts.iter().enumerate() {
                            match part {
                                MessageContentPart::Text { text } => info!(
                                    "  PART_{}_TEXT: length={}, text={:?}",
                                    j, text.len(), text
                                ),
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

        self.load_tracker.record(complexity_score);

        let _small_url = self
            .is_backend_available(&self.config.small_mllm_url, &self.config.large_text_url)
            .await;
        let large_url = self
            .is_backend_available(&self.config.large_text_url, &self.config.large_mllm_url)
            .await;

        let mut target_override = if let Some(url) = cached_route {
            info!("SESSION AFFINITY: candidate from cache target_url={}", url);
            Some(url)
        } else if history_has_tools {
            info!(
                "SESSION AFFINITY: History has tools but no cached route. Forcing large model target_url={}",
                large_url
            );
            if let Some(ref key) = session_key {
                self.session_cache
                    .insert(key.clone(), large_url.clone())
                    .await;
            }
            Some(large_url.clone())
        } else {
            None
        };

        if let Some(ref _url) = target_override {
            if has_image && !self.config.large_model_multimodal {
                warn!("SESSION AFFINITY INVALIDATED: request has images but large model is text-only");
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
                Ok(_) => {
                    self.circuit_breaker.record_success(&target).await;
                }
                Err(_) => {
                    self.circuit_breaker.record_failure(&target).await;
                }
            }
            self.metrics.record_request("session_affinity");
            return result;
        }

        if has_image && has_tools {
            info!("IMAGE + TOOLS: describing images with small vision model first");

            let image_urls = self.extract_image_urls(&injected_payload.messages);

            let description_futures = image_urls.iter().map(|url| async {
                match self.describe_image(url, language).await {
                    Some(desc) => {
                        info!(
                            "Image description: {}",
                            desc.chars().take(100).collect::<String>()
                        );
                        desc
                    }
                    None => "[Image could not be described]".to_string(),
                }
            });
            let descriptions = futures_util::future::join_all(description_futures).await;

            let mut modified_payload = injected_payload.clone();
            self.replace_images_with_text(&mut modified_payload, &descriptions);

            info!(
                "IMAGE + TOOLS: routing text+descriptions+tools to large text model ({})",
                large_url
            );
            let result = self
                .proxy_to_backend(&modified_payload, &large_url, is_streaming)
                .await;
            match &result {
                Ok(_) => {
                    self.circuit_breaker.record_success(&large_url).await;
                    if let Some(ref key) = session_key {
                        self.session_cache
                            .insert(key.clone(), large_url.clone())
                            .await;
                    }
                }
                Err(_) => {
                    self.circuit_breaker.record_failure(&large_url).await;
                }
            }
            self.metrics.record_request("large");
            return result;
        }

        if has_tools && self.config.route_tools_to_large {
            let target = &large_url;
            info!("TOOLS DETECTED + route_tools_to_large=true: routing to large text model");
            let result = self.proxy_to_backend(&injected_payload, target, is_streaming).await;
            match &result {
                Ok(_) => {
                    self.circuit_breaker.record_success(target).await;
                    if let Some(ref key) = session_key {
                        self.session_cache
                            .insert(key.clone(), target.clone())
                            .await;
                    }
                }
                Err(_) => {
                    self.circuit_breaker.record_failure(target).await;
                }
            }
            self.metrics.record_request("large");
            return result;
        }

        info!(
            "Routing decision: has_image={}, complexity_score={:.2}, threshold={}, large_multimodal={}",
            has_image,
            complexity_score,
            self.config.router_threshold,
            self.config.large_model_multimodal
        );

        let (use_small, target_url) = self.pick_model(has_image, complexity_score, tier);
        let target_url = target_url.to_owned();
        info!("SELECTED_URL: {}", target_url);

        if !use_small {
            let result = self
                .proxy_to_backend(&injected_payload, &target_url, is_streaming)
                .await;
            match &result {
                Ok(_) => {
                    self.circuit_breaker.record_success(&target_url).await;
                    if let Some(ref key) = session_key {
                        self.session_cache
                            .insert(key.clone(), target_url.clone())
                            .await;
                    }
                }
                Err(_) => {
                    self.circuit_breaker.record_failure(&target_url).await;
                }
            }
            self.metrics.record_request(
                if has_image && self.config.large_model_multimodal {
                    "large_multimodal"
                } else {
                    "large"
                },
            );
            return result;
        }

        info!("SMALL_MODEL_PATH: target_url={}, use_small=true", target_url);
        let mut small_payload = injected_payload.clone();
        info!(
            "SMALL_MODEL_PAYLOAD_SENT: messages={}",
            small_payload.messages.len()
        );

        if small_payload.max_tokens.is_none() {
            small_payload.max_tokens = Some(4096);
        }

        if is_streaming {
            let result = self
                .proxy_to_backend(&small_payload, &target_url, true)
                .await;
            match &result {
                Ok(_) => {
                    self.circuit_breaker.record_success(&target_url).await;
                    if let Some(ref key) = session_key {
                        self.session_cache
                            .insert(key.clone(), target_url.clone())
                            .await;
                    }
                }
                Err(_) => {
                    self.circuit_breaker.record_failure(&target_url).await;
                }
            }
            self.metrics.record_request("small");
            return result;
        }

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

        if !status.is_success() {
            info!(
                "Small model returned HTTP {}, rerouting original request to large model",
                status
            );
            self.circuit_breaker.record_failure(&target_url).await;
            self.metrics.record_fallback("primary_failed");
            let result = self
                .proxy_to_backend(&injected_payload, &large_url, false)
                .await;
            if result.is_ok() {
                self.circuit_breaker.record_success(&large_url).await;
            }
            self.metrics.record_request("large");
            return result;
        }

        self.circuit_breaker.record_success(&target_url).await;

        let confidence = self.extract_confidence(&body_bytes);
        let keep_small = match confidence {
            Some(c) if c >= self.config.confidence_threshold => {
                info!(
                    "SMALL MODEL CONFIDENCE: {:.4} >= threshold {:.4}, keeping response",
                    c, self.config.confidence_threshold
                );
                true
            }
            Some(c) => {
                info!(
                    "SMALL MODEL CONFIDENCE: {:.4} < threshold {:.4}, rerouting to large model",
                    c, self.config.confidence_threshold
                );
                false
            }
            None => {
                info!("No logprobs in small model response, keeping response");
                true
            }
        };

        if keep_small {
            if let Some(ref key) = session_key {
                self.session_cache
                    .insert(key.clone(), target_url.clone())
                    .await;
            }
            let mut headers = HeaderMap::new();
            headers.insert("content-type", HeaderValue::from_static("application/json"));
            if let Some(c) = confidence {
                let val = format!("{:.4}", c);
                if let Ok(hv) = HeaderValue::from_str(&val) {
                    headers.insert("x-confidence", hv);
                }
            }
            self.metrics.record_request("small");
            return Ok((headers, Body::from(body_bytes)));
        }

        info!("Rerouting original request to large text model");
        self.metrics.record_fallback("quality_low");
        let result = self
            .proxy_to_backend(&injected_payload, &large_url, false)
            .await;
        if result.is_ok() {
            self.circuit_breaker.record_success(&large_url).await;
            if let Some(ref key) = session_key {
                self.session_cache
                    .insert(key.clone(), large_url.clone())
                    .await;
            }
        }
        self.metrics.record_request("large");
        result
    }

    fn detect_image(&self, messages: &[ChatMessage]) -> bool {
        for msg in messages {
            if let Some(ref content) = msg.content {
                if let MessageContent::Parts(parts) = content {
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
