use crate::config::AppConfig;
use crate::db::Db;
use crate::language;
use crate::types::*;

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use futures_util::stream::StreamExt;
use moka::future::Cache;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing::{info, warn};

pub async fn fetch_large_model_multimodal_async(capability_url: &str) -> bool {
    let url = capability_url.trim_end_matches('/');
    match reqwest::Client::new().get(url).send().await {
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if status.is_success() && body.trim() == "multimodal" {
                info!("Auto-detected: large model supports multimodal via capability signal");
                true
            } else {
                if status.is_success() {
                    info!("Auto-detected: large model is text-only (capability endpoint returned: {})", body.trim());
                } else {
                    warn!("Multimodal capability endpoint returned HTTP {}: {}", status, body.trim());
                }
                false
            }
        }
        Err(e) => {
            warn!("Failed to fetch multimodal capability endpoint (main inference server not ready?): {}", e);
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
    /// Registry of extraction backends for `/v1/extraction`.
    pub extraction_backends: Arc<tokio::sync::RwLock<Vec<ExtractionBackendEntry>>>,
    /// Runtime upstream overrides keyed by role (inference|auxiliary|ocr).
    pub upstream_overrides:
        Arc<tokio::sync::RwLock<std::collections::HashMap<String, crate::types::UpstreamOverride>>>,
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
                .connect_timeout(Duration::from_secs(5))
                .timeout(Duration::from_secs(600))
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
            extraction_backends: Arc::new(tokio::sync::RwLock::new(
                config.build_extraction_backends(),
            )),
            upstream_overrides: Arc::new(tokio::sync::RwLock::new(
                std::collections::HashMap::new(),
            )),
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
                                MessageContentPart::InputAudio { .. } => {
                                    total_chars += 500;
                                }
                                MessageContentPart::File { .. } => {
                                    total_chars += 300;
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

    fn pick_model(&self, has_image: bool, complexity: f64, tier: &str) -> (bool, &str) {
        if tier == "premium" {
            info!("PREMIUM TIER: routing to large model");
            return (false, &self.config.large_text_url);
        }

        if has_image && !self.config.large_model_multimodal {
            info!(
                "MODEL SELECTION: image present but large model is text-only, routing to auxiliary multimodal model"
            );
            return (true, &self.config.small_mllm_url);
        }

        if has_image && self.config.large_model_multimodal {
            if complexity > self.config.router_threshold {
                info!(
                    "MODEL SELECTION: image present, complexity {:.2} > threshold {}, routing to large multimodal model",
                    complexity, self.config.router_threshold
                );
                (false, &self.config.large_text_url)
            } else {
                info!(
                    "MODEL SELECTION: image present but complexity {:.2} <= threshold {}, routing to small multimodal model",
                    complexity, self.config.router_threshold
                );
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

    // =========================================================================
    // MULTI-ENDPOINT ROUTING  (OCR / Document / Agent-Compression / Inference)
    // =========================================================================

    /// Extracts a backend route hint from request headers.
    fn header_route_hint(&self, headers: &HeaderMap) -> Option<String> {
        for name in [
            "x-cascade-route",
            "x-cascade-mode",
            "x-router-mode",
            "x-route-mode",
            "x-router",
        ] {
            if let Some(v) = headers.get(name) {
                if let Ok(s) = v.to_str() {
                    let s = s.trim().to_lowercase();
                    if !s.is_empty() {
                        return Some(s);
                    }
                }
            }
        }
        None
    }

    /// Extracts a backend route hint from the request body (metadata / top-level `route`).
    fn payload_route_hint(&self, payload: &ChatCompletionRequest) -> Option<String> {
        if let Some(r) = &payload.route_hint {
            let r = r.trim().to_lowercase();
            if !r.is_empty() {
                return Some(r);
            }
        }
        if let Some(meta) = &payload.metadata {
            if let Some(obj) = meta.as_object() {
                for key in ["route", "mode", "target", "routing"] {
                    if let Some(v) = obj.get(key).and_then(|v| v.as_str()) {
                        let v = v.trim().to_lowercase();
                        if !v.is_empty() {
                            return Some(v);
                        }
                    }
                }
            }
        }
        None
    }

    /// Maps a hint string to a concrete route decision.
    fn hint_to_decision(&self, hint: &str) -> RouteDecision {
        match hint {
            "ocr" | "vision" | "document" | "doc" | "docparse" | "vlm" | "image" | "images" => {
                RouteDecision::Ocr
            }
            "aux" | "auxiliary" | "compression" | "compress" | "compaction" | "summary"
            | "summarize" | "subagent" | "agent" => RouteDecision::Auxiliary,
            "auto" | "adaptive" | "complexity" | "legacy" => RouteDecision::Auto,
            _ => RouteDecision::Inference,
        }
    }

    /// True when the conversation carries image content (Condition A signal).
    fn has_image(&self, payload: &ChatCompletionRequest) -> bool {
        payload.messages.iter().any(|m| match &m.content {
            Some(MessageContent::Parts(parts)) => parts.iter().any(
                |p| matches!(p, MessageContentPart::ImageUrl { .. }),
            ),
            _ => false,
        })
    }

    /// True when the conversation contains uploaded file attachments / document artifacts.
    fn has_file_attachment(&self, payload: &ChatCompletionRequest) -> bool {
        payload.messages.iter().any(|m| match &m.content {
            Some(MessageContent::Parts(parts)) => parts.iter().any(|p| matches!(p, MessageContentPart::File { .. })),
            _ => false,
        })
    }

    /// Condition A — document / OCR parsing payloads (PaddleOCR-VL).
    ///
    /// Only *document* traffic (uploaded files, explicit `document`/`ocr` flags,
    /// system-level OCR markers) is routed to the OCR backend. Plain image_url
    /// chats are general vision and fall through to the multimodal inference
    /// model (Qwythos-9B w/ mmproj), which can describe photos — PaddleOCR-VL
    /// is a document/OCR model, not a chat VLM.
    fn is_ocr_payload(&self, payload: &ChatCompletionRequest) -> bool {
        if self.has_file_attachment(payload) {
            // Uploaded documents (PDF/scans) and image attachments go to the OCR server.
            info!("ROUTE_OCR: file attachment detected in payload");
            return true;
        }

        if let Some(meta) = &payload.metadata {
            if let Some(obj) = meta.as_object() {
                for key in ["document", "ocr", "parse_document", "parseDocument"] {
                    if let Some(v) = obj.get(key) {
                        if v.as_bool().unwrap_or(false) {
                            info!("ROUTE_OCR: explicit '{}' flag in metadata", key);
                            return true;
                        }
                        if let Some(s) = v.as_str() {
                            if !s.is_empty() && !s.eq_ignore_ascii_case("false") {
                                info!("ROUTE_OCR: explicit '{}' flag in metadata", key);
                                return true;
                            }
                        }
                    }
                }
            }
        }

        const DOC_MARKERS: &[&str] = &[
            "paddleocr",
            "paddleocr-vl",
            "[ocr]",
            "[document]",
            "<ocr>",
            "<document>",
            "parse document",
            "parse this document",
            "convert document",
            "convert this document",
            "document to markdown",
            "extract text from this document",
            "extract tables",
            "dokument parsen",
            "dokument in markdown",
        ];
        for m in &payload.messages {
            if m.role == "system" {
                if let Some(MessageContent::Text(t)) = &m.content {
                    let low = t.to_lowercase();
                    if DOC_MARKERS.iter().any(|k| low.contains(k)) {
                        info!("ROUTE_OCR: document parsing marker in system message");
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Condition B — agent context compression / summarization payloads.
    /// Only system-level messages carry Hermes compression hooks, so the detector
    /// is scoped to `role == "system"` to avoid hijacking ordinary user content.
    fn is_agent_compression_payload(&self, payload: &ChatCompletionRequest) -> bool {
        const COMPRESS_MARKERS: &[&str] = &[
            "compression",
            "compress the following",
            "compress the conversation",
            "conversation summary",
            "conversation_summary",
            "summarize the conversation",
            "summarise the conversation",
            "condense the conversation",
            "compact the conversation",
            "context compaction",
            "context_compaction",
            "agent scratchpad",
            "sub-agent",
            "sub_agent",
            "subagent",
            "hermes compression",
        ];
        for m in &payload.messages {
            if m.role == "system" {
                if let Some(MessageContent::Text(t)) = &m.content {
                    let low = t.to_lowercase();
                    if COMPRESS_MARKERS.iter().any(|k| low.contains(k)) {
                        info!("ROUTE_AUXILIARY: compression marker in system message");
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Returns the routing decision for a request based on headers, metadata
    /// and payload structure — the three-condition multi-endpoint dispatcher.
    fn inspect_route(&self, payload: &ChatCompletionRequest, headers: &HeaderMap) -> RouteDecision {
        // Explicit header hint wins for OCR / aux / auto.
        if let Some(hint) = self.header_route_hint(headers) {
            let decision = self.hint_to_decision(&hint);
            if decision == RouteDecision::Ocr {
                info!("ROUTE_HEADER_HINT: '{hint}' -> OCR");
                return decision;
            }
            if decision == RouteDecision::Inference {
                // Symmetric with the body hint: an explicit inference hint pins
                // the main backend (skips OCR/aux content detection).
                info!("ROUTE_HEADER_HINT: '{hint}' -> Inference");
                return decision;
            }
            if (decision == RouteDecision::Auxiliary && !self.has_image(payload))
                || decision == RouteDecision::Auto
            {
                info!("ROUTE_HEADER_HINT: '{hint}' -> {:?}", decision);
                return decision;
            }
        }

        // Metadata / top-level body hint.
        if let Some(hint) = self.payload_route_hint(payload) {
            let decision = self.hint_to_decision(&hint);
            if decision == RouteDecision::Auxiliary && !self.has_image(payload) {
                info!("ROUTE_BODY_HINT: '{hint}' -> auxiliary");
                return decision;
            }
            if decision == RouteDecision::Auto {
                info!("ROUTE_BODY_HINT: '{hint}' -> auto (legacy)");
                return decision;
            }
            if decision == RouteDecision::Inference {
                info!("ROUTE_BODY_HINT: '{hint}' -> inference");
                return decision;
            }
        }

        // Condition A: vision / OCR / document parsing.
        if self.is_ocr_payload(payload) {
            return RouteDecision::Ocr;
        }
        // Condition B: agent context compression / sub-agent execution.
        if self.is_agent_compression_payload(payload) {
            return RouteDecision::Auxiliary;
        }
        // Condition C: default — main inference / RAG / reasoning.
        RouteDecision::Inference
    }

    async fn proxy_to_backend(
        &self,
        payload: &ChatCompletionRequest,
        url: &str,
        bearer: Option<&str>,
        is_streaming: bool,
        _origin: &str,
    ) -> Result<(HeaderMap, Body), ProxyError> {
        // Strip router-internal fields so they are never forwarded to the backend.
        let mut clean = payload.clone();
        clean.metadata = None;
        clean.route_hint = None;

        let mut request = self.http_client.post(url).json(&clean);
        if let Some(token) = bearer {
            request = request.bearer_auth(token);
        }
        let backend_response = request.send()
            .await
            .map_err(|e| {
                ProxyError::unreachable(StatusCode::SERVICE_UNAVAILABLE, url, format!("backend unreachable: {}", e))
            })?;

        let status = backend_response.status();
        if !status.is_success() {
            let err_body = backend_response.text().await.unwrap_or_default();
            warn!("Backend error HTTP {} from {}: {}", status, url, err_body);
            let error_code = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            return Err(ProxyError::new(
                error_code.as_u16(),
                url,
                format!("backend returned HTTP {}: {}", status, err_body.chars().take(300).collect::<String>()),
            ));
        }

        let mut headers = HeaderMap::new();
        if is_streaming {
            headers.insert("content-type", HeaderValue::from_static("text/event-stream"));
            headers.insert("cache-control", HeaderValue::from_static("no-cache"));
            headers.insert("connection", HeaderValue::from_static("keep-alive"));
        } else {
            headers.insert("content-type", HeaderValue::from_static("application/json"));
        }

        fn normalize_event_json(raw: &str) -> Option<String> {
            let trimmed = raw.trim();
            if trimmed.is_empty() || trimmed == "[DONE]" {
                return None;
            }
            let trimmed = trimmed.strip_prefix("data: ").unwrap_or(trimmed);
            let trimmed = trimmed.strip_prefix("data:").unwrap_or(trimmed).trim();
            if trimmed.is_empty() || trimmed == "[DONE]" {
                return None;
            }

            let mut event: serde_json::Value = serde_json::from_str(trimmed).ok()?;
            let delta = match event.get_mut("delta") {
                Some(delta) => delta,
                None => {
                    return Some(format!("data: {}", trimmed));
                }
            };

            info!(target: "cascade_llm::state", "NORMALIZE delta={}", serde_json::to_string(delta).unwrap_or_default());

            if delta.get("tool_calls").is_some() || delta.get("function_call").is_some() {
                return Some(format!("data: {}", trimmed));
            }

            if delta.get("role").is_none() {
                delta["role"] = serde_json::Value::String("assistant".to_string());
            }

            let mut normalized = false;
            if let Some(reasoning) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                if !reasoning.is_empty() {
                    delta["content"] = serde_json::Value::String(reasoning.to_string());
                    delta.as_object_mut()?.remove("reasoning_content");
                    normalized = true;
                }
            } else if let Some(reasoning) = delta.get("reasoning").and_then(|v| v.as_str()) {
                if !reasoning.is_empty() {
                    delta["content"] = serde_json::Value::String(reasoning.to_string());
                    delta.as_object_mut()?.remove("reasoning");
                    normalized = true;
                }
            }

            if normalized {
                Some(format!("data: {}", serde_json::to_string(&event).unwrap_or_default()))
            } else {
                Some(format!("data: {}", trimmed))
            }
        }

        fn emit_complete_sse_events(buffer: &mut Vec<u8>) -> Vec<u8> {
            info!(target: "cascade_llm::state", "EMIT: buffer_len={} first_bytes={:?}", buffer.len(), std::str::from_utf8(&buffer[..buffer.len().min(100)]).unwrap_or_default());
            let mut out = Vec::new();
            while let Some(idx) = buffer.windows(2).position(|w| w == b"\n\n") {
                let end = idx + 2;
                let frame = buffer[..end].to_vec();
                buffer.drain(..end);
                let text = match std::str::from_utf8(&frame) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                if text.trim() == "data: [DONE]" {
                    // OpenAI-strict clients need the terminator; pass it through.
                    out.extend_from_slice(b"data: [DONE]\n\n");
                    continue;
                }
                if let Some(normalized) = normalize_event_json(text) {
                    out.extend_from_slice(normalized.as_bytes());
                    out.extend_from_slice(b"\n\n");
                }
            }
            while let Some(idx) = buffer.iter().position(|&b| b == b'\n') {
                let line = buffer[..idx].to_vec();
                buffer.drain(..=idx);
                let text = match std::str::from_utf8(&line) {
                    Ok(s) => s.trim(),
                    Err(_) => continue,
                };
                if text.is_empty() || text == "[DONE]" {
                    continue;
                }
                if let Some(normalized) = normalize_event_json(text) {
                    out.extend_from_slice(normalized.as_bytes());
                    out.extend_from_slice(b"\n\n");
                }
            }
            out
        }

        if is_streaming {
            let pending = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
            let stream = backend_response.bytes_stream().map(move |item| {
                let mut chunk = match item {
                    Ok(c) => c,
                    Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::Other, e)),
                };

                let mut local = pending.lock().unwrap();
                local.extend_from_slice(&chunk);
                let ready = emit_complete_sse_events(&mut local);
                drop(local);

                if ready.is_empty() {
                    return Ok(axum::body::Bytes::new());
                }

                Ok(axum::body::Bytes::from(ready))
            });
            let body = Body::from_stream(stream);
            return Ok((headers, body));
        }

        let body_bytes = backend_response
            .bytes()
            .await
            .map_err(|_| ProxyError::unreachable(StatusCode::SERVICE_UNAVAILABLE, url, "failed to read backend response"))?;

        if let Ok(mut event) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
            if let Some(choices) = event.get_mut("choices").and_then(|c| c.as_array_mut()) {
                if let Some(choice) = choices.get_mut(0) {
                    if let Some(delta) = choice.get_mut("delta") {
                        if delta.get("tool_calls").is_some() || delta.get("function_call").is_some() {
                            return Ok((headers, Body::from(body_bytes)));
                        }
                        if let Some(reasoning) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                            if !reasoning.is_empty() {
                                delta["content"] = serde_json::Value::String(reasoning.to_string());
                                delta.as_object_mut().unwrap().remove("reasoning_content");
                            }
                        } else if let Some(reasoning) = delta.get("reasoning").and_then(|v| v.as_str()) {
                            if !reasoning.is_empty() {
                                delta["content"] = serde_json::Value::String(reasoning.to_string());
                                delta.as_object_mut().unwrap().remove("reasoning");
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

    // --------------------------------------------------------------------------
    // Main entry point: three-condition routing dispatch
    // --------------------------------------------------------------------------

    pub async fn route_request_with_fallback(
        &self,
        payload: ChatCompletionRequest,
        is_streaming: bool,
        tier: &str,
        headers: &HeaderMap,
        origin: &str,
    ) -> Result<(HeaderMap, Body), ProxyError> {
        let decision = self.inspect_route(&payload, headers);
        info!(
            "ROUTE_OUTCOME: decision={:?}, model={}, messages={}, tier={}, streaming={}",
            decision,
            payload.model,
            payload.messages.len(),
            tier,
            is_streaming
        );

        match decision {
            RouteDecision::Ocr => self.route_ocr(&payload, is_streaming, origin).await,
            RouteDecision::Auxiliary => self.route_auxiliary(&payload, is_streaming, origin).await,
            RouteDecision::Inference => self.route_inference(&payload, is_streaming, origin).await,
            RouteDecision::Auto => self.route_auto(payload, is_streaming, tier, headers, origin).await,
        }
    }

    async fn private_proxy(
        &self,
        payload: &ChatCompletionRequest,
        url: &str,
        backend: RouteBackend,
        is_streaming: bool,
        origin: &str,
    ) -> Result<(HeaderMap, Body), ProxyError> {
        self.private_proxy_with_bearer(payload, url, None, backend, is_streaming, origin)
            .await
    }

    async fn private_proxy_with_bearer(
        &self,
        payload: &ChatCompletionRequest,
        url: &str,
        bearer: Option<&str>,
        backend: RouteBackend,
        is_streaming: bool,
        origin: &str,
    ) -> Result<(HeaderMap, Body), ProxyError> {
        let backend_name = match backend {
            RouteBackend::OcrServer => "ocr-server",
            RouteBackend::AuxiliaryServer => "auxiliary-server",
            RouteBackend::InferenceServer => "inference-server",
        };
        let metric_name = match backend {
            RouteBackend::OcrServer => "ocr",
            RouteBackend::AuxiliaryServer => "auxiliary",
            RouteBackend::InferenceServer => "inference",
        };

        if self.circuit_breaker.is_open(url).await {
            warn!("Circuit breaker OPEN for {}, refusing {}", url, backend_name);
            self.metrics.record_fallback("backend_unavailable");
            return Err(ProxyError::unreachable(
                StatusCode::SERVICE_UNAVAILABLE,
                backend_name,
                format!("{} circuit breaker open (too many recent failures)", backend_name),
            ));
        }

        match self.proxy_to_backend(payload, url, bearer, is_streaming, origin).await {
            Ok(mut parts) => {
                if let Ok(v) = HeaderValue::from_str(backend_name) {
                    parts.0.insert("x-cascade-route", v);
                }
                self.circuit_breaker.record_success(url).await;
                self.metrics.record_request(metric_name);
                info!("PROXY_OK: backend={}, url={}", backend_name, url);
                Ok(parts)
            }
            Err(e) => {
                self.circuit_breaker.record_failure(url).await;
                self.metrics.record_fallback("backend_unavailable");
                if e.unreachable {
                    Err(ProxyError::unreachable(
                        StatusCode::SERVICE_UNAVAILABLE,
                        backend_name,
                        format!("{} server unavailable: {}", backend_name, e.context),
                    ))
                } else {
                    Err(ProxyError::new(
                        e.status,
                        backend_name,
                        format!("{} returned: {}", backend_name, e.context),
                    ))
                }
            }
        }
    }

    /// Resolve the effective upstream for a role: runtime override if present,
    /// otherwise the static config URL. Returns (url, bearer).
    async fn resolve_upstream(&self, role: &str, default_url: &str) -> (String, Option<String>) {
        let map = self.upstream_overrides.read().await;
        match map.get(role) {
            Some(o) => {
                info!("UPSTREAM_OVERRIDE: {} -> {}", role, o.url);
                (o.url.clone(), o.bearer.clone())
            }
            None => (default_url.to_string(), None),
        }
    }

    /// Condition A — dedicated vision / document parsing (OCR) endpoint.
    async fn route_ocr(
        &self,
        payload: &ChatCompletionRequest,
        is_streaming: bool,
        origin: &str,
    ) -> Result<(HeaderMap, Body), ProxyError> {
        let mut payload = payload.clone();
        self.inline_all_images(&mut payload).await;
        let (url, bearer) = self.resolve_upstream("ocr", &self.config.ocr_server_url).await;
        info!(
            "ROUTE_A: OCR -> {} ({} messages)",
            url,
            payload.messages.len()
        );
        let _ = self.extract_image_urls(&payload.messages);
        self.private_proxy_with_bearer(&payload, &url, bearer.as_deref(), RouteBackend::OcrServer, is_streaming, origin).await
    }

    /// Condition B — agent context compression / fast text (auxiliary) endpoint.
    async fn route_auxiliary(
        &self,
        payload: &ChatCompletionRequest,
        is_streaming: bool,
        origin: &str,
    ) -> Result<(HeaderMap, Body), ProxyError> {
        let mut payload = payload.clone();
        // If a remote image slipped in, inline it before forwarding.
        self.inline_all_images(&mut payload).await;
        let (url, bearer) = self.resolve_upstream("auxiliary", &self.config.small_mllm_url).await;
        info!(
            "ROUTE_B: Auxiliary(compression) -> {} ({} messages)",
            url,
            payload.messages.len()
        );
        self.private_proxy_with_bearer(&payload, &url, bearer.as_deref(), RouteBackend::AuxiliaryServer, is_streaming, origin).await
    }

    /// Condition C — default chat & reasoning (main inference) endpoint.
    async fn route_inference(
        &self,
        payload: &ChatCompletionRequest,
        is_streaming: bool,
        origin: &str,
    ) -> Result<(HeaderMap, Body), ProxyError> {
        if self.has_image(payload) && !self.config.large_model_multimodal {
            info!(
                "ROUTE_C: image detected but large model is text-only -> auxiliary multimodal backend"
            );
            return self.route_auxiliary(payload, is_streaming, origin).await;
        }

        let mut payload = payload.clone();
        // Qwythos-9B is multimodal — inline any remote image URLs before forwarding.
        self.inline_all_images(&mut payload).await;
        let lang = language::detect_language(&payload.messages);
        payload = language::inject_language_prompt(lang, payload);
        let (url, bearer) = self.resolve_upstream("inference", &self.config.large_text_url).await;
        info!(
            "ROUTE_C: Inference(default) -> {} ({} messages, lang={})",
            url,
            payload.messages.len(),
            lang
        );
        self.private_proxy_with_bearer(&payload, &url, bearer.as_deref(), RouteBackend::InferenceServer, is_streaming, origin).await
    }

    // --------------------------------------------------------------------------
    // Legacy complexity/session routing, reachable via explicit "auto" mode.
    // --------------------------------------------------------------------------

    async fn route_auto(
        &self,
        payload: ChatCompletionRequest,
        is_streaming: bool,
        tier: &str,
        headers: &HeaderMap,
        origin: &str,
    ) -> Result<(HeaderMap, Body), ProxyError> {
        let has_image = self.has_image(&payload);
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
            "[AUTO_REQ_START] has_image={}, has_tools={}, history_has_tools={}, session={:?}, cached={:?}, tier={}",
            has_image, has_tools, history_has_tools, session_key, cached_route, tier
        );
        info!("AUTO: complexity={:.2}, threshold={:.2}", complexity_score, self.config.router_threshold);

        let injected_payload = language::inject_language_prompt(language, payload);
        let mut injected_payload = injected_payload;
        injected_payload.metadata = None;
        injected_payload.route_hint = None;

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
            info!("SESSION AFFINITY: History has tools but no cached route. Forcing large model target_url={}", large_url);
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
            info!("SESSION AFFINITY ROUTE: target={}", target);
            let result = self.proxy_to_backend(&injected_payload, &target, None, is_streaming, origin).await;
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

        let (use_small, target_url) = self.pick_model(has_image, complexity_score, tier);
        let target_url = target_url.to_owned();
        info!("AUTO SELECTED_URL: {}", target_url);

        if !use_small {
            let result = self.proxy_to_backend(&injected_payload, &target_url, None, is_streaming, origin).await;
            match &result {
                Ok(_) => {
                    self.circuit_breaker.record_success(&target_url).await;
                    if let Some(ref key) = session_key {
                        self.session_cache.insert(key.clone(), target_url.clone()).await;
                    }
                }
                Err(_) => {
                    self.circuit_breaker.record_failure(&target_url).await;
                }
            }
            self.metrics.record_request("large");
            return result;
        }

        info!("AUTO SMALL_MODEL_PATH: target_url={}", target_url);
        let mut small_payload = injected_payload.clone();
        if small_payload.max_tokens.is_none() {
            small_payload.max_tokens = Some(4096);
        }

        if is_streaming {
            let result = self.proxy_to_backend(&small_payload, &target_url, None, true, origin).await;
            match &result {
                Ok(_) => {
                    self.circuit_breaker.record_success(&target_url).await;
                    if let Some(ref key) = session_key {
                        self.session_cache.insert(key.clone(), target_url.clone()).await;
                    }
                }
                Err(_) => {
                    self.circuit_breaker.record_failure(&target_url).await;
                }
            }
            self.metrics.record_request("small");
            return result;
        }

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
            .map_err(|e| ProxyError::unreachable(StatusCode::BAD_GATEWAY, &target_url, format!("{}", e)))?;

        let status = backend_response.status();
        let body_bytes = backend_response
            .bytes()
            .await
            .map_err(|_| ProxyError::unreachable(StatusCode::BAD_GATEWAY, &target_url, "read failure"))?;

        if !status.is_success() {
            info!("Small model returned HTTP {}, rerouting original request to large model", status);
            self.circuit_breaker.record_failure(&target_url).await;
            self.metrics.record_fallback("primary_failed");
            let result = self.proxy_to_backend(&injected_payload, &large_url, None, false, origin).await;
            if result.is_ok() {
                self.circuit_breaker.record_success(&large_url).await;
                if let Some(ref key) = session_key {
                    self.session_cache.insert(key.clone(), large_url.clone()).await;
                }
            }
            self.metrics.record_request("large");
            return result;
        }

        self.circuit_breaker.record_success(&target_url).await;

        let confidence = self.extract_confidence(&body_bytes);
        let keep_small = match confidence {
            Some(c) if c >= self.config.confidence_threshold => {
                info!("SMALL MODEL CONFIDENCE: {:.4} >= threshold {:.4}, keeping", c, self.config.confidence_threshold);
                true
            }
            Some(c) => {
                info!("SMALL MODEL CONFIDENCE: {:.4} < threshold {:.4}, rerouting to large", c, self.config.confidence_threshold);
                false
            }
            None => {
                info!("No logprobs in small model response, keeping response");
                true
            }
        };

        if keep_small {
            if let Some(ref key) = session_key {
                self.session_cache.insert(key.clone(), target_url.clone()).await;
            }
            let mut hdrs = HeaderMap::new();
            hdrs.insert("content-type", HeaderValue::from_static("application/json"));
            if let Some(c) = confidence {
                let val = format!("{:.4}", c);
                if let Ok(hv) = HeaderValue::from_str(&val) {
                    hdrs.insert("x-confidence", hv);
                }
            }
            self.metrics.record_request("small");
            return Ok((hdrs, Body::from(body_bytes)));
        }

        info!("Rerouting original request to large text model");
        self.metrics.record_fallback("quality_low");
        let result = self.proxy_to_backend(&injected_payload, &large_url, None, false, origin).await;
        if result.is_ok() {
            self.circuit_breaker.record_success(&large_url).await;
            if let Some(ref key) = session_key {
                self.session_cache.insert(key.clone(), large_url.clone()).await;
            }
        }
        self.metrics.record_request("large");
        result
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

    // =========================================================================
    // EXTRACTION ENDPOINT  (POST /v1/extraction)
    // =========================================================================

    /// Selects the best available extraction backend: lowest enabled priority
    /// whose circuit breaker is closed, sorted by `max_cost_per_hour` when
    /// tied.
    async fn select_extraction_backend(&self) -> Option<ExtractionBackendEntry> {
        let backends = self.extraction_backends.read().await;
        let mut candidates: Vec<&ExtractionBackendEntry> = backends
            .iter()
            .filter(|b| b.enabled && b.healthy)
            .collect();
        candidates.sort_by_key(|b| b.priority);

        for candidate in candidates {
            if !self.circuit_breaker.is_open(&candidate.url).await {
                return Some(candidate.clone());
            }
            info!(
                "EXTRACT_SKIP: backend '{}' (priority {}) circuit breaker open",
                candidate.id, candidate.priority
            );
        }

        warn!("EXTRACT: no healthy extraction backend available");
        None
    }

    /// Registers or updates an extraction backend.
    pub async fn register_extraction_backend(&self, entry: ExtractionBackendEntry) {
        let mut backends = self.extraction_backends.write().await;
        if let Some(existing) = backends.iter_mut().find(|b| b.id == entry.id) {
            *existing = entry.clone();
            info!("EXTRACT_REGISTER: updated backend '{}' (priority {})", entry.id, entry.priority);
        } else {
            info!("EXTRACT_REGISTER: added backend '{}' (priority {})", entry.id, entry.priority);
            backends.push(entry);
        }
    }

    /// Removes an extraction backend by id.
    pub async fn remove_extraction_backend(&self, id: &str) -> bool {
        let mut backends = self.extraction_backends.write().await;
        let len_before = backends.len();
        backends.retain(|b| b.id != id);
        let removed = backends.len() < len_before;
        if removed {
            info!("EXTRACT_REGISTER: removed backend '{}'", id);
        }
        removed
    }

    /// Routes an extraction request to the best available backend with
    /// fallback.  The payload is forwarded as-is to the backend's
    /// `/v1/chat/completions` endpoint.
    pub async fn route_extraction(
        &self,
        payload: ChatCompletionRequest,
        is_streaming: bool,
    ) -> Result<(HeaderMap, Body), ProxyError> {
        let backend = self
            .select_extraction_backend()
            .await
            .ok_or_else(|| {
                ProxyError::unreachable(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "extraction",
                    "no healthy extraction backend available",
                )
            })?;

        let url = backend.completions_url();
        info!(
            "EXTRACT_ROUTE: backend='{}' type={:?} url={}",
            backend.id, backend.backend_type, url
        );

        // Strip internal fields before forwarding
        let mut clean = payload.clone();
        clean.metadata = None;
        clean.route_hint = None;

        // Forward with API key if provided
        let mut req_builder = self.http_client.post(&url).json(&clean);
        if let Some(api_key) = &backend.api_key {
            if !api_key.is_empty() {
                req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
            }
        }

        let backend_response = match req_builder.send().await {
            Ok(resp) => resp,
            Err(e) => {
                self.circuit_breaker.record_failure(&url).await;
                return Err(ProxyError::unreachable(
                    StatusCode::SERVICE_UNAVAILABLE,
                    &backend.id,
                    format!("extraction backend unreachable: {}", e),
                ));
            }
        };

        let status = backend_response.status();
        if !status.is_success() {
            self.circuit_breaker.record_failure(&url).await;
            let err_body = backend_response.text().await.unwrap_or_default();
            let error_code =
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            return Err(ProxyError::new(
                error_code.as_u16(),
                &backend.id,
                format!(
                    "extraction backend returned HTTP {}: {}",
                    status,
                    err_body.chars().take(300).collect::<String>()
                ),
            ));
        }

        self.circuit_breaker.record_success(&url).await;
        self.metrics.record_request("extraction");

        let mut headers = HeaderMap::new();
        if is_streaming {
            headers.insert("content-type", HeaderValue::from_static("text/event-stream"));
            headers.insert("cache-control", HeaderValue::from_static("no-cache"));
            headers.insert("connection", HeaderValue::from_static("keep-alive"));
        } else {
            headers.insert("content-type", HeaderValue::from_static("application/json"));
        }

        if is_streaming {
            let stream = backend_response
                .bytes_stream()
                .map(|item| item.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)));
            return Ok((headers, Body::from_stream(stream)));
        }

        let body_bytes = backend_response
            .bytes()
            .await
            .map_err(|_| {
                ProxyError::unreachable(
                    StatusCode::SERVICE_UNAVAILABLE,
                    &backend.id,
                    "failed to read extraction response",
                )
            })?;

        Ok((headers, Body::from(body_bytes)))
    }
}