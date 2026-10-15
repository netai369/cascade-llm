use crate::config::AppConfig;
use crate::db::Db;
use crate::intent_routing;
use crate::language;
use crate::types::*;
use crate::usage::UsageEvent;

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

/// Highest keyword score (see `keyword_score`) that still counts as a "simple"
/// prompt. Any complexity marker at all pushes the request to the main model:
/// one code fence is 0.25, one list item 0.1, one keyword 0.2. Keeping this at
/// 0.0 makes the rule "no marker means simple", which is the conservative
/// direction for a router whose whole job is to avoid downgrading real work.
const HYBRID_MAX_KEYWORD_SCORE: f64 = 0.0;

fn native_service_action(engine: &str) -> Option<String> {
    let action = match engine.to_ascii_lowercase().as_str() {
        "vmlx" | "mlx" => "Start the vmlx/MLX server on the registered Mac host URL.",
        "inferencer" => "Start the Inferencer service on the registered Mac host URL.",
        "llamacpp" | "llama.cpp" => "Start llama-server on the registered host URL.",
        "sd.cpp" | "sd-cpp" => "Start sd.cpp on the registered host URL.",
        _ => return None,
    };
    Some(action.to_string())
}

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
    /// Round-robin cursor for `/v1/extraction` backend selection.
    extraction_counter: AtomicU64,
    /// Dynamic upstream registry (role -> weighted node pool), managed via the
    /// admin API and probed by the background health checker.
    pub registry: Arc<tokio::sync::RwLock<crate::registry::UpstreamRegistry>>,
    /// Settings-page-controlled runtime values (persisted via Db).
    pub runtime: Arc<tokio::sync::RwLock<crate::types::RuntimeSettings>>,
    /// Usage metering sink (see `usage.rs`). Disabled until `main` installs a
    /// live recorder, so tests and library users pay nothing.
    pub usage: Arc<crate::usage::UsageRecorder>,
    /// Per-tenant fair-share concurrency gate for LLM generation routes.
    pub tenant_limiter: Arc<crate::tenant_limit::TenantLimiter>,
}


fn match_marker(content: &str, marker: &str, mode: &str) -> bool {
    if mode == "prefix" {
        content.trim_start().to_lowercase().starts_with(&marker.to_lowercase())
    } else {
        content.to_lowercase().contains(&marker.to_lowercase())
    }
}

impl GatewayState {
    pub fn new(
        config: AppConfig,
        metrics: Arc<crate::cascade_features::MetricsRegistry>,
        db: Arc<Db>,
    ) -> Self {
        let http_client = Arc::new(
            reqwest::Client::builder()
                // Keep-alive across requests: pool_max_idle_per_host(0)
                // disabled reuse and forced a new TCP+TLS handshake per call.
                .pool_max_idle_per_host(32)
                .pool_idle_timeout(Duration::from_secs(90))
                .connect_timeout(Duration::from_secs(5))
                // Total ceiling for batch calls; long SSE streams are bounded
                // by the idle read_timeout below rather than an absolute cap.
                .timeout(Duration::from_secs(1800))
                .read_timeout(Duration::from_secs(180))
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
            extraction_counter: AtomicU64::new(0),
            registry: Arc::new(tokio::sync::RwLock::new(
                crate::registry::UpstreamRegistry::default(),
            )),
            usage: crate::usage::UsageRecorder::disabled(),
            tenant_limiter: Arc::new(crate::tenant_limit::TenantLimiter::new(
                crate::tenant_limit::TenantLimitConfig::from_env(),
            )),
            runtime: Arc::new(tokio::sync::RwLock::new(crate::types::RuntimeSettings {
                router_threshold: config.router_threshold,
                confidence_threshold: config.confidence_threshold,
                route_tools_to_large: config.route_tools_to_large,
                default_route: config.default_route.clone(),
                hybrid_auto: config.hybrid_auto,
                hybrid_aux_max_chars: config.hybrid_aux_max_chars,
                marker_mode: config.marker_mode.clone(),
                tts_url: config.tts_url.clone(),
                stt_url: config.stt_url.clone(),
                providers: Vec::new(),
            })),
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

    async fn pick_model(&self, has_image: bool, complexity: f64, tier: &str) -> (bool, String) {
        let rt = self.runtime.read().await;
        let router_threshold = rt.router_threshold;
        let large_url = self.config.large_text_url.clone();
        let small_url = self.config.small_mllm_url.clone();

        if tier == "premium" {
            info!("PREMIUM TIER: routing to large model");
            return (false, large_url);
        }

        if has_image && !self.config.large_model_multimodal {
            info!("MODEL SELECTION: image + text-only large -> auxiliary multimodal");
            return (true, small_url);
        }

        let use_small = complexity <= router_threshold;
        info!(
            "MODEL SELECTION: complexity {:.2} vs threshold {:.2} -> {}",
            complexity,
            router_threshold,
            if use_small { "small" } else { "large" }
        );
        if use_small {
            (true, small_url)
        } else {
            (false, large_url)
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
    ///
    /// Accepts both the short aliases and the canonical resource names the
    /// settings page documents as `x-cascade-route` values
    /// (`inference-server`, `auxiliary-server`, `auto-small`, `auto-large`,
    /// `session-affinity`). Without the canonical names, `auxiliary-server`
    /// silently fell through to `Inference` — the opposite of what the docs
    /// promise (regression caught wiring voice-chat-v2's realtime path).
    fn hint_to_decision(hint: &str) -> RouteDecision {
        match hint {
            "ocr" | "vision" | "document" | "doc" | "docparse" | "vlm" | "image" | "images" => {
                RouteDecision::Ocr
            }
            "aux" | "auxiliary" | "auxiliary-server" | "small" | "small-server"
            | "compression" | "compress" | "compaction" | "summary" | "summarize"
            | "subagent" | "agent" => RouteDecision::Auxiliary,
            "auto" | "adaptive" | "complexity" | "legacy"
            | "auto-small" | "auto-large" | "session-affinity" => RouteDecision::Auto,
            "inference" | "inference-server" | "main" | "large" | "large-server" => {
                RouteDecision::Inference
            }
            // Unknown hints keep the historical behavior: treat as main.
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
    fn is_ocr_payload(&self, payload: &ChatCompletionRequest, marker_mode: &str) -> bool {
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
                    if DOC_MARKERS.iter().any(|k| match_marker(&low, k, marker_mode)) {
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
    fn is_agent_compression_payload(&self, payload: &ChatCompletionRequest, marker_mode: &str) -> bool {
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
                    if COMPRESS_MARKERS.iter().any(|k| match_marker(&low, k, marker_mode)) {
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
    async fn inspect_route(&self, payload: &ChatCompletionRequest, headers: &HeaderMap) -> RouteDecision {
        // Settings-page deterministic short-circuit (default_route).
        if let Some(dr) = self.runtime.read().await.default_route.clone() {
            let d = match dr.to_lowercase().as_str() {
                "inference" => Some(RouteDecision::Inference),
                "auxiliary" => Some(RouteDecision::Auxiliary),
                // "ocr" => Some(RouteDecision::Ocr),  // Route A retired — falls through to None
                _ => None,
            };
            if let Some(d) = d {
                info!("ROUTE_DEFAULT: settings pin {:?} (payload ignored)", d);
                return d;
            }
        }
        // Explicit header hint wins for OCR / aux / auto.
        if let Some(hint) = self.header_route_hint(headers) {
            let decision = Self::hint_to_decision(&hint);
            if decision == RouteDecision::Ocr {
                // Dynamic OCR route: honoured whenever an active `ocr` node is
                // registered; otherwise content detection decides (the static
                // default OCR URL is only used by explicit document payloads).
                if self.has_active_role("ocr").await {
                    info!("ROUTE_HEADER_HINT: '{hint}' -> Ocr (active ocr node)");
                    return decision;
                }
                info!("ROUTE_HEADER_HINT: '{hint}' requested OCR — no active ocr node, using content detection");
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
            let decision = Self::hint_to_decision(&hint);
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

        let marker_mode = self.runtime.read().await.marker_mode.clone();
        // Condition B: agent context compression / sub-agent execution.
        if self.is_agent_compression_payload(payload, &marker_mode) {
            return RouteDecision::Auxiliary;
        }
        // CONDITION A (dynamic): document/OCR payloads go to the dedicated OCR
        // node — but only when one is actually registered and healthy. Without
        // a registry node the multimodal main/auxiliary backends handle scans
        // natively (Route A stays retired by default).
        if self.has_active_role("ocr").await && self.is_ocr_payload(payload, &marker_mode) {
            return RouteDecision::Ocr;
        }
        // Condition C: hybrid — when the effective default route is "auto",
        // the complexity router picks the backend: simple text goes to the
        // fast auxiliary model, everything else to the main model.
        // Disabled by default (CASCADE_HYBRID_AUTO=false) so the vllm fold
        // (aux == main) and the llamacpp rollback keep the old hard default.
        if self.hybrid_enabled().await && self.hybrid_pick_aux(payload).await {
            return RouteDecision::Auxiliary;
        }
        // Condition C: default — main inference / RAG / reasoning.
        RouteDecision::Inference
    }

    /// True when the hybrid complexity router is allowed to override the
    /// hard "everything goes to main" default. Requires BOTH:
    ///   * `hybrid_auto` (env `CASCADE_HYBRID_AUTO` / settings checkbox), and
    ///   * a default route that is unset or "auto" — an explicit
    ///     `inference`/`auxiliary` pin from the settings page wins outright.
    async fn hybrid_enabled(&self) -> bool {
        let rt = self.runtime.read().await;
        if !rt.hybrid_auto {
            return false;
        }
        match rt.default_route.as_deref().map(|s| s.to_lowercase()) {
            None => true,
            Some(ref r) if r == "auto" || r.is_empty() => true,
            Some(_) => false,
        }
    }

    /// Complexity gate for the hybrid router: "is this a short, plain,
    /// text-only prompt that the small model can answer on its own?".
    ///
    /// Reuses `evaluate_complexity` (the same score `route_auto` /
    /// `pick_model` compare against `router_threshold`) but deliberately on the
    /// **last user message only** — the full transcript includes system prompts,
    /// tool schemas and chat history, which would push every real conversation
    /// over the threshold through the `total_chars` term alone.
    ///
    /// Everything that must reach the main model is rejected up front:
    /// images, audio, files, tools (declared or present in the history),
    /// reasoning/code markers, long prompts and multi-turn conversations.
    async fn hybrid_pick_aux(&self, payload: &ChatCompletionRequest) -> bool {
        let (threshold, max_chars) = {
            let rt = self.runtime.read().await;
            (rt.router_threshold, rt.hybrid_aux_max_chars)
        };

        // Images/audio/files are the multimodal/aux-VLM path, not the fast path.
        if self.has_image(payload) {
            return false;
        }
        if payload.messages.iter().any(|m| {
            m.role == "tool"
                || m.tool_calls.is_some()
                || matches!(&m.content, Some(MessageContent::Parts(parts)) if parts.iter().any(|p| !matches!(p, MessageContentPart::Text { .. })))
        }) {
            return false;
        }
        // A declared tool list is a hard "main" signal (see also ROUTE_TOOLS_TO_LARGE).
        if payload.tools.is_some() || payload.functions.is_some() {
            return false;
        }

        let Some(last_user) = payload.messages.iter().rev().find(|m| m.role == "user") else {
            return false;
        };
        let Some(MessageContent::Text(text)) = &last_user.content else {
            return false;
        };
        let text = text.trim();
        if text.is_empty() {
            return false;
        }
        // Length gate: `evaluate_complexity` caps its char term at 1.0 and
        // halves it, so on its own a long keyword-free prompt can never cross
        // the threshold — a 5 kB paste would still read as "simple".
        if text.chars().count() > max_chars {
            info!(
                "HYBRID: prompt of {} chars exceeds the simple-prompt budget ({}) -> main",
                text.chars().count(),
                max_chars
            );
            return false;
        }
        // Only the newest turn counts as "simple" — a follow-up question needs
        // the history and therefore the main model's context handling.
        let user_turns = payload.messages.iter().filter(|m| m.role == "user").count();
        if user_turns > 1 {
            return false;
        }

        // Reuse the existing scorer, but `evaluate_complexity` halves BOTH
        // components (`0.5*char + 0.5*keyword`), so on a single short message
        // the char term is near zero and the score can only approach the 0.5
        // default threshold by accumulating keywords — which makes a
        // threshold comparison useless exactly where the hybrid router
        // operates. "```python" scored 0.13 and "analyze deeply …" 0.19,
        // i.e. a code block and a reasoning prompt both read as simple.
        //
        // So compare the KEYWORD component on its own here (0 = no
        // complexity marker of any kind, whatever the message length) and keep
        // the combined score as a second, looser condition. The length budget
        // above already covers "long but plain".
        let complexity = self.evaluate_complexity(std::slice::from_ref(last_user));
        let keyword_score = self.keyword_score(std::slice::from_ref(last_user));
        let use_aux = complexity <= threshold && keyword_score <= HYBRID_MAX_KEYWORD_SCORE;
        info!(
            "HYBRID: last user message ({} chars): complexity={:.2} (threshold {:.2}), keywords={:.2} (limit {:.2}) -> {}",
            text.chars().count(),
            complexity,
            threshold,
            keyword_score,
            HYBRID_MAX_KEYWORD_SCORE,
            if use_aux { "auxiliary" } else { "main" }
        );
        use_aux
    }

    /// The keyword half of `evaluate_complexity`, on its own. Returned
    /// separately because the combined score mixes it with a length term that
    /// is ~0 for the single short prompt the hybrid router looks at.
    fn keyword_score(&self, messages: &[ChatMessage]) -> f64 {
        let detected_lang = language::detect_language(messages);
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
            _ => vec![
                "analyze deeply", "write code", "expert", "reasoning", "logic", "complex",
                "step 1", "step 2", "first,", "second,", "third,", "four", "five",
                "architecture", "infrastructure", "debug", "optimize", "refactor",
                "theorem", "proof", "calculate", "compute", "equation", "derivative",
            ],
        };

        for msg in messages {
            let Some(ref content) = msg.content else { continue };
            let texts: Vec<&str> = match content {
                MessageContent::Text(text) => vec![text.as_str()],
                MessageContent::Parts(parts) => parts
                    .iter()
                    .filter_map(|p| match p {
                        MessageContentPart::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect(),
            };
            for text in texts {
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
                keyword_score += (text.matches("```").count() / 2) as f64 * 0.25;
                let list_patterns = ["\n1.", "\n2.", "\n3.", "\n1)", "\na)", "\na."];
                let list_count = list_patterns.iter().map(|p| text.matches(p).count()).sum::<usize>();
                keyword_score += list_count as f64 * 0.1;
            }
        }
        keyword_score
    }

    #[allow(clippy::too_many_arguments)]
    async fn proxy_to_backend(
        &self,
        payload: &ChatCompletionRequest,
        url: &str,
        bearer: Option<&str>,
        is_streaming: bool,
        usage_base: UsageEvent,
        header_latency_ms: &mut Option<u64>,
    ) -> Result<(HeaderMap, Body), ProxyError> {
        // Strip router-internal fields so they are never forwarded to the backend.
        let mut clean = payload.clone();
        clean.metadata = None;
        clean.route_hint = None;
        // Usage metering (see `usage.rs`): every streamed request asks the
        // backend for the terminal usage chunk, whether or not the client did.
        // The SSE rewriter strips that chunk again for clients that did not
        // opt in. Non-streamed requests never carry stream_options upstream
        // (vLLM rejects it there).
        clean.stream_options = if is_streaming {
            Some(serde_json::json!({"include_usage": true}))
        } else {
            None
        };
        let client_wants_usage = payload.client_wants_stream_usage();

        let mut request = self.http_client.post(url).json(&clean);
        if let Some(token) = bearer {
            request = request.bearer_auth(token);
        }
        let sent_at = Instant::now();
        let backend_response = request.send()
            .await
            .map_err(|e| {
                ProxyError::unreachable(StatusCode::SERVICE_UNAVAILABLE, url, format!("backend unreachable: {}", e))
            })?;
        // Latency-to-first-byte (response headers), not full stream duration —
        // keeps the routing EMA comparable between streaming and batch calls.
        *header_latency_ms = Some(sent_at.elapsed().as_millis() as u64);

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
        // Unified model identity header on every proxied response.
        if let Ok(hv) = HeaderValue::from_str(&self.config.cascade_model_name) {
            headers.insert("x-model-name", hv);
        }

        fn emit_complete_sse_events(
            buffer: &mut Vec<u8>,
            model_name: &str,
            usage: &mut ChatUsageTracker,
        ) -> Vec<u8> {
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
                let Some(kept) = usage.observe_frame(text) else { continue };
                if let Some(normalized) = normalize_event_json(&kept, model_name) {
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
                let Some(kept) = usage.observe_frame(text) else { continue };
                if let Some(normalized) = normalize_event_json(&kept, model_name) {
                    out.extend_from_slice(normalized.as_bytes());
                    out.extend_from_slice(b"\n\n");
                }
            }
            out
        }

        let cascade_model_name = self.config.cascade_model_name.clone();
        if is_streaming {
            let pending = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
            let model_name = cascade_model_name.clone();
            let lp_monitor = std::sync::Arc::new(std::sync::Mutex::new(
                crate::intent_routing::LogprobMonitor::new(),
            ));
            // Owned by the stream closure: records one usage event when the
            // stream ends or the client disconnects (Drop).
            let mut usage = ChatUsageTracker::new(
                self.usage.clone(),
                usage_base.clone(),
                payload,
                sent_at,
                client_wants_usage,
            );
            let stream = backend_response.bytes_stream().map(move |item| {
                let chunk = match item {
                    Ok(c) => c,
                    Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::Other, e)),
                };

                let mut local = pending.lock().unwrap();
                let chunk_start = local.len();
                local.extend_from_slice(&chunk);

                // Cascade fallback: strip <.CASCADE_FALLBACK>. markers and
                // monitor token logprobs (MA5 < -1.8) as a second uncertainty
                // signal.  Rerouting a stream that is already in-flight is not
                // possible — the intent-based pre-bypass (and the non-streaming
                // FALLBACK_TAG reroute) are the primary mechanisms, this only
                // keeps the marker out of the client's text.
                let (tag_hit, lp_hit) = {
                    let text = std::str::from_utf8(&local[..]).unwrap_or_default();
                    (
                        text.contains(crate::intent_routing::FALLBACK_TAG),
                        {
                            let mut m = lp_monitor.lock().unwrap();
                            crate::intent_routing::check_chunk_logprobs(text, &mut m)
                        },
                    )
                };
                if tag_hit {
                    tracing::warn!("CASCADE_FALLBACK: small-model marker detected — stripping from stream");
                    let cleaned: Vec<u8> = {
                        let text = &local[..];
                        let tag: &[u8] = crate::intent_routing::FALLBACK_TAG.as_bytes();
                        let mut out = Vec::with_capacity(text.len());
                        let mut i = 0;
                        while i < text.len() {
                            if text[i..].starts_with(tag) {
                                i += tag.len();
                            } else {
                                out.push(text[i]);
                                i += 1;
                            }
                        }
                        out
                    };
                    local.clear();
                    local.extend_from_slice(&cleaned);
                } else if lp_hit {
                    // The chunk pushed average logprobs below threshold —
                    // drop just this Network chunk, keep prior buffered text.
                    tracing::warn!("CASCADE_FALLBACK: low-logprob content dropped");
                    local.truncate(chunk_start);
                }

                let ready = emit_complete_sse_events(&mut local, &model_name, &mut usage);
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

        // Records once when this function returns (every branch below).
        let mut usage = ChatUsageTracker::new(
            self.usage.clone(),
            usage_base.clone(),
            payload,
            sent_at,
            client_wants_usage,
        );
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
            usage.observe_value(&v);
        }

        if let Ok(mut event) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
            // Unified model identity: rewrite backend model id.
            if let Some(model) = event.get_mut("model") {
                if model.as_str() != Some(cascade_model_name.as_str()) {
                    *model = serde_json::Value::String(cascade_model_name.clone());
                }
            }
            if let Some(choices) = event.get_mut("choices").and_then(|c| c.as_array_mut()) {
                if let Some(choice) = choices.get_mut(0) {
                    // Tool-call payloads are forwarded verbatim (model rewrite
                    // skipped) so client tool orchestration sees ids/functions
                    // exactly as the backend produced them.
                    if let Some(delta) = choice.get("delta") {
                        if delta.get("tool_calls").is_some() || delta.get("function_call").is_some() {
                            return Ok((headers, Body::from(body_bytes)));
                        }
                    }
                    // Reasoning (reasoning_content/reasoning/… ) is intentionally
                    // passed through UNMERGED — it lives in its own field so
                    // consumers that hide reasoning (Hermes show_reasoning:false,
                    // LibreChat) can do so. Never fold it into `content`.
                    if let Some(msg) = choice.get("message") {
                        if msg.get("tool_calls").is_some() || msg.get("function_call").is_some() {
                            return Ok((headers, Body::from(body_bytes)));
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
        _origin: &str,
    ) -> Result<(HeaderMap, Body), ProxyError> {
        let mut decision = self.inspect_route(&payload, headers).await;
        // Intent-based routing: planning/design/security/meta → force large model.
        let intent = intent_routing::inspect_intent(&payload);
        if intent.direct_to_large && matches!(decision, RouteDecision::Auxiliary | RouteDecision::Auto) {
            info!(
                "INTENT_ROUTING: intent={} → forcing Inference (was {:?})",
                intent.intent.label(),
                decision,
            );
            decision = RouteDecision::Inference;
        }
        // Settings: tools-bearing requests go to the large model (deterministic).
        let has_tools = payload.tools.is_some() || payload.functions.is_some();
        if has_tools && self.runtime.read().await.route_tools_to_large {
            if decision == RouteDecision::Auxiliary {
                info!("ROUTE_TOOLS_TO_LARGE: aux hint overridden -> Inference");
                decision = RouteDecision::Inference;
            }
        }
        info!(
            "ROUTE_OUTCOME: decision={:?}, model={}, messages={}, tier={}, streaming={}",
            decision,
            payload.model,
            payload.messages.len(),
            tier,
            is_streaming
        );

        match decision {
            RouteDecision::Ocr => self.route_ocr(&payload, is_streaming, headers).await,
            RouteDecision::Auxiliary => self.route_auxiliary(&payload, is_streaming, headers).await,
            RouteDecision::Inference => self.route_inference(&payload, is_streaming, headers).await,
            RouteDecision::Auto => self.route_auto(payload, is_streaming, tier, headers).await,
        }
    }

    #[allow(dead_code)]
    async fn private_proxy(
        &self,
        payload: &ChatCompletionRequest,
        url: &str,
        backend: RouteBackend,
        is_streaming: bool,
        headers: &HeaderMap,
    ) -> Result<(HeaderMap, Body), ProxyError> {
        self.private_proxy_with_bearer(payload, url, None, backend, is_streaming, headers, None)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn private_proxy_with_bearer(
        &self,
        payload: &ChatCompletionRequest,
        url: &str,
        bearer: Option<&str>,
        backend: RouteBackend,
        is_streaming: bool,
        headers: &HeaderMap,
        node_id: Option<&str>,
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
            let action = if let Some(id) = node_id {
                self.registry.read().await.get(id).and_then(|n| n.engine.as_deref().and_then(native_service_action))
            } else { None };
            return Err(ProxyError::unreachable(
                StatusCode::SERVICE_UNAVAILABLE,
                backend_name,
                format!("{} circuit breaker open (too many recent failures)", backend_name),
            ).with_service_action(action));
        }

        let usage_base = UsageEvent::from_headers(headers, "chat", metric_name);
        let mut header_latency = None;
        match self.proxy_to_backend(payload, url, bearer, is_streaming, usage_base, &mut header_latency).await {
            Ok(mut parts) => {
                if let Ok(v) = HeaderValue::from_str(backend_name) {
                    parts.0.insert("x-cascade-route", v);
                }
                if let Some(id) = node_id {
                    if let Ok(v) = HeaderValue::from_str(id) {
                        parts.0.insert("x-cascade-node", v);
                    }
                    if let Some(ms) = header_latency {
                        self.registry.write().await.record_request_latency(id, ms);
                    }
                }
                self.circuit_breaker.record_success(url).await;
                self.metrics.record_request(metric_name);
                info!("PROXY_OK: backend={}, url={}", backend_name, url);
                Ok(parts)
            }
            Err(e) => {
                // Only transport failures and 5xx indicate a sick backend; a
                // 4xx is a client error and must not open the breaker.
                if e.unreachable || e.status >= 500 {
                    self.circuit_breaker.record_failure(url).await;
                    self.metrics.record_fallback("backend_unavailable");
                }
                if e.unreachable {
                    let action = if let Some(id) = node_id {
                        self.registry.read().await.get(id).and_then(|n| n.engine.as_deref().and_then(native_service_action))
                    } else { None };
                    Err(ProxyError::unreachable(
                        StatusCode::SERVICE_UNAVAILABLE,
                        backend_name,
                        format!("{} server unavailable: {}", backend_name, e.context),
                    ).with_service_action(action))
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

    /// Resolve the effective upstream for a role. The dynamic registry wins
    /// (weighted round-robin over active + healthy nodes); the static config
    /// URL is the fallback when no node is registered.
    /// Static fallbacks carry LLAMA_API_KEY as bearer when set (None = no
    /// header, today's behavior).
    /// Returns (url, bearer, node_id).
    async fn resolve_upstream(&self, role: &str, default_url: &str) -> (String, Option<String>, Option<String>) {
        // Read-only: pick() advances an atomic counter, no writer needed.
        let registry = self.registry.read().await;
        if let Some(node) = registry.pick(role) {
            info!("UPSTREAM_NODE: {} -> {} ({})", role, node.endpoint_url, node.id);
            return (node.endpoint_url.clone(), node.bearer_token.clone(), Some(node.id));
        }
        (default_url.to_string(), self.config.llama_api_key.clone(), None)
    }

    /// Bearer + node id for an already-resolved endpoint URL. Used by the
    /// session-affinity path, which carries a URL (not a role) forward. Falls
    /// back to the static LLAMA_API_KEY for the configured backends.
    async fn resolve_target(&self, url: &str) -> (Option<String>, Option<String>) {
        {
            let registry = self.registry.read().await;
            for node in registry.nodes() {
                if node.endpoint_url == url {
                    return (node.bearer_token.clone(), Some(node.id.clone()));
                }
            }
        }
        if url == self.config.large_text_url || url == self.config.small_mllm_url {
            return (self.config.llama_api_key.clone(), None);
        }
        (None, None)
    }

    /// Picks a node from the registry without falling back to static config.
    /// Used by media/RAG handlers that need to know whether a node exists.
    pub async fn pick_node(&self, role: &str) -> Option<crate::registry::UpstreamNode> {
        self.registry.read().await.pick(role)
    }

    /// True when at least one active + healthy node is registered for the role.
    pub async fn has_active_role(&self, role: &str) -> bool {
        self.registry.read().await.has_active(role)
    }

    /// Hosted APIs (Pollinations etc.) don't know client model ids — when the
    /// target node declares a `model`, force it onto the outgoing payload.
    async fn apply_node_model(&self, payload: &mut ChatCompletionRequest, node_id: &Option<String>) {
        let Some(id) = node_id else { return };
        let model = self.registry.read().await.get(id).and_then(|n| n.model.clone());
        if let Some(m) = model {
            if !m.is_empty() && payload.model != m {
                info!("NODE_MODEL_OVERRIDE: {} -> {}", payload.model, m);
                payload.model = m;
            }
        }
    }

    /// The hybrid router picked the auxiliary role on its own, so the outgoing
    /// `model` must follow — otherwise the "auxiliary" request is just another
    /// main-model call (under OVMS both models live behind one URL and are
    /// selected purely by name, so a forwarded `netai-hybrid` would silently
    /// ignore the whole decision).
    ///
    /// Narrow on purpose: it only fires when the caller asked for the
    /// **configured main model** and `SMALL_MODEL_NAME` names a *different*
    /// one. An explicit `x-cascade-route: auxiliary-server` from a consumer that
    /// already sends the aux name is left alone, and any other name is still
    /// forwarded verbatim — the static-upstream pass-through that
    /// `test_static_upstream_forwards_a_foreign_model_name` pins. It is also a
    /// no-op whenever aux and main are the same model (the vLLM fold) or the
    /// engine ignores the field (llama.cpp rollback).
    async fn retarget_aux_model(&self, payload: &mut ChatCompletionRequest) {
        if !self.hybrid_enabled().await {
            return;
        }
        let main = self.config.main_model_name.trim();
        let aux = self.config.small_model_name.trim();
        if main.is_empty() || aux.is_empty() || aux == main || payload.model != main {
            return;
        }
        info!("AUX_MODEL_RETARGET: {} -> {} (hybrid routing)", payload.model, aux);
        payload.model = aux.to_string();
    }

    /// Condition A — dedicated vision / document parsing (OCR) endpoint.
    async fn route_ocr(
        &self,
        payload: &ChatCompletionRequest,
        is_streaming: bool,
        headers: &HeaderMap,
    ) -> Result<(HeaderMap, Body), ProxyError> {
        let mut payload = payload.clone();
        self.inline_all_images(&mut payload).await;
        let (url, bearer, node) = self.resolve_upstream("ocr", &self.config.ocr_server_url).await;
        self.apply_node_model(&mut payload, &node).await;
        info!(
            "ROUTE_A: OCR -> {} ({} messages)",
            url,
            payload.messages.len()
        );
        let _ = self.extract_image_urls(&payload.messages);
        self.private_proxy_with_bearer(&payload, &url, bearer.as_deref(), RouteBackend::OcrServer, is_streaming, headers, node.as_deref()).await
    }

    /// Condition B — agent context compression / fast text (auxiliary) endpoint.
    async fn route_auxiliary(
        &self,
        payload: &ChatCompletionRequest,
        is_streaming: bool,
        headers: &HeaderMap,
    ) -> Result<(HeaderMap, Body), ProxyError> {
        let mut payload = payload.clone();
        // If a remote image slipped in, inline it before forwarding.
        self.inline_all_images(&mut payload).await;
        let (url, bearer, node) = self.resolve_upstream("auxiliary", &self.config.small_mllm_url).await;
        self.apply_node_model(&mut payload, &node).await;
        self.retarget_aux_model(&mut payload).await;
        info!(
            "ROUTE_B: Auxiliary(compression) -> {} ({} messages)",
            url,
            payload.messages.len()
        );
        self.private_proxy_with_bearer(&payload, &url, bearer.as_deref(), RouteBackend::AuxiliaryServer, is_streaming, headers, node.as_deref()).await
    }

    /// Condition C — default chat & reasoning (main inference) endpoint.
    async fn route_inference(
        &self,
        payload: &ChatCompletionRequest,
        is_streaming: bool,
        headers: &HeaderMap,
    ) -> Result<(HeaderMap, Body), ProxyError> {
        if self.has_image(payload) && !self.config.large_model_multimodal {
            info!(
                "ROUTE_C: image detected but large model is text-only -> auxiliary multimodal backend"
            );
            return self.route_auxiliary(payload, is_streaming, headers).await;
        }

        let mut payload = payload.clone();
        // Qwythos-9B is multimodal — inline any remote image URLs before forwarding.
        self.inline_all_images(&mut payload).await;
        let lang = language::detect_language(&payload.messages);
        payload = language::inject_language_prompt(lang, payload);
        let (url, bearer, node) = self.resolve_upstream("main", &self.config.large_text_url).await;
        self.apply_node_model(&mut payload, &node).await;
        info!(
            "ROUTE_C: Inference(default) -> {} ({} messages, lang={})",
            url,
            payload.messages.len(),
            lang
        );
        self.private_proxy_with_bearer(&payload, &url, bearer.as_deref(), RouteBackend::InferenceServer, is_streaming, headers, node.as_deref()).await
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
    ) -> Result<(HeaderMap, Body), ProxyError> {
        // Intent-based bypass: planning/design/security/meta → skip small model entirely.
        let intent = intent_routing::inspect_intent(&payload);
        if intent.direct_to_large {
            let language = language::detect_language(&payload.messages);
            info!(
                "AUTO_INTENT: intent={} → routing directly to large model (skipping small)",
                intent.intent.label(),
            );
            let mut payload = payload.clone();
            self.inline_all_images(&mut payload).await;
            payload = language::inject_language_prompt(language, payload);
            let (url, bearer, node) = self.resolve_upstream("main", &self.config.large_text_url).await;
            self.apply_node_model(&mut payload, &node).await;
            return self.private_proxy_with_bearer(&payload, &url, bearer.as_deref(), RouteBackend::InferenceServer, is_streaming, headers, node.as_deref()).await;
        }
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

        let route_tools_to_large = self.runtime.read().await.route_tools_to_large;
        let mut target_override = if history_has_tools {
            // Tool-carrying conversations ALWAYS go to the large model — the
            // small aux backend struggles with multi-step tool orchestration.
            // This takes precedence over any cached small-model affinity.
            info!("SESSION AFFINITY: History has tools -> forcing large model target_url={}", large_url);
            if let Some(ref key) = session_key {
                self.session_cache
                    .insert(key.clone(), large_url.clone())
                    .await;
            }
            Some(large_url.clone())
        } else if let Some(url) = cached_route {
            info!("SESSION AFFINITY: candidate from cache target_url={}", url);
            Some(url)
        } else {
            None
        };

        // ROUTE_TOOLS_TO_LARGE (settings page): keep tool *selection* on the
        // big model even when the fresh payload (not the history) declares tools.
        if has_tools && route_tools_to_large && target_override.is_none() {
            info!("ROUTE_TOOLS_TO_LARGE: payload has tools, routing to large directly");
            target_override = Some(large_url.clone());
        }

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
            let (bearer, node) = self.resolve_target(&target).await;
            let backend = if target == self.config.small_mllm_url {
                RouteBackend::AuxiliaryServer
            } else {
                RouteBackend::InferenceServer
            };
            let mut payload = injected_payload.clone();
            self.apply_node_model(&mut payload, &node).await;
            let mut result = self
                .private_proxy_with_bearer(
                    &payload, &target, bearer.as_deref(), backend, is_streaming, headers,
                    node.as_deref(),
                )
                .await;
            if let Ok(parts) = result.as_mut() {
                if let Ok(v) = HeaderValue::from_str("session-affinity") {
                    parts.0.insert("x-cascade-route", v);
                }
            }
            self.metrics.record_request("session_affinity");
            return result;
        }

        let (use_small, static_target) = self.pick_model(has_image, complexity_score, tier).await;
        // Resolve through the dynamic registry (weighted round-robin) and carry
        // its bearer + circuit breaker. The static-config fallback uses
        // LLAMA_API_KEY. Previously this path posted straight to the static
        // URL with `bearer None`, bypassing auth and the breaker.
        let target_role = if use_small { "auxiliary" } else { "main" };
        let (target_url, target_bearer, target_node) =
            self.resolve_upstream(target_role, &static_target).await;
        let (large_url, large_bearer, large_node) =
            self.resolve_upstream("main", &self.config.large_text_url).await;
        info!("AUTO SELECTED_URL: {} (role={})", target_url, target_role);

        if !use_small {
            let mut payload = injected_payload.clone();
            self.apply_node_model(&mut payload, &target_node).await;
            let mut result = self
                .private_proxy_with_bearer(
                    &payload, &target_url, target_bearer.as_deref(),
                    RouteBackend::InferenceServer, is_streaming, headers,
                    target_node.as_deref(),
                )
                .await;
            if let Ok(parts) = result.as_mut() {
                if let Ok(v) = HeaderValue::from_str("auto-large") {
                    parts.0.insert("x-cascade-route", v);
                }
            }
            if result.is_ok() {
                if let Some(ref key) = session_key {
                    self.session_cache.insert(key.clone(), target_url.clone()).await;
                }
            }
            return result;
        }

        info!("AUTO SMALL_MODEL_PATH: target_url={}", target_url);
        let mut small_payload = injected_payload.clone();
        self.apply_node_model(&mut small_payload, &target_node).await;
        if small_payload.max_tokens.is_none() {
            small_payload.max_tokens = Some(4096);
        }
        // Small model gets the cascade fallback instruction: emit
        // <CASCADE_FALLBACK> when out of its depth (stream parser reroutes).
        language::append_fallback_instruction(&mut small_payload);

        if is_streaming {
            let mut result = self
                .private_proxy_with_bearer(
                    &small_payload, &target_url, target_bearer.as_deref(),
                    RouteBackend::AuxiliaryServer, true, headers, target_node.as_deref(),
                )
                .await;
            if let Ok(parts) = result.as_mut() {
                if let Ok(v) = HeaderValue::from_str("auto-small") {
                    parts.0.insert("x-cascade-route", v);
                }
            }
            if result.is_ok() {
                if let Some(ref key) = session_key {
                    self.session_cache.insert(key.clone(), target_url.clone()).await;
                }
            }
            return result;
        }

        small_payload.logprobs = Some(true);
        small_payload.top_logprobs = Some(0);
        if let Some(max_tokens) = injected_payload.max_tokens {
            small_payload.max_tokens = Some(max_tokens);
        }

        let mut small_request = self.http_client.post(&target_url).json(&small_payload);
        if let Some(token) = target_bearer.as_deref() {
            small_request = small_request.bearer_auth(token);
        }
        let backend_response = small_request
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
            let mut payload = injected_payload.clone();
            self.apply_node_model(&mut payload, &large_node).await;
            let mut result = self
                .private_proxy_with_bearer(
                    &payload, &large_url, large_bearer.as_deref(),
                    RouteBackend::InferenceServer, false, headers, large_node.as_deref(),
                )
                .await;
            if let Ok(parts) = result.as_mut() {
                if let Ok(v) = HeaderValue::from_str("auto-large") {
                    parts.0.insert("x-cascade-route", v);
                }
            }
            if result.is_ok() {
                if let Some(ref key) = session_key {
                    self.session_cache.insert(key.clone(), large_url.clone()).await;
                }
            }
            return result;
        }

        self.circuit_breaker.record_success(&target_url).await;

        // Shared reroute helper: re-sends the ORIGINAL (language-injected,
        // pre-small) payload to the large model and tags the response.
        let reroute_to_large = async {
            self.metrics.record_fallback("quality_low");
            let mut payload = injected_payload.clone();
            self.apply_node_model(&mut payload, &large_node).await;
            let mut result = self
                .private_proxy_with_bearer(
                    &payload, &large_url, large_bearer.as_deref(),
                    RouteBackend::InferenceServer, false, headers, large_node.as_deref(),
                )
                .await;
            if let Ok(parts) = result.as_mut() {
                if let Ok(v) = HeaderValue::from_str("auto-large") {
                    parts.0.insert("x-cascade-route", v);
                }
            }
            if result.is_ok() {
                if let Some(ref key) = session_key {
                    self.session_cache.insert(key.clone(), large_url.clone()).await;
                }
            }
            result
        };

        // The small model may have declared the task beyond its capability
        // (<CASCADE_FALLBACK> tag from get_small_model_system_prompt) — in
        // that case reroute the ORIGINAL request to the large model.
        let body_text = std::str::from_utf8(&body_bytes).unwrap_or_default();
        if body_text.contains(crate::intent_routing::FALLBACK_TAG) {
            info!("SMALL MODEL fallback tag detected, rerouting original request to large model");
            return reroute_to_large.await;
        }

        let confidence = self.extract_confidence(&body_bytes);
        let keep_small = match confidence {
            Some(c) if c >= self.runtime.read().await.confidence_threshold => {
                info!("SMALL MODEL CONFIDENCE: {:.4} >= threshold, keeping", c);
                true
            }
            Some(c) => {
                info!("SMALL MODEL CONFIDENCE: {:.4} < threshold, rerouting to large", c);
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
            // Documented response header (docs/routing-settings.md): the
            // small-success path used to return no x-cascade-route at all,
            // making adaptive routing invisible to callers.
            hdrs.insert("x-cascade-route", HeaderValue::from_static("auto-small"));
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
        reroute_to_large.await
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
    pub(crate) async fn select_extraction_backend(&self) -> Option<ExtractionBackendEntry> {
        let enabled: Vec<ExtractionBackendEntry> = {
            let backends = self.extraction_backends.read().await;
            backends
                .iter()
                .filter(|b| b.enabled && b.healthy)
                .cloned()
                .collect()
        };

        // Drop circuit-open backends before selection.
        let mut candidates: Vec<ExtractionBackendEntry> = Vec::with_capacity(enabled.len());
        for backend in enabled {
            if self.circuit_breaker.is_open(&backend.url).await {
                info!(
                    "EXTRACT_SKIP: backend '{}' (priority {}) circuit breaker open",
                    backend.id, backend.priority
                );
                continue;
            }
            candidates.push(backend);
        }

        if candidates.is_empty() {
            warn!("EXTRACT: no healthy extraction backend available");
            return None;
        }

        match self.config.extraction_strategy {
            ExtractionStrategy::Priority => {
                candidates.sort_by_key(|b| b.priority);
                candidates.into_iter().next()
            }
            ExtractionStrategy::RoundRobin => {
                // Deterministic pool order so the cursor maps to a stable node.
                candidates.sort_by(|a, b| a.id.cmp(&b.id));
                let expanded: Vec<&ExtractionBackendEntry> = candidates
                    .iter()
                    .flat_map(|b| std::iter::repeat_n(b, b.weight.max(1) as usize))
                    .collect();
                let idx = self.extraction_counter.fetch_add(1, Ordering::Relaxed);
                let picked = expanded[(idx as usize) % expanded.len()].clone();
                info!(
                    "EXTRACT_RR: cycle={} picked='{}' (priority {})",
                    idx, picked.id, picked.priority
                );
                Some(picked)
            }
        }
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
        request_headers: &HeaderMap,
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
        // Usage metering (same rule as `proxy_to_backend`): streams always ask
        // for the terminal usage chunk; the SSE rewriter strips it again for
        // clients that did not opt in. Non-streamed calls carry no stream_options.
        clean.stream_options = if is_streaming {
            Some(serde_json::json!({"include_usage": true}))
        } else {
            None
        };
        let client_wants_usage = payload.client_wants_stream_usage();
        let usage_base = UsageEvent::from_headers(request_headers, "chat", "extraction");

        // Forward with API key if provided
        let mut req_builder = self.http_client.post(&url).json(&clean);
        if let Some(api_key) = &backend.api_key {
            if !api_key.is_empty() {
                req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
            }
        }

        let sent_at = Instant::now();
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
            // Owned by the stream closure: records one usage event when the
            // stream ends or the client disconnects.
            let mut usage = ChatUsageTracker::new(
                self.usage.clone(),
                usage_base,
                &payload,
                sent_at,
                client_wants_usage,
            );
            let mut pending: Vec<u8> = Vec::new();
            let stream = backend_response.bytes_stream().map(move |item| {
                let chunk = item.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                pending.extend_from_slice(&chunk);
                let mut out: Vec<u8> = Vec::new();
                while let Some(idx) = pending.windows(2).position(|w| w == b"\n\n") {
                    let frame: Vec<u8> = pending.drain(..idx + 2).collect();
                    let text = String::from_utf8_lossy(&frame);
                    if let Some(kept) = usage.observe_frame(text.trim_end()) {
                        out.extend_from_slice(kept.as_bytes());
                        out.extend_from_slice(b"\n\n");
                    }
                }
                Ok::<_, std::io::Error>(axum::body::Bytes::from(out))
            });
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

        // Records once when this function returns.
        let mut usage = ChatUsageTracker::new(
            self.usage.clone(),
            usage_base,
            &payload,
            sent_at,
            client_wants_usage,
        );
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
            usage.observe_value(&v);
        }

        Ok((headers, Body::from(body_bytes)))
    }

    // =========================================================================
    // RAG WORKER ROUTING  (POST /v1/rag/extract)
    // =========================================================================

    /// Forwards a LightRAG graph-extraction job to an active `rag_worker`
    /// node. The body is passed through untouched (extraction payloads are
    /// worker-specific); responses are streamed back. Returns `None`-style
    /// 503 upstream when no node is registered so callers/UI can react.
    pub async fn route_rag_extract(
        &self,
        content_type: Option<&str>,
        body: axum::body::Bytes,
    ) -> Result<(HeaderMap, Body, String), ProxyError> {
        let node = self.pick_node("rag_worker").await.ok_or_else(|| {
            ProxyError::unreachable(
                StatusCode::SERVICE_UNAVAILABLE,
                "rag_worker",
                "no active rag_worker node registered (provision one via PUT /web/api/upstreams/rag_worker)",
            )
        })?;

        if self.circuit_breaker.is_open(&node.endpoint_url).await {
            self.metrics.record_fallback("backend_unavailable");
            return Err(ProxyError::unreachable(
                StatusCode::SERVICE_UNAVAILABLE,
                "rag_worker",
                format!("rag_worker '{}' circuit breaker open", node.id),
            ));
        }

        let mut request = self.http_client.post(&node.endpoint_url).body(body);
        if let Some(token) = &node.bearer_token {
            request = request.bearer_auth(token);
        }
        let ct = content_type.unwrap_or("application/json");
        request = request.header("content-type", ct);

        let backend_response = match request.send().await {
            Ok(resp) => resp,
            Err(e) => {
                self.circuit_breaker
                    .record_failure(&node.endpoint_url)
                    .await;
                return Err(ProxyError::unreachable(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "rag_worker",
                    format!("rag_worker unreachable: {}", e),
                ));
            }
        };

        let status = backend_response.status();
        if !status.is_success() {
            self.circuit_breaker.record_failure(&node.endpoint_url).await;
            let err_body = backend_response.text().await.unwrap_or_default();
            return Err(ProxyError::new(
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY).as_u16(),
                "rag_worker",
                format!("rag_worker returned HTTP {}: {}", status, err_body.chars().take(300).collect::<String>()),
            ));
        }

        self.circuit_breaker.record_success(&node.endpoint_url).await;
        self.metrics.record_request("rag_worker");

        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            HeaderValue::from_str(
                backend_response
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("application/json"),
            )
            .unwrap_or(HeaderValue::from_static("application/json")),
        );
        if let Ok(v) = HeaderValue::from_str(&node.id) {
            headers.insert("x-cascade-node", v);
        }
        headers.insert("x-cascade-route", HeaderValue::from_static("rag-worker"));

        let stream = backend_response
            .bytes_stream()
            .map(|item| item.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)));
        Ok((headers, Body::from_stream(stream), node.id))
    }

}

/// Normalize a single SSE chunk forwarded to the client:
/// - rewrite the backend model id to the unified `cascade_model_name`
/// - move `reasoning_content`/`reasoning` deltas into `content`
/// Returns `None` for non-chunk frames (keepalives, `[DONE]`).
fn normalize_event_json(raw: &str, model_name: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return None;
    }
    let trimmed = trimmed.strip_prefix("data: ").unwrap_or(trimmed);
    let trimmed = trimmed.strip_prefix("data:").unwrap_or(trimmed).trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return None;
    }

    let Ok(mut event) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        // Not a JSON chat chunk. OpenAI-compatible clients JSON.parse every
        // `data:` event, so forwarding raw non-JSON (llama.cpp SSE keep-alive
        // comments like a bare `:`) made LibreChat agents fatal:
        // "Unexpected token ':' is not valid JSON". Drop instead of wrapping.
        return None;
    };

    // Unified model identity: rewrite backend model id in every chunk.
    if let Some(model) = event.get_mut("model") {
        if model.as_str() != Some(model_name) {
            *model = serde_json::Value::String(model_name.to_string());
        }
    }

    // Delta lives under choices[0].delta in the OpenAI SSE shape.
    let Some(delta) = event
        .get_mut("choices")
        .and_then(|c| c.as_array_mut())
        .and_then(|c| c.get_mut(0))
        .and_then(|c| c.get_mut("delta"))
    else {
        return Some(format!("data: {}", serde_json::to_string(&event).unwrap_or_default()));
    };

    // Keep the NORMALIZE trace log minimal: log shapes, never reasoning
    // content (chain-of-thought must not land in gateway logs).
    if delta.get("reasoning_content").is_some() || delta.get("reasoning").is_some() {
        info!(target: "cascade_llm::state", "NORMALIZE chunk with reasoning delta (passthrough)");
    }

    if delta.get("tool_calls").is_some() || delta.get("function_call").is_some() {
        return Some(format!("data: {}", serde_json::to_string(&event).unwrap_or_default()));
    }

    if delta.get("role").is_none() {
        delta["role"] = serde_json::Value::String("assistant".to_string());
    }

    // Reasoning/thinking is deliberately NOT merged into `content`.  The small
    // and large backends surface chain-of-thought via `reasoning_content`
    // (llama.cpp Qwen/Gemma-style); OpenAI-compatible clients like Hermes
    // render only `content` and optionally expose reasoning separately
    // (show_reasoning).  Merging here leaks the whole CoT into the visible
    // answer (regression 2026-09-10: Hermes "Opencode ssh" session).
    Some(format!("data: {}", serde_json::to_string(&event).unwrap_or_default()))
}

/// Usage event skeleton for a non-chat call proxied by a handler (embeddings,
/// rerank, audio, image). The caller fills the kind-specific counters.
pub(crate) fn passthrough_usage(
    headers: &HeaderMap,
    kind: &str,
    route: &str,
    status: u16,
    started: Instant,
) -> UsageEvent {
    let mut ev = UsageEvent::from_headers(headers, kind, route);
    ev.status = status;
    ev.latency_ms = started.elapsed().as_millis() as u64;
    ev
}

/// Estimated characters per token for the usage fallback when the upstream
/// returned no usage block (matches the `chars/4` rule used across the stack).
const USAGE_CHARS_PER_TOKEN: u64 = 4;

/// Characters of prompt text in a chat request: message text parts, tool-call
/// payloads and the tool/function schemas. Used only for the estimate fallback.
fn prompt_char_len(payload: &ChatCompletionRequest) -> u64 {
    let mut n: u64 = 0;
    for m in &payload.messages {
        n += match &m.content {
            Some(MessageContent::Text(t)) => t.chars().count() as u64,
            Some(MessageContent::Parts(parts)) => parts
                .iter()
                .map(|p| match p {
                    MessageContentPart::Text { text } => text.chars().count() as u64,
                    _ => 0,
                })
                .sum(),
            None => 0,
        };
        if let Some(tc) = &m.tool_calls {
            n += tc.to_string().chars().count() as u64;
        }
    }
    for schema in [&payload.tools, &payload.functions].into_iter().flatten() {
        n += schema.to_string().chars().count() as u64;
    }
    n
}

/// Usage metering for one chat call (streamed or not). Captures the upstream's
/// usage block when present, otherwise counts completion text for an estimate.
/// Records exactly one `UsageEvent` when dropped, which for a stream is when
/// the stream finishes or the client disconnects.
pub(crate) struct ChatUsageTracker {
    recorder: Arc<crate::usage::UsageRecorder>,
    event: UsageEvent,
    started: Instant,
    client_wants_usage: bool,
    prompt_chars: u64,
    usage_prompt: Option<u64>,
    usage_completion: Option<u64>,
    completion_chars: u64,
    served_model: Option<String>,
}

impl ChatUsageTracker {
    fn new(
        recorder: Arc<crate::usage::UsageRecorder>,
        mut event: UsageEvent,
        payload: &ChatCompletionRequest,
        started: Instant,
        client_wants_usage: bool,
    ) -> Self {
        event.model = payload.model.clone();
        Self {
            recorder,
            event,
            started,
            client_wants_usage,
            prompt_chars: prompt_char_len(payload),
            usage_prompt: None,
            usage_completion: None,
            completion_chars: 0,
            served_model: None,
        }
    }

    /// Inspect one parsed chunk or full response body.
    fn observe_value(&mut self, v: &serde_json::Value) {
        if self.served_model.is_none() {
            if let Some(m) = v.get("model").and_then(|m| m.as_str()) {
                self.served_model = Some(m.to_string());
            }
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            if let Some(p) = u.get("prompt_tokens").and_then(|x| x.as_u64()) {
                self.usage_prompt = Some(p);
            }
            if let Some(c) = u.get("completion_tokens").and_then(|x| x.as_u64()) {
                self.usage_completion = Some(c);
            }
        }
        // Choice-less frames (the usage-only chunk) simply contribute nothing here.
        let Some(choices) = v.get("choices").and_then(|c| c.as_array()) else {
            return;
        };
        for choice in choices {
            let Some(msg) = choice.get("delta").or_else(|| choice.get("message")) else {
                continue;
            };
            for key in ["content", "reasoning_content", "reasoning"] {
                if let Some(s) = msg.get(key).and_then(|s| s.as_str()) {
                    self.completion_chars += s.chars().count() as u64;
                }
            }
            for key in ["tool_calls", "function_call"] {
                if let Some(tc) = msg.get(key).filter(|t| !t.is_null()) {
                    self.completion_chars += tc.to_string().chars().count() as u64;
                }
            }
        }
    }

    /// Inspect one SSE `data:` frame before it is normalized. Returns the text
    /// to keep, or `None` to drop the frame. Frames this cannot parse pass
    /// through unchanged (the normalizer drops non-JSON frames itself).
    ///
    /// Usage-only frames (`"choices": []` plus a `usage` block) are dropped for
    /// clients that did not ask for `stream_options.include_usage`, so they see
    /// the same stream as before. A frame that carries both content and usage
    /// keeps its content and loses only the `usage` key.
    fn observe_frame(&mut self, raw: &str) -> Option<String> {
        let body = raw.trim();
        let body = body.strip_prefix("data:").unwrap_or(body).trim();
        if body.is_empty() || body == "[DONE]" {
            return Some(raw.to_string());
        }
        let Ok(mut v) = serde_json::from_str::<serde_json::Value>(body) else {
            return Some(raw.to_string());
        };
        self.observe_value(&v);
        let usage_present = v.get("usage").map_or(false, |u| !u.is_null());
        if !usage_present || self.client_wants_usage {
            return Some(raw.to_string());
        }
        let choice_less = v
            .get("choices")
            .and_then(|c| c.as_array())
            .map_or(true, |a| a.is_empty());
        if choice_less {
            return None;
        }
        if let Some(obj) = v.as_object_mut() {
            obj.remove("usage");
        }
        Some(format!("data: {}", v))
    }
}

impl Drop for ChatUsageTracker {
    fn drop(&mut self) {
        let (prompt_tokens, prompt_est) = match self.usage_prompt {
            Some(p) => (p, false),
            None => (self.prompt_chars / USAGE_CHARS_PER_TOKEN, true),
        };
        let (completion_tokens, completion_est) = match self.usage_completion {
            Some(c) => (c, false),
            None => (self.completion_chars / USAGE_CHARS_PER_TOKEN, true),
        };
        let mut ev = self.event.clone();
        if let Some(m) = self.served_model.take() {
            ev.model = m;
        }
        ev.prompt_tokens = prompt_tokens;
        ev.completion_tokens = completion_tokens;
        ev.estimated = prompt_est || completion_est;
        ev.status = 200;
        ev.latency_ms = self.started.elapsed().as_millis() as u64;
        self.recorder.record(ev);
    }
}

#[cfg(test)]
mod normalize_tests {
    use super::normalize_event_json;

    #[test]
    fn rewrites_model_and_preserves_reasoning_content() {
        let chunk = r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"reasoning_content":"\nThe"}}],"created":1788978989,"id":"chatcmpl-x","model":"/models/gguf.backend.gguf","object":"chat.completion.chunk"}"#;
        let out = normalize_event_json(chunk, "cascade-hybrid-v1").unwrap();
        assert!(out.contains("\"model\":\"cascade-hybrid-v1\""), "model rewritten: {out}");
        assert!(
            out.contains("reasoning_content"),
            "reasoning must stay separate (NOT merged into content): {out}"
        );
        assert!(
            !out.contains("\"content\":\"\\nThe\""),
            "CoT must not leak into content: {out}"
        );
    }

    #[test]
    fn preserves_other_reasoning_field_names() {
        let chunk = r#"data: {"choices":[{"delta":{"reasoning":"<thinking>plan</thinking>"}}],"id":"chatcmpl-x","model":"/models/raw.gguf","object":"chat.completion.chunk"}"#;
        let out = normalize_event_json(chunk, "cascade-hybrid-v1").unwrap();
        assert!(out.contains("\"reasoning\":\"<thinking>"));
        assert!(!out.contains("\"content\":\"<thinking>"));
    }

    #[test]
    fn rewrites_model_on_plain_content_chunk() {
        let chunk = r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"role":"assistant","content":null}}],"created":1,"id":"chatcmpl-x","model":"/models/raw.gguf","object":"chat.completion.chunk"}"#;
        let out = normalize_event_json(chunk, "cascade-hybrid-v1").unwrap();
        assert!(out.contains("\"model\":\"cascade-hybrid-v1\""));
        assert!(!out.contains("/models/raw.gguf"));
    }

    #[test]
    fn rewrites_model_on_tool_call_chunk() {
        let chunk = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1}"}}]},"finish_reason":null,"index":0}],"id":"chatcmpl-x","model":"/models/raw.gguf","object":"chat.completion.chunk"}"#;
        let out = normalize_event_json(chunk, "cascade-hybrid-v1").unwrap();
        assert!(out.contains("\"model\":\"cascade-hybrid-v1\""));
        assert!(out.contains("tool_calls"));
    }

    #[test]
    fn drops_non_json_frames() {
        assert!(normalize_event_json("data: not json at all", "cascade-hybrid-v1").is_none());
        assert!(normalize_event_json(":", "cascade-hybrid-v1").is_none());
        assert!(normalize_event_json(": keepalive", "cascade-hybrid-v1").is_none());
        assert!(normalize_event_json("data: :", "cascade-hybrid-v1").is_none());
    }

    #[test]
    fn drops_done_frame() {
        assert!(normalize_event_json("data: [DONE]", "cascade-hybrid-v1").is_none());
        assert!(normalize_event_json("", "cascade-hybrid-v1").is_none());
    }

    #[test]
    fn preserves_reasoning_model_rewrite_when_no_reasoning_key() {
        // Backend /v1/completions chunks (non-chat) have no delta at all —
        // model rewrite must still be applied and serialized.
        let chunk = r#"data: {"choices":[{"text":"Hello","finish_reason":"length","index":0}],"model":"/models/base.bin","object":"text_completion","id":"cmpl"}"#;
        let out = normalize_event_json(chunk, "cascade-hybrid-v1").unwrap();
        assert!(out.contains("\"model\":\"cascade-hybrid-v1\""));
        assert!(out.contains("\"text\":\"Hello\""));
    }
}

#[cfg(test)]
mod route_hint_tests {
    use super::{GatewayState, RouteDecision};

    // GatewayState::hint_to_decision is the single mapping used by both header and
    // body route hints. The canonical values documented in the settings page
    // and docs/routing-settings.md must resolve to their real backends.
    #[test]
    fn canonical_resource_names_map_to_expected_decisions() {
        use RouteDecision::*;
        for (hint, want) in [
            ("inference-server", Inference),
            ("main", Inference),
            ("large", Inference),
            ("auxiliary-server", Auxiliary),
            ("small", Auxiliary),
            ("auxiliary", Auxiliary),
            ("compression", Auxiliary),
            ("auto-small", Auto),
            ("auto-large", Auto),
            ("session-affinity", Auto),
            ("auto", Auto),
            ("ocr", Ocr),
            ("vision", Ocr),
            ("rag-worker", Inference),
        ] {
            assert_eq!(
                GatewayState::hint_to_decision(hint),
                want,
                "hint {hint:?} mapped incorrectly"
            );
        }
    }

    #[test]
    fn is_case_sensitive_after_lowercasing_callers() {
        // header_route_hint lowercases before calling; document that contract.
        assert_eq!(
            GatewayState::hint_to_decision("auxiliary-server"),
            RouteDecision::Auxiliary
        );
    }
}

#[cfg(test)]
mod extraction_routing_tests {
    use super::*;
    use crate::cascade_features::MetricsRegistry;
    use crate::db::Db;

    fn backend(id: &str, priority: u32, weight: u32) -> ExtractionBackendEntry {
        ExtractionBackendEntry {
            id: id.to_string(),
            backend_type: ExtractionBackendType::Local,
            name: id.to_string(),
            url: format!("http://{id}.test:8080"),
            model: None,
            api_key: None,
            enabled: true,
            priority,
            weight,
            max_cost_per_hour: None,
            last_validated: None,
            healthy: true,
        }
    }

    fn state_with(strategy: ExtractionStrategy) -> GatewayState {
        let mut cfg = AppConfig::from_env();
        cfg.extraction_strategy = strategy;
        GatewayState::new(
            cfg,
            Arc::new(MetricsRegistry::init()),
            Arc::new(Db::new_in_memory().unwrap()),
        )
    }

    async fn picks(state: &GatewayState, n: usize) -> Vec<String> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(state.select_extraction_backend().await.unwrap().id);
        }
        out
    }

    #[test]
    fn strategy_parses_aliases() {
        assert_eq!(ExtractionStrategy::parse("round_robin"), Some(ExtractionStrategy::RoundRobin));
        assert_eq!(ExtractionStrategy::parse("priority"), Some(ExtractionStrategy::Priority));
        assert_eq!(ExtractionStrategy::parse("bogus"), None);
        assert_eq!(ExtractionStrategy::default(), ExtractionStrategy::Priority);
    }

    #[tokio::test]
    async fn priority_mode_always_prefers_lower_priority() {
        let state = state_with(ExtractionStrategy::Priority);
        state.register_extraction_backend(backend("gpu", 10, 1)).await;
        state.register_extraction_backend(backend("local", 99, 1)).await;
        assert_eq!(picks(&state, 4).await, vec!["gpu", "gpu", "gpu", "gpu"]);
    }

    #[tokio::test]
    async fn round_robin_alternates_local_and_gpu() {
        let state = state_with(ExtractionStrategy::RoundRobin);
        state.register_extraction_backend(backend("gpu", 10, 1)).await;
        state.register_extraction_backend(backend("local", 99, 1)).await;
        assert_eq!(picks(&state, 4).await, vec!["gpu", "local", "gpu", "local"]);
    }

    #[tokio::test]
    async fn round_robin_honours_weight() {
        let state = state_with(ExtractionStrategy::RoundRobin);
        state.register_extraction_backend(backend("gpu", 10, 2)).await;
        state.register_extraction_backend(backend("local", 99, 1)).await;
        let got = picks(&state, 6).await;
        assert_eq!(got.iter().filter(|p| *p == "gpu").count(), 4);
        assert_eq!(got.iter().filter(|p| *p == "local").count(), 2);
    }

    #[tokio::test]
    async fn round_robin_skips_unhealthy_backend() {
        let state = state_with(ExtractionStrategy::RoundRobin);
        state.register_extraction_backend(backend("gpu", 10, 1)).await;
        let mut local = backend("local", 99, 1);
        local.healthy = false;
        state.register_extraction_backend(local).await;
        assert_eq!(picks(&state, 3).await, vec!["gpu", "gpu", "gpu"]);
    }
}

#[cfg(test)]
mod usage_stream_tests {
    use super::{normalize_event_json, ChatUsageTracker};
    use crate::types::ChatCompletionRequest;
    use crate::usage::{UsageEvent, UsageRecorder};
    use serde_json::json;
    use std::time::Instant;

    fn payload(stream_options: Option<serde_json::Value>) -> ChatCompletionRequest {
        let mut v = json!({
            "model": "client-model",
            "messages": [{"role": "user", "content": "abcdefgh"}],
            "stream": true,
        });
        if let Some(o) = stream_options {
            v["stream_options"] = o;
        }
        serde_json::from_value(v).unwrap()
    }

    fn tracker(
        wants: bool,
    ) -> (ChatUsageTracker, tokio::sync::mpsc::UnboundedReceiver<UsageEvent>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let t = ChatUsageTracker::new(
            UsageRecorder::with_sender(tx),
            UsageEvent::default(),
            &payload(None),
            Instant::now(),
            wants,
        );
        (t, rx)
    }

    const CONTENT: &str = r#"data: {"id":"c1","model":"/models/backend.gguf","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#;
    const USAGE_ONLY: &str = r#"data: {"id":"c1","model":"/models/backend.gguf","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}}"#;

    #[test]
    fn usage_only_frame_is_dropped_for_client_that_did_not_opt_in() {
        let (mut t, _rx) = tracker(false);
        assert!(t.observe_frame(CONTENT).is_some());
        assert!(t.observe_frame(USAGE_ONLY).is_none());
        assert_eq!(t.usage_prompt, Some(12));
        assert_eq!(t.usage_completion, Some(5));
    }

    #[test]
    fn usage_only_frame_is_passed_through_for_client_that_opted_in() {
        let (mut t, _rx) = tracker(true);
        let kept = t.observe_frame(USAGE_ONLY).expect("opted-in client keeps usage frame");
        assert_eq!(kept, USAGE_ONLY);
        let normalized = normalize_event_json(&kept, "cascade-hybrid-v1").unwrap();
        assert!(normalized.contains("\"model\":\"cascade-hybrid-v1\""));
        assert!(normalized.contains("\"prompt_tokens\":12"));
        assert!(normalized.contains("\"choices\":[]"));
    }

    #[test]
    fn usage_on_a_content_frame_is_stripped_but_content_kept() {
        let frame = r#"data: {"model":"m","choices":[{"index":0,"delta":{"content":"end"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}"#;
        let (mut t, _rx) = tracker(false);
        let kept = t.observe_frame(frame).expect("content frame must survive");
        assert!(!kept.contains("usage"), "usage removed for old clients: {kept}");
        assert!(kept.contains("\"content\":\"end\""));
        assert_eq!(t.usage_completion, Some(1));
    }

    #[test]
    fn choiceless_frame_without_usage_is_not_dropped() {
        let (mut t, _rx) = tracker(false);
        let frame = r#"data: {"id":"c1","choices":[]}"#;
        assert_eq!(t.observe_frame(frame).as_deref(), Some(frame));
    }

    #[test]
    fn malformed_and_done_frames_pass_through_unchanged() {
        let (mut t, _rx) = tracker(false);
        assert_eq!(t.observe_frame("data: not json at all").as_deref(), Some("data: not json at all"));
        assert_eq!(t.observe_frame(": keepalive").as_deref(), Some(": keepalive"));
        assert_eq!(t.observe_frame("data: [DONE]").as_deref(), Some("data: [DONE]"));
    }

    #[test]
    fn recorded_event_uses_upstream_usage_when_present() {
        let (t, mut rx) = tracker(false);
        let mut t = t;
        t.observe_frame(CONTENT);
        t.observe_frame(USAGE_ONLY);
        drop(t);
        let ev = rx.try_recv().expect("one usage event on drop");
        assert_eq!(ev.prompt_tokens, 12);
        assert_eq!(ev.completion_tokens, 5);
        assert!(!ev.estimated);
        assert_eq!(ev.model, "/models/backend.gguf");
        assert_eq!(ev.status, 200);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn recorded_event_estimates_when_upstream_sent_no_usage() {
        let (t, mut rx) = tracker(false);
        let mut t = t;
        // "abcdefgh" = 8 prompt chars -> 2 tokens; "Hi" + "there" = 7 chars -> 1 token.
        t.observe_frame(CONTENT);
        t.observe_frame(r#"data: {"model":"m","choices":[{"index":0,"delta":{"content":"there"}}]}"#);
        drop(t);
        let ev = rx.try_recv().unwrap();
        assert!(ev.estimated);
        assert_eq!(ev.prompt_tokens, 2);
        assert_eq!(ev.completion_tokens, 1);
        // First served model seen wins (the first frame names the backend).
        assert_eq!(ev.model, "/models/backend.gguf");
    }

    #[test]
    fn non_stream_body_is_metered_from_its_usage_block() {
        let (mut t, mut rx) = tracker(false);
        t.observe_value(&json!({
            "model": "m",
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2}
        }));
        drop(t);
        let ev = rx.try_recv().unwrap();
        assert_eq!((ev.prompt_tokens, ev.completion_tokens, ev.estimated), (7, 2, false));
    }

    #[test]
    fn prompt_estimate_counts_text_parts_and_tools() {
        let p: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "abcd"}, {"type": "image_url", "image_url": {"url": "x"}}]}],
            "tools": [{"type": "function"}]
        }))
        .unwrap();
        assert!(super::prompt_char_len(&p) >= 4);
    }
}

#[cfg(test)]
mod extraction_usage_tests {
    use super::*;
    use crate::cascade_features::MetricsRegistry;
    use crate::db::Db;
    use crate::types::{ExtractionBackendType, ExtractionBackendEntry};
    use crate::usage::{UsageEvent, UsageRecorder};
    use axum::http::HeaderValue;
    use axum::response::IntoResponse;
    use axum::routing::post;
    use axum::Json;
    use serde_json::json;
    use std::sync::Mutex;

    const STREAM_BODY: &str = "data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7,\"total_tokens\":18}}\n\n\
data: [DONE]\n\n";

    /// Mock OpenAI-compatible backend. Records the `stream_options` it received.
    async fn spawn_backend(seen: Arc<Mutex<Vec<Option<serde_json::Value>>>>) -> String {
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            post(move |Json(body): Json<serde_json::Value>| {
                let seen = seen.clone();
                async move {
                    seen.lock().unwrap().push(body.get("stream_options").cloned());
                    if body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false) {
                        ([("content-type", "text/event-stream")], STREAM_BODY.to_string()).into_response()
                    } else {
                        Json(json!({
                            "id": "c2", "model": "m", "object": "chat.completion",
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
                        }))
                        .into_response()
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn state_with_backend(url: String) -> (GatewayState, tokio::sync::mpsc::UnboundedReceiver<UsageEvent>) {
        let mut cfg = AppConfig::from_env();
        cfg.extraction_strategy = ExtractionStrategy::Priority;
        let mut state = GatewayState::new(
            cfg,
            Arc::new(MetricsRegistry::init()),
            Arc::new(Db::new_in_memory().unwrap()),
        );
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        state.usage = UsageRecorder::with_sender(tx);
        state
            .register_extraction_backend(ExtractionBackendEntry {
                id: "mock-extract".to_string(),
                backend_type: ExtractionBackendType::Local,
                name: "mock".to_string(),
                url,
                model: None,
                api_key: None,
                enabled: true,
                priority: 1,
                weight: 1,
                max_cost_per_hour: None,
                last_validated: None,
                healthy: true,
            })
            .await;
        (state, rx)
    }

    fn billing_headers() -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert("x-billing-user-id", HeaderValue::from_static("user-42"));
        h.insert("x-billing-tenant", HeaderValue::from_static("tenant-a"));
        h.insert("x-billing-source", HeaderValue::from_static("knowledge-engine"));
        h
    }

    fn request(stream: bool) -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "extract-model",
            "messages": [{"role": "user", "content": "extract entities"}],
            "stream": stream,
        }))
        .unwrap()
    }

    fn assert_attribution(ev: &UsageEvent) {
        assert_eq!(ev.kind, "chat");
        assert_eq!(ev.route, "extraction");
        assert_eq!(ev.user_id.as_deref(), Some("user-42"));
        assert_eq!(ev.tenant_id.as_deref(), Some("tenant-a"));
        assert_eq!(ev.source.as_deref(), Some("knowledge-engine"));
        assert_eq!(ev.prompt_tokens, 11);
        assert_eq!(ev.completion_tokens, 7);
        assert!(!ev.estimated);
        assert_eq!(ev.status, 200);
    }

    #[tokio::test]
    async fn non_stream_extraction_emits_exactly_one_usage_event() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (state, mut rx) = state_with_backend(spawn_backend(seen.clone()).await).await;

        let (_hdrs, body) = state
            .route_extraction(request(false), false, &billing_headers())
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        assert!(!bytes.is_empty());

        let ev = rx.recv().await.expect("one usage event");
        assert_attribution(&ev);
        assert!(rx.try_recv().is_err(), "exactly one event");
        assert_eq!(seen.lock().unwrap().as_slice(), &[None], "no stream_options on non-stream");
    }

    #[tokio::test]
    async fn stream_extraction_forces_usage_and_emits_exactly_one_event() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (state, mut rx) = state_with_backend(spawn_backend(seen.clone()).await).await;

        let (_hdrs, body) = state
            .route_extraction(request(true), true, &billing_headers())
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        let out = String::from_utf8(bytes.to_vec()).unwrap();

        // Client did not opt in: the usage-only chunk is stripped, content and DONE kept.
        assert!(out.contains("\"content\":\"Hi\""), "{out}");
        assert!(out.contains("[DONE]"), "{out}");
        assert!(!out.contains("prompt_tokens"), "usage chunk must be stripped: {out}");

        let ev = rx.recv().await.expect("one usage event");
        assert_attribution(&ev);
        assert!(rx.try_recv().is_err(), "exactly one event");
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            &[Some(json!({"include_usage": true}))],
            "stream must force include_usage upstream"
        );
    }
}
