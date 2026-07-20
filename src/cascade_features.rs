// Cascade LLM Gateway - Advanced Features Module
// Implements: In-Flight Fallback, Streaming Quality Filter, Prometheus Observability

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Router,
};
use prometheus::{
    opts,
    register_counter_vec,
    Encoder, TextEncoder,
};
use tracing::{error, warn};

// ============================================================================
// PROMETHEUS METRICS
// ============================================================================

#[allow(dead_code)]
#[derive(Debug)]
pub struct MetricsRegistry {
    pub requests_total: prometheus::CounterVec,
    pub fallback_triggered: prometheus::CounterVec,
}

#[allow(dead_code)]
impl MetricsRegistry {
    pub fn init() -> Self {
        let requests_total = register_counter_vec!(
            opts!(
                "cascade_requests_total",
                "Total number of requests routed to each backend"
            ),
            &["selected_backend"]
        )
        .unwrap();

        let fallback_triggered = register_counter_vec!(
            opts!(
                "cascade_fallback_triggered_total",
                "Total number of fallback activations"
            ),
            &["reason"]
        )
        .unwrap();

        // Pre-initialize label values so metrics appear in /metrics from startup
        // (prometheus prunes empty MetricFamilies during gather)
        for backend in &["small", "large", "large_multimodal", "large_text", "fallback"] {
            requests_total.with_label_values(&[backend]);
        }
        for reason in &["primary_failed", "quality_low", "timeout"] {
            fallback_triggered.with_label_values(&[reason]);
        }

        Self {
            requests_total,
            fallback_triggered,
        }
    }

    pub fn record_request(&self, backend: &str) {
        self.requests_total.with_label_values(&[backend]).inc();
    }

    pub fn record_fallback(&self, reason: &str) {
        self.fallback_triggered.with_label_values(&[reason]).inc();
    }
}

// ============================================================================
// FALLBACK MANAGER (In-Flight Fallback Mechanism)
// ============================================================================

/// Manages automatic fallback to secondary backend on failure
#[allow(dead_code)]
pub struct FallbackManager {
    primary_url: String,
    fallback_url: String,
    client: Arc<reqwest::Client>,
}

#[allow(dead_code)]
impl FallbackManager {
    pub fn new(
        primary_url: String,
        fallback_url: String,
        client: Arc<reqwest::Client>,
    ) -> Self {
        Self {
            primary_url,
            fallback_url,
            client,
        }
    }

    /// Attempts to call primary backend, falls back automatically on failure
    pub async fn execute_with_fallback(
        &self,
        request: &serde_json::Value,
    ) -> Result<reqwest::Response, StatusCode> {
        match self.call_primary(request).await {
            Ok(response) => Ok(response),
            Err(StatusCode::BAD_GATEWAY)
            | Err(StatusCode::SERVICE_UNAVAILABLE)
            | Err(StatusCode::GATEWAY_TIMEOUT) => {
                warn!(
                    "Primary backend failed: {}, attempting fallback to {}",
                    self.primary_url, self.fallback_url
                );
                self.call_fallback(request).await
            }
            Err(status) => Err(status),
        }
    }

    async fn call_primary(&self, request: &serde_json::Value) -> Result<reqwest::Response, StatusCode> {
        let req = self.client.post(&self.primary_url).json(request);
        req.send()
            .await
            .inspect_err(|e| error!("Primary backend network error: {}", e))
            .map(|resp| resp.error_for_status())
            .unwrap_or_else(|err| {
                let status = err.status().unwrap_or(StatusCode::BAD_GATEWAY);
                error!("Primary backend HTTP error: {} - {:?}", status, err);
                panic!("{}", status.as_str());
            })
            .map_err(|err| {
                let status = err.status().unwrap_or(StatusCode::BAD_GATEWAY);
                error!("Primary backend HTTP error: {} - {:?}", status, err);
                StatusCode::BAD_GATEWAY
            })
    }

    async fn call_fallback(&self, request: &serde_json::Value) -> Result<reqwest::Response, StatusCode> {
        let req = self.client.post(&self.fallback_url).json(request);
        req.send()
            .await
            .inspect_err(|e| error!("Fallback backend network error: {}", e))
            .map(|resp| resp.error_for_status())
            .unwrap_or_else(|err| {
                let status = err.status().unwrap_or(StatusCode::BAD_GATEWAY);
                error!("Fallback backend HTTP error: {} - {:?}", status, err);
                panic!("{}", status.as_str());
            })
            .map_err(|err| {
                let status = err.status().unwrap_or(StatusCode::BAD_GATEWAY);
                error!("Fallback backend HTTP error: {} - {:?}", status, err);
                StatusCode::BAD_GATEWAY
            })
    }
}

// ============================================================================
// STREAMING QUALITY FILTER (LLM-as-a-Judge Lite)
// ============================================================================

/// Monitors streaming responses and detects quality issues
#[allow(dead_code)]
pub struct QualityFilter {
    sample_size: usize,
}

#[allow(dead_code)]
impl QualityFilter {
    pub fn new() -> Self {
        Self { sample_size: 3 }
    }

    /// Validates tool calls in stream chunks
    pub fn validate_tool_calls(&self, chunks: Vec<String>) -> (bool, Option<String>) {
        if chunks.is_empty() {
            return (false, Some("No streaming data received".to_string()));
        }

        let mut tool_chunk_count = 0;
        for (idx, chunk) in chunks.iter().enumerate() {
            if idx >= self.sample_size {
                break;
            }
            if chunk.contains(r#""name":"#) || chunk.contains(r#""arguments":"#) {
                tool_chunk_count += 1;
            }
        }

        if tool_chunk_count >= self.sample_size / 2 {
            (true, None)
        } else {
            (
                false,
                Some(format!(
                    "Tool call validation failed: insufficient tool markers in {} chunks",
                    self.sample_size
                )),
            )
        }
    }

    /// Creates a stream wrapper for quality filtering
    /// Note: Full streaming quality filter implementation requires additional
    /// stream transformation logic to be added based on specific requirements
    pub fn apply_quality_filter(
        &self,
        input_stream: std::pin::Pin<Box<dyn futures_util::stream::Stream<Item = Result<axum::body::Bytes, std::io::Error>> + Send + Unpin>>,
        _backend: String,
    ) -> std::pin::Pin<Box<dyn futures_util::stream::Stream<Item = Result<axum::body::Bytes, std::io::Error>> + Send + Unpin>> {
        input_stream
    }
}

// ============================================================================
// AXUM ROUTES
// ============================================================================

pub async fn metrics_handler() -> impl IntoResponse {
    let families = prometheus::default_registry().gather();
    let encoder = TextEncoder::new();
    let mut buffer = Vec::new();
    encoder.encode(&families, &mut buffer).unwrap();
    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        String::from_utf8(buffer).unwrap(),
    )
}

pub fn build_router<S: Clone + Send + Sync + 'static>() -> Router<S> {
    Router::new()
        .route("/metrics", get(metrics_handler))
}
