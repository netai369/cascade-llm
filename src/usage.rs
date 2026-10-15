//! Usage metering — one `UsageEvent` per billable call (chat, completions,
//! embeddings, audio, images, realtime voice sessions).
//!
//! Cascade is the one place every LLM/audio consumer passes through (stack
//! gotcha 86), so it is the only place usage can be counted for ALL of them —
//! LibreChat, Hermes, voice-chat, dictation, LightRAG, buchhaltung. The ledger
//! (credits, plans, Stripe) lives in the NetAI billing service; Cascade only
//! measures and ships events there (`CASCADE_USAGE_WEBHOOK_URL`).
//!
//! Attribution comes from request headers:
//!   `X-Billing-Tenant`, `X-Billing-User-Id` (or `X-LibreChat-User-Id`),
//!   `X-Billing-Source` (consumer name, e.g. `voice-chat`),
//!   `X-Billing-Metered: 1` — the billing gateway already debits this call;
//!   the event is still recorded (cost analytics) but flagged so the billing
//!   service does not debit twice.
//!
//! INTERFACE CONTRACT (other modules code against this; keep it stable):
//!   `UsageEvent::from_headers(&HeaderMap, kind, route)` -> pre-filled event
//!   `state.usage.record(ev)` — non-blocking, never fails the request.

use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

use crate::db::Db;
use crate::state::GatewayState;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct UsageEvent {
    /// Unique id (uuid-ish hex) — the billing service dedupes on it.
    pub id: String,
    /// Unix epoch milliseconds.
    pub ts_ms: u64,
    /// `chat` | `completions` | `embeddings` | `rerank` | `audio_transcription`
    /// | `audio_speech` | `image` | `realtime`
    pub kind: String,
    /// Cascade route / upstream role that served it (`inference`, `auxiliary`, `stt`, …).
    pub route: String,
    /// Model id as served (response `model`, else request `model`).
    pub model: String,
    pub tenant_id: Option<String>,
    pub user_id: Option<String>,
    pub source: Option<String>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Audio seconds (transcription / realtime / speech output).
    pub audio_seconds: f64,
    /// Characters synthesized (TTS input).
    pub characters: u64,
    /// Images generated.
    pub images: u64,
    /// True when token counts were estimated (upstream returned no usage).
    pub estimated: bool,
    /// True when the billing gateway already debited this call (`X-Billing-Metered: 1`).
    pub metered_upstream: bool,
    pub status: u16,
    pub latency_ms: u64,
}

fn header(h: &HeaderMap, name: &str) -> Option<String> {
    h.get(name)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

impl UsageEvent {
    pub fn from_headers(h: &HeaderMap, kind: &str, route: &str) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        Self {
            id: format!("{:x}{:08x}", now.as_nanos(), rand_u32()),
            ts_ms: now.as_millis() as u64,
            kind: kind.to_string(),
            route: route.to_string(),
            tenant_id: header(h, "x-billing-tenant"),
            user_id: header(h, "x-billing-user-id").or_else(|| header(h, "x-librechat-user-id")),
            source: header(h, "x-billing-source"),
            metered_upstream: header(h, "x-billing-metered").as_deref() == Some("1"),
            ..Default::default()
        }
    }
}

fn rand_u32() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static C: AtomicU32 = AtomicU32::new(0);
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    n ^ C.fetch_add(0x9E37_79B9, Ordering::Relaxed)
}

/// Recorder handle stored on `GatewayState.usage`. Persistence (SQLite) and the
/// webhook sink are implemented behind `record`; callers never await I/O.
#[derive(Debug)]
pub struct UsageRecorder {
    tx: Option<tokio::sync::mpsc::UnboundedSender<UsageEvent>>,
}

impl UsageRecorder {
    /// A recorder that drops everything (tests, or metering disabled).
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self { tx: None })
    }

    pub fn with_sender(tx: tokio::sync::mpsc::UnboundedSender<UsageEvent>) -> Arc<Self> {
        Arc::new(Self { tx: Some(tx) })
    }

    pub fn record(&self, ev: UsageEvent) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(ev);
        }
    }
}

const FLUSH_BATCH: usize = 500;
const MAX_BACKOFF: Duration = Duration::from_secs(60);
const CLEANUP_EVERY: Duration = Duration::from_secs(3600);

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Start the usage pipeline: an ingest task (channel -> SQLite), a delivery
/// task (SQLite -> billing webhook, only when `CASCADE_USAGE_WEBHOOK_URL` is
/// set) and an hourly retention task. Returns a disabled recorder when
/// `CASCADE_USAGE_ENABLED=false`. Must be called inside a Tokio runtime.
pub fn spawn_worker(db: Arc<Db>, http: Arc<reqwest::Client>) -> Arc<UsageRecorder> {
    let enabled = env_nonempty("CASCADE_USAGE_ENABLED")
        .map(|v| !matches!(v.to_ascii_lowercase().as_str(), "false" | "0" | "no" | "off"))
        .unwrap_or(true);
    if !enabled {
        info!("Usage metering disabled (CASCADE_USAGE_ENABLED=false)");
        return UsageRecorder::disabled();
    }

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<UsageEvent>();

    // Ingest: every event lands in SQLite, off the request path.
    let ingest_db = db.clone();
    tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            let db = ingest_db.clone();
            let res = tokio::task::spawn_blocking(move || db.insert_usage(&ev)).await;
            match res {
                Ok(Ok(())) => {}
                Ok(Err(e)) => warn!("usage insert failed: {}", e),
                Err(e) => warn!("usage insert task failed: {}", e),
            }
        }
    });

    // Delivery to the billing webhook.
    let delivering = env_nonempty("CASCADE_USAGE_WEBHOOK_URL").is_some();
    match env_nonempty("CASCADE_USAGE_WEBHOOK_URL") {
        Some(url) => {
            let key = env_nonempty("CASCADE_USAGE_WEBHOOK_KEY");
            let flush_secs = env_nonempty("CASCADE_USAGE_FLUSH_SECS")
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(5);
            let flush = Duration::from_secs(flush_secs);
            let db = db.clone();
            tokio::spawn(delivery_loop(db, http, url, key, flush));
        }
        None => info!("Usage webhook not configured; events stored but not delivered"),
    }

    // Retention: delete rows older than CASCADE_USAGE_RETENTION_DAYS. With a
    // webhook only DELIVERED rows go (an outage must not lose billable events);
    // without one nothing is ever delivered, so age alone decides — otherwise
    // the table grows without bound.
    let retention_days = env_nonempty("CASCADE_USAGE_RETENTION_DAYS")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30);
    let retention_ms = retention_days.saturating_mul(86_400_000);
    let cleanup_db = db.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(CLEANUP_EVERY);
        loop {
            tick.tick().await;
            let db = cleanup_db.clone();
            match tokio::task::spawn_blocking(move || db.cleanup_usage(retention_ms, !delivering)).await {
                Ok(Ok(n)) if n > 0 => info!("usage retention: removed {} events", n),
                Ok(Ok(_)) => {}
                Ok(Err(e)) => warn!("usage cleanup failed: {}", e),
                Err(e) => warn!("usage cleanup task failed: {}", e),
            }
        }
    });

    Arc::new(UsageRecorder { tx: Some(tx) })
}

async fn delivery_loop(
    db: Arc<Db>,
    http: Arc<reqwest::Client>,
    url: String,
    key: Option<String>,
    flush: Duration,
) {
    let max_wait = MAX_BACKOFF.max(flush);
    let mut wait = flush;
    loop {
        tokio::time::sleep(wait).await;
        match flush_once(&db, &http, &url, key.as_deref()).await {
            Ok(_) => wait = flush,
            Err(e) => {
                warn!("usage delivery failed, retrying in {:?}: {}", (wait * 2).min(max_wait), e);
                wait = (wait * 2).min(max_wait);
            }
        }
    }
}

/// Deliver all undelivered events in batches of `FLUSH_BATCH`. Rows are marked
/// delivered only after a 2xx response; on any failure they stay pending.
async fn flush_once(
    db: &Arc<Db>,
    http: &reqwest::Client,
    url: &str,
    key: Option<&str>,
) -> Result<usize, String> {
    let mut total = 0;
    loop {
        let batch_db = db.clone();
        let batch = tokio::task::spawn_blocking(move || batch_db.undelivered_usage(FLUSH_BATCH))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        if batch.is_empty() {
            return Ok(total);
        }
        let n = batch.len();
        let ids: Vec<String> = batch.iter().map(|e| e.id.clone()).collect();
        let body = serde_json::json!({ "events": batch });

        let mut req = http
            .post(url)
            .timeout(Duration::from_secs(15))
            .json(&body);
        if let Some(k) = key {
            req = req.header("X-Usage-Key", k);
        }
        let resp = req.send().await.map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("webhook answered HTTP {}", resp.status()));
        }

        let mark_db = db.clone();
        tokio::task::spawn_blocking(move || mark_db.mark_usage_delivered(&ids))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        total += n;
        if n < FLUSH_BATCH {
            return Ok(total);
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct UsageQuery {
    /// Look-back window in hours (default 24).
    pub since_hours: Option<u64>,
    /// `tenant` (default) | `model` | `kind` | `source` | `day`.
    pub group_by: Option<String>,
}

const GROUP_BY_VALUES: [&str; 5] = ["tenant", "model", "kind", "source", "day"];

/// `GET /api/v1/admin/usage` — aggregated usage plus the delivery backlog.
pub async fn admin_usage(
    State(state): State<Arc<GatewayState>>,
    Query(q): Query<UsageQuery>,
) -> Response {
    let since_hours = q.since_hours.unwrap_or(24).min(24 * 366);
    let group_by = q.group_by.unwrap_or_else(|| "tenant".to_string());
    if !GROUP_BY_VALUES.contains(&group_by.as_str()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid group_by",
                "allowed": GROUP_BY_VALUES,
            })),
        )
            .into_response();
    }
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let since_ms = now_ms.saturating_sub(since_hours * 3_600_000);

    let db = state.db.clone();
    let gb = group_by.clone();
    let result = tokio::task::spawn_blocking(move || {
        let rows = db.usage_summary(since_ms, &gb)?;
        let pending = db.pending_usage_count()?;
        Ok::<_, rusqlite::Error>((rows, pending))
    })
    .await;

    match result {
        Ok(Ok((rows, pending))) => Json(serde_json::json!({
            "since_ms": since_ms,
            "group_by": group_by,
            "rows": rows,
            "pending_delivery": pending,
        }))
        .into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn from_headers_parses_billing_attribution_and_metered_flag() {
        let mut h = HeaderMap::new();
        h.insert("x-billing-tenant", HeaderValue::from_static(" acme "));
        h.insert("x-billing-user-id", HeaderValue::from_static("u-1"));
        h.insert("x-billing-source", HeaderValue::from_static("voice-chat"));
        h.insert("x-billing-metered", HeaderValue::from_static("1"));
        let ev = UsageEvent::from_headers(&h, "chat", "inference");
        assert_eq!(ev.kind, "chat");
        assert_eq!(ev.route, "inference");
        assert_eq!(ev.tenant_id.as_deref(), Some("acme"));
        assert_eq!(ev.user_id.as_deref(), Some("u-1"));
        assert_eq!(ev.source.as_deref(), Some("voice-chat"));
        assert!(ev.metered_upstream);
        assert!(!ev.id.is_empty());
        assert!(ev.ts_ms > 0);
    }

    #[test]
    fn from_headers_falls_back_to_librechat_user_and_defaults_unmetered() {
        let mut h = HeaderMap::new();
        h.insert("x-librechat-user-id", HeaderValue::from_static("lc-7"));
        h.insert("x-billing-metered", HeaderValue::from_static("0"));
        h.insert("x-billing-tenant", HeaderValue::from_static("   "));
        let ev = UsageEvent::from_headers(&h, "embeddings", "embeddings");
        assert_eq!(ev.user_id.as_deref(), Some("lc-7"));
        assert_eq!(ev.tenant_id, None);
        assert_eq!(ev.source, None);
        assert!(!ev.metered_upstream);
    }

    #[test]
    fn disabled_recorder_drops_events_without_panicking() {
        UsageRecorder::disabled().record(UsageEvent::default());
    }
}
