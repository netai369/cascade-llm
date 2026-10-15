//! Per-tenant fair-share concurrency limit for LLM generation routes.
//!
//! Applies to `/v1/chat/completions` and `/v1/messages` only. Embeddings,
//! rerank, audio, images, video, RAG extraction and realtime are NOT gated.
//!
//! Tenant identity: `X-Billing-Tenant`, falling back to `X-Knowledge-User-Tenant`.
//! A request with no tenant header is NOT limited (it is internal or
//! unattributed traffic; one shared bucket would starve the whole stack).
//! Plan: `X-Billing-Plan` selects an entry of `CASCADE_TENANT_CONCURRENCY_BY_PLAN`;
//! unknown or absent plans use `CASCADE_TENANT_MAX_CONCURRENCY`.
//! A limit of 0 disables gating for that tenant.
//!
//! When a tenant is at its limit, the request waits up to
//! `CASCADE_TENANT_QUEUE_WAIT_SECS` for a permit, then receives HTTP 429 with an
//! OpenAI-style error body and a `Retry-After` header. The permit is held for
//! the whole response body and released when the body ends or is dropped
//! (client disconnect).

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use futures_util::{stream, StreamExt};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Prune idle buckets once the map grows past this many tenants.
const PRUNE_AT: usize = 4096;

#[derive(Debug, Clone)]
pub struct TenantLimitConfig {
    pub default_max: usize,
    pub by_plan: HashMap<String, usize>,
    pub queue_wait: Duration,
}

impl Default for TenantLimitConfig {
    fn default() -> Self {
        Self {
            default_max: 0,
            by_plan: HashMap::new(),
            queue_wait: Duration::from_secs(30),
        }
    }
}

impl TenantLimitConfig {
    pub fn from_env() -> Self {
        let default_max = std::env::var("CASCADE_TENANT_MAX_CONCURRENCY")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0);
        let by_plan = std::env::var("CASCADE_TENANT_CONCURRENCY_BY_PLAN")
            .map(|v| parse_plan_map(&v))
            .unwrap_or_default();
        let queue_wait = std::env::var("CASCADE_TENANT_QUEUE_WAIT_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(30));
        Self {
            default_max,
            by_plan,
            queue_wait,
        }
    }
}

/// Parses `"starter=2,professional=4,enterprise=16"`. Keys are lowercased;
/// malformed entries are skipped.
pub fn parse_plan_map(raw: &str) -> HashMap<String, usize> {
    raw.split(',')
        .filter_map(|entry| {
            let (k, v) = entry.split_once('=')?;
            let key = k.trim().to_ascii_lowercase();
            let limit = v.trim().parse::<usize>().ok()?;
            (!key.is_empty()).then_some((key, limit))
        })
        .collect()
}

fn header_value(h: &HeaderMap, name: &str) -> Option<String> {
    h.get(name)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Tenant key for a request, or `None` when the request is unattributed.
pub fn tenant_of(h: &HeaderMap) -> Option<String> {
    header_value(h, "x-billing-tenant").or_else(|| header_value(h, "x-knowledge-user-tenant"))
}

fn plan_of(h: &HeaderMap) -> Option<String> {
    header_value(h, "x-billing-plan").map(|p| p.to_ascii_lowercase())
}

/// A request was refused because the tenant stayed at its limit for the whole
/// queue wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rejected {
    pub retry_after_secs: u64,
}

#[derive(Debug)]
struct Bucket {
    limit: usize,
    sem: Arc<Semaphore>,
}

#[derive(Debug)]
pub struct TenantLimiter {
    cfg: TenantLimitConfig,
    buckets: Mutex<HashMap<String, Bucket>>,
}

impl TenantLimiter {
    pub fn new(cfg: TenantLimitConfig) -> Self {
        Self {
            cfg,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// Effective limit for a plan (0 = unlimited).
    pub fn limit_for(&self, plan: Option<&str>) -> usize {
        plan.and_then(|p| self.cfg.by_plan.get(p).copied())
            .unwrap_or(self.cfg.default_max)
    }

    /// `Ok(None)` = not gated (disabled, or no tenant). `Ok(Some(permit))` =
    /// admitted; hold the permit for the life of the response body.
    pub async fn acquire(&self, headers: &HeaderMap) -> Result<Option<OwnedSemaphorePermit>, Rejected> {
        let Some(tenant) = tenant_of(headers) else {
            return Ok(None);
        };
        let limit = self.limit_for(plan_of(headers).as_deref());
        if limit == 0 {
            return Ok(None);
        }
        let sem = self.bucket(&tenant, limit);
        match tokio::time::timeout(self.cfg.queue_wait, sem.acquire_owned()).await {
            Ok(Ok(permit)) => Ok(Some(permit)),
            Ok(Err(_)) => Ok(None),
            Err(_) => Err(Rejected {
                retry_after_secs: self.cfg.queue_wait.as_secs().max(1),
            }),
        }
    }

    fn bucket(&self, tenant: &str, limit: usize) -> Arc<Semaphore> {
        let mut map = self.buckets.lock().unwrap_or_else(|e| e.into_inner());
        if map.len() >= PRUNE_AT {
            // A semaphore referenced only by the map has no permits out and no waiters.
            map.retain(|_, b| Arc::strong_count(&b.sem) > 1);
        }
        if let Some(b) = map.get(tenant) {
            if b.limit == limit {
                return b.sem.clone();
            }
        }
        // Limit changed (plan switch): new permits use the new capacity; permits
        // already out on the old semaphore finish there.
        let sem = Arc::new(Semaphore::new(limit));
        map.insert(
            tenant.to_string(),
            Bucket {
                limit,
                sem: sem.clone(),
            },
        );
        sem
    }

    #[cfg(test)]
    fn available(&self, tenant: &str) -> Option<usize> {
        self.buckets
            .lock()
            .unwrap()
            .get(tenant)
            .map(|b| b.sem.available_permits())
    }

    #[cfg(test)]
    fn bucket_count(&self) -> usize {
        self.buckets.lock().unwrap().len()
    }
}

/// Wraps a response body so the permit lives exactly as long as the stream:
/// released when the stream ends, or when the body is dropped (client gone).
pub fn guard_response_body(body: Body, permit: Option<OwnedSemaphorePermit>) -> Body {
    let Some(permit) = permit else {
        return body;
    };
    let guarded = stream::unfold(
        Some((body.into_data_stream(), permit)),
        |state| async move {
            let (mut data, permit) = state?;
            data.next()
                .await
                .map(|item| (item, Some((data, permit))))
        },
    );
    Body::from_stream(guarded)
}

pub fn rejection_response(rej: Rejected) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": format!(
                "Too many concurrent requests for this organisation; retry after {}s",
                rej.retry_after_secs
            ),
            "type": "rate_limit_exceeded",
            "param": serde_json::Value::Null,
            "code": "rate_limit_exceeded"
        }
    });
    let mut resp = Response::new(Body::from(body.to_string()));
    *resp.status_mut() = StatusCode::TOO_MANY_REQUESTS;
    resp.headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    if let Ok(v) = HeaderValue::from_str(&rej.retry_after_secs.to_string()) {
        resp.headers_mut().insert("retry-after", v);
    }
    resp
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Bytes;
    use axum::http::HeaderValue;
    use http_body_util::BodyExt;

    fn hdrs(tenant: Option<&str>, plan: Option<&str>) -> HeaderMap {
        let mut h = HeaderMap::new();
        if let Some(t) = tenant {
            h.insert("x-billing-tenant", HeaderValue::from_str(t).unwrap());
        }
        if let Some(p) = plan {
            h.insert("x-billing-plan", HeaderValue::from_str(p).unwrap());
        }
        h
    }

    fn cfg(default_max: usize, wait_ms: u64) -> TenantLimitConfig {
        TenantLimitConfig {
            default_max,
            by_plan: HashMap::new(),
            queue_wait: Duration::from_millis(wait_ms),
        }
    }

    #[tokio::test]
    async fn limit_enforced_per_tenant() {
        let lim = TenantLimiter::new(cfg(2, 20));
        let h = hdrs(Some("acme"), None);
        let _p1 = lim.acquire(&h).await.unwrap().unwrap();
        let _p2 = lim.acquire(&h).await.unwrap().unwrap();
        let third = lim.acquire(&h).await;
        assert_eq!(third.unwrap_err(), Rejected { retry_after_secs: 1 });
    }

    #[tokio::test]
    async fn tenants_are_isolated() {
        let lim = TenantLimiter::new(cfg(1, 20));
        let _a = lim.acquire(&hdrs(Some("acme"), None)).await.unwrap().unwrap();
        assert!(lim.acquire(&hdrs(Some("acme"), None)).await.is_err());
        // A different tenant is unaffected by acme being full.
        assert!(lim.acquire(&hdrs(Some("globex"), None)).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn queued_request_admitted_when_permit_frees_within_wait() {
        let lim = Arc::new(TenantLimiter::new(cfg(1, 500)));
        let held = lim.acquire(&hdrs(Some("acme"), None)).await.unwrap().unwrap();
        let lim2 = lim.clone();
        let waiter = tokio::spawn(async move {
            lim2.acquire(&hdrs(Some("acme"), None)).await.map(|p| p.is_some())
        });
        tokio::time::sleep(Duration::from_millis(30)).await;
        drop(held);
        assert_eq!(waiter.await.unwrap(), Ok(true));
    }

    #[tokio::test]
    async fn permit_released_when_stream_ends() {
        let lim = TenantLimiter::new(cfg(1, 20));
        let h = hdrs(Some("acme"), None);
        let permit = lim.acquire(&h).await.unwrap();
        let src = Body::from_stream(stream::iter(vec![
            Ok::<_, std::io::Error>(Bytes::from_static(b"data: a\n\n")),
            Ok(Bytes::from_static(b"data: b\n\n")),
        ]));
        let body = guard_response_body(src, permit);
        assert_eq!(lim.available("acme"), Some(0));
        let collected = body.collect().await.unwrap().to_bytes();
        assert_eq!(&collected[..], b"data: a\n\ndata: b\n\n");
        assert_eq!(lim.available("acme"), Some(1));
    }

    #[tokio::test]
    async fn permit_released_when_body_dropped_mid_stream() {
        let lim = TenantLimiter::new(cfg(1, 20));
        let h = hdrs(Some("acme"), None);
        let permit = lim.acquire(&h).await.unwrap();
        let src = Body::from_stream(stream::iter(vec![
            Ok::<_, std::io::Error>(Bytes::from_static(b"a")),
            Ok(Bytes::from_static(b"b")),
        ]));
        let mut body = guard_response_body(src, permit);
        let _first = body.frame().await;
        assert_eq!(lim.available("acme"), Some(0));
        drop(body); // client disconnect
        assert_eq!(lim.available("acme"), Some(1));
        assert!(lim.acquire(&h).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn disabled_when_zero() {
        let lim = TenantLimiter::new(cfg(0, 20));
        let h = hdrs(Some("acme"), None);
        let mut permits = Vec::new();
        for _ in 0..50 {
            permits.push(lim.acquire(&h).await.unwrap());
        }
        assert!(permits.iter().all(|p| p.is_none()));
        assert_eq!(lim.bucket_count(), 0);
    }

    #[tokio::test]
    async fn no_tenant_header_is_not_gated() {
        let lim = TenantLimiter::new(cfg(1, 20));
        let h = HeaderMap::new();
        assert!(lim.acquire(&h).await.unwrap().is_none());
        assert!(lim.acquire(&h).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn knowledge_tenant_header_is_fallback() {
        let lim = TenantLimiter::new(cfg(1, 20));
        let mut h = HeaderMap::new();
        h.insert("x-knowledge-user-tenant", HeaderValue::from_static("acme"));
        let _p = lim.acquire(&h).await.unwrap().unwrap();
        assert!(lim.acquire(&h).await.is_err());
        // X-Billing-Tenant wins when both are present.
        h.insert("x-billing-tenant", HeaderValue::from_static("other"));
        assert!(lim.acquire(&h).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn plan_override_applies() {
        let mut by_plan = HashMap::new();
        by_plan.insert("starter".to_string(), 1);
        by_plan.insert("enterprise".to_string(), 3);
        let lim = TenantLimiter::new(TenantLimitConfig {
            default_max: 2,
            by_plan,
            queue_wait: Duration::from_millis(20),
        });
        let s = hdrs(Some("small"), Some("Starter"));
        let _p = lim.acquire(&s).await.unwrap().unwrap();
        assert!(lim.acquire(&s).await.is_err());

        let e = hdrs(Some("big"), Some("enterprise"));
        let mut held = Vec::new();
        for _ in 0..3 {
            held.push(lim.acquire(&e).await.unwrap().unwrap());
        }
        assert!(lim.acquire(&e).await.is_err());

        // Unknown plan falls back to the default of 2.
        let u = hdrs(Some("mid"), Some("pro-unknown"));
        assert_eq!(lim.limit_for(Some("pro-unknown")), 2);
        let _u1 = lim.acquire(&u).await.unwrap().unwrap();
        let _u2 = lim.acquire(&u).await.unwrap().unwrap();
        assert!(lim.acquire(&u).await.is_err());
    }

    #[test]
    fn parse_plan_map_skips_malformed() {
        let m = parse_plan_map(" starter=2, Professional = 4,enterprise=16,bad,x=y,=3");
        assert_eq!(m.get("starter"), Some(&2));
        assert_eq!(m.get("professional"), Some(&4));
        assert_eq!(m.get("enterprise"), Some(&16));
        assert_eq!(m.len(), 3);
    }

    #[tokio::test]
    async fn rejection_response_is_429_with_retry_after() {
        let resp = rejection_response(Rejected { retry_after_secs: 7 });
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(resp.headers().get("retry-after").unwrap(), "7");
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_exceeded");
        assert_eq!(v["error"]["code"], "rate_limit_exceeded");
    }
}
