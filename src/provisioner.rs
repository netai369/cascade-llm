//! Server-side passthrough to the stack's GPU provisioner.
//!
//! ## Why this lives in the gateway
//!
//! The GPU fleet is *not* an independent product: a rented instance only becomes
//! routable once the provisioner registers it into this gateway's
//! [`crate::registry::UpstreamRegistry`] (gotcha: the registry is in-memory, so a
//! gateway restart empties it until `reconcile_once` repopulates it). Users were
//! therefore managing one system from two dashboards — the provisioner UI for the
//! fleet, the gateway UI for the pool the fleet lands in — and the second half of
//! the state (why a node is absent, what it costs, whether it is still billing)
//! was invisible from the gateway side.
//!
//! Everything the fleet does, the gateway can therefore drive, and this module is
//! the transport for that.
//!
//! ## Why a passthrough instead of a new client
//!
//! The provisioner's request/response schema is large (instances, cost records,
//! flavor filters, model catalog) and belongs to the provisioner's own version
//! cadence. Duplicating it here would guarantee drift. So we forward verbatim
//! behind three hard rules:
//!
//! 1. **Path allowlist.** Only the exact `(method, path)` pairs the UI needs are
//!    forwarded. Anything else is a 404 *from the gateway* and never reaches the
//!    provisioner — this is not an open proxy even for an admin.
//! 2. **Admin key.** The routes sit behind [`crate::handlers::require_admin`].
//!    Through Caddy the key is injected server-side after `forward_auth`; on the
//!    raw loopback port the caller must present it. Either way the browser holds
//!    exactly one credential, and it is the same one it already uses.
//! 3. **Key stays here.** The provisioner's `X-API-Key` (`KNOWLEDGE_API_KEY`) is
//!    attached server-side and never echoed. Only bodies come back, and the
//!    provisioner already masks provider tokens in its listings.

use axum::{
    body::Body,
    extract::{Path, Request, State},
    http::{HeaderValue, StatusCode},
    response::Response,
};
use std::sync::Arc;

use crate::state::GatewayState;

/// `(method, path)` pairs the gateway forwards. GET reads, the rest are the
/// lifecycle calls whose effect is visible in our own registry.
fn allowed(method: &str, path: &str) -> bool {
    if path.is_empty() || path.split('/').any(|s| s.is_empty() || s == "." || s == "..") {
        return false;
    }
    match (method, path) {
        // --- reads -------------------------------------------------------
        ("GET", "health")
        | ("GET", "instances")
        | ("GET", "inference-nodes")
        | ("GET", "budget")
        | ("GET", "costs")
        | ("GET", "providers")
        | ("GET", "settings")
        | ("GET", "flavors")
        | ("GET", "model-catalog")
        | ("GET", "external-llms")
        | ("GET", "provider-catalog")
        | ("GET", "pollinations/models") => true,

        // --- fleet lifecycle ---------------------------------------------
        ("POST", "instances")
        | ("POST", "inference-nodes")
        | ("POST", "inference-nodes/pollinations")
        | ("POST", "cloud-toggle")
        | ("POST", "reconcile")
        | ("PUT", "budget")
        | ("PUT", "settings")
        | ("POST", "external-llms")
        | ("POST", "external-llms/discover")
        | ("POST", "external-llms/reconcile") => true,

        // --- generic OpenAI-compatible endpoints (per entry) --------------
        // PUT/DELETE `external-llms/{id}` and POST `external-llms/{id}/test`.
        ("PUT" | "DELETE" | "POST", p) if external_entry_path(method, p) => true,

        // --- per-instance / per-node ------------------------------------
        // GET/POST/DELETE on `/inference-nodes/{id}[/{verb}]`. A create call
        // (`POST inference-nodes`) is matched by the explicit arm above; the
        // per-node verbs (`activate`, `deactivate`, `terminate`, `logs`) are the
        // third segment.
        //
        // The id segment must be a `gpu_<hex>` / provider-chosen slug. Anything
        // that could re-target the request — query strings, traversal,
        // percent-encoding, extra depth — is refused rather than
        // pattern-matched around.
        ("GET" | "POST" | "DELETE", p) => {
            let segs: Vec<&str> = p.split('/').collect();
            segs[0] == "inference-nodes"
                && segs.len() >= 2
                && segs.len() <= 3
                && segs.iter().skip(1).all(|s| {
                    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
                })
        }

        _ => false,
    }
}

/// `external-llms/{id}` (PUT, DELETE) and `external-llms/{id}/test` (POST).
/// The id must be a plain slug, same rule as inference-node ids.
fn external_entry_path(method: &str, p: &str) -> bool {
    let segs: Vec<&str> = p.split('/').collect();
    if segs.first() != Some(&"external-llms") || segs.len() < 2 || segs.len() > 3 {
        return false;
    }
    let id_ok = segs[1].chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        && !segs[1].is_empty();
    match (method, segs.len()) {
        ("PUT" | "DELETE", 2) => id_ok,
        ("POST", 3) => id_ok && segs[2] == "test",
        _ => false,
    }
}

/// Re-uses the gateway's shared client so keep-alive and the configured
/// timeouts apply. `request` is consumed; `tail` is the allowlisted sub-path.
pub async fn proxy(
    State(state): State<Arc<GatewayState>>,
    method: axum::http::Method,
    Path(tail): Path<String>,
    request: Request<Body>,
) -> Response {
    let tail = tail.trim_start_matches('/').trim_end_matches('/').to_string();
    let method_str = method.as_str().to_string();
    if !allowed(&method_str, &tail) {
        return json(
            serde_json::json!({
                "error": "not an allowed provisioner endpoint",
                "method": method_str,
                "path": tail,
            }),
            StatusCode::NOT_FOUND,
        );
    }

    let Some(base) = state.config.provisioner_url.as_deref() else {
        return json(
            serde_json::json!({
                "error": "GPU provisioner integration is disabled",
                "hint": "set GPU_PROVISIONER_URL on the gateway to enable it",
            }),
            StatusCode::SERVICE_UNAVAILABLE,
        );
    };

    let (parts, body) = request.into_parts();
    let url = format!("{}/api/v1/{}", base, tail);
    tracing::debug!("provisioner proxy: {} {}", method_str, url);

    let mut builder = state
        .http_client
        .request(method.clone(), &url)
        .timeout(std::time::Duration::from_secs(120));
    if let Some(key) = &state.config.provisioner_api_key {
        builder = builder.header("x-api-key", key);
    }
    if let Some(ct) = parts.headers.get("content-type") {
        builder = builder.header("content-type", ct.clone());
    }

    // Collect the body first: axum and reqwest have distinct body types, and
    // these payloads are small control-plane JSON.
    let bytes = match axum::body::to_bytes(body, 2 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return json(
                serde_json::json!({"error": format!("could not read request body: {}", e)}),
                StatusCode::BAD_REQUEST,
            )
        }
    };

    let resp = match builder.body(bytes).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("provisioner call {} failed: {}", url, e);
            return json(
                serde_json::json!({
                    "error": format!("provisioner unreachable: {}", e),
                    "url": url,
                }),
                StatusCode::BAD_GATEWAY,
            );
        }
    };

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    // Pass the upstream body through unchanged so the UI renders the
    // provisioner's own error text (FastAPI's `{"detail": …}`) instead of a
    // flattened message that hides the status.
    let mut out = Response::new(Body::from(text));
    *out.status_mut() = status;
    out.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("application/json"),
    );
    out
}

/// Single liveness probe used by the overview header (the UI shows one dot for
/// "can the gateway reach the provisioner"), separate from the richer
/// [`status`] call the fleet section makes.
pub async fn reachable(state: &Arc<GatewayState>, base: &str) -> bool {
    let mut req = state
        .http_client
        .get(format!("{}/api/v1/health", base.trim_end_matches('/')));
    if let Some(key) = &state.config.provisioner_api_key {
        req = req.header("x-api-key", key);
    }
    matches!(
        req.timeout(std::time::Duration::from_secs(5)).send().await,
        Ok(r) if r.status().is_success()
    )
}

/// Whether the integration is configured at all — the UI uses this to render an
/// honest empty state instead of a stream of 502s.
pub async fn status(State(state): State<Arc<GatewayState>>) -> Response {
    let configured = state.config.provisioner_url.is_some();
    let mut reachable = false;
    let mut detail = serde_json::Value::Null;
    if let Some(base) = state.config.provisioner_url.as_deref() {
        let mut req = state
            .http_client
            .get(format!("{}/api/v1/health", base.trim_end_matches('/')));
        if let Some(key) = &state.config.provisioner_api_key {
            req = req.header("x-api-key", key);
        }
        match req.timeout(std::time::Duration::from_secs(5)).send().await {
            Ok(r) if r.status().is_success() => {
                reachable = true;
                detail = r.json::<serde_json::Value>().await.unwrap_or(serde_json::Value::Null);
            }
            Ok(r) => {
                detail = serde_json::json!({"status": r.status().as_u16()});
            }
            Err(e) => {
                tracing::debug!("provisioner status probe failed: {}", e);
                detail = serde_json::json!({"error": e.to_string()});
            }
        }
    }
    json(
        serde_json::json!({
            "configured": configured,
            "url": state.config.provisioner_url,
            "reachable": reachable,
            "health": detail,
            // The key is configured, never its value.
            "api_key_set": state.config.provisioner_api_key.is_some(),
        }),
        StatusCode::OK,
    )
}

fn json(body: serde_json::Value, status: StatusCode) -> Response {
    let mut resp = Response::new(Body::from(body.to_string()));
    *resp.status_mut() = status;
    resp.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("application/json"),
    );
    resp
}

#[cfg(test)]
mod tests {
    use super::allowed;

    #[test]
    fn external_llm_lifecycle_is_allowed() {
        assert!(allowed("GET", "external-llms"));
        assert!(allowed("POST", "external-llms"));
        assert!(allowed("POST", "external-llms/discover"));
        assert!(allowed("POST", "external-llms/reconcile"));
        assert!(allowed("GET", "pollinations/models"));
        assert!(allowed("PUT", "external-llms/ext-1a2b3c4d"));
        assert!(allowed("DELETE", "external-llms/ext-1a2b3c4d"));
        assert!(allowed("POST", "external-llms/ext-1a2b3c4d/test"));
    }

    #[test]
    fn provider_catalog_read_is_allowed_and_only_as_get() {
        assert!(allowed("GET", "provider-catalog"));
        assert!(allowed("POST", "external-llms/discover"));
        assert!(!allowed("POST", "provider-catalog"));
        assert!(!allowed("PUT", "provider-catalog"));
        assert!(!allowed("GET", "provider-catalog/../budget"));
    }

    #[test]
    fn external_llm_rejects_traversal_and_wrong_verbs() {
        assert!(!allowed("POST", "external-llms/ext-1/../../budget"));
        assert!(!allowed("PUT", "external-llms/ext-1/test"));
        assert!(!allowed("POST", "external-llms/ext-1"));
        assert!(!allowed("DELETE", "external-llms/ext-1/test"));
        assert!(!allowed("PUT", "external-llms"));
        assert!(!allowed("POST", "external-llms/ext%2F1/test"));
        assert!(!allowed("POST", "external-llms/ext?x=1/test"));
    }
}
