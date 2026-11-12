//! External-access panel data + per-user API key provisioning.
//!
//! The console used to have no answer to "where do I point my app, and with
//! which key?". The URL it showed for the LLM was whatever the last engine
//! wrote into `.env`, and the only credential was the stack-wide
//! `CASCADE_API_KEY`. This module makes the access surface explicit:
//!
//! * the public base URL is derived from the request Caddy forwarded (and from
//!   `CASCADE_PUBLIC_URL` / `DOMAIN` when set), so an engine or domain change
//!   cannot leave a stale hostname on screen;
//! * both API dialects the gateway serves are listed (OpenAI-compatible and
//!   Anthropic Messages);
//! * the internal service key is shown on demand (masked by default);
//! * personal `nbk_…` keys are listed/created/revoked through the billing
//!   service, which already owns the tenant-scoped key store. Cascade proxies
//!   the call with the caller's forward_auth identity, so a key always belongs
//!   to the signed-in user and is metered against that user's tenant.
//!
//! Nothing secret is ever placed in a list response: keys are only returned by
//! the create call (shown once) or the explicit service-key reveal.

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::{json, Value};
use std::sync::Arc;

use crate::state::GatewayState;

const IDENTITY_HEADERS: [&str; 12] = [
    "x-knowledge-user-id",
    "x-knowledge-user-email",
    "x-knowledge-user-name",
    "x-knowledge-user-role",
    "x-knowledge-user-tenant",
    "x-knowledge-user-package",
    "x-knowledge-user-industry",
    "x-org-role",
    "x-org-path",
    "x-org-groups",
    "x-org-overrides",
    "x-knowledge-user-locale",
];

pub fn json_response(body: Value, status: StatusCode) -> Response {
    (
        status,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

fn err(status: StatusCode, message: &str) -> Response {
    json_response(json!({"error": message}), status)
}

fn mask(key: &str) -> String {
    if key.len() <= 8 {
        return "***".to_string();
    }
    let tail: String = key
        .chars()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("***{tail}")
}

/// The URL an external client should use. Explicit configuration wins; then
/// the headers Caddy sets for this very request; only then a local fallback.
fn public_base(state: &GatewayState, headers: &HeaderMap) -> String {
    if let Some(url) = state
        .config
        .public_url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return url.trim_end_matches('/').to_string();
    }
    let domain = std::env::var("DOMAIN").unwrap_or_default();
    let domain = domain.trim();
    if !domain.is_empty() && domain != "localhost" && !domain.starts_with("127.") {
        return format!("https://{}", domain);
    }
    let proto = headers
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("https");
    let host = headers
        .get("x-forwarded-host")
        .or_else(|| headers.get(axum::http::header::HOST))
        .and_then(|v| v.to_str().ok())
        .unwrap_or("localhost");
    format!("{}://{}", proto, host.trim_end_matches('/'))
}

fn endpoint_catalog() -> Value {
    // Keep this list complete: it is the only place an operator can discover a
    // route without reading the router. `service` = the stack-wide bearer
    // (/api2), `user` = a personal nbk_ key (/api/llm/v1). The billing gateway
    // only proxies the OpenAI subset, so Anthropic and rerank are service-only.
    json!([
        {"method": "POST", "path": "/v1/chat/completions", "family": "openai", "auth": ["service", "user"], "purpose": "Chat — the default entrypoint"},
        {"method": "POST", "path": "/v1/completions", "family": "openai", "auth": ["service", "user"], "purpose": "Legacy text completions"},
        {"method": "GET",  "path": "/v1/models", "family": "openai", "auth": ["service", "user"], "purpose": "Model discovery"},
        {"method": "POST", "path": "/v1/embeddings", "family": "openai", "auth": ["service", "user"], "purpose": "Embeddings (bge-m3)"},
        {"method": "POST", "path": "/v1/rerank", "family": "openai", "auth": ["service"], "purpose": "Cross-encoder rerank"},
        {"method": "POST", "path": "/v1/images/generations", "family": "openai", "auth": ["service", "user"], "purpose": "Image generation"},
        {"method": "POST", "path": "/v1/audio/speech", "family": "openai", "auth": ["service", "user"], "purpose": "Text-to-speech"},
        {"method": "POST", "path": "/v1/audio/transcriptions", "family": "openai", "auth": ["service", "user"], "purpose": "Speech-to-text"},
        {"method": "POST", "path": "/v1/audio/diarize", "family": "openai", "auth": ["service"], "purpose": "Speaker diarization"},
        {"method": "WS",   "path": "/v1/realtime", "family": "openai", "auth": ["service"], "purpose": "OpenAI-Realtime-compatible streaming STT"},
        {"method": "POST", "path": "/v1/rag/extract", "family": "rag", "auth": ["service"], "purpose": "Heavy RAG extraction job"},
        {"method": "POST", "path": "/v1/messages", "family": "anthropic", "auth": ["service"], "purpose": "Anthropic Messages API"},
        {"method": "POST", "path": "/v1/messages/count_tokens", "family": "anthropic", "auth": ["service"], "purpose": "Anthropic token counting"},
    ])
}

pub async fn access_info(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
) -> Response {
    let base = public_base(&state, &headers);
    let service_key = state.config.service_api_key.clone();
    let identity = headers
        .get("x-knowledge-user-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let email = headers
        .get("x-knowledge-user-email")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    json_response(
        json!({
            "public_base": base,
            "signed_in": {"user_id": identity, "email": email},
            "service": {
                "base_url": format!("{}/api2", base),
                "internal_url": "http://netai-cascade-llm:3000/v1",
                "header": "Authorization: Bearer <service key>",
                "key_set": service_key.is_some(),
                "key_masked": service_key.as_deref().map(mask),
                "scope": "Full gateway surface (all endpoints, including Anthropic and realtime). Share only inside the trust boundary.",
            },
            "user_keys": {
                "base_url": format!("{}/api/llm/v1", base),
                "header": "Authorization: Bearer nbk_…",
                "billing_url": "/billing/",
                "billing_available": state.config.billing_url.is_some(),
                "scope": "Tenant-scoped, metered per user. OpenAI-compatible chat, models, embeddings, images and audio.",
            },
            "endpoints": endpoint_catalog(),
            "endpoints_note": "`service` = the /api2 base URL and reaches every row. `personal` = the /api/llm/v1 base URL with an nbk_ key; it goes through the metered billing gateway, which proxies only chat, completions, models, embeddings, images and audio. Anthropic Messages, rerank, diarize, realtime and RAG extraction are service-key only.",
            "claude_code": {
                "base_url": format!("{}/api2", base),
                "env": [
                    format!("export ANTHROPIC_BASE_URL={}/api2", base),
                    "export ANTHROPIC_AUTH_TOKEN=<service key>".to_string(),
                    "export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1".to_string(),
                ],
                "models": ["claude-netai-hybrid", "claude-netai-main", "claude-netai-aux"],
                "note": "Claude Code only accepts discovered ids that start with `claude` or `anthropic`; these three are advertised as aliases and the Messages route retargets them to the gateway identity. Without discovery, the built-in claude-* aliases also work.",
            },
        }),
        StatusCode::OK,
    )
}

/// Explicit reveal of the stack service key. Admin-gated by the router layer
/// (Caddy injects x-cascade-admin-key only for an authenticated admin session).
pub async fn reveal_service_key(State(state): State<Arc<GatewayState>>) -> Response {
    match state.config.service_api_key.as_deref() {
        Some(key) if !key.is_empty() => json_response(json!({"key": key}), StatusCode::OK),
        _ => err(StatusCode::NOT_FOUND, "CASCADE_API_KEY is not configured on this gateway"),
    }
}

fn billing_url(state: &GatewayState) -> Option<String> {
    state
        .config
        .billing_url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_end_matches('/').to_string())
}

fn upstream_request(
    state: &GatewayState,
    headers: &HeaderMap,
    method: reqwest::Method,
    path: &str,
) -> Option<reqwest::RequestBuilder> {
    let base = billing_url(state)?;
    let mut req = state.http_client.request(method, format!("{}{}", base, path));
    for name in IDENTITY_HEADERS {
        if let Some(value) = headers.get(name) {
            req = req.header(name, value);
        }
    }
    Some(req)
}

fn billing_error(e: reqwest::Error) -> Response {
    json_response(
        json!({
            "error": format!("billing service unavailable: {}", e),
            "hint": "The billing profile owns the per-user key store. Enable it (scripts/compose-up.sh --billing up -d) or create keys at /billing/.",
        }),
        StatusCode::BAD_GATEWAY,
    )
}

async fn relay(resp: reqwest::Response) -> Response {
    let status = resp.status();
    let body: Value = resp
        .json()
        .await
        .unwrap_or_else(|_| json!({"error": "billing returned a non-JSON response"}));
    json_response(body, status)
}

pub async fn list_keys(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
) -> Response {
    let Some(req) = upstream_request(&state, &headers, reqwest::Method::GET, "/billing/api/keys")
    else {
        return err(StatusCode::SERVICE_UNAVAILABLE, "billing integration disabled (BILLING_URL empty)");
    };
    match req.send().await {
        Ok(resp) => relay(resp).await,
        Err(e) => billing_error(e),
    }
}

pub async fn create_key(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let Some(req) = upstream_request(&state, &headers, reqwest::Method::POST, "/billing/api/keys")
    else {
        return err(StatusCode::SERVICE_UNAVAILABLE, "billing integration disabled (BILLING_URL empty)");
    };
    match req
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
    {
        Ok(resp) => relay(resp).await,
        Err(e) => billing_error(e),
    }
}

pub async fn delete_key(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    // Billing ids are opaque but plain (`k_<hex>`); percent-encode anything
    // outside the unreserved set so a hostile id cannot reshape the URL.
    let encoded: String = id
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            other => format!("%{:02X}", other),
        })
        .collect();
    let path = format!("/billing/api/keys/{}", encoded);
    let Some(req) = upstream_request(&state, &headers, reqwest::Method::DELETE, &path) else {
        return err(StatusCode::SERVICE_UNAVAILABLE, "billing integration disabled (BILLING_URL empty)");
    };
    match req.send().await {
        Ok(resp) => relay(resp).await,
        Err(e) => billing_error(e),
    }
}
