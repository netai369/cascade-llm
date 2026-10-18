//! Pass-through proxies for the stack's embedding and reranking services.

use axum::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    response::Response,
};
use http_body_util::BodyExt;
use std::sync::Arc;
use tracing::{info, warn};

use crate::registry;
use crate::state::{passthrough_usage, GatewayState};
use crate::usage::UsageEvent;
use axum::http::HeaderMap;
use serde_json::Value;
use std::time::Instant;

pub async fn embeddings_handler(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    proxy_json_api(state, req, "embeddings").await
}

pub async fn rerank_handler(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    proxy_json_api(state, req, "rerank").await
}

async fn proxy_json_api(state: Arc<GatewayState>, req: Request<Body>, service: &str) -> Response {
    let started = Instant::now();
    let (default_url, endpoint) = match service {
        "embeddings" => (
            state.config.embeddings_url.clone(),
            "/v1/embeddings",
        ),
        "rerank" => (state.config.rerank_url.clone(), "/v1/rerank"),
        _ => unreachable!("unsupported embedding service"),
    };
    let role = service;
    let (node, had_registered_pool) = {
        let registry = state.registry.read().await;
        let had_pool = registry.nodes().iter().any(|n| n.role == role);
        (registry.pick(role), had_pool)
    };
    if had_registered_pool && node.is_none() {
        let body = serde_json::json!({"error": {
            "message": format!("No healthy {role} backend is available; start the configured native/container service and check its health."),
            "type": "cascade_backend_unavailable", "backend": role, "code": 503
        }});
        return Response::builder().status(StatusCode::SERVICE_UNAVAILABLE)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string())).unwrap();
    }
    let node_model = node.as_ref().and_then(|n| n.model.clone()).filter(|m| !m.is_empty());
    let (target_url, bearer, node_id) = if let Some(node) = node {
        let base = node.endpoint_url.trim_end_matches('/');
        let target = if base.ends_with(endpoint) {
            base.to_string()
        } else if base.ends_with("/v1") {
            format!("{}{}", base.trim_end_matches("/v1"), endpoint)
        } else {
            format!("{}{}", base, endpoint)
        };
        (target, node.bearer_token, Some(node.id))
    } else {
        (format!("{}{}", default_url.trim_end_matches('/'), endpoint), None, None)
    };
    info!("{} proxy: {} -> {}", service, req.uri(), target_url);

    let method = req.method().clone();
    let headers = req.headers().clone();
    let body = match req.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from("Failed to read request body"))
                .unwrap()
        }
    };
    let request_body = body.clone();
    // Per-node model alias (hosted APIs): rewrite the JSON body's `model`.
    let send_body = match node_model.as_deref() {
        Some(model) => registry::force_model_json(&body, model)
            .map(axum::body::Bytes::from)
            .unwrap_or_else(|| body.clone()),
        None => body,
    };

    // reqwest recomputes content-length; the rewritten body may differ in size.
    let mut upstream = state.http_client.request(method, target_url);
    for (key, value) in &headers {
        if key != "host" && key != "authorization" && key != "content-length" {
            upstream = upstream.header(key, value);
        }
    }
    if let Some(token) = bearer.as_deref().filter(|t| !t.is_empty()) {
        upstream = upstream.bearer_auth(token);
    } else if let Some(value) = headers.get("authorization") {
        upstream = upstream.header("authorization", value);
    }

    match upstream.body(send_body).send().await {
        Ok(response) => {
            let status = response.status();
            let response_headers = response.headers().clone();
            let body = match response.bytes().await {
                Ok(body) => body,
                Err(_) => {
                    return Response::builder()
                        .status(StatusCode::BAD_GATEWAY)
                        .body(Body::from("Failed to read upstream response"))
                        .unwrap()
                }
            };
            if status.is_success() {
                state.usage.record(meter_json_api(
                    &headers, service, status.as_u16(), &request_body, &body, started,
                ));
            }
            let mut result = Response::new(Body::from(body));
            *result.status_mut() = status;
            for (key, value) in &response_headers {
                if key != "content-length" && key != "transfer-encoding" {
                    result.headers_mut().insert(key.clone(), value.clone());
                }
            }
            result
        }
        Err(error) => {
            warn!("{} proxy error: {}", service, error);
            Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::json!({"error": {
                    "message": format!("{} backend {} is unreachable: {}. Start or restart that native/container service, then retry.", role, node_id.as_deref().unwrap_or("default"), error),
                    "type": "cascade_backend_unavailable", "backend": role, "code": 503
                }}).to_string()))
                .unwrap()
        }
    }
}

/// Usage for a pass-through embeddings or rerank call. Embeddings count prompt
/// tokens (upstream `usage.prompt_tokens`, else an input-length estimate); rerank
/// is recorded as a call only.
fn meter_json_api(
    headers: &HeaderMap,
    service: &str,
    status: u16,
    request: &[u8],
    response: &[u8],
    started: Instant,
) -> UsageEvent {
    let kind = if service == "rerank" { "rerank" } else { "embeddings" };
    let mut ev = passthrough_usage(headers, kind, service, status, started);
    let req_json: Value = serde_json::from_slice(request).unwrap_or(Value::Null);
    let resp_json: Value = serde_json::from_slice(response).unwrap_or(Value::Null);
    ev.model = resp_json
        .get("model")
        .and_then(|m| m.as_str())
        .or_else(|| req_json.get("model").and_then(|m| m.as_str()))
        .unwrap_or_default()
        .to_string();
    if kind == "embeddings" {
        match resp_json.pointer("/usage/prompt_tokens").and_then(|v| v.as_u64()) {
            Some(p) => ev.prompt_tokens = p,
            None => {
                ev.prompt_tokens = json_input_chars(req_json.get("input")) / 4;
                ev.estimated = true;
            }
        }
    }
    ev
}

/// Characters in an OpenAI-style `input` (a string, or an array of strings).
fn json_input_chars(input: Option<&Value>) -> u64 {
    match input {
        Some(Value::String(s)) => s.chars().count() as u64,
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.chars().count() as u64)
            .sum(),
        _ => 0,
    }
}
