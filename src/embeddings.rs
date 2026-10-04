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

use crate::state::GatewayState;

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
    let (default_url, endpoint) = match service {
        "embeddings" => (
            state.config.embeddings_url.clone(),
            "/v1/embeddings",
        ),
        "rerank" => (state.config.rerank_url.clone(), "/v1/rerank"),
        _ => unreachable!("unsupported embedding service"),
    };
    let (node, had_registered_pool) = {
        let registry = state.registry.read().await;
        let had_pool = registry.nodes().iter().any(|n| n.role == "embeddings");
        (registry.pick("embeddings"), had_pool)
    };
    if had_registered_pool && node.is_none() {
        let body = serde_json::json!({"error": {
            "message": format!("No healthy embeddings backend is available for {service}; start the configured native/container service and check its health."),
            "type": "cascade_backend_unavailable", "backend": "embeddings", "code": 503
        }});
        return Response::builder().status(StatusCode::SERVICE_UNAVAILABLE)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string())).unwrap();
    }
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

    let mut upstream = state.http_client.request(method, target_url);
    for (key, value) in &headers {
        if key != "host" && key != "authorization" {
            upstream = upstream.header(key, value);
        }
    }
    if let Some(token) = bearer.as_deref().filter(|t| !t.is_empty()) {
        upstream = upstream.bearer_auth(token);
    } else if let Some(value) = headers.get("authorization") {
        upstream = upstream.header("authorization", value);
    }

    match upstream.body(body).send().await {
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
                    "message": format!("{} backend {} is unreachable: {}. Start or restart that native/container service, then retry.", service, node_id.as_deref().unwrap_or("default"), error),
                    "type": "cascade_backend_unavailable", "backend": "embeddings", "code": 503
                }}).to_string()))
                .unwrap()
        }
    }
}
