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
    let (base_url, endpoint) = match service {
        "embeddings" => (
            state.config.embeddings_url.clone(),
            "/v1/embeddings",
        ),
        "rerank" => (state.config.rerank_url.clone(), "/v1/rerank"),
        _ => unreachable!("unsupported embedding service"),
    };
    let target_url = format!("{}{}", base_url.trim_end_matches('/'), endpoint);
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
        if key != "host" {
            upstream = upstream.header(key, value);
        }
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
                .status(StatusCode::BAD_GATEWAY)
                .body(Body::from(format!("Proxy error: {}", error)))
                .unwrap()
        }
    }
}
