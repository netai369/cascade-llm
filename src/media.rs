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

pub async fn image_generation_handler(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    let target_url = std::env::var("IMAGE_GENERATION_URL")
        .unwrap_or_else(|_| "http://localhost:8080/v1/images/generations".to_string());
    info!("Image generation proxy: {} -> {}", req.uri(), target_url);
    proxy_request(state, req, &target_url).await
}

pub async fn video_generation_handler(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    let target_url = std::env::var("VIDEO_GENERATION_URL")
        .unwrap_or_else(|_| "http://localhost:8080/v1/video/generations".to_string());
    info!("Video generation proxy: {} -> {}", req.uri(), target_url);
    proxy_request(state, req, &target_url).await
}

async fn proxy_request(state: Arc<GatewayState>, req: Request<Body>, target_url: &str) -> Response {
    let method = req.method().clone();
    let headers = req.headers().clone();

    let body_bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from("Failed to read request body"))
                .unwrap();
        }
    };

    let mut proxy_req = state.http_client.request(method, target_url);
    for (key, value) in headers.iter() {
        if key != "host" {
            proxy_req = proxy_req.header(key.clone(), value.clone());
        }
    }

    match proxy_req.body(body_bytes).send().await {
        Ok(resp) => {
            let status = resp.status();
            let resp_headers = resp.headers().clone();
            let body = match resp.bytes().await {
                Ok(b) => b,
                Err(_) => {
                    return Response::builder()
                        .status(StatusCode::BAD_GATEWAY)
                        .body(Body::from("Failed to read upstream response"))
                        .unwrap();
                }
            };
            let mut response = Response::new(Body::from(body));
            *response.status_mut() = status;
            for (key, value) in resp_headers.iter() {
                if key != "content-length" && key != "transfer-encoding" {
                    response
                        .headers_mut()
                        .insert(key.clone(), value.clone());
                }
            }
            response
        }
        Err(e) => {
            warn!("Media proxy error: {}", e);
            Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Body::from(format!("Proxy error: {}", e)))
                .unwrap()
        }
    }
}
