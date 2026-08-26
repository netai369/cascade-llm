//! Media generation proxies.
//!
//! Image routing (spec route A): an active `image` registry node wins; the
//! static IMAGE_GENERATION_URL is the fallback. Nodes with
//! `provider: pollinations` get an adapter that translates the OpenAI images
//! request to Pollinations' GET-based API and returns OpenAI-shaped JSON.

use axum::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    response::Response,
};
use http_body_util::BodyExt;
use std::sync::Arc;
use tracing::{info, warn};

use crate::registry::UpstreamNode;
use crate::state::GatewayState;

const POLLINATIONS_DEFAULT_MODEL: &str = "flux";

pub async fn image_generation_handler(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    let node = state.pick_node("image").await;
    if let Some(node) = &node {
        if is_pollinations(node) {
            return pollinations_image(&state, req, node).await;
        }
    }
    // Dynamic routing: an active `image` registry node wins; the static
    // IMAGE_GENERATION_URL is the fallback for deployments without one.
    let target_url = match &node {
        Some(node) => {
            info!("Image generation via node '{}' -> {}", node.id, node.endpoint_url);
            node.endpoint_url.clone()
        }
        None => state.config.image_generation_url.clone(),
    };
    info!("Image generation proxy: {} -> {}", req.uri(), target_url);
    proxy_request(state, req, &target_url).await
}

pub async fn video_generation_handler(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    let target_url = match state.pick_node("image").await {
        Some(node) if node.endpoint_url.contains("/v1/video") => node.endpoint_url,
        _ => state.config.video_generation_url.clone(),
    };
    info!("Video generation proxy: {} -> {}", req.uri(), target_url);
    proxy_request(state, req, &target_url).await
}

fn is_pollinations(node: &UpstreamNode) -> bool {
    node.provider.as_deref() == Some("pollinations")
        || node.endpoint_url.contains("pollinations.ai")
}

fn json_response(body: serde_json::Value, status: StatusCode) -> Response {
    let mut resp = Response::new(Body::from(body.to_string()));
    *resp.status_mut() = status;
    resp.headers_mut()
        .insert("content-type", "application/json".parse().unwrap());
    resp
}

/// Percent-encodes a prompt for the Pollinations URL path.
fn encode_path_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// Builds the Pollinations GET URL from the OpenAI-style parameters.
/// A registered endpoint containing `/prompt/` is used verbatim; otherwise it
/// is treated as the service base URL (`https://image.pollinations.ai`).
fn build_pollinations_url(
    base: &str,
    prompt: &str,
    model: Option<&str>,
    width: u32,
    height: u32,
    seed: Option<u64>,
) -> String {
    let mut url = if base.contains("/prompt/") {
        base.to_string()
    } else {
        let base = base.trim_end_matches('/');
        format!("{base}/prompt/{}", encode_path_segment(prompt))
    };
    let mut params: Vec<String> = vec![
        format!("width={width}"),
        format!("height={height}"),
        format!("model={}", model.unwrap_or(POLLINATIONS_DEFAULT_MODEL)),
    ];
    if let Some(seed) = seed {
        params.push(format!("seed={seed}"));
    }
    params.push("nologo=true".to_string());
    // Verbatim endpoints may carry their own params — never duplicate a key.
    let existing = url.split_once('?').map(|(_, q)| q).unwrap_or("");
    params.retain(|p| {
        !existing
            .split('&')
            .any(|kv| kv.split('=').next() == p.split('=').next())
    });
    if params.is_empty() {
        return url;
    }
    url.push(if url.contains('?') { '&' } else { '?' });
    url.push_str(&params.join("&"));
    url
}

/// Parses an OpenAI `size` string ("1024x1024"); falls back to 1024².
fn parse_size(size: Option<&str>) -> (u32, u32) {
    const MAX: u32 = 2048;
    size.and_then(|s| {
        s.split_once('x').and_then(|(w, h)| {
            Some((
                w.trim().parse::<u32>().ok()?,
                h.trim().parse::<u32>().ok()?,
            ))
        })
    })
    .map(|(w, h)| (w.clamp(64, MAX), h.clamp(64, MAX)))
    .unwrap_or((1024, 1024))
}

async fn pollinations_image(state: &Arc<GatewayState>, req: Request<Body>, node: &UpstreamNode) -> Response {
    let body_bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => return json_response(serde_json::json!({"error": {"message": "invalid body", "code": 400}}), StatusCode::BAD_REQUEST),
    };
    let payload: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(_) => return json_response(serde_json::json!({"error": {"message": "invalid JSON", "code": 400}}), StatusCode::BAD_REQUEST),
    };
    let prompt = payload.get("prompt").and_then(|p| p.as_str()).unwrap_or_default().to_string();
    if prompt.trim().is_empty() {
        return json_response(
            serde_json::json!({"error": {"message": "'prompt' is required", "code": 400}}),
            StatusCode::BAD_REQUEST,
        );
    }
    let (width, height) = parse_size(payload.get("size").and_then(|s| s.as_str()));
    let model = payload.get("model").and_then(|m| m.as_str()).filter(|m| !m.is_empty());
    let seed = payload.get("seed").and_then(|s| s.as_u64());
    let want_b64 = payload.get("response_format").and_then(|r| r.as_str()) != Some("url");
    let pollinations_url = build_pollinations_url(
        &node.endpoint_url, &prompt, model, width, height, seed,
    );
    info!("POLLINATIONS_IMAGE: {} (node {})", &pollinations_url, node.id);

    let mut request = state.http_client.get(&pollinations_url);
    if let Some(token) = &node.bearer_token {
        request = request.bearer_auth(token);
    }
    match request.send().await {
        Ok(resp) if resp.status().is_success() => {
            let content_type = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("image/jpeg")
                .to_string();
            match resp.bytes().await {
                Ok(bytes) => {
                    state.metrics.record_request("image");
                    let data_item = if want_b64 {
                        use base64::Engine as _;
                        serde_json::json!({ "b64_json": base64::engine::general_purpose::STANDARD.encode(&bytes) })
                    } else {
                        serde_json::json!({ "url": pollinations_url })
                    };
                    json_response(
                        serde_json::json!({
                            "created": chrono::Utc::now().timestamp(),
                            "model": model.unwrap_or(POLLINATIONS_DEFAULT_MODEL),
                            "data": [data_item],
                        }),
                        StatusCode::OK,
                    )
                    .tap_content_type(&content_type)
                }
                Err(e) => {
                    warn!("Pollinations read error: {e}");
                    json_response(
                        serde_json::json!({"error": {"message": format!("pollinations read failed: {e}"), "type": "image_backend_error", "code": 502}}),
                        StatusCode::BAD_GATEWAY,
                    )
                }
            }
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            warn!("Pollinations HTTP {}: {}", status, body.chars().take(200).collect::<String>());
            json_response(
                serde_json::json!({
                    "error": {"message": format!("pollinations returned HTTP {status}"), "type": "image_backend_error", "code": status.as_u16()}
                }),
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            )
        }
        Err(e) => {
            warn!("Pollinations unreachable: {e}");
            json_response(
                serde_json::json!({"error": {"message": format!("pollinations unreachable: {e}"), "type": "image_backend_unavailable", "code": 502}}),
                StatusCode::BAD_GATEWAY,
            )
        }
    }
}

/// Small helper so success responses can carry through a header without a
/// builder dance at every call site.
trait TapContentType {
    fn tap_content_type(self, ct: &str) -> Response;
}

impl TapContentType for Response {
    fn tap_content_type(mut self, ct: &str) -> Response {
        if let Ok(v) = ct.parse() {
            self.headers_mut().insert("content-type", v);
        }
        self
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_path_segments() {
        assert_eq!(encode_path_segment("a cat & dog!"), "a%20cat%20%26%20dog%21");
        assert_eq!(encode_path_segment("Straße"), "Stra%C3%9Fe");
        assert_eq!(encode_path_segment("-_.~"), "-_.~");
    }

    #[test]
    fn builds_pollinations_urls() {
        let url = build_pollinations_url("https://image.pollinations.ai", "a cat", None, 1024, 1024, None);
        assert_eq!(url, "https://image.pollinations.ai/prompt/a%20cat?width=1024&height=1024&model=flux&nologo=true");

        let url = build_pollinations_url("https://image.pollinations.ai/", "x", Some("turbo"), 512, 768, Some(42));
        assert!(url.ends_with("/prompt/x?width=512&height=768&model=turbo&seed=42&nologo=true"), "{url}");

        // Full /prompt/ endpoints are extended with & (not a second ?).
        assert_eq!(
            build_pollinations_url("https://img.example.com/prompt/pre?width=1", "ignored", None, 1, 1, None),
            "https://img.example.com/prompt/pre?width=1&height=1&model=flux&nologo=true"
        );
    }

    #[test]
    fn parses_sizes_with_clamp_and_fallback() {
        assert_eq!(parse_size(Some("512x768")), (512, 768));
        assert_eq!(parse_size(Some("4096x10")), (2048, 64), "clamped into range");
        assert_eq!(parse_size(None), (1024, 1024));
        assert_eq!(parse_size(Some("garbage")), (1024, 1024));
    }
}
