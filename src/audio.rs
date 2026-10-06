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

/// The audio path prefix every STT/TTS service in this stack exposes.
pub const AUDIO_PREFIX: &str = "/v1/audio";

/// Compose an upstream audio URL from a configured base and a request path.
///
/// ★ **THE CONVENTION LIVES HERE, AND IT IS IDEMPOTENT ON PURPOSE.**
///
/// `STT_URL` / `TTS_URL` are **bases**. Their trailing `/v1/audio` is
/// optional — the binary default carries it, `compose/base.yml` sets it, and
/// both forms have been shipped. This function therefore normalises *both*
/// into the same result, which is what makes the double-prefix 404
/// unrepresentable instead of merely untested:
///
/// ```text
/// base .../v1/audio  +  /v1/audio/transcriptions  ->  .../v1/audio/transcriptions
/// base (no suffix)   +  /v1/audio/transcriptions  ->  .../v1/audio/transcriptions
/// base .../v1/audio  +  /transcriptions            ->  .../v1/audio/transcriptions
/// base (no suffix)   +  /transcriptions            ->  .../v1/audio/transcriptions
/// ```
///
/// **Why this is a bug class and not a typo.** The handlers used to inline
/// `format!("{}{}", stt_url, path.replacen("/v1/audio", "", 1))` — strip —
/// while a second audio path in `handlers.rs` built
/// `format!("{}/v1/audio/transcriptions", stt_url)` — append — from the *same*
/// variable. Commit `2c6ed0b` (2026-08-25) added the suffix to fix the strip
/// side (`was /transcriptions -> 404`, its own words) and broke the append
/// side in the same line. Nothing noticed for 36 days because nothing tested
/// the composition. The path that appended was removed 2026-10-06
/// (`6197506`); this function is the structural answer, so the next one has
/// nothing left to get wrong.
pub fn audio_target(base: &str, path: &str) -> String {
    let host = {
        let b = base.trim_end_matches('/');
        b.strip_suffix(AUDIO_PREFIX).unwrap_or(b)
    };
    let leaf = {
        let p = path.trim_start_matches('/').trim_end_matches('/');
        p.strip_prefix("v1/audio")
            .map(|r| r.trim_start_matches('/'))
            .unwrap_or(p)
    };
    format!("{}{}/{}", host, AUDIO_PREFIX, leaf)
}

pub async fn tts_handler(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    let target_url = audio_target(
        &state.runtime.read().await.tts_url,
        req.uri().path(),
    );
    info!("TTS proxy: {} -> {}", req.uri(), target_url);
    proxy_request(state, req, &target_url).await
}

pub async fn stt_handler(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    let target_url = audio_target(
        &state.runtime.read().await.stt_url,
        req.uri().path(),
    );
    info!("STT proxy: {} -> {}", req.uri(), target_url);
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
            warn!("Audio proxy error: {}", e);
            Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Body::from(format!("Proxy error: {}", e)))
                .unwrap()
        }
    }
}
