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

/// Where one audio call goes: a registry node (roles `speech` /
/// `transcription`) or the static `TTS_URL` / `STT_URL` fallback.
struct AudioUpstream {
    url: String,
    bearer: Option<String>,
    node_id: Option<String>,
    /// Model forced onto the request body (hosted/alias node), if any.
    model: Option<String>,
}

/// Registry-first selection with the same weighted round-robin / least-latency
/// logic as chat. Without an active node the static base URL is used exactly
/// as before this routing existed.
async fn select_audio_upstream(
    state: &Arc<GatewayState>,
    role: &str,
    static_base: &str,
    path: &str,
) -> AudioUpstream {
    if let Some(node) = state.pick_node(role).await {
        info!("AUDIO_NODE: {} -> {} ({})", role, node.endpoint_url, node.id);
        return AudioUpstream {
            url: node.endpoint_url,
            bearer: node.bearer_token,
            node_id: Some(node.id),
            model: node.model,
        };
    }
    AudioUpstream {
        url: audio_target(static_base, path),
        bearer: None,
        node_id: None,
        model: None,
    }
}

pub async fn tts_handler(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    let static_base = state.runtime.read().await.tts_url.clone();
    let mut upstream = select_audio_upstream(&state, "speech", &static_base, req.uri().path()).await;
    // ★ Inject the TTS bearer when neither the node nor the caller brings one.
    //
    // The key sits in the container environment (`POCKET_API_KEY`) and was
    // never read — see the field docstring in config.rs. Without this the
    // route answers 401 to every client that does not send it, which is what
    // it did until 2026-10-07. A client-supplied Authorization is kept as-is.
    if upstream.bearer.is_none() && !req.headers().contains_key(axum::http::header::AUTHORIZATION) {
        upstream.bearer = state.config.tts_api_key.clone();
    }
    info!("TTS proxy: {} -> {}", req.uri(), upstream.url);
    // Speech output is streamed to the client; usage is metered from the request.
    proxy_request(state, req, upstream, "audio_speech", "tts", true).await
}

pub async fn stt_handler(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    let static_base = state.runtime.read().await.stt_url.clone();
    let upstream = select_audio_upstream(&state, "transcription", &static_base, req.uri().path()).await;
    info!("STT proxy: {} -> {}", req.uri(), upstream.url);
    proxy_request(state, req, upstream, "audio_transcription", "stt", false).await
}

/// Speaker diarization (`/v1/audio/diarize`) — identical to the STT proxy.
///
/// Kept as its own entry point rather than folding into `stt_handler` so the
/// two leaves stay independently visible in the logs; the URL is composed by
/// the same `audio_target()`, so there is still exactly one composition rule.
/// Diarization always uses the static `STT_URL`: the `transcription` registry
/// role serves `/v1/audio/transcriptions` only.
pub async fn stt_diarize_handler(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    let target_url = audio_target(
        &state.runtime.read().await.stt_url,
        req.uri().path(),
    );
    info!("STT diarize proxy: {} -> {}", req.uri(), target_url);
    let upstream = AudioUpstream { url: target_url, bearer: None, node_id: None, model: None };
    proxy_request(state, req, upstream, "audio_transcription", "stt", false).await
}

async fn proxy_request(
    state: Arc<GatewayState>,
    req: Request<Body>,
    upstream: AudioUpstream,
    usage_kind: &'static str,
    usage_route: &'static str,
    stream_response: bool,
) -> Response {
    let started = Instant::now();
    let method = req.method().clone();
    let headers = req.headers().clone();
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();

    let body_bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from("Failed to read request body"))
                .unwrap();
        }
    };
    // Metering reads the caller's original body; the node receives the rewrite.
    let request_bytes = body_bytes.clone();
    let send_bytes: Vec<u8> = match upstream.model.as_deref().filter(|m| !m.is_empty()) {
        Some(model) => {
            let rewritten = if content_type.starts_with("multipart/form-data") {
                registry::force_model_multipart(&body_bytes, &content_type, model)
            } else {
                registry::force_model_json(&body_bytes, model)
            };
            rewritten.unwrap_or_else(|| body_bytes.to_vec())
        }
        None => body_bytes.to_vec(),
    };

    // Content-length is recomputed by reqwest: a model rewrite changes it.
    let mut proxy_req = state.http_client.request(method, &upstream.url);
    for (key, value) in headers.iter() {
        if key == "host" || key == "content-length" {
            continue;
        }
        if key == "authorization" && upstream.bearer.is_some() {
            continue;
        }
        proxy_req = proxy_req.header(key.clone(), value.clone());
    }
    if let Some(token) = upstream.bearer.as_deref().filter(|t| !t.is_empty()) {
        proxy_req = proxy_req.bearer_auth(token);
    }

    match proxy_req.body(send_bytes).send().await {
        Ok(resp) => {
            let status = resp.status();
            let resp_headers = resp.headers().clone();
            if let Some(id) = &upstream.node_id {
                let latency = started.elapsed().as_millis() as u64;
                state.registry.write().await.record_request_latency(id, latency);
            }
            let mut response = if stream_response && status.is_success() {
                // One usage event per call, recorded when the stream starts:
                // speech bills the request's characters, not the audio bytes.
                state.usage.record(meter_audio(
                    &headers, usage_kind, usage_route, status.as_u16(),
                    &request_bytes, &[], started,
                ));
                Response::new(Body::from_stream(resp.bytes_stream()))
            } else {
                let body = match resp.bytes().await {
                    Ok(b) => b,
                    Err(_) => {
                        return Response::builder()
                            .status(StatusCode::BAD_GATEWAY)
                            .body(Body::from("Failed to read upstream response"))
                            .unwrap();
                    }
                };
                if status.is_success() {
                    state.usage.record(meter_audio(
                        &headers, usage_kind, usage_route, status.as_u16(),
                        &request_bytes, &body, started,
                    ));
                }
                Response::new(Body::from(body))
            };
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

/// Usage for a proxied audio call. TTS (`audio_speech`) bills input characters;
/// STT (`audio_transcription`) bills the response's `duration` in seconds.
fn meter_audio(
    headers: &HeaderMap,
    kind: &str,
    route: &str,
    status: u16,
    request: &[u8],
    response: &[u8],
    started: Instant,
) -> UsageEvent {
    let mut ev = passthrough_usage(headers, kind, route, status, started);
    let req_json: Value = serde_json::from_slice(request).unwrap_or(Value::Null);
    let resp_json: Value = serde_json::from_slice(response).unwrap_or(Value::Null);
    ev.model = resp_json
        .get("model")
        .and_then(|m| m.as_str())
        .or_else(|| req_json.get("model").and_then(|m| m.as_str()))
        .unwrap_or_default()
        .to_string();
    match kind {
        "audio_speech" => {
            ev.characters = req_json
                .get("input")
                .and_then(|v| v.as_str())
                .map(|s| s.chars().count() as u64)
                .unwrap_or(0);
        }
        _ => {
            ev.audio_seconds = resp_json.get("duration").and_then(|v| v.as_f64()).unwrap_or(0.0);
        }
    }
    ev
}
