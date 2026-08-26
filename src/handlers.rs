use crate::registry::RegisterUpstreamRequest;
use crate::state::GatewayState;
use crate::types::*;
use axum::{
    body::Body,
    extract::{Path, Request, State},
    http::{HeaderValue, StatusCode},
    response::Response,
    Json,
};
use http_body_util::BodyExt;
use base64::Engine;
use std::sync::Arc;
use tracing::info;

fn json_response(body: serde_json::Value, status: StatusCode) -> Response {
    let body_str = body.to_string();
    let mut resp = Response::new(Body::from(body_str));
    *resp.status_mut() = status;
    resp.headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    resp
}

fn extract_audio_from_messages(messages: &[ChatMessage]) -> Option<(String, String)> {
    for msg in messages {
        if let Some(MessageContent::Parts(parts)) = &msg.content {
            for part in parts {
                if let MessageContentPart::InputAudio { input_audio } = part {
                    let format = input_audio.format.clone().unwrap_or_else(|| "mp3".to_string());
                    return Some((input_audio.data.clone(), format));
                }
            }
        }
    }
    None
}

async fn transcribe_audio(state: &Arc<GatewayState>, audio_data: &str, format: &str) -> Result<String, String> {
    let stt_url = &state.config.stt_url;
    
    let audio_bytes = base64::engine::general_purpose::STANDARD
        .decode(audio_data)
        .map_err(|e| format!("Failed to decode base64 audio: {}", e))?;
    
    let mime_type = match format {
        "mp3" | "mpeg" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        "webm" => "audio/webm",
        _ => "audio/mpeg",
    };
    
    let part_name = format!("file.{}", format);
    let file_part = reqwest::multipart::Part::bytes(audio_bytes)
        .file_name(part_name)
        .mime_str(mime_type)
        .map_err(|e| format!("Failed to create multipart part: {}", e))?;
    
    let form = reqwest::multipart::Form::new()
        .part("file", file_part)
        .text("model", "parakeet-tdt-0.6b-v3")
        .text("response_format", "text");
    
    let url = format!("{}/v1/audio/transcriptions", stt_url);
    info!("STT proxy: transcribing audio via {}", url);
    
    let resp = state.http_client
        .post(&url)
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("STT request failed: {}", e))?;
    
    let status = resp.status();
    let body = resp.text().await.map_err(|e| format!("Failed to read STT response: {}", e))?;
    
    if !status.is_success() {
        return Err(format!("STT returned {}: {}", status, body));
    }
    
    Ok(body)
}

pub async fn chat_completions(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    let headers = req.headers().clone();
    let origin = headers
        .get("x-request-origin")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown");
    state.metrics.record_request_origin(&origin);
    let body_bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => {
            return json_response(
                serde_json::json!({
                    "error": {
                        "message": "Invalid request body",
                        "type": "cascade_proxy_error",
                        "param": serde_json::Value::Null,
                        "code": 400
                    }
                }),
                StatusCode::BAD_REQUEST,
            );
        }
    };

    let json: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(j) => j,
        Err(_) => {
            return json_response(
                serde_json::json!({
                    "error": {
                        "message": "Invalid JSON",
                        "type": "cascade_proxy_error",
                        "param": serde_json::Value::Null,
                        "code": 400
                    }
                }),
                StatusCode::BAD_REQUEST,
            );
        }
    };

    let is_streaming = json.stream.unwrap_or(false);
    let tier = headers
        .get("x-tier")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("standard");

    if let Some((audio_data, format)) = extract_audio_from_messages(&json.messages) {
        info!("Audio content detected, routing to STT");
        match transcribe_audio(&state, &audio_data, &format).await {
            Ok(transcription) => {
                state.metrics.record_request("stt_proxy");
                let response = serde_json::json!({
                    "id": "cascade-audio-transcription",
                    "object": "chat.completion",
                    "created": chrono::Utc::now().timestamp(),
                    "model": json.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": transcription
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                });
                return json_response(response, StatusCode::OK);
            }
            Err(e) => {
                return json_response(
                    serde_json::json!({
                        "error": {
                            "message": format!("Audio transcription failed: {}", e),
                            "type": "cascade_proxy_error",
                            "param": serde_json::Value::Null,
                            "code": 500
                        }
                    }),
                    StatusCode::INTERNAL_SERVER_ERROR,
                );
            }
        }
    }

    match state
        .route_request_with_fallback(json, is_streaming, tier, &headers, origin)
        .await
    {
        Ok((hdrs, res_body)) => {
            let mut response = Response::new(res_body);
            *response.headers_mut() = hdrs;
            response
        }
        Err(e) => {
            let status = StatusCode::from_u16(e.status).unwrap_or(StatusCode::BAD_GATEWAY);
            let (err_type, details) = if e.unreachable {
                ("cascade_backend_unavailable".to_string(), format!("{} is unreachable: {}", e.backend, e.context))
            } else {
                ("cascade_backend_error".to_string(), format!("{}: {}", e.backend, e.context))
            };
            json_response(
                serde_json::json!({
                    "error": {
                        "message": details,
                        "type": err_type,
                        "backend": e.backend,
                        "param": serde_json::Value::Null,
                        "code": serde_json::Value::Number(status.as_u16().into())
                    }
                }),
                status,
            )
        }
    }
}

async fn fetch_models_from(client: &reqwest::Client, base_url: &str, model_type: &str) -> Vec<serde_json::Value> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<serde_json::Value>().await {
                Ok(val) => {
                    let mut models = val.get("data")
                        .and_then(|d| d.as_array())
                        .cloned()
                        .unwrap_or_default();
                    for m in &mut models {
                        if let Some(obj) = m.as_object_mut() {
                            obj.insert("model_type".to_string(), serde_json::Value::String(model_type.to_string()));
                        }
                    }
                    models
                }
                Err(e) => { tracing::warn!("Failed to parse models from {}: {}", url, e); vec![] }
            }
        }
        Ok(resp) => { tracing::warn!("Models fetch from {} returned HTTP {}", url, resp.status()); vec![] }
        Err(e) => { tracing::warn!("Failed to connect to {}: {}", url, e); vec![] }
    }
}

pub async fn list_models(State(state): State<Arc<GatewayState>>) -> Response {
    let large_models = fetch_models_from(&state.http_client, &state.config.large_text_url.replace("/v1/chat/completions", ""), "Main").await;
    let small_models = fetch_models_from(&state.http_client, &state.config.small_mllm_url.replace("/v1/chat/completions", ""), "Compression").await;
    let ocr_models = fetch_models_from(&state.http_client, &state.config.ocr_server_url.replace("/v1/chat/completions", ""), "OCR").await;

    let mut all_models = large_models;
    all_models.extend(small_models);
    all_models.extend(if ocr_models.is_empty() {
        // Backend not reachable yet — still advertise the configured OCR model id.
        serde_json::to_value(build_model_info(&state.config.ocr_model_name))
            .ok()
            .filter(|m| m.is_object())
            .into_iter()
            .collect()
    } else {
        ocr_models
    });

    json_response(
        serde_json::json!({
            "data": all_models,
            "object": "list"
        }),
        StatusCode::OK,
    )
}

pub async fn get_model(State(state): State<Arc<GatewayState>>) -> Response {
    let model = build_model_info(&state.config.main_model_name);
    json_response(serde_json::to_value(model).unwrap(), StatusCode::OK)
}

pub async fn health_check(State(state): State<Arc<GatewayState>>) -> Response {
    let extract_backends = state.extraction_backends.read().await;
    let extract_summary: Vec<serde_json::Value> = extract_backends
        .iter()
        .map(|b| {
            serde_json::json!({
                "id": b.id,
                "type": b.backend_type,
                "enabled": b.enabled,
                "healthy": b.healthy,
                "priority": b.priority,
            })
        })
        .collect();

    json_response(
        serde_json::json!({
            "status": "ok",
            "large_model_multimodal": state.config.large_model_multimodal,
            "router_threshold": state.config.router_threshold,
            "confidence_threshold": state.config.confidence_threshold,
            "ocr_server_url": state.config.ocr_server_url,
            "auxiliary_server_url": state.config.small_mllm_url,
            "inference_server_url": state.config.large_text_url,
            "extract_fallback_url": state.config.extract_fallback_url,
            "session_cache_entries": state.session_cache.entry_count() as u64,
            "uptime_seconds": state.start_time.elapsed().as_secs(),
            "providers": state.config.providers.len(),
            "extraction_backends": extract_summary,
        }),
        StatusCode::OK,
    )
}

pub async fn tts(State(state): State<Arc<GatewayState>>, req: Request<Body>) -> Response {
    crate::audio::tts_handler(State(state), req).await
}

pub async fn stt(State(state): State<Arc<GatewayState>>, req: Request<Body>) -> Response {
    crate::audio::stt_handler(State(state), req).await
}

pub async fn image_generation(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    crate::media::image_generation_handler(State(state), req).await
}

pub async fn video_generation(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    crate::media::video_generation_handler(State(state), req).await
}

pub async fn dashboard(_state: State<Arc<GatewayState>>) -> Response {
    let html = include_str!("web/dashboard.html");
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .body(Body::from(html))
        .unwrap()
}

pub async fn settings_page(_state: State<Arc<GatewayState>>) -> Response {
    let html = include_str!("web/settings.html");
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .body(Body::from(html))
        .unwrap()
}

pub async fn dashboard_api(State(state): State<Arc<GatewayState>>) -> Response {
    let uptime = state.start_time.elapsed().as_secs();
    let cache_entries = state.session_cache.entry_count() as u64;

    let known_backends = ["small", "large", "large_multimodal", "session_affinity", "stt_proxy", "ocr", "auxiliary", "inference"];
    let mut requests_by_backend = std::collections::HashMap::new();
    let mut total_requests: u64 = 0;
    for backend in &known_backends {
        let val = state
            .metrics
            .requests_total
            .with_label_values(&[backend])
            .get() as u64;
        total_requests += val;
        if val > 0 {
            requests_by_backend.insert(backend.to_string(), val);
        }
    }

    let origin_counts = state.metrics.get_origin_counts().clone();
    let requests_by_origin: std::collections::HashMap<String, u64> = origin_counts
        .iter()
        .filter(|(_, count)| **count > 0)
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    let metrics = DashboardMetrics {
        requests_total: total_requests,
        requests_by_backend,
        fallback_count: ["primary_failed", "quality_low", "timeout",
                         "backend_unavailable", "extraction_backend_unavailable"]
            .iter()
            .map(|r| state.metrics.fallback_triggered.with_label_values(&[r]).get() as u64)
            .sum(),
        uptime_seconds: uptime,
        session_cache_entries: cache_entries,
        large_model_multimodal: state.config.large_model_multimodal,
        requests_by_origin,
    };
    json_response(serde_json::to_value(metrics).unwrap(), StatusCode::OK)
}

pub async fn get_settings(State(state): State<Arc<GatewayState>>) -> Response {
    let mut s = state.config.to_settings();
    let rt = state.runtime.read().await;
    // runtime wins over env defaults
    s.routing.router_threshold = rt.router_threshold;
    s.routing.confidence_threshold = rt.confidence_threshold;
    s.routing.route_tools_to_large = rt.route_tools_to_large;
    s.routing.default_route = rt.default_route.clone();
    s.routing.marker_mode = rt.marker_mode.clone();
    s.audio.tts_url = Some(rt.tts_url.clone());
    s.audio.stt_url = Some(rt.stt_url.clone());
    if !rt.providers.is_empty() {
        s.providers = rt.providers.clone();
    }
    json_response(serde_json::to_value(s).unwrap(), StatusCode::OK)
}

pub async fn update_settings(
    State(state): State<Arc<GatewayState>>,
    Json(settings): Json<Settings>,
) -> Response {
    // Apply to the live runtime…
    {
        let mut rt = state.runtime.write().await;
        rt.router_threshold = settings.routing.router_threshold.clamp(0.0, 1.0);
        rt.confidence_threshold = settings.routing.confidence_threshold.clamp(0.0, 1.0);
        rt.route_tools_to_large = settings.routing.route_tools_to_large;
        rt.default_route = settings
            .routing
            .default_route
            .as_deref()
            .map(|s| s.to_lowercase())
            .filter(|s| matches!(s.as_str(), "inference" | "auxiliary" | "auto" | ""));  // ocr retired
        if rt.default_route.as_deref() == Some("") {
            rt.default_route = None;
        }
        rt.marker_mode = match settings.routing.marker_mode.as_str() {
            "prefix" => "prefix".to_string(),
            _ => "substring".to_string(),
        };
        if let Some(u) = &settings.audio.tts_url {
            rt.tts_url = u.clone();
        }
        if let Some(u) = &settings.audio.stt_url {
            rt.stt_url = u.clone();
        }
        if !settings.providers.is_empty() {
            rt.providers = settings.providers.clone();
        }
    }
    // …chat URLs additionally flow into the dynamic registry as manual nodes
    // so the routing layer actually uses them.
    for (role, url_opt, default) in [
        ("auxiliary", &settings.small_mllm_url, &state.config.small_mllm_url),
        ("main", &settings.large_text_url, &state.config.large_text_url),
    ] {
        if let Some(url) = url_opt {
            if url != default && url.starts_with("http") {
                let req = RegisterUpstreamRequest {
                    endpoint_url: Some(url.clone()),
                    id: Some(format!("manual-{}", role)),
                    ..Default::default()
                };
                state.registry.write().await.upsert(role, req);
            }
        }
    }
    // Persist for next boot.
    if let Err(e) = state
        .db
        .save_config("settings", &serde_json::to_string(&settings).unwrap_or_default())
    {
        return json_response(
            serde_json::json!({"error": format!("Failed to save: {}", e)}),
            StatusCode::INTERNAL_SERVER_ERROR,
        );
    }
    info!("SETTINGS updated via UI/API");
    json_response(serde_json::json!({"status": "ok"}), StatusCode::OK)
}

pub async fn list_providers(State(state): State<Arc<GatewayState>>) -> Response {
    let rt = state.runtime.read().await;
    json_response(
        serde_json::to_value(&rt.providers).unwrap_or(serde_json::Value::Array(vec![])),
        StatusCode::OK,
    )
}

pub async fn add_provider(
    State(state): State<Arc<GatewayState>>,
    Json(provider): Json<ProviderConfig>,
) -> Response {
    if let Err(e) = state.db.save_provider(&provider) {
        return json_response(
            serde_json::json!({"error": format!("Failed to save: {}", e)}),
            StatusCode::INTERNAL_SERVER_ERROR,
        );
    }
    {
        let mut rt = state.runtime.write().await;
        rt.providers.retain(|p| p.id != provider.id);
        rt.providers.push(provider.clone());
    }
    json_response(
        serde_json::json!({"status": "created", "id": provider.id}),
        StatusCode::CREATED,
    )
}

pub async fn get_provider(
    State(state): State<Arc<GatewayState>>,
    Path(id): Path<String>,
) -> Response {
    match state.runtime.read().await.providers.iter().find(|p| p.id == id) {
        Some(p) => json_response(serde_json::to_value(p).unwrap(), StatusCode::OK),
        None => json_response(
            serde_json::json!({"error": "Provider not found"}),
            StatusCode::NOT_FOUND,
        ),
    }
}

pub async fn delete_provider(
    State(state): State<Arc<GatewayState>>,
    Path(id): Path<String>,
) -> Response {
    state.db.delete_provider(&id);
    state.runtime.write().await.providers.retain(|p| p.id != id);
    json_response(
        serde_json::json!({"status": "deleted", "id": id}),
        StatusCode::OK,
    )
}

// ============================================================================
// EXTRACTION ENDPOINT  (/v1/extraction)
// ============================================================================

pub async fn extraction_completions(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    let body_bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => {
            return json_response(
                serde_json::json!({
                    "error": { "message": "Invalid request body", "type": "proxy_error", "code": 400 }
                }),
                StatusCode::BAD_REQUEST,
            );
        }
    };

    let json: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(j) => j,
        Err(_) => {
            return json_response(
                serde_json::json!({
                    "error": { "message": "Invalid JSON", "type": "proxy_error", "code": 400 }
                }),
                StatusCode::BAD_REQUEST,
            );
        }
    };

    let is_streaming = json.stream.unwrap_or(false);

    match state.route_extraction(json, is_streaming).await {
        Ok((hdrs, body)) => {
            let mut response = Response::new(body);
            *response.headers_mut() = hdrs;
            response
        }
        Err(e) => {
            let status = StatusCode::from_u16(e.status).unwrap_or(StatusCode::BAD_GATEWAY);
            json_response(
                serde_json::json!({
                    "error": {
                        "message": e.context,
                        "type": if e.unreachable { "extraction_backend_unavailable" } else { "extraction_backend_error" },
                        "backend": e.backend,
                        "param": serde_json::Value::Null,
                        "code": status.as_u16()
                    }
                }),
                status,
            )
        }
    }
}

pub async fn list_extraction_backends(
    State(state): State<Arc<GatewayState>>,
) -> Response {
    let backends = state.extraction_backends.read().await;
    // Redact secrets: never echo full API keys back out.
    let masked: Vec<serde_json::Value> = backends
        .iter()
        .map(|b| {
            let mut v = serde_json::to_value(b).unwrap_or(serde_json::Value::Null);
            if let Some(obj) = v.as_object_mut() {
                if let Some(Some(key)) = obj.get("api_key").map(|k| k.as_str().map(String::from)) {
                    let tail: String = key.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
                    obj.insert("api_key".to_string(), serde_json::Value::String(format!("***{tail}")));
                }
            }
            v
        })
        .collect();
    json_response(
        serde_json::json!({ "backends": masked }),
        StatusCode::OK,
    )
}

// ---------------------------------------------------------------------------
// Dynamic upstream registry API (admin; guarded by CASCADE_ADMIN_KEY)
// ---------------------------------------------------------------------------

fn admin_authorized(state: &GatewayState, headers: &axum::http::HeaderMap) -> bool {
    match state.config.admin_key.as_deref() {
        Some(expected) => headers
            .get("x-cascade-admin-key")
            .and_then(|v| v.to_str().ok())
            .map(|v| v == expected)
            .unwrap_or(false),
        None => false,
    }
}

fn mask_bearer(b: &Option<String>) -> Option<String> {
    b.as_ref().map(|k| {
        let tail: String = k.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
        format!("***{tail}")
    })
}

/// Public JSON view of a node — never leaks the bearer token.
fn node_view(node: &crate::registry::UpstreamNode) -> serde_json::Value {
    let mut v = serde_json::to_value(node).unwrap_or(serde_json::Value::Null);
    if let Some(obj) = v.as_object_mut() {
        obj.insert("bearer_set".into(), serde_json::json!(node.bearer_token.is_some()));
        obj.insert("bearer_masked".into(), serde_json::json!(mask_bearer(&node.bearer_token)));
    }
    v
}

pub async fn list_upstreams(
    State(state): State<Arc<GatewayState>>,
    headers: axum::http::HeaderMap,
) -> Response {
    if !admin_authorized(&state, &headers) {
        return json_response(serde_json::json!({"error": "forbidden"}), StatusCode::FORBIDDEN);
    }
    let registry = state.registry.read().await;
    // Backward-compatible flat view (first node per role) alongside the pool.
    let overrides: serde_json::Map<String, serde_json::Value> = crate::registry::ROLES
        .iter()
        .filter_map(|role| {
            registry
                .nodes()
                .into_iter()
                .find(|n| n.role == *role)
                .map(|n| (role.to_string(), serde_json::json!({"url": n.endpoint_url})))
        })
        .collect();
    json_response(
        serde_json::json!({
            "roles": crate::registry::ROLES,
            "overrides": overrides,
            "nodes": registry.nodes().iter().map(|n| node_view(n)).collect::<Vec<_>>(),
            "strategy": registry.strategy(),
            "defaults": {
                "main": state.config.large_text_url,
                "auxiliary": state.config.small_mllm_url,
                "ocr": state.config.ocr_server_url,
                "image": state.config.image_generation_url,
                "rag_worker": null,
            }
        }),
        StatusCode::OK,
    )
}

pub async fn put_upstream(
    State(state): State<Arc<GatewayState>>,
    Path(role): Path<String>,
    headers: axum::http::HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    if !admin_authorized(&state, &headers) {
        return json_response(serde_json::json!({"error": "forbidden"}), StatusCode::FORBIDDEN);
    }
    let role = match crate::registry::canonical_role(&role) {
        Some(r) => r.to_string(),
        None => {
            return json_response(
                serde_json::json!({"error": format!("role must be one of {}", crate::registry::ROLES.join("|"))}),
                StatusCode::BAD_REQUEST,
            )
        }
    };
    let req: RegisterUpstreamRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(
                serde_json::json!({"error": format!("invalid payload: {}", e)}),
                StatusCode::BAD_REQUEST,
            )
        }
    };
    let url = match req.endpoint_url.as_deref() {
        Some(u) if u.starts_with("http://") || u.starts_with("https://") => u.to_string(),
        _ => {
            return json_response(
                serde_json::json!({"error": "endpoint_url must be http(s)"}),
                StatusCode::BAD_REQUEST,
            )
        }
    };
    let req = RegisterUpstreamRequest { endpoint_url: Some(url), ..req };

    let result = state.registry.write().await.upsert(&role, req);
    info!("UPSTREAM_PUT: role={} id={} created={}", role, result.node.id, result.created);
    json_response(
        serde_json::json!({
            "status": "ok",
            "role": role,
            "created": result.created,
            "node": node_view(&result.node),
        }),
        if result.created { StatusCode::CREATED } else { StatusCode::OK },
    )
}

pub async fn delete_upstream(
    State(state): State<Arc<GatewayState>>,
    Path(role): Path<String>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(query): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    if !admin_authorized(&state, &headers) {
        return json_response(serde_json::json!({"error": "forbidden"}), StatusCode::FORBIDDEN);
    }
    let role = match crate::registry::canonical_role(&role) {
        Some(r) => r.to_string(),
        None => {
            return json_response(
                serde_json::json!({"error": format!("role must be one of {}", crate::registry::ROLES.join("|"))}),
                StatusCode::BAD_REQUEST,
            )
        }
    };
    // `?id=` targets a single node; without it the whole role pool is cleared.
    if let Some(id) = query.get("id") {
        let removed = state.registry.write().await.remove_id(id);
        return match removed {
            Some(n) if n.role == role => json_response(
                serde_json::json!({"status": "cleared", "id": id}),
                StatusCode::OK,
            ),
            _ => json_response(
                serde_json::json!({"error": "node not found for this role"}),
                StatusCode::NOT_FOUND,
            ),
        };
    }
    let removed = state.registry.write().await.remove_role(&role);
    info!("UPSTREAM_DELETE: role={} removed={}", role, removed);
    json_response(
        serde_json::json!({"status": "cleared", "removed": removed}),
        StatusCode::OK,
    )
}

async fn set_node_active(
    State(state): State<Arc<GatewayState>>,
    Path(id): Path<String>,
    headers: axum::http::HeaderMap,
    active: bool,
) -> Response {
    if !admin_authorized(&state, &headers) {
        return json_response(serde_json::json!({"error": "forbidden"}), StatusCode::FORBIDDEN);
    }
    let found = state.registry.read().await.get(&id).is_some();
    if !found {
        return json_response(
            serde_json::json!({"error": format!("node '{}' not found", id)}),
            StatusCode::NOT_FOUND,
        );
    }
    state.registry.write().await.set_active(&id, active);
    info!("NODE_TOGGLE: id={} active={}", id, active);
    json_response(
        serde_json::json!({"status": "ok", "id": id, "active": active}),
        StatusCode::OK,
    )
}

pub async fn activate_node(
    state: State<Arc<GatewayState>>,
    path: Path<String>,
    headers: axum::http::HeaderMap,
) -> Response {
    set_node_active(state, path, headers, true).await
}

pub async fn deactivate_node(
    state: State<Arc<GatewayState>>,
    path: Path<String>,
    headers: axum::http::HeaderMap,
) -> Response {
    set_node_active(state, path, headers, false).await
}

/// Pool status for admins/UI: nodes incl. health, latency stats and cloud
/// cost badges, plus effective strategy and static defaults.
pub async fn admin_upstreams(
    State(state): State<Arc<GatewayState>>,
    headers: axum::http::HeaderMap,
) -> Response {
    if !admin_authorized(&state, &headers) {
        return json_response(serde_json::json!({"error": "forbidden"}), StatusCode::FORBIDDEN);
    }
    let registry = state.registry.read().await;
    let mut active_roles = serde_json::Map::new();
    for role in crate::registry::ROLES {
        active_roles.insert(
            role.to_string(),
            serde_json::json!(registry.has_active(role)),
        );
    }
    json_response(
        serde_json::json!({
            "roles": crate::registry::ROLES,
            "active_roles": active_roles,
            "strategy": registry.strategy(),
            "nodes": registry.nodes().iter().map(|n| node_view(n)).collect::<Vec<_>>(),
            "defaults": {
                "main": state.config.large_text_url,
                "auxiliary": state.config.small_mllm_url,
                "ocr": state.config.ocr_server_url,
                "image": state.config.image_generation_url,
                "video": state.config.video_generation_url,
            },
            "health": {
                "interval_secs": state.config.health_interval_secs,
                "failure_threshold": state.config.health_failure_threshold,
                "timeout_secs": state.config.health_timeout_secs,
            }
        }),
        StatusCode::OK,
    )
}

/// Triggers an immediate health sweep over all registered nodes.
pub async fn probe_upstreams(
    State(state): State<Arc<GatewayState>>,
    headers: axum::http::HeaderMap,
) -> Response {
    if !admin_authorized(&state, &headers) {
        return json_response(serde_json::json!({"error": "forbidden"}), StatusCode::FORBIDDEN);
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(state.config.health_timeout_secs))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    crate::registry::run_health_sweep(&state, &client).await;
    let registry = state.registry.read().await;
    json_response(
        serde_json::json!({
            "status": "ok",
            "nodes": registry.nodes().iter().map(|n| node_view(n)).collect::<Vec<_>>(),
        }),
        StatusCode::OK,
    )
}

// ---------------------------------------------------------------------------
// RAG worker endpoint (heavy graph-extraction jobs)
// ---------------------------------------------------------------------------

pub async fn rag_extract(State(state): State<Arc<GatewayState>>, req: Request<Body>) -> Response {
    let content_type = req
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(String::from);
    let body_bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => {
            return json_response(
                serde_json::json!({
                    "error": { "message": "Invalid request body", "type": "proxy_error", "code": 400 }
                }),
                StatusCode::BAD_REQUEST,
            );
        }
    };

    match state.route_rag_extract(content_type.as_deref(), body_bytes).await {
        Ok((hdrs, body, _node_id)) => {
            let mut response = Response::new(body);
            *response.headers_mut() = hdrs;
            response
        }
        Err(e) => {
            let status = StatusCode::from_u16(e.status).unwrap_or(StatusCode::BAD_GATEWAY);
            json_response(
                serde_json::json!({
                    "error": {
                        "message": e.context,
                        "type": if e.unreachable { "rag_worker_unavailable" } else { "rag_worker_error" },
                        "backend": e.backend,
                        "param": serde_json::Value::Null,
                        "code": status.as_u16()
                    }
                }),
                status,
            )
        }
    }
}

pub async fn register_extraction_backend_handler(
    State(state): State<Arc<GatewayState>>,
    Json(entry): Json<ExtractionBackendEntry>,
) -> Response {
    state.register_extraction_backend(entry.clone()).await;
    json_response(
        serde_json::json!({ "status": "ok", "id": entry.id }),
        StatusCode::OK,
    )
}

pub async fn remove_extraction_backend_handler(
    State(state): State<Arc<GatewayState>>,
    Path(id): Path<String>,
) -> Response {
    let removed = state.remove_extraction_backend(&id).await;
    if removed {
        json_response(serde_json::json!({ "status": "deleted", "id": id }), StatusCode::OK)
    } else {
        json_response(
            serde_json::json!({ "error": "Backend not found" }),
            StatusCode::NOT_FOUND,
        )
    }
}
