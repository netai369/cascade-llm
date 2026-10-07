use crate::registry::{RegisterUpstreamRequest, UpstreamNode};
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
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

fn json_response(body: serde_json::Value, status: StatusCode) -> Response {
    let body_str = body.to_string();
    let mut resp = Response::new(Body::from(body_str));
    *resp.status_mut() = status;
    resp.headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    resp
}

fn redact_provider(mut provider: ProviderConfig) -> ProviderConfig {
    provider.api_key = None;
    provider
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
    state.metrics.record_request_origin(origin);
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
        Err(e) => {
            // Name the parse failure. The message used to be a flat "Invalid
            // JSON" for *every* deserialization problem, so the single most
            // common cause — omitting `model`, which this struct requires —
            // read as a client bug with no way to tell.
            return json_response(
                serde_json::json!({
                    "error": {
                        "message": format!("Invalid request body: {}", e),
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
                        "service_action": e.service_action,
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
    let base = base_url.trim_end_matches('/');
    // vLLM and OVMS serve /v1/models; llama.cpp serves both /v1/models and
    // /models. Fetching only /models got a 400 from OVMS ("Invalid request URL")
    // and left the dashboard's Local Models table empty while it kept showing
    // stale llama.cpp entries.
    let mut models: Vec<serde_json::Value> = Vec::new();
    for path in ["/v1/models", "/models"] {
        let url = format!("{base}{path}");
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<serde_json::Value>().await {
                    Ok(val) => {
                        models = val.get("data")
                            .and_then(|d| d.as_array())
                            .cloned()
                            .unwrap_or_default();
                        if !models.is_empty() {
                            break;
                        }
                    }
                    Err(e) => { tracing::warn!("Failed to parse models from {}: {}", url, e); }
                }
            }
            Ok(resp) => { tracing::warn!("Models fetch from {} returned HTTP {}", url, resp.status()); }
            Err(e) => { tracing::warn!("Failed to connect to {}: {}", url, e); }
        }
    }
    for m in &mut models {
        if let Some(obj) = m.as_object_mut() {
            obj.insert("model_type".to_string(), serde_json::Value::String(model_type.to_string()));
        }
    }
    models
}

pub async fn list_models(State(state): State<Arc<GatewayState>>) -> Response {
    let large_base = state.config.large_text_url.replace("/v1/chat/completions", "");
    let small_base = state.config.small_mllm_url.replace("/v1/chat/completions", "");
    let ocr_base = state.config.ocr_server_url.replace("/v1/chat/completions", "");

    // Each fetch reports where it found the model rather than pretending to know
    // what it can do. Under OVMS one process serves the chat AND the embedding
    // graphs, so "found on the main origin" is the only honest claim available —
    // calling `bge-m3` a "Main" model (the old `model_type` tag) is a guess that
    // was wrong on the live system.
    let large_models = fetch_models_from(&state.http_client, &large_base, "Main").await;
    let small_models = fetch_models_from(&state.http_client, &small_base, "Auxiliary").await;
    let ocr_models = fetch_models_from(&state.http_client, &ocr_base, "OCR").await;

    // Unified model identity: the cascade-hybrid model is what clients
    // (LibreChat, Hermes, …) send, so it leads the list. It carries an explicit
    // flag rather than relying on `owned_by`: the configured OCR fallback is
    // built by the same helper and is also `netai-stack`, and the console used
    // to label it "gateway identity — send this one".
    let hybrid = build_model_info(&state.config.cascade_model_name);
    let mut all_models = vec![serde_json::to_value(hybrid)
        .map(|mut m| {
            if let Some(obj) = m.as_object_mut() {
                obj.insert("advertised_identity".into(), serde_json::json!(true));
            }
            m
        })
        .unwrap_or_default()];

    let mut seen: std::collections::HashSet<String> = all_models
        .iter()
        .filter_map(|m| m.get("id").and_then(|v| v.as_str()).map(str::to_string))
        .collect();

    let append = |models: Vec<serde_json::Value>, origin: &str, seen: &mut std::collections::HashSet<String>, out: &mut Vec<serde_json::Value>| {
        for mut m in models {
            if let Some(obj) = m.as_object_mut() {
                obj.insert("discovered_via".into(), serde_json::json!(origin));
            }
            let id = m.get("id").and_then(|v| v.as_str()).unwrap_or_default().to_string();
            // De-duplicate by id. The gateway identity wins the slot: it is the
            // id that actually works upstream (gotcha 40/68), so a second entry
            // with the same name — which the live OVMS setup produced — is noise
            // in every consumer's model picker.
            if id.is_empty() || !seen.insert(id) {
                continue;
            }
            out.push(m);
        }
    };

    append(large_models, &large_base, &mut seen, &mut all_models);
    append(small_models, &small_base, &mut seen, &mut all_models);
    if ocr_models.is_empty() {
        // Backend not reachable yet — still advertise the configured OCR model id.
        if seen.insert(state.config.ocr_model_name.clone()) {
            all_models.push(serde_json::to_value(build_model_info(&state.config.ocr_model_name))
                .ok()
                .filter(|m| m.is_object())
                .map(|mut m| {
                    if let Some(obj) = m.as_object_mut() {
                        obj.insert("discovered_via".into(), serde_json::json!(ocr_base));
                    }
                    m
                })
                .unwrap_or_default());
        }
    } else {
        append(ocr_models, &ocr_base, &mut seen, &mut all_models);
    }

    json_response(
        serde_json::json!({
            "data": all_models,
            "object": "list"
        }),
        StatusCode::OK,
    )
}

pub async fn get_model(State(state): State<Arc<GatewayState>>) -> Response {
    let model = build_model_info(&state.config.cascade_model_name);
    json_response(serde_json::to_value(model).unwrap(), StatusCode::OK)
}

pub async fn active_models(State(state): State<Arc<GatewayState>>) -> Response {
    let nodes: Vec<UpstreamNode> = {
        let registry = state.registry.read().await;
        registry.nodes().into_iter().cloned().collect()
    };
    let providers = state.runtime.read().await.providers.clone();
    // Which model id does this role actually serve? A registered node declares
    // its own; otherwise it is the configured default. This is what we look for
    // in the upstream's `/v1/models` — see `fetch_reported_model`.
    let declared: std::collections::HashMap<&str, Option<String>> = {
        let registry = state.registry.read().await;
        ["main", "auxiliary"]
            .iter()
            .map(|r| (*r, registry.get_by_role_active(r).and_then(|n| n.model.clone())))
            .collect()
    };
    let main_preferred = declared
        .get("main")
        .cloned()
        .flatten()
        .unwrap_or_else(|| state.config.main_model_name.clone());
    let aux_preferred = declared
        .get("auxiliary")
        .cloned()
        .flatten()
        .unwrap_or_else(|| state.config.small_model_name.clone());
    let main = role_active_status(
        &state.http_client,
        &nodes,
        &providers,
        "main",
        &state.config.large_text_url,
        &state.config.main_model_name,
        &main_preferred,
    )
    .await;
    let auxiliary = role_active_status(
        &state.http_client,
        &nodes,
        &providers,
        "auxiliary",
        &state.config.small_mllm_url,
        &state.config.small_model_name,
        &aux_preferred,
    )
    .await;
    json_response(
        serde_json::json!({
            "cascade_model": state.config.cascade_model_name,
            "main": main,
            "auxiliary": auxiliary,
        }),
        StatusCode::OK,
    )
}

fn pretty_model_name(id: &str) -> String {
    let id = id.rsplit('/').next().unwrap_or(id);
    id.strip_suffix(".gguf").unwrap_or(id).to_string()
}

async fn fetch_reported_model(
    client: &reqwest::Client,
    endpoint: &str,
    preferred: &str,
) -> Option<String> {
    let origin = crate::registry::origin_of(endpoint);
    for path in ["/v1/models", "/models"] {
        let url = format!("{}{}", origin, path);
        let Ok(resp) = client.get(&url).send().await else {
            continue;
        };
        if !resp.status().is_success() {
            continue;
        }
        let Ok(val) = resp.json::<serde_json::Value>().await else {
            continue;
        };
        let ids: Vec<String> = val
            .get("data")
            .and_then(|d| d.as_array())
            .map(|a| a.iter().filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(pretty_model_name)).collect())
            .or_else(|| {
                val.get("models").and_then(|m| m.as_array()).map(|a| {
                    a.iter()
                        .filter_map(|m| m.get("model").and_then(|i| i.as_str()).map(pretty_model_name))
                        .collect()
                })
            })
            .unwrap_or_default();
        if ids.is_empty() {
            continue;
        }
        // Prefer the model this role is configured to serve.
        //
        // The old code returned `data[0]`, which on the live OVMS node is
        // `bge-m3` — the console then reported the main chat model as an
        // *embedding* model. A single OVMS process serves every graph from one
        // port, so list order carries no meaning; membership does.
        if !preferred.is_empty() {
            if let Some(hit) = ids.iter().find(|id| id.as_str() == preferred) {
                return Some(hit.clone());
            }
        }
        return Some(ids[0].clone());
    }
    None
}

async fn role_active_status(
    client: &reqwest::Client,
    nodes: &[UpstreamNode],
    providers: &[ProviderConfig],
    role: &str,
    fallback_endpoint: &str,
    fallback_model: &str,
    preferred_model: &str,
) -> serde_json::Value {
    let node = nodes
        .iter()
        .find(|n| n.role == role && n.active && n.healthy);
    let endpoint = node
        .map(|n| n.endpoint_url.clone())
        .unwrap_or_else(|| fallback_endpoint.to_string());
    let provider = node.and_then(|node| {
        providers.iter().find(|p| {
            p.use_as.as_deref() == Some(role) && node.label.as_deref() == Some(p.name.as_str())
        })
    });
    let reported = fetch_reported_model(client, &endpoint, preferred_model).await;
    serde_json::json!({
        "role": role,
        "model": reported.clone().unwrap_or_else(|| fallback_model.to_string()),
        "source": provider
            .map(|p| p.name.clone())
            .or_else(|| node.and_then(|n| n.label.clone()))
            .unwrap_or_else(|| "local".to_string()),
        "endpoint": endpoint,
        "active": reported.is_some(),
        "provider": provider.is_some(),
    })
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

    // Lightweight liveness probe so /health reflects a dead backend instead
    // of always reporting "ok" (the compose healthcheck would otherwise never
    // notice an outage). The configured URLs include the API path
    // (/v1/chat/completions), so probe the origin's /health.
    fn origin_of(url: &str) -> String {
        let trimmed = url.trim_end_matches('/');
        match trimmed.find("://") {
            Some(i) => {
                let rest = &trimmed[i + 3..];
                match rest.find('/') {
                    Some(j) => trimmed[..i + 3 + j].to_string(),
                    None => trimmed.to_string(),
                }
            }
            None => trimmed.to_string(),
        }
    }
    async fn probe(client: &reqwest::Client, base: &str) -> bool {
        let url = format!("{}/health", origin_of(base));
        matches!(
            tokio::time::timeout(Duration::from_secs(3), client.get(&url).send()).await,
            Ok(Ok(r)) if r.status().is_success()
        )
    }
    let (large_up, small_up) = tokio::join!(
        probe(&state.http_client, &state.config.large_text_url),
        probe(&state.http_client, &state.config.small_mllm_url),
    );
    let status = if large_up || small_up { "ok" } else { "degraded" };

    json_response(
        serde_json::json!({
            "status": status,
            "large_model_multimodal": state.config.large_model_multimodal,
            "router_threshold": state.config.router_threshold,
            "confidence_threshold": state.config.confidence_threshold,
            "ocr_server_url": state.config.ocr_server_url,
            "auxiliary_server_url": state.config.small_mllm_url,
            "inference_server_url": state.config.large_text_url,
            "inference_reachable": large_up,
            "auxiliary_reachable": small_up,
            "extract_fallback_url": state.config.extract_fallback_url,
            "session_cache_entries": state.session_cache.entry_count(),
            "uptime_seconds": state.start_time.elapsed().as_secs(),
            // The runtime list, not `config.providers` — that vec is initialised
            // empty and never written, so reading it reported "0 providers" on
            // every healthy install with providers registered.
            "providers": state.runtime.read().await.providers.len(),
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

/// Speaker diarization — same byte proxy as STT, different leaf.
pub async fn stt_diarize(State(state): State<Arc<GatewayState>>, req: Request<Body>) -> Response {
    crate::audio::stt_diarize_handler(State(state), req).await
}

pub async fn embeddings(State(state): State<Arc<GatewayState>>, req: Request<Body>) -> Response {
    crate::embeddings::embeddings_handler(State(state), req).await
}

pub async fn rerank(State(state): State<Arc<GatewayState>>, req: Request<Body>) -> Response {
    crate::embeddings::rerank_handler(State(state), req).await
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

/// The single-page console. One document for every section — the previous pair
/// of pages (`dashboard.html` + `settings.html`) duplicated the chrome, the
/// styles and the fetch layer, and had already drifted apart.
pub async fn dashboard(_state: State<Arc<GatewayState>>) -> Response {
    let html = include_str!("web/app.html");
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .header("cache-control", "no-store")
        .body(Body::from(html))
        .unwrap()
}

/// `/web/settings` was the second page's canonical URL. It now serves the same
/// document and lets the client-side section routing pick the view, so existing
/// bookmarks keep working.
pub async fn settings_page(_state: State<Arc<GatewayState>>) -> Response {
    dashboard(_state).await
}

const DOCS: &[(&str, &str)] = &[
    (
        "routing-settings.md",
        include_str!("../docs/routing-settings.md"),
    ),
    (
        "dynamic-upstreams.md",
        include_str!("../docs/dynamic-upstreams.md"),
    ),
];

/// Serves the built-in markdown docs (settings reference / dynamic upstreams),
/// so the settings-page links resolve on the offline stack.
pub async fn docs_page(axum::extract::Path(name): axum::extract::Path<String>) -> Response {
    for (doc, body) in DOCS {
        if *doc == name {
            return Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/markdown; charset=utf-8")
                .body(Body::from(*body))
                .unwrap();
        }
    }
    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .header("content-type", "application/json")
        .body(Body::from(r#"{"error":"doc not found"}"#))
        .unwrap()
}

/// Canonical settings mount is /web/settings; redirect keeps relative links
/// working when the dashboard is served from "/".
pub async fn settings_redirect(_state: State<Arc<GatewayState>>) -> Response {
    Response::builder()
        .status(StatusCode::TEMPORARY_REDIRECT)
        .header("location", "/web/settings")
        .body(Body::empty())
        .unwrap()
}

/// Collects the counter snapshot shared by `/web/api/dashboard` and
/// `/web/api/overview`, so the two can never disagree.
fn dashboard_metrics(state: &Arc<GatewayState>) -> DashboardMetrics {
    let cache_entries = state.session_cache.entry_count();

    // ★ EVERY endpoint, including the idle ones.
    //
    // This used to drop `val == 0`, so the chart showed only what had already
    // been used — and an endpoint that is wired but never reached (the entire
    // `ocr`, `rag_worker`, `image`, `extraction` set, and `small` vs
    // `auxiliary`, which are two counters for one upstream) simply did not
    // appear. A reader then cannot tell "no traffic" from "does not exist",
    // which is exactly the question the owner asked about `small`. Zeroes are
    // now part of the answer.
    let mut requests_by_backend = std::collections::HashMap::new();
    let mut total_requests: u64 = 0;
    for backend in crate::cascade_features::MetricsRegistry::backend_labels() {
        let val = state
            .metrics
            .requests_total
            .with_label_values(&[backend])
            .get() as u64;
        total_requests += val;
        requests_by_backend.insert(backend.to_string(), val);
    }

    let mut fallback_by_reason = std::collections::HashMap::new();
    for reason in crate::cascade_features::MetricsRegistry::fallback_reasons() {
        let v = state
            .metrics
            .fallback_triggered
            .with_label_values(&[reason])
            .get() as u64;
        if v > 0 {
            fallback_by_reason.insert(reason.to_string(), v);
        }
    }

    let origin_counts = state.metrics.get_origin_counts().clone();
    let requests_by_origin: std::collections::HashMap<String, u64> = origin_counts
        .iter()
        .filter(|(_, count)| **count > 0)
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    DashboardMetrics {
        requests_total: total_requests,
        requests_by_backend,
        fallback_count: fallback_by_reason.values().sum(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        session_cache_entries: cache_entries,
        large_model_multimodal: state.config.large_model_multimodal,
        requests_by_origin,
        fallback_by_reason,
    }
}

pub async fn dashboard_api(State(state): State<Arc<GatewayState>>) -> Response {
    json_response(
        serde_json::to_value(dashboard_metrics(&state)).unwrap(),
        StatusCode::OK,
    )
}

pub async fn get_settings(State(state): State<Arc<GatewayState>>) -> Response {
    let mut s = state.config.to_settings();
    let rt = state.runtime.read().await;
    // runtime wins over env defaults
    s.routing.router_threshold = rt.router_threshold;
    s.routing.confidence_threshold = rt.confidence_threshold;
    s.routing.route_tools_to_large = rt.route_tools_to_large;
    s.routing.default_route = rt.default_route.clone();
    s.routing.hybrid_auto = Some(rt.hybrid_auto);
    s.routing.hybrid_aux_max_chars = Some(rt.hybrid_aux_max_chars);
    s.routing.marker_mode = rt.marker_mode.clone();
    s.audio.tts_url = Some(rt.tts_url.clone());
    s.audio.stt_url = Some(rt.stt_url.clone());
    if !rt.providers.is_empty() {
        s.providers = rt.providers.iter().cloned().map(redact_provider).collect();
    }
    // Report the *effective* chat URLs. `to_settings()` returns the env
    // defaults, so a UI that reloaded the form after saving showed the old URL
    // again while a `manual-main` / `manual-auxiliary` node carrying the new one
    // sat in the pool — an override that looked like it had been reverted.
    {
        let registry = state.registry.read().await;
        if let Some(n) = registry.get("manual-main") {
            s.large_text_url = Some(n.endpoint_url.clone());
        }
        if let Some(n) = registry.get("manual-auxiliary") {
            s.small_mllm_url = Some(n.endpoint_url.clone());
        }
    }
    s.routing_strategy = Some(state.registry.read().await.strategy().to_string());
    json_response(serde_json::to_value(s).unwrap(), StatusCode::OK)
}

pub async fn update_settings(
    State(state): State<Arc<GatewayState>>,
    headers: axum::http::HeaderMap,
    Json(settings): Json<Settings>,
) -> Response {
    if admin_forbidden(&state, &headers) {
        return json_response(serde_json::json!({"error": "forbidden"}), StatusCode::FORBIDDEN);
    }
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
        // `None` = field not submitted (e.g. an API caller that doesn't know about
        // it): keep the current value instead of silently resetting it.
        if let Some(v) = settings.routing.hybrid_auto {
            rt.hybrid_auto = v;
        }
        // 0 means "no length gate"; anything else is clamped so a typo in the
        // settings page cannot turn a 4 kB paste into a "simple" prompt.
        if let Some(n) = settings.routing.hybrid_aux_max_chars {
            rt.hybrid_aux_max_chars = n.min(100_000);
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
        // Providers are managed through the tested provider endpoint. Do not
        // replace stored credentials with the redacted settings-page copy.
    }
    // …chat URLs additionally flow into the dynamic registry as manual nodes
    // so the routing layer actually uses them.
    //
    // Reconciled in both directions: previously only the "create" branch
    // existed, so a user could set an override, clear the field and never get
    // the old node back out of the pool — the override was write-only.
    for (role, url_opt, default) in [
        ("auxiliary", &settings.small_mllm_url, &state.config.small_mllm_url),
        ("main", &settings.large_text_url, &state.config.large_text_url),
    ] {
        let node_id = format!("manual-{}", role);
        let wanted = url_opt
            .as_deref()
            .map(str::trim)
            .filter(|u| !u.is_empty() && u.starts_with("http"));
        match wanted {
            Some(url) if url != default => {
                let req = RegisterUpstreamRequest {
                    endpoint_url: Some(url.to_string()),
                    id: Some(node_id),
                    ..Default::default()
                };
                state.registry.write().await.upsert(role, req);
            }
            _ => {
                // Back to the default (or explicitly cleared): drop the override.
                state.registry.write().await.remove_id(&node_id);
            }
        }
    }
    // Persist only the runtime-owned subset. Sending the whole `Settings`
    // back would overwrite the stored providers list with the settings page's
    // *redacted* copy (api_key: null), i.e. a read would silently delete every
    // stored credential.
    let persisted = serde_json::json!({
        "routing": settings.routing,
        "audio": settings.audio,
    });
    if let Err(e) = state
        .db
        .save_config("settings", &persisted.to_string())
    {
        return json_response(
            serde_json::json!({"error": format!("Failed to save: {}", e)}),
            StatusCode::INTERNAL_SERVER_ERROR,
        );
    }
    info!("SETTINGS updated via UI/API");
    json_response(serde_json::json!({"status": "ok"}), StatusCode::OK)
}

/// Public view of a provider: never the key itself, but *whether* one is set and
/// its last four characters — the same masking convention as `mask_bearer` and
/// `list_extraction_backends`. Without this the settings page could not tell
/// "no key" from "key stripped by redaction", and a user re-entering a key on
/// every page load had no way to know one was already stored.
fn provider_view(p: &ProviderConfig) -> serde_json::Value {
    let mut v = serde_json::to_value(redact_provider(p.clone())).unwrap_or(serde_json::Value::Null);
    if let Some(obj) = v.as_object_mut() {
        obj.insert("has_key".into(), serde_json::json!(p.api_key.is_some()));
        obj.insert(
            "key_masked".into(),
            serde_json::json!(mask_bearer(&p.api_key)),
        );
    }
    v
}

pub async fn list_providers(State(state): State<Arc<GatewayState>>) -> Response {
    let rt = state.runtime.read().await;
    json_response(
        serde_json::to_value(rt.providers.iter().map(provider_view).collect::<Vec<_>>())
            .unwrap_or(serde_json::Value::Array(vec![])),
        StatusCode::OK,
    )
}

pub async fn add_provider(
    State(state): State<Arc<GatewayState>>,
    headers: axum::http::HeaderMap,
    Json(provider): Json<ProviderConfig>,
) -> Response {
    if admin_forbidden(&state, &headers) {
        return json_response(serde_json::json!({"error": "forbidden"}), StatusCode::FORBIDDEN);
    }
    let role = match provider.use_as.as_deref() {
        Some("main") | Some("auxiliary") => provider.use_as.clone().unwrap(),
        Some(_) => return json_response(serde_json::json!({"error": "use_as must be main or auxiliary"}), StatusCode::BAD_REQUEST),
        None => return json_response(serde_json::json!({"error": "Choose whether this provider is used as main or auxiliary"}), StatusCode::BAD_REQUEST),
    };
    let model = provider.models.first().cloned().unwrap_or_default();
    if model.is_empty() {
        return json_response(serde_json::json!({"error": "At least one model is required"}), StatusCode::BAD_REQUEST);
    }
    let (_, answer) = match probe_provider(&state.http_client, &provider, &model).await {
        Ok(ok) => ok,
        Err((status, body)) => return json_response(body, status),
    };
    let endpoint = format!("{}/chat/completions", provider.base_url.trim_end_matches('/'));
    let register = RegisterUpstreamRequest {
        endpoint_url: Some(endpoint),
        bearer_token: provider.api_key.clone(),
        id: Some(format!("provider-{}", provider.id)),
        label: Some(provider.name.clone()),
        model: Some(model),
        enabled: Some(true),
        ..Default::default()
    };
    state.registry.write().await.upsert(&role, register);
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
        serde_json::json!({"status": "created", "id": provider.id, "tested": true, "answer": answer}),
        StatusCode::CREATED,
    )
}

pub async fn get_provider(
    State(state): State<Arc<GatewayState>>,
    Path(id): Path<String>,
) -> Response {
    match state.runtime.read().await.providers.iter().find(|p| p.id == id) {
        Some(p) => json_response(provider_view(p), StatusCode::OK),
        None => json_response(
            serde_json::json!({"error": "Provider not found"}),
            StatusCode::NOT_FOUND,
        ),
    }
}

/// Live chat probe against a provider's own endpoint, using a caller-supplied
/// model. Reused by `add_provider` and exposed standalone so an *existing*
/// provider can be tested — previously "test" only happened as a side effect of
/// adding, which meant a provider whose key later expired could only be
/// diagnosed by deleting and re-adding it.
async fn probe_provider(
    client: &reqwest::Client,
    provider: &ProviderConfig,
    model: &str,
) -> Result<(StatusCode, String), (StatusCode, serde_json::Value)> {
    let endpoint = format!("{}/chat/completions", provider.base_url.trim_end_matches('/'));
    let mut request = client
        .post(&endpoint)
        .json(&serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "Hello, are you there? respond with yes"}],
            "max_tokens": 16,
            "temperature": 0
        }));
    if let Some(key) = &provider.api_key {
        request = request.bearer_auth(key);
    }
    let response = match request.send().await {
        Ok(r) => r,
        Err(error) => {
            return Err((
                StatusCode::BAD_GATEWAY,
                serde_json::json!({"error": format!("Provider is unreachable: {}", error)}),
            ))
        }
    };
    let status = response.status();
    let body = response.json::<serde_json::Value>().await.unwrap_or_default();
    let answer = body
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if !status.is_success() || answer.is_empty() {
        return Err((
            StatusCode::BAD_GATEWAY,
            serde_json::json!({
                "error": "Provider did not return a valid chat response",
                "upstream_status": status.as_u16(),
            }),
        ));
    }
    Ok((status, answer))
}

/// Test body. Either a bare `{"model": …}` against a stored provider, or the
/// full draft of a *not yet registered* provider so a key can be checked before
/// it is committed to the DB.
#[derive(serde::Deserialize, Default)]
pub struct ProviderTestBody {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub models: Vec<String>,
}

pub async fn test_provider(
    State(state): State<Arc<GatewayState>>,
    headers: axum::http::HeaderMap,
    Path(id): Path<String>,
    body: Option<Json<ProviderTestBody>>,
) -> Response {
    if admin_forbidden(&state, &headers) {
        return json_response(serde_json::json!({"error": "forbidden"}), StatusCode::FORBIDDEN);
    }
    let body = body.map(|Json(b)| b).unwrap_or_default();
    let stored = state
        .runtime
        .read()
        .await
        .providers
        .iter()
        .find(|p| p.id == id)
        .cloned();

    let provider = match stored {
        Some(p) => {
            // A posted key wins over the stored one — that is how a user verifies
            // a *new* key before committing it.
            let mut p = p;
            if let Some(k) = &body.api_key {
                p.api_key = if k.is_empty() { None } else { Some(k.clone()) };
            }
            if body.base_url.is_some() {
                p.base_url = body.base_url.clone().unwrap_or(p.base_url);
            }
            p
        }
        None if body.base_url.is_some() => ProviderConfig {
            id: id.clone(),
            name: id.clone(),
            provider_type: ProviderType::Custom,
            base_url: body.base_url.clone().unwrap_or_default(),
            api_key: body.api_key.clone().filter(|k| !k.is_empty()),
            models: body.models.clone(),
            enabled: true,
            priority: 10,
            capabilities: Default::default(),
            use_as: None,
        },
        None => {
            return json_response(
                serde_json::json!({"error": "Provider not found"}),
                StatusCode::NOT_FOUND,
            )
        }
    };

    let model = body
        .model
        .filter(|m| !m.is_empty())
        .or_else(|| provider.models.first().cloned())
        .unwrap_or_default();
    if provider.base_url.is_empty() || model.is_empty() {
        return json_response(
            serde_json::json!({"error": "provider needs a base_url and at least one model"}),
            StatusCode::BAD_REQUEST,
        );
    }
    match probe_provider(&state.http_client, &provider, &model).await {
        Ok((_, answer)) => json_response(
            serde_json::json!({"status": "ok", "model": model, "answer": answer}),
            StatusCode::OK,
        ),
        Err((status, body)) => json_response(body, status),
    }
}

/// Partial update. `api_key` semantics are the only subtle part:
/// * omitted / null → keep the stored key;
/// * `""` → clear it;
/// * anything else → replace it.
///
/// Every change that moves the upstream (base_url / models / role / enabled)
/// re-registers or withdraws the registry node, because a provider that is not
/// in the pool is a provider that never receives traffic no matter what the
/// settings page says.
#[derive(serde::Deserialize)]
pub struct ProviderUpdate {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub models: Option<Vec<String>>,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub use_as: Option<String>,
}

pub async fn update_provider(
    State(state): State<Arc<GatewayState>>,
    headers: axum::http::HeaderMap,
    Path(id): Path<String>,
    Json(update): Json<ProviderUpdate>,
) -> Response {
    if admin_forbidden(&state, &headers) {
        return json_response(serde_json::json!({"error": "forbidden"}), StatusCode::FORBIDDEN);
    }
    let Some(mut provider) = state
        .runtime
        .read()
        .await
        .providers
        .iter()
        .find(|p| p.id == id)
        .cloned()
    else {
        return json_response(
            serde_json::json!({"error": "Provider not found"}),
            StatusCode::NOT_FOUND,
        );
    };

    if let Some(v) = update.name {
        provider.name = v;
    }
    if let Some(v) = update.base_url {
        if !v.starts_with("http") {
            return json_response(
                serde_json::json!({"error": "base_url must be http(s)"}),
                StatusCode::BAD_REQUEST,
            );
        }
        provider.base_url = v;
    }
    match update.api_key {
        Some(ref k) if k.is_empty() => provider.api_key = None,
        Some(k) => provider.api_key = Some(k),
        None => {}
    }
    if let Some(v) = update.models {
        if v.is_empty() {
            return json_response(
                serde_json::json!({"error": "At least one model is required"}),
                StatusCode::BAD_REQUEST,
            );
        }
        provider.models = v;
    }
    if let Some(v) = update.enabled {
        provider.enabled = v;
    }
    if let Some(v) = update.use_as {
        match crate::registry::canonical_role(&v) {
            Some(r) if matches!(r, "main" | "auxiliary") => provider.use_as = Some(r.to_string()),
            Some(r) => {
                return json_response(
                    serde_json::json!({"error": format!(
                        "providers back the main/auxiliary chat roles only (got '{}')",
                        r
                    )}),
                    StatusCode::BAD_REQUEST,
                )
            }
            None => {
                return json_response(
                    serde_json::json!({"error": "use_as must be main or auxiliary"}),
                    StatusCode::BAD_REQUEST,
                )
            }
        }
    }

    // Save first, then reconcile the pool — a failed write must not leave the
    // registry pointing at a configuration that was never stored.
    if let Err(e) = state.db.save_provider(&provider) {
        return json_response(
            serde_json::json!({"error": format!("Failed to save: {}", e)}),
            StatusCode::INTERNAL_SERVER_ERROR,
        );
    }
    sync_provider_node(&state, &provider).await;
    {
        let mut rt = state.runtime.write().await;
        rt.providers.retain(|p| p.id != provider.id);
        rt.providers.push(provider.clone());
    }
    json_response(
        serde_json::json!({"status": "ok", "provider": provider_view(&provider)}),
        StatusCode::OK,
    )
}

/// Adds or withdraws the registry node that makes a provider reachable, and
/// keeps its model name in step with the stored list.
///
/// The single source of truth for the provider ⇄ registry relationship: called
/// by `add_provider`, `update_provider` and the boot-time restore alike. The
/// restore path used to register unconditionally, which resurrected a provider
/// the operator had explicitly disabled — the setting looked persisted right up
/// until the next gateway restart put it back into the pool.
pub async fn sync_provider_node(state: &Arc<GatewayState>, provider: &ProviderConfig) {
    let node_id = format!("provider-{}", provider.id);
    let role = match provider.use_as.as_deref().and_then(crate::registry::canonical_role) {
        Some(r) => r.to_string(),
        None => {
            state.registry.write().await.remove_id(&node_id);
            return;
        }
    };
    if !provider.enabled {
        state.registry.write().await.remove_id(&node_id);
        return;
    }
    let Some(model) = provider.models.first().cloned() else {
        state.registry.write().await.remove_id(&node_id);
        return;
    };
    let register = RegisterUpstreamRequest {
        endpoint_url: Some(format!(
            "{}/chat/completions",
            provider.base_url.trim_end_matches('/')
        )),
        bearer_token: provider.api_key.clone(),
        id: Some(node_id),
        label: Some(provider.name.clone()),
        model: Some(model),
        enabled: Some(true),
        ..Default::default()
    };
    state.registry.write().await.upsert(&role, register);
}

pub async fn delete_provider(
    State(state): State<Arc<GatewayState>>,
    headers: axum::http::HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if admin_forbidden(&state, &headers) {
        return json_response(serde_json::json!({"error": "forbidden"}), StatusCode::FORBIDDEN);
    }
    state.db.delete_provider(&id);
    state.runtime.write().await.providers.retain(|p| p.id != id);
    // The registry node created when the provider was added has its own id and
    // was never tied to the provider row, so deleting the provider used to leave
    // a live node in the pool pointing at a removed provider — traffic kept
    // going to it until the gateway restarted.
    state.registry.write().await.remove_id(&format!("provider-{}", id));
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

/// True when a mutating route must be rejected: the admin key is configured
/// but the request did not present it. When no key is configured the guard is
/// treated as *not armed* (legacy installs stay usable) — `setup.sh` warns
/// loudly about that state.
fn admin_forbidden(state: &GatewayState, headers: &axum::http::HeaderMap) -> bool {
    state.config.admin_key.is_some() && !admin_authorized(state, headers)
}

/// Route-level guard for the mutating dashboard/pipeline endpoints. Runs
/// before body extraction, so an unauthenticated request is rejected with
/// 403 even when its payload is malformed.
pub async fn require_admin(
    State(state): State<Arc<GatewayState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if admin_forbidden(&state, req.headers()) {
        return json_response(serde_json::json!({"error": "forbidden"}), StatusCode::FORBIDDEN);
    }
    next.run(req).await
}

fn mask_bearer(b: &Option<String>) -> Option<String> {
    b.as_ref().map(|k| {
        let tail: String = k.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
        format!("***{tail}")
    })
}

async fn persist_upstream_registry(state: &Arc<GatewayState>) {
    let nodes: Vec<_> = state.registry.read().await.nodes().into_iter().cloned().collect();
    if let Err(e) = state.db.save_upstreams(&nodes) {
        warn!("Could not persist upstream registry: {}", e);
    }
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
                "vision": state.config.ocr_server_url,
                "ocr": state.config.ocr_server_url,
                "image": state.config.image_generation_url,
                "embeddings": state.config.embeddings_url,
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
    let host_gateway = std::env::var("CASCADE_HOST_GATEWAY").is_ok_and(|v| v.eq_ignore_ascii_case("true"));
    let url = crate::registry::resolve_host_gateway_url(&url, host_gateway);
    let url = crate::registry::normalize_endpoint_url(&role, &url);
    let health_url = req.health_url.as_deref()
        .map(|u| crate::registry::resolve_host_gateway_url(u, host_gateway));
    let req = RegisterUpstreamRequest { endpoint_url: Some(url), health_url, ..req };

    let result = state.registry.write().await.upsert(&role, req);
    persist_upstream_registry(&state).await;
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
        let belongs = state.registry.read().await.get(id).is_some_and(|n| n.role == role);
        if !belongs {
            return json_response(serde_json::json!({"error": "node not found for this role"}), StatusCode::NOT_FOUND);
        }
        let removed = state.registry.write().await.remove_id(id);
        persist_upstream_registry(&state).await;
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
    persist_upstream_registry(&state).await;
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
    persist_upstream_registry(&state).await;
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
                "vision": state.config.ocr_server_url,
                "ocr": state.config.ocr_server_url,
                "image": state.config.image_generation_url,
                "embeddings": state.config.embeddings_url,
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

/// Refreshes model names from all registered upstream nodes by calling
/// `/v1/models` on each node's base URL and updating the registry.
pub async fn refresh_models_upstreams(
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

    // Collect all nodes first (registry read lock)
    let nodes: Vec<UpstreamNode> = {
        let registry = state.registry.read().await;
        registry.nodes().into_iter().cloned().collect()
    };

    let mut updated = 0;
    let mut errors = Vec::new();

    for node in nodes {
        let base_url = crate::registry::origin_of(&node.endpoint_url);
        let models_url = format!("{}/v1/models", base_url);
        let mut req = client.get(&models_url);
        if let Some(token) = &node.bearer_token {
            if !token.is_empty() {
                req = req.bearer_auth(token);
            }
        }
        match req.send().await {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<serde_json::Value>().await {
                    Ok(val) => {
                        if let Some(model_id) = val
                            .get("data")
                            .and_then(|d| d.as_array())
                            .and_then(|a| a.first())
                            .and_then(|m| m.get("id"))
                            .and_then(|i| i.as_str())
                        {
                            let pretty_name = pretty_model_name(model_id);
                            // Update the node in the registry
                            state.registry.write().await.update_model(&node.role, &node.id, &pretty_name);
                            updated += 1;
                            info!("MODEL_REFRESH: role={} id={} model={}", node.role, node.id, pretty_name);
                        } else {
                            errors.push(format!("{} ({}): no model id in response", node.id, node.role));
                        }
                    }
                    Err(e) => errors.push(format!("{} ({}): parse error: {}", node.id, node.role, e)),
                }
            }
            Ok(resp) => {
                errors.push(format!("{} ({}): HTTP {}", node.id, node.role, resp.status()));
            }
            Err(e) => {
                errors.push(format!("{} ({}): connection error: {}", node.id, node.role, e));
            }
        }
    }

    let registry = state.registry.read().await;
    json_response(
        serde_json::json!({
            "status": "ok",
            "updated": updated,
            "errors": errors,
            "nodes": registry.nodes().iter().map(|n| node_view(n)).collect::<Vec<_>>(),
        }),
        StatusCode::OK,
    )
}

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

/// Everything the UI needs for one coherent header render, in one round trip.
///
/// The old dashboard assembled its header from three independent endpoints on a
/// 5-second timer, so a partial outage left a mixture of live numbers and `—`
/// with no indication which was which. One payload means one atomic snapshot and
/// one place to report "could not load".
///
/// Deliberately *excludes* per-node details: those carry endpoints and token
/// state and stay behind the admin guard (`admin_upstreams`).
pub async fn overview(
    State(state): State<Arc<GatewayState>>,
    headers: axum::http::HeaderMap,
) -> Response {
    let metrics = dashboard_metrics(&state);
    let rt = state.runtime.read().await;
    let (providers_total, providers_enabled, providers_with_key) = {
        let p = &rt.providers;
        (p.len(), p.iter().filter(|x| x.enabled).count(), p.iter().filter(|x| x.api_key.is_some()).count())
    };
    let (routing_strategy, node_count, active_nodes, roles) = {
        let registry = state.registry.read().await;
        let nodes = registry.nodes();
        let mut roles = serde_json::Map::new();
        for role in crate::registry::ROLES {
            let pool: Vec<_> = nodes.iter().filter(|n| n.role == *role).collect();
            roles.insert(
                role.to_string(),
                serde_json::json!({
                    "total": pool.len(),
                    "active": pool.iter().filter(|n| n.active).count(),
                    "healthy": pool.iter().filter(|n| n.active && n.healthy).count(),
                    "serving": registry.has_active(role),
                }),
            );
        }
        (
            registry.strategy().to_string(),
            nodes.len(),
            nodes.iter().filter(|n| n.active && n.healthy).count(),
            roles,
        )
    };
    drop(rt);

    // Health is derived from the same probes /health uses — one definition of
    // "degraded", reused by the compose healthcheck and by this header.
    let (large_up, small_up) = tokio::join!(
        probe_origin(&state.http_client, &state.config.large_text_url),
        probe_origin(&state.http_client, &state.config.small_mllm_url),
    );

    let (configured, reachable) = match &state.config.provisioner_url {
        Some(url) => (true, crate::provisioner::reachable(&state, url).await),
        None => (false, false),
    };

    json_response(
        serde_json::json!({
            "status": if large_up || small_up { "ok" } else { "degraded" },
            "uptime_seconds": state.start_time.elapsed().as_secs(),
            "inference_reachable": large_up,
            "auxiliary_reachable": small_up,
            "cascade_model": state.config.cascade_model_name,
            "main_model": state.config.main_model_name,
            "small_model": state.config.small_model_name,
            "ocr_model": state.config.ocr_model_name,
            "large_model_multimodal": state.config.large_model_multimodal,
            "large_text_url": state.config.large_text_url,
            "small_mllm_url": state.config.small_mllm_url,
            "ocr_server_url": state.config.ocr_server_url,
            "image_generation_url": state.config.image_generation_url,
            "tts_url": state.config.tts_url,
            "stt_url": state.config.stt_url,
            "embeddings_url": state.config.embeddings_url,
            "rerank_url": state.config.rerank_url,
            "admin_key_required": state.config.admin_key.is_some(),
            // Whether *this* caller may write. Through Caddy the key is injected
            // after forward_auth; on the raw port the browser has to supply it.
            "caller_is_admin": admin_authorized(&state, &headers) || state.config.admin_key.is_none(),
            "routing": {
                "strategy": routing_strategy,
                "default_route": state.runtime.read().await.default_route,
                "hybrid_auto": state.runtime.read().await.hybrid_auto,
                "hybrid_aux_max_chars": state.runtime.read().await.hybrid_aux_max_chars,
                "marker_mode": state.runtime.read().await.marker_mode,
                "router_threshold": state.runtime.read().await.router_threshold,
                "confidence_threshold": state.runtime.read().await.confidence_threshold,
                "route_tools_to_large": state.runtime.read().await.route_tools_to_large,
            },
            "registry": {
                "nodes": node_count,
                "serving": active_nodes,
                "roles": roles,
            },
            "providers": {
                "total": providers_total,
                "enabled": providers_enabled,
                "with_key": providers_with_key,
            },
            "provisioner": { "configured": configured, "reachable": reachable },
            "metrics": metrics,
            "extraction_backends": state.extraction_backends.read().await.len(),
        }),
        StatusCode::OK,
    )
}

async fn probe_origin(client: &reqwest::Client, url: &str) -> bool {
    let trimmed = url.trim_end_matches('/');
    let probe = match trimmed.find("://") {
        Some(i) => {
            let rest = &trimmed[i + 3..];
            match rest.find('/') {
                Some(j) => format!("{}/health", &trimmed[..i + 3 + j]),
                None => format!("{}/health", trimmed),
            }
        }
        None => format!("{}/health", trimmed),
    };
    matches!(
        tokio::time::timeout(Duration::from_secs(3), client.get(probe).send()).await,
        Ok(Ok(r)) if r.status().is_success()
    )
}

#[derive(serde::Deserialize)]
pub struct StrategyBody {
    pub strategy: String,
}

/// Node selection across a role pool. Previously reachable only through the
/// `CASCADE_ROUTING_STRATEGY` env var at boot, while the dashboard *displayed*
/// the current value as if it were a setting.
pub async fn set_routing_strategy(
    State(state): State<Arc<GatewayState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<StrategyBody>,
) -> Response {
    if admin_forbidden(&state, &headers) {
        return json_response(serde_json::json!({"error": "forbidden"}), StatusCode::FORBIDDEN);
    }
    match crate::registry::RoutingStrategy::parse(&body.strategy) {
        Some(s) => {
            state.registry.write().await.set_strategy(s);
            json_response(
                serde_json::json!({"status": "ok", "strategy": s.to_string()}),
                StatusCode::OK,
            )
        }
        None => json_response(
            serde_json::json!({"error": "strategy must be round_robin or least_latency"}),
            StatusCode::BAD_REQUEST,
        ),
    }
}

pub async fn register_extraction_backend_handler(
    State(state): State<Arc<GatewayState>>,
    headers: axum::http::HeaderMap,
    Json(entry): Json<ExtractionBackendEntry>,
) -> Response {
    if admin_forbidden(&state, &headers) {
        return json_response(serde_json::json!({"error": "forbidden"}), StatusCode::FORBIDDEN);
    }
    state.register_extraction_backend(entry.clone()).await;
    json_response(
        serde_json::json!({ "status": "ok", "id": entry.id }),
        StatusCode::OK,
    )
}

pub async fn remove_extraction_backend_handler(
    State(state): State<Arc<GatewayState>>,
    headers: axum::http::HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if admin_forbidden(&state, &headers) {
        return json_response(serde_json::json!({"error": "forbidden"}), StatusCode::FORBIDDEN);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The defect this pins: `dashboard_api` summed a hand-written, *shorter*
    /// list of backend labels than the one `MetricsRegistry::init` creates, so
    /// "Total Requests" under-counted and the charts silently omitted every
    /// backend the router only reaches through the newer paths (`main`,
    /// `image`, `rag_worker`, `extraction`, `large_text`, `fallback`) — while
    /// listing two labels (`session_affinity`, `stt_proxy`) that were never
    /// recorded at all. Both sides now read one list; this asserts the KPI is
    /// exactly the sum of every counter prometheus can report.
    #[test]
    fn dashboard_total_is_the_sum_of_every_backend_counter() {
        let metrics = crate::cascade_features::MetricsRegistry::init();
        let labels = crate::cascade_features::MetricsRegistry::backend_labels();
        // Record on every label, so "the total is the sum" cannot pass by
        // accident on an empty fixture.
        for (i, label) in labels.iter().enumerate() {
            for _ in 0..=i {
                metrics.record_request(label);
            }
        }
        let expected: u64 = (0..labels.len() as u64).map(|i| i + 1).sum();

        let state = Arc::new(GatewayState::new(
            crate::config::AppConfig::from_env(),
            Arc::new(metrics),
            Arc::new(crate::db::Db::new_in_memory().unwrap()),
        ));
        let snapshot = dashboard_metrics(&state);
        assert_eq!(snapshot.requests_total, expected);
        assert_eq!(snapshot.requests_by_backend.len(), labels.len());
        for (i, label) in labels.iter().enumerate() {
            assert_eq!(
                snapshot.requests_by_backend.get(*label).copied(),
                Some(i as u64 + 1),
                "label {label} missing from the dashboard payload"
            );
        }
    }

    #[test]
    fn fallback_total_is_the_sum_of_every_reason() {
        let metrics = crate::cascade_features::MetricsRegistry::init();
        let reasons = crate::cascade_features::MetricsRegistry::fallback_reasons();
        metrics.record_fallback(reasons[0]);
        metrics.record_fallback(reasons[2]);

        let state = Arc::new(GatewayState::new(
            crate::config::AppConfig::from_env(),
            Arc::new(metrics),
            Arc::new(crate::db::Db::new_in_memory().unwrap()),
        ));
        let snapshot = dashboard_metrics(&state);
        assert_eq!(snapshot.fallback_count, 2);
        assert_eq!(snapshot.fallback_by_reason.len(), 2);
    }
}
