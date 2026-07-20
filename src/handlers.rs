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

fn json_response(body: serde_json::Value, status: StatusCode) -> Response {
    let body_str = body.to_string();
    let mut resp = Response::new(Body::from(body_str));
    *resp.status_mut() = status;
    resp.headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    resp
}

pub async fn chat_completions(
    State(state): State<Arc<GatewayState>>,
    req: Request<Body>,
) -> Response {
    let headers = req.headers().clone();
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

    match state
        .route_request_with_fallback(json, is_streaming, tier, &headers)
        .await
    {
        Ok((hdrs, res_body)) => {
            let mut response = Response::new(res_body);
            *response.headers_mut() = hdrs;
            response
        }
        Err(status) => json_response(
            serde_json::json!({
                "error": {
                    "message": format!("HTTP {}", status.as_u16()),
                    "type": "cascade_proxy_error",
                    "param": serde_json::Value::Null,
                    "code": serde_json::Value::Number(status.as_u16().into())
                }
            }),
            status,
        ),
    }
}

pub async fn list_models(State(state): State<Arc<GatewayState>>) -> Response {
    let mut model_ids = vec![state.config.main_model_name.clone(), state.config.small_model_name.clone()];
    for provider in &state.config.providers {
        for model in &provider.models {
            if !model_ids.contains(model) {
                model_ids.push(model.clone());
            }
        }
    }
    let models: Vec<ModelInfo> = model_ids.iter().map(|id| build_model_info(id)).collect();
    let model_list = ModelList {
        object: "list".to_string(),
        data: models,
    };
    json_response(serde_json::to_value(model_list).unwrap(), StatusCode::OK)
}

pub async fn get_model(State(state): State<Arc<GatewayState>>) -> Response {
    let model = build_model_info(&state.config.main_model_name);
    json_response(serde_json::to_value(model).unwrap(), StatusCode::OK)
}

pub async fn health_check(State(state): State<Arc<GatewayState>>) -> Response {
    json_response(
        serde_json::json!({
            "status": "ok",
            "large_model_multimodal": state.config.large_model_multimodal,
            "router_threshold": state.config.router_threshold,
            "confidence_threshold": state.config.confidence_threshold,
            "session_cache_entries": state.session_cache.entry_count() as u64,
            "uptime_seconds": state.start_time.elapsed().as_secs(),
            "providers": state.config.providers.len(),
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

    let mut requests_by_backend = std::collections::HashMap::new();
    for backend in &["small", "large", "large_multimodal", "session_affinity"] {
        let val = state
            .metrics
            .requests_total
            .with_label_values(&[backend])
            .get() as u64;
        if val > 0 {
            requests_by_backend.insert(backend.to_string(), val);
        }
    }

    let metrics = DashboardMetrics {
        requests_total: state.metrics.requests_total.with_label_values(&[""]).get() as u64,
        requests_by_backend,
        fallback_count: state
            .metrics
            .fallback_triggered
            .with_label_values(&[""])
            .get() as u64,
        uptime_seconds: uptime,
        session_cache_entries: cache_entries,
        large_model_multimodal: state.config.large_model_multimodal,
    };
    json_response(serde_json::to_value(metrics).unwrap(), StatusCode::OK)
}

pub async fn get_settings(State(state): State<Arc<GatewayState>>) -> Response {
    let settings = state.config.to_settings();
    json_response(serde_json::to_value(settings).unwrap(), StatusCode::OK)
}

pub async fn update_settings(
    State(state): State<Arc<GatewayState>>,
    Json(settings): Json<Settings>,
) -> Response {
    if let Err(e) = state
        .db
        .save_config("settings", &serde_json::to_string(&settings).unwrap_or_default())
    {
        return json_response(
            serde_json::json!({"error": format!("Failed to save: {}", e)}),
            StatusCode::INTERNAL_SERVER_ERROR,
        );
    }
    json_response(serde_json::json!({"status": "ok"}), StatusCode::OK)
}

pub async fn list_providers(State(state): State<Arc<GatewayState>>) -> Response {
    json_response(
        serde_json::to_value(&state.config.providers).unwrap(),
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
    json_response(
        serde_json::json!({"status": "created", "id": provider.id}),
        StatusCode::CREATED,
    )
}

pub async fn get_provider(
    State(state): State<Arc<GatewayState>>,
    Path(id): Path<String>,
) -> Response {
    match state.config.providers.iter().find(|p| p.id == id) {
        Some(p) => json_response(serde_json::to_value(p).unwrap(), StatusCode::OK),
        None => json_response(
            serde_json::json!({"error": "Provider not found"}),
            StatusCode::NOT_FOUND,
        ),
    }
}

pub async fn delete_provider(
    _state: State<Arc<GatewayState>>,
    Path(id): Path<String>,
) -> Response {
    json_response(
        serde_json::json!({"status": "deleted", "id": id}),
        StatusCode::OK,
    )
}
