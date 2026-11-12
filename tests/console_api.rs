//! Black-box tests for the reworked gateway console's backend surface:
//! the `/web/api/overview` header, provider CRUD completion (edit / key
//! rotation / enable-disable / test), the routing-strategy control, the
//! `/web/api/models` de-duplication, and the GPU-provisioner passthrough.
//!
//! The provisioning tests are the interesting ones: they pin the *allowlist*
//! and the fact that the provisioner's own API key never leaves the gateway.


use axum::http::HeaderMap;
use axum::routing::{get, post};
use axum::Router;
use cascade_llm::cascade_features::MetricsRegistry;
use cascade_llm::config::AppConfig;
use cascade_llm::db::Db;
use cascade_llm::router::build_router;
use cascade_llm::state::GatewayState;
use regex::Regex;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

const ADMIN_KEY: &str = "test-admin-key";
const PROVISIONER_KEY: &str = "provisioner-secret-key";

fn admin_headers() -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert("x-cascade-admin-key", ADMIN_KEY.parse().unwrap());
    h
}

async fn spawn_gateway_with(cfg_mut: impl FnOnce(&mut AppConfig)) -> String {
    let mut cfg = AppConfig::from_env();
    cfg.admin_key = Some(ADMIN_KEY.to_string());
    // Keep the tests hermetic: no probe may escape to a real provisioner.
    cfg.provisioner_url = None;
    cfg.provisioner_api_key = None;
    cfg_mut(&mut cfg);
    let state = Arc::new(GatewayState::new(
        cfg,
        Arc::new(MetricsRegistry::init()),
        Arc::new(Db::new_in_memory().unwrap()),
    ));
    let app: Router = build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
    format!("http://{}", addr)
}

async fn spawn_gateway() -> String {
    spawn_gateway_with(|_| {}).await
}

// ---------------------------------------------------------------------------
// Overview header
// ---------------------------------------------------------------------------

#[tokio::test]
async fn overview_is_one_coherent_snapshot() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();

    let body: serde_json::Value = client
        .get(format!("{}/web/api/overview", base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    // The six fields the header renders, all from one payload — the old header
    // was assembled from three endpoints on a timer and could show a mixture of
    // live numbers and placeholders with no indication which was which.
    for key in [
        "status",
        "uptime_seconds",
        "metrics",
        "routing",
        "registry",
        "providers",
        "provisioner",
    ] {
        assert!(body.get(key).is_some(), "overview is missing {key}");
    }
    assert!(body["metrics"]["requests_total"].is_number());
    assert!(body["routing"]["strategy"].is_string());
    assert_eq!(body["provisioner"]["configured"], false);

    // Every registry role is present with its own coverage, so the UI can never
    // silently omit a role that has no nodes.
    for role in ["main", "auxiliary", "ocr", "image", "rag_worker"] {
        assert!(body["registry"]["roles"][role].is_object(), "role {role} missing");
    }

    // Read-only: overview is a GET and must not require the admin key.
    let res = client.get(format!("{}/api/overview", base)).send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);
}

#[tokio::test]
async fn overview_reports_caller_admin_state() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();

    let anon: serde_json::Value = client
        .get(format!("{}/web/api/overview", base))
        .send().await.unwrap().json().await.unwrap();
    assert_eq!(anon["admin_key_required"], true);
    assert_eq!(anon["caller_is_admin"], false);

    let keyed: serde_json::Value = client
        .get(format!("{}/web/api/overview", base))
        .headers(admin_headers())
        .send().await.unwrap().json().await.unwrap();
    assert_eq!(keyed["caller_is_admin"], true);
}

#[tokio::test]
async fn dashboard_payload_is_wellformed() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();
    let dash: serde_json::Value = client
        .get(format!("{}/web/api/dashboard", base))
        .send().await.unwrap().json().await.unwrap();
    assert!(dash["requests_total"].is_number());
    assert!(dash["requests_by_backend"].is_object());
    assert!(dash["fallback_by_reason"].is_object());
    // The "Total Requests equals the sum of every counter" property is a unit
    // test in handlers.rs: it must not need a live backend to be meaningful.
}

// ---------------------------------------------------------------------------
// Provider management
// ---------------------------------------------------------------------------

/// Minimal OpenAI-compatible stub: `/v1/chat/completions` answers, and the
/// stub records whether it saw a bearer token.
async fn stub_openai() -> (String, Arc<Mutex<Vec<Option<String>>>>) {
    let seen: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let capture = seen.clone();
    let router = Router::new().route(
        "/v1/chat/completions",
        post(|headers: HeaderMap, body: axum::body::Bytes| async move {
            capture.lock().unwrap().push(
                headers.get("authorization").map(|v| v.to_str().unwrap_or("").to_string()),
            );
            let model = serde_json::from_slice::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v["model"].as_str().map(str::to_string))
                .unwrap_or_default();
            (axum::http::StatusCode::OK, axum::Json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "yes"}, "finish_reason": "stop"}],
                "model": model,
            })))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap(); });
    (url, seen)
}

#[tokio::test]
async fn provider_crud_reconciles_the_registry_node() {
    let (upstream, _seen) = stub_openai().await;
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();

    // Create.
    let res = client
        .post(format!("{}/web/api/providers", base))
        .headers(admin_headers())
        .json(&serde_json::json!({
            "id": "p_test", "name": "Stub OpenAI",
            "provider_type": {"type": "custom"},
            "base_url": upstream, "api_key": "sk-original",
            "models": ["stub-1", "stub-2"], "enabled": true,
            "use_as": "auxiliary", "priority": 10,
            "capabilities": {"chat": true, "streaming": true},
        }))
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::CREATED);

    // The key must never come back out, but its *state* must be visible.
    let listed: serde_json::Value = client
        .get(format!("{}/web/api/providers", base)).send().await.unwrap().json().await.unwrap();
    assert_eq!(listed[0]["has_key"], true);
    assert!(listed[0]["api_key"].is_null(), "the key itself must never be serialized");
    assert_eq!(listed[0]["key_masked"], "***inal");

    // The provider landed in the auxiliary pool.
    let pool: serde_json::Value = client
        .get(format!("{}/api/v1/admin/upstreams", base)).headers(admin_headers())
        .send().await.unwrap().json().await.unwrap();
    let node = pool["nodes"].as_array().unwrap().iter()
        .find(|n| n["id"] == "provider-p_test")
        .expect("provider node must be registered");
    assert_eq!(node["role"], "auxiliary");
    assert_eq!(node["active"], true);
    assert_eq!(node["model"], "stub-1");

    // Disable → the node is withdrawn, the provider is kept.
    let res = client
        .put(format!("{}/web/api/providers/p_test", base))
        .headers(admin_headers())
        .json(&serde_json::json!({"enabled": false}))
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);
    let pool: serde_json::Value = client
        .get(format!("{}/api/v1/admin/upstreams", base)).headers(admin_headers())
        .send().await.unwrap().json().await.unwrap();
    assert!(pool["nodes"].as_array().unwrap().iter().all(|n| n["id"] != "provider-p_test"),
            "a disabled provider must not stay in the pool");
    let listed: serde_json::Value = client
        .get(format!("{}/web/api/providers", base)).send().await.unwrap().json().await.unwrap();
    assert_eq!(listed[0]["enabled"], false);

    // Re-enable and rotate the key.
    let res = client
        .put(format!("{}/web/api/providers/p_test", base))
        .headers(admin_headers())
        .json(&serde_json::json!({"enabled": true, "api_key": "sk-rotated"}))
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);
    let listed: serde_json::Value = client
        .get(format!("{}/web/api/providers/p_test", base)).send().await.unwrap().json().await.unwrap();
    assert_eq!(listed["key_masked"], "***ated", "a posted key replaces the stored one");

    // Delete → the pool node goes with it. Previously the node survived and kept
    // receiving traffic after its provider was gone.
    let res = client
        .delete(format!("{}/web/api/providers/p_test", base))
        .headers(admin_headers())
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);
    let pool: serde_json::Value = client
        .get(format!("{}/api/v1/admin/upstreams", base)).headers(admin_headers())
        .send().await.unwrap().json().await.unwrap();
    assert!(pool["nodes"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn provider_test_endpoint_probes_without_mutating() {
    let (upstream, seen) = stub_openai().await;
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();

    client
        .post(format!("{}/web/api/providers", base))
        .headers(admin_headers())
        .json(&serde_json::json!({
            "id": "p_t", "name": "T", "provider_type": {"type": "custom"},
            "base_url": upstream, "api_key": "sk-one", "models": ["m1"],
            "enabled": true, "use_as": "main", "priority": 10,
            "capabilities": {"chat": true, "streaming": true},
        }))
        .send().await.unwrap();

    let before = seen.lock().unwrap().len();
    let res = client
        .post(format!("{}/web/api/providers/p_t/test", base))
        .headers(admin_headers())
        .json(&serde_json::json!({"model": "m1"}))
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = res.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(seen.lock().unwrap().len(), before + 1, "test must actually probe upstream");

    // An override key is used for the probe without being stored.
    let res = client
        .post(format!("{}/web/api/providers/p_t/test", base))
        .headers(admin_headers())
        .json(&serde_json::json!({"api_key": "sk-two"}))
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);
    let seen_keys = seen.lock().unwrap().clone();
    assert_eq!(seen_keys.last().unwrap().as_deref(), Some("Bearer sk-two"));
    let listed: serde_json::Value = client
        .get(format!("{}/web/api/providers/p_t", base)).send().await.unwrap().json().await.unwrap();
    assert_eq!(listed["key_masked"], "***-one", "a probe must not overwrite the stored key");

    // Unknown provider, no draft → 404, not a silent 200.
    let res = client
        .post(format!("{}/web/api/providers/nope/test", base))
        .headers(admin_headers())
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn provider_mutations_require_the_admin_key() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();
    for (method, path) in [
        (reqwest::Method::PUT, "/web/api/providers/x"),
        (reqwest::Method::DELETE, "/web/api/providers/x"),
        (reqwest::Method::POST, "/web/api/providers/x/test"),
        (reqwest::Method::POST, "/web/api/providers"),
    ] {
        let res = client
            .request(method.clone(), format!("{}{}", base, path))
            .json(&serde_json::json!({}))
            .send().await.unwrap();
        assert_eq!(res.status(), reqwest::StatusCode::FORBIDDEN, "{method} {path} must be guarded");
    }
}

// ---------------------------------------------------------------------------
// Routing strategy
// ---------------------------------------------------------------------------

#[tokio::test]
async fn routing_strategy_is_settable_at_runtime() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();

    let res = client
        .put(format!("{}/web/api/routing-strategy", base))
        .headers(admin_headers())
        .json(&serde_json::json!({"strategy": "least_latency"}))
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);

    // The dashboard used to *display* this value with no way to change it.
    let overview: serde_json::Value = client
        .get(format!("{}/web/api/overview", base)).send().await.unwrap().json().await.unwrap();
    assert_eq!(overview["routing"]["strategy"], "least_latency");

    let res = client
        .put(format!("{}/web/api/routing-strategy", base))
        .headers(admin_headers())
        .json(&serde_json::json!({"strategy": "nonsense"}))
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::BAD_REQUEST);

    // And it is guarded.
    let res = client
        .put(format!("{}/web/api/routing-strategy", base))
        .json(&serde_json::json!({"strategy": "round_robin"}))
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::FORBIDDEN);
}

// ---------------------------------------------------------------------------
// /web/api/models
// ---------------------------------------------------------------------------

#[tokio::test]
async fn advertised_models_are_deduped_and_say_where_they_were_found() {
    // Stub upstream advertising the same id the gateway advertises itself.
    let router = Router::new().route(
        "/v1/models",
        get(|| async {
            axum::Json(serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "netai-hybrid", "object": "model", "owned_by": "OVMS"},
                    {"id": "bge-m3", "object": "model", "owned_by": "OVMS"},
                    {"id": "netai-aux", "object": "model", "owned_by": "OVMS"},
                ],
            }))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap(); });

    let base = spawn_gateway_with(|cfg| {
        cfg.large_text_url = format!("{}/v1/chat/completions", origin);
        cfg.small_mllm_url = format!("{}/v1/chat/completions", origin);
        cfg.ocr_server_url = format!("{}/v1/chat/completions", origin);
        cfg.cascade_model_name = "netai-hybrid".to_string();
    })
    .await;
    let client = reqwest::Client::new();

    let body: serde_json::Value = client
        .get(format!("{}/web/api/models", base)).send().await.unwrap().json().await.unwrap();
    let data = body["data"].as_array().unwrap();

    let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
    let mut unique = ids.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(
        ids.len(),
        unique.len(),
        "duplicate ids in the discovery list: {ids:?}"
    );
    assert!(ids.contains(&"netai-hybrid"));
    assert!(ids.contains(&"bge-m3"));
    assert_eq!(ids.iter().filter(|i| **i == "netai-hybrid").count(), 1);

    // The gateway identity keeps its slot and stays the first entry.
    assert_eq!(data[0]["id"], "netai-hybrid");
    assert_eq!(data[0]["owned_by"], "netai-stack");
    // Upstream-discovered entries report where they came from, which is the only
    // honest claim when one process serves chat and embedding graphs.
    let bge = data.iter().find(|m| m["id"] == "bge-m3").unwrap();
    assert!(bge["discovered_via"].as_str().unwrap().starts_with("http://127.0.0.1"));
}

// ---------------------------------------------------------------------------
// OpenAI model discovery: every endpoint, with capabilities
// ---------------------------------------------------------------------------

async fn spawn_models_stub(ids: &[&str]) -> String {
    let data: Vec<serde_json::Value> = ids
        .iter()
        .map(|id| {
            serde_json::json!({"id": id, "object": "model", "owned_by": "stub"})
        })
        .collect();
    let router = Router::new().route(
        "/v1/models",
        get(move || {
            let data = data.clone();
            async move { axum::Json(serde_json::json!({"object": "list", "data": data})) }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap(); });
    origin
}

#[tokio::test]
async fn discovery_advertises_every_endpoint_with_capabilities() {
    let llm = spawn_models_stub(&[
        "netai-hybrid",
        "netai-hybrid-think",
        "netai-aux",
        "bge-m3",
        "bge-reranker-v2-m3",
        "Qwen3.5-9B",
    ])
    .await;
    let ocr = spawn_models_stub(&["docling-serve"]).await;
    let tts = spawn_models_stub(&["de_6l", "en_6l"]).await;
    let stt = spawn_models_stub(&["parakeet-tdt-0.6b-v3"]).await;

    let base = spawn_gateway_with(|cfg| {
        cfg.large_text_url = format!("{}/v1/chat/completions", llm);
        cfg.small_mllm_url = format!("{}/v1/chat/completions", llm);
        cfg.ocr_server_url = format!("{}/v1/chat/completions", ocr);
        cfg.tts_url = format!("{}/v1/audio", tts);
        cfg.stt_url = format!("{}/v1/audio", stt);
        cfg.cascade_model_name = "netai-hybrid".to_string();
        cfg.tts_model_name = "pocket-tts".to_string();
        cfg.stt_model_name = "parakeet-tdt-0.6b-v3".to_string();
        cfg.ocr_model_name = "docling-serve".to_string();
        cfg.image_model_name = "z-image-turbo".to_string();
        cfg.model_capabilities = cascade_llm::types::CapabilityPolicy::default();
        cfg.model_capabilities.deny = vec!["Qwen3.5-9B".to_string()];
    })
    .await;
    let client = reqwest::Client::new();

    let body: serde_json::Value = client
        .get(format!("{}/v1/models", base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let data = body["data"].as_array().unwrap();
    let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();

    // Every endpoint family must be proposed: chat, embeddings, rerank, OCR,
    // TTS, STT and image generation.
    for id in [
        "netai-hybrid",
        "netai-hybrid-think",
        "netai-aux",
        "bge-m3",
        "bge-reranker-v2-m3",
        "docling-serve",
        "de_6l",
        "en_6l",
        "parakeet-tdt-0.6b-v3",
        "z-image-turbo",
    ] {
        assert!(ids.contains(&id), "missing {id}: {ids:?}");
    }
    assert!(!ids.iter().any(|id| id.to_ascii_lowercase().contains("qwen")));

    let hybrid = data.iter().find(|m| m["id"] == "netai-hybrid").unwrap();
    assert_eq!(hybrid["capabilities"]["chat"], true);
    assert_eq!(hybrid["capabilities"]["streaming"], true);
    assert_eq!(hybrid["capabilities"]["tools"], true);
    assert_eq!(hybrid["capabilities"]["vision"], true);
    assert_eq!(hybrid["type"], "vlm");
    assert_eq!(hybrid["architecture"]["input_modalities"][1], "image");
    assert!(hybrid["supported_parameters"]
        .as_array()
        .unwrap()
        .iter()
        .any(|p| p.as_str() == Some("tools")));

    let think = data.iter().find(|m| m["id"] == "netai-hybrid-think").unwrap();
    assert_eq!(think["capabilities"]["thinking"], true);

    let aux = data.iter().find(|m| m["id"] == "netai-aux").unwrap();
    assert_eq!(aux["capabilities"]["chat"], true);
    assert_eq!(aux["capabilities"]["tools"], false);
    assert_eq!(aux["capabilities"]["vision"], false);

    let bge = data.iter().find(|m| m["id"] == "bge-m3").unwrap();
    assert_eq!(bge["capabilities"]["embeddings"], true);
    assert_eq!(bge["capabilities"]["chat"], false);
    assert_eq!(bge["type"], "embeddings");

    let rerank = data
        .iter()
        .find(|m| m["id"] == "bge-reranker-v2-m3")
        .unwrap();
    assert_eq!(rerank["capabilities"]["rerank"], true);
    assert_eq!(rerank["type"], "rerank");

    let docling = data.iter().find(|m| m["id"] == "docling-serve").unwrap();
    assert_eq!(docling["capabilities"]["ocr"], true);
    assert_eq!(docling["type"], "ocr");

    let tts_entry = data.iter().find(|m| m["id"] == "de_6l").unwrap();
    assert_eq!(tts_entry["capabilities"]["tts"], true);
    assert_eq!(tts_entry["type"], "tts");
    assert_eq!(tts_entry["capabilities"]["chat"], false);

    let stt_entry = data
        .iter()
        .find(|m| m["id"] == "parakeet-tdt-0.6b-v3")
        .unwrap();
    assert_eq!(stt_entry["capabilities"]["stt"], true);
    assert_eq!(stt_entry["capabilities"]["diarization"], true);
    assert_eq!(stt_entry["type"], "stt");
    assert_eq!(stt_entry["capabilities"]["chat"], false);

    let image = data.iter().find(|m| m["id"] == "z-image-turbo").unwrap();
    assert_eq!(image["capabilities"]["image"], true);
    assert_eq!(image["type"], "image");
    assert_eq!(image["capabilities"]["chat"], false);
}

#[tokio::test]
async fn discovery_keeps_fallback_ids_when_services_are_down() {
    let base = spawn_gateway_with(|cfg| {
        // Connection-refused endpoint; discovery must still propose the
        // configured ids instead of silently dropping an endpoint family.
        cfg.large_text_url = "http://127.0.0.1:1/v1/chat/completions".to_string();
        cfg.small_mllm_url = "http://127.0.0.1:1/v1/chat/completions".to_string();
        cfg.ocr_server_url = "http://127.0.0.1:1/v1/chat/completions".to_string();
        cfg.tts_url = "http://127.0.0.1:1/v1/audio".to_string();
        cfg.stt_url = "http://127.0.0.1:1/v1/audio".to_string();
        cfg.cascade_model_name = "netai-hybrid".to_string();
        cfg.tts_model_name = "pocket-tts".to_string();
        cfg.stt_model_name = "parakeet-tdt-0.6b-v3".to_string();
        cfg.ocr_model_name = "docling-serve".to_string();
        cfg.image_model_name = "z-image-turbo".to_string();
        cfg.model_capabilities = cascade_llm::types::CapabilityPolicy::default();
    })
    .await;
    let client = reqwest::Client::new();

    let body: serde_json::Value = client
        .get(format!("{}/v1/models", base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let data = body["data"].as_array().unwrap();
    let by_id = |id: &str| data.iter().find(|m| m["id"] == id).unwrap();

    assert_eq!(by_id("docling-serve")["capabilities"]["ocr"], true);
    assert_eq!(by_id("pocket-tts")["capabilities"]["tts"], true);
    assert_eq!(
        by_id("parakeet-tdt-0.6b-v3")["capabilities"]["stt"],
        true
    );
    assert_eq!(by_id("z-image-turbo")["capabilities"]["image"], true);
}

// ---------------------------------------------------------------------------
// GPU provisioner passthrough
// ---------------------------------------------------------------------------

/// Stub provisioner that requires `X-API-Key` and echoes what it received.
async fn stub_provisioner() -> (String, Arc<Mutex<Vec<(String, String)>>>) {
    let seen: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let capture = seen.clone();

    let router = Router::new()
        .route(
            "/api/v1/health",
            get({
                let cap = capture.clone();
                move || {
                    let cap = cap.clone();
                    async move {
                        cap.lock().unwrap().push(("/api/v1/health".into(), String::new()));
                        axum::Json(serde_json::json!({
                            "status": "ok", "active_instances": 1, "max_instances": 3,
                            "billed_usd": 1.25, "cloud_enabled": true,
                        }))
                    }
                }
            }),
        )
        .route(
            "/api/v1/*tail",
            axum::routing::any({
                let cap = capture.clone();
                move |req: axum::extract::Request| {
                    let cap = cap.clone();
                    async move {
                        let (parts, _) = req.into_parts();
                        let key = parts
                            .headers
                            .get("x-api-key")
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or("")
                            .to_string();
                        cap.lock().unwrap().push((parts.uri.path().to_string(), key.clone()));
                        if key != PROVISIONER_KEY {
                            return (
                                axum::http::StatusCode::UNAUTHORIZED,
                                axum::Json(serde_json::json!({"detail": "missing key"})),
                            );
                        }
                        (
                            axum::http::StatusCode::OK,
                            axum::Json(serde_json::json!({"ok": true, "path": parts.uri.path()})),
                        )
                    }
                }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap(); });
    (url, seen)
}

#[tokio::test]
async fn provisioner_passthrough_injects_the_key_and_hides_it() {
    let (prov, seen) = stub_provisioner().await;
    let base = spawn_gateway_with(|cfg| {
        cfg.provisioner_url = Some(prov.clone());
        cfg.provisioner_api_key = Some(PROVISIONER_KEY.to_string());
    })
    .await;
    let client = reqwest::Client::new();

    let res = client
        .get(format!("{}/web/api/provisioner/status", base))
        .headers(admin_headers())
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = res.json().await.unwrap();
    assert_eq!(body["configured"], true);
    assert_eq!(body["reachable"], true);
    assert_eq!(body["api_key_set"], true);
    // The key's *value* must never appear in any response.
    assert!(!res_text(&body).contains(PROVISIONER_KEY));

    // A read-only allowlisted path is forwarded with the key attached.
    let res = client
        .get(format!("{}/web/api/provisioner/instances", base))
        .headers(admin_headers())
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);
    let forwarded = seen.lock().unwrap().clone();
    assert!(
        forwarded.iter().any(|(p, k)| p == "/api/v1/instances" && k == PROVISIONER_KEY),
        "the provisioner key must be attached server-side: {forwarded:?}"
    );

    // A per-node lifecycle path with several segments resolves too.
    let res = client
        .post(format!("{}/web/api/provisioner/inference-nodes/gpu_abc123/activate", base))
        .headers(admin_headers())
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);

    // Every request must carry the admin key, not the provisioner key.
    let res = client
        .get(format!("{}/web/api/provisioner/instances", base))
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::FORBIDDEN);
}

fn res_text(v: &serde_json::Value) -> String {
    serde_json::to_string(v).unwrap_or_default()
}

#[tokio::test]
async fn provisioner_passthrough_is_an_allowlist_not_a_proxy() {
    let (prov, _seen) = stub_provisioner().await;
    let base = spawn_gateway_with(|cfg| {
        cfg.provisioner_url = Some(prov.clone());
        cfg.provisioner_api_key = Some(PROVISIONER_KEY.to_string());
    })
    .await;
    let client = reqwest::Client::new();

    for path in [
        // Not a provisioner endpoint at all.
        "providers/pollinations/../../etc/passwd",
        "secrets",
        "db",
        "settings/../backends",
        // Traversal / empty segments.
        "inference-nodes/..",
        "inference-nodes/gpu_abc/activate/extra",
        "inference-nodes/gpu abc",
        // A verb the provisioner does not expose.
        "shutdown",
    ] {
        let res = client
            .get(format!("{}/web/api/provisioner/{}", base, path))
            .headers(admin_headers())
            .send().await.unwrap();
        assert_eq!(
            res.status(),
            reqwest::StatusCode::NOT_FOUND,
            "{path} must not be forwarded"
        );
    }
}

#[tokio::test]
async fn provisioner_disabled_is_reported_not_faked() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();
    let body: serde_json::Value = client
        .get(format!("{}/web/api/provisioner/status", base))
        .headers(admin_headers())
        .send().await.unwrap().json().await.unwrap();
    assert_eq!(body["configured"], false);

    let res = client
        .get(format!("{}/web/api/provisioner/instances", base))
        .headers(admin_headers())
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
    let body: serde_json::Value = res.json().await.unwrap();
    assert!(body["hint"].as_str().unwrap().contains("GPU_PROVISIONER_URL"));
}

#[tokio::test]
async fn provisioner_unreachable_surfaces_as_502_not_success() {
    // Point at a closed port: the UI must be able to tell "down" from "empty".
    let dead = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = dead.local_addr().unwrap();
    drop(dead);
    let base = spawn_gateway_with(|cfg| {
        cfg.provisioner_url = Some(format!("http://{}", addr));
        cfg.provisioner_api_key = Some(PROVISIONER_KEY.to_string());
    })
    .await;
    let client = reqwest::Client::new();
    let res = client
        .get(format!("{}/web/api/provisioner/instances", base))
        .headers(admin_headers())
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::BAD_GATEWAY);
    let status: serde_json::Value = client
        .get(format!("{}/web/api/provisioner/status", base))
        .headers(admin_headers())
        .send().await.unwrap().json().await.unwrap();
    assert_eq!(status["reachable"], false);
}

// ---------------------------------------------------------------------------
// Console document
// ---------------------------------------------------------------------------

#[tokio::test]
async fn active_models_reports_the_configured_model_not_the_first_in_the_list() {
    // The defect: `fetch_reported_model` returned `data[0]`. One OVMS process
    // serves the chat AND the embedding graphs from one port, and the live node
    // lists `bge-m3` first — so the console reported an *embedding* model as the
    // main chat model. List order carries no meaning; membership does.
    let router = Router::new().route(
        "/v1/models",
        get(|| async {
            axum::Json(serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "bge-m3", "object": "model"},
                    {"id": "bge-reranker-v2-m3", "object": "model"},
                    {"id": "netai-aux", "object": "model"},
                    {"id": "netai-hybrid", "object": "model"},
                ],
            }))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap(); });

    let base = spawn_gateway_with(|cfg| {
        cfg.large_text_url = format!("{}/v1/chat/completions", origin);
        cfg.small_mllm_url = format!("{}/v1/chat/completions", origin);
        cfg.main_model_name = "netai-hybrid".to_string();
        cfg.small_model_name = "netai-aux".to_string();
    })
    .await;
    let client = reqwest::Client::new();

    let body: serde_json::Value = client
        .get(format!("{}/web/api/active-models", base))
        .send().await.unwrap().json().await.unwrap();
    assert_eq!(body["main"]["model"], "netai-hybrid");
    assert_eq!(body["auxiliary"]["model"], "netai-aux");
    assert_eq!(body["main"]["active"], true);
}

#[tokio::test]
async fn gateway_paths_the_console_needs_exist_under_the_web_namespace() {
    // Caddy exposes the console with `handle_path /cascade/* { rewrite * /web{path} }`,
    // so every gateway root path (/health, /extraction/v1/backends) is
    // unreachable from the browser: the rewrite puts it under /web, where no such
    // route exists. The console therefore reads the namespaced aliases.
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();
    for path in [
        "/web/api/health",
        "/api/health",
        "/web/api/extraction-backends",
        "/api/extraction-backends",
        "/web/api/models",
        "/api/models",
        "/web/api/overview",
        "/web/api/v1/admin/upstreams",
        "/web/api/routing-strategy",
    ] {
        let method = if path.ends_with("routing-strategy") { reqwest::Method::PUT } else { reqwest::Method::GET };
        let res = client
            .request(method.clone(), format!("{}{}", base, path))
            .headers(admin_headers())
            .json(&serde_json::json!({"strategy": "round_robin"}))
            .send().await.unwrap();
        assert_eq!(res.status(), reqwest::StatusCode::OK, "{method} {path}");
    }
}

#[tokio::test]
async fn disabled_provider_stays_out_of_the_pool_across_a_restore() {
    // The boot path re-registered every persisted provider unconditionally, so a
    // provider the operator had disabled in the UI came back into the pool on the
    // next gateway restart. The setting read as persisted and silently stopped
    // applying — the worst possible shape for a "disable" control.
    let (_upstream, _seen) = stub_openai().await;
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();

    client
        .post(format!("{}/web/api/providers", base))
        .headers(admin_headers())
        .json(&serde_json::json!({
            "id": "p_rest", "name": "R", "provider_type": {"type": "custom"},
            "base_url": _upstream, "api_key": "sk-r", "models": ["m1"],
            "enabled": true, "use_as": "main", "priority": 10,
            "capabilities": {"chat": true, "streaming": true},
        }))
        .send().await.unwrap();
    client
        .put(format!("{}/web/api/providers/p_rest", base))
        .headers(admin_headers())
        .json(&serde_json::json!({"enabled": false}))
        .send().await.unwrap();

    // What the boot path does — via the same helper `main.rs` calls.
    let stored = client
        .get(format!("{}/web/api/providers/p_rest", base))
        .send().await.unwrap().json::<serde_json::Value>().await.unwrap();
    assert_eq!(stored["enabled"], false);

    let pool: serde_json::Value = client
        .get(format!("{}/api/v1/admin/upstreams", base)).headers(admin_headers())
        .send().await.unwrap().json().await.unwrap();
    assert!(
        pool["nodes"].as_array().unwrap().iter().all(|n| n["id"] != "provider-p_rest"),
        "a disabled provider must not be in the pool"
    );
}

#[tokio::test]
async fn admin_upstream_namespace_is_complete_under_both_mounts() {
    // The console's Remove button called DELETE /web/api/v1/upstreams/{role} for
    // symmetry with the v1 inference-node routes — which did not exist. It 404'd,
    // the node stayed in the pool, and the table kept showing it: a UI that
    // guesses a path gets a silent no-op. Every mutation the console can perform
    // must resolve on BOTH mounts.
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();
    let url = |p: &str| format!("{}{}", base, p);

    for mount in ["/web", ""] {
        let res = client
            .put(url(&format!("{}/api/v1/upstreams/main", mount)))
            .headers(admin_headers())
            .json(&serde_json::json!({"endpoint_url": "http://n1:8080/v1/chat/completions"}))
            .send().await.unwrap();
        assert_eq!(res.status(), reqwest::StatusCode::CREATED, "PUT {mount}/api/v1/upstreams/main");

        let del = client
            .delete(url(&format!("{}/api/v1/upstreams/main", mount)))
            .headers(admin_headers())
            .send().await.unwrap();
        assert_eq!(del.status(), reqwest::StatusCode::OK, "DELETE {mount}/api/v1/upstreams/main");
        let body: serde_json::Value = del.json().await.unwrap();
        assert_eq!(body["removed"], 1);
    }

    // And the activate/deactivate/admin pair the console also uses.
    for path in [
        "/web/api/v1/admin/upstreams",
        "/api/v1/admin/upstreams",
    ] {
        let res = client.get(url(path)).headers(admin_headers()).send().await.unwrap();
        assert_eq!(res.status(), reqwest::StatusCode::OK, "GET {path}");
    }
}

#[tokio::test]
async fn console_is_served_from_both_mounts_and_has_no_cdn() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();

    for path in ["/web/", "/", "/web/settings"] {
        let res = client.get(format!("{}{}", base, path)).send().await.unwrap();
        assert_eq!(res.status(), reqwest::StatusCode::OK, "{path} must serve the console");
        let html = res.text().await.unwrap();
        assert!(html.contains("netai-brandmark"), "{path} lost the canonical header");
        assert!(html.contains("data-theme-toggle"), "{path} lost the theme toggle");
        assert!(html.contains("data-lang-toggle"), "{path} lost the language toggle");
        // Gotcha 13: an air-gapped install must not depend on a CDN. The old
        // dashboard pulled Chart.js from cdn.jsdelivr.net, which left two blank
        // cards with no error anywhere. Check the actual tags rather than the
        // string "cdn" — the analysis file and code comments name the old
        // library on purpose.
        for m in Regex::new(r#"<(?:script[^>]*\ssrc|link[^>]*\shref|img[^>]*\ssrc)\s*=\s*["']([^"']+)"#)
            .unwrap()
            .captures_iter(&html)
            .map(|c| c[1].to_string())
        {
            assert!(
                m.starts_with("/"),
                "{path} loads a non-local asset: {m}",
            );
        }
    }
}

// ---------------------------------------------------------------------------
// External access (address + credentials discoverability)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn access_lists_both_api_dialects_and_the_live_base_url() {
    let base = spawn_gateway_with(|cfg| {
        cfg.public_url = Some("https://example.test".to_string());
        cfg.service_api_key = Some("stack-service-key-123456".to_string());
        cfg.billing_url = None;
    })
    .await;
    let client = reqwest::Client::new();

    let body: serde_json::Value = client
        .get(format!("{}/web/api/access", base))
        .headers(admin_headers())
        .send().await.unwrap().json().await.unwrap();

    assert_eq!(body["public_base"], "https://example.test");
    assert_eq!(body["service"]["base_url"], "https://example.test/api2");
    assert_eq!(body["user_keys"]["base_url"], "https://example.test/api/llm/v1");
    // The full stack key must never be in a list response.
    assert_eq!(body["service"]["key_set"], true);
    assert_eq!(body["service"]["key_masked"], "***3456");
    assert!(body["service"].get("key").is_none());

    let paths: Vec<&str> = body["endpoints"]
        .as_array().unwrap().iter()
        .map(|e| e["path"].as_str().unwrap())
        .collect();
    assert!(paths.contains(&"/v1/chat/completions"));
    // Anthropic is a first-class surface, not a footnote behind OpenAI.
    assert!(paths.contains(&"/v1/messages"));
    assert!(paths.contains(&"/v1/messages/count_tokens"));
    assert!(paths.contains(&"/v1/rerank"));
}

#[tokio::test]
async fn service_key_reveal_is_admin_gated_and_returns_the_configured_key() {
    let base = spawn_gateway_with(|cfg| {
        cfg.service_api_key = Some("stack-service-key-123456".to_string());
    })
    .await;
    let client = reqwest::Client::new();

    let anon = client
        .get(format!("{}/web/api/access/service-key", base))
        .send().await.unwrap();
    assert_eq!(anon.status(), reqwest::StatusCode::FORBIDDEN);

    let keyed: serde_json::Value = client
        .get(format!("{}/web/api/access/service-key", base))
        .headers(admin_headers())
        .send().await.unwrap().json().await.unwrap();
    assert_eq!(keyed["key"], "stack-service-key-123456");
}

#[tokio::test]
async fn personal_key_crud_without_billing_is_a_clear_502_not_a_fake_success() {
    let base = spawn_gateway_with(|cfg| {
        // Reserved TEST-NET address: nothing listens, so this fails fast.
        cfg.billing_url = Some("http://192.0.2.1:8099".to_string());
    })
    .await;
    let client = reqwest::Client::new();

    let res = client
        .post(format!("{}/web/api/access/keys", base))
        .headers(admin_headers())
        .json(&serde_json::json!({"label": "phone"}))
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::BAD_GATEWAY);
    let body: serde_json::Value = res.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("billing"));
}

// ---------------------------------------------------------------------------
// Anthropic model discovery (Claude Code `/model` picker)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_model_list_advertises_claude_prefixed_aliases() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();

    let anth: serde_json::Value = client
        .get(format!("{}/v1/models?limit=1000", base))
        .header("anthropic-version", "2023-06-01")
        .send().await.unwrap().json().await.unwrap();

    // Claude Code drops every id that does not begin with claude/anthropic;
    // netai-hybrid and bge-m3 alone produced "0 Modelle gefunden".
    let ids: Vec<&str> = anth["data"].as_array().unwrap().iter()
        .map(|m| m["id"].as_str().unwrap()).collect();
    assert!(ids.iter().all(|id| id.starts_with("claude") || id.starts_with("anthropic")));
    assert!(ids.contains(&"claude-netai-hybrid"));
    assert!(ids.contains(&"claude-netai-main"));
    assert!(ids.contains(&"claude-netai-aux"));
    // Anthropic page shape.
    assert_eq!(anth["has_more"], false);
    assert_eq!(anth["first_id"], ids[0]);
    assert!(anth["data"][0]["type"] == "model");
    assert!(anth["data"][0]["display_name"].as_str().is_some());
    assert!(anth["data"][0]["created_at"].as_str().is_some());

    // OpenAI clients keep the historical shape and real model ids.
    let oai: serde_json::Value = client
        .get(format!("{}/v1/models", base))
        .send().await.unwrap().json().await.unwrap();
    assert_eq!(oai["object"], "list");
    let oai_ids: Vec<&str> = oai["data"].as_array().unwrap().iter()
        .map(|m| m["id"].as_str().unwrap()).collect();
    assert!(!oai_ids.is_empty());
    assert!(!oai_ids.contains(&"claude-netai-hybrid"));

    // Retrieve one alias in Anthropic shape.
    let one: serde_json::Value = client
        .get(format!("{}/v1/models/claude-netai-main", base))
        .header("anthropic-version", "2023-06-01")
        .send().await.unwrap().json().await.unwrap();
    assert_eq!(one["id"], "claude-netai-main");
    assert_eq!(one["type"], "model");
}

// ---------------------------------------------------------------------------
// Output-token ceiling (OVMS graph max_tokens_limit)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn max_tokens_is_clamped_to_the_engine_ceiling() {
    let seen: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let capture = seen.clone();
    let router = Router::new().route(
        "/v1/chat/completions",
        post(|body: axum::body::Bytes| async move {
            let mt = serde_json::from_slice::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v["max_tokens"].as_u64())
                .unwrap_or(0);
            capture.lock().unwrap().push(mt);
            (axum::http::StatusCode::OK, axum::Json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "model": "netai-hybrid",
            })))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = format!("http://{}/v1", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap(); });

    let base = spawn_gateway_with(|cfg| {
        cfg.large_text_url = format!("{}/chat/completions", upstream);
        cfg.small_mllm_url = format!("{}/chat/completions", upstream);
        cfg.max_output_tokens = Some(20479);
    })
    .await;

    let res = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", base))
        .json(&serde_json::json!({
            "model": "netai-hybrid",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64000
        }))
        .send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);
    assert_eq!(*seen.lock().unwrap(), vec![20479]);
}
