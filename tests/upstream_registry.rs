//! Black-box tests for the dynamic upstream registry API:
//! PUT/DELETE /web/api/upstreams/{role}, activate/deactivate,
//! GET /api/v1/admin/upstreams and POST /v1/rag/extract fallbacks.

use axum::routing::post;
use axum::Router;
use cascade_llm::cascade_features::MetricsRegistry;
use cascade_llm::config::AppConfig;
use cascade_llm::db::Db;
use cascade_llm::router::build_router;
use cascade_llm::state::GatewayState;
use std::net::SocketAddr;
use std::sync::Arc;

const ADMIN_KEY: &str = "test-admin-key";

fn admin_headers() -> reqwest::header::HeaderMap {
    let mut h = reqwest::header::HeaderMap::new();
    h.insert("x-cascade-admin-key", ADMIN_KEY.parse().unwrap());
    h
}

/// Boots a gateway on an ephemeral port; returns its base URL.
async fn spawn_gateway() -> String {
    let mut cfg = AppConfig::from_env();
    cfg.admin_key = Some(ADMIN_KEY.to_string());
    let state = Arc::new(GatewayState::new(
        cfg,
        Arc::new(MetricsRegistry::init()),
        Arc::new(Db::new_in_memory().unwrap()),
    ));
    let app: Router = build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{}", addr)
}

#[tokio::test]
async fn admin_api_requires_key() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();

    let res = client.get(format!("{}/api/v1/admin/upstreams", base)).send().await.unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::FORBIDDEN);

    let res = client
        .get(format!("{}/api/v1/admin/upstreams", base))
        .headers(admin_headers())
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = res.json().await.unwrap();
    assert!(body["nodes"].is_array());
    assert!(body["roles"].as_array().unwrap().iter().any(|r| r == "rag_worker"));
}

#[tokio::test]
async fn put_get_delete_upstream_lifecycle() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();
    let url = |p: &str| format!("{}{}", base, p);

    // Invalid role rejected.
    let res = client
        .put(url("/web/api/upstreams/bogus"))
        .headers(admin_headers())
        .json(&serde_json::json!({"endpoint_url": "http://x:1/v1"}))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::BAD_REQUEST);

    // Register with the new payload shape.
    let res = client
        .put(url("/web/api/upstreams/main"))
        .headers(admin_headers())
        .json(&serde_json::json!({
            "endpoint_url": "http://10.0.0.45:8080/v1/chat/completions",
            "bearer_token": "node-secret",
            "weight": 2,
            "max_context_length": 131072,
            "label": "Vast.ai RTX 4070",
            "provider": "vast.ai",
            "cost_per_hour": 0.076
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::CREATED);
    let created: serde_json::Value = res.json().await.unwrap();
    let node_id = created["node"]["id"].as_str().unwrap().to_string();

    // Legacy payload shape still accepted (provisioner compat) and idempotent:
    // same URL must update, not duplicate.
    let res = client
        .put(url("/api/upstreams/main"))
        .headers(admin_headers())
        .json(&serde_json::json!({"url": "http://10.0.0.45:8080/v1/chat/completions", "bearer": "node-secret-2"}))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);

    let status: serde_json::Value = client
        .get(url("/api/v1/admin/upstreams"))
        .headers(admin_headers())
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let nodes = status["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 1, "re-PUT with same URL must not create a second node");
    assert_eq!(nodes[0]["id"].as_str(), Some(node_id.as_str()));
    assert_eq!(nodes[0]["weight"], 2);
    assert_eq!(nodes[0]["max_context_length"], 131072);
    assert_eq!(nodes[0]["bearer_masked"], "***et-2", "legacy bearer update must be applied");
    assert!(nodes[0].get("bearer_token").is_none(), "bearer token must never be serialized");
    assert_eq!(status["active_roles"]["main"], true);

    // Toggle the node out of the pool.
    let res = client
        .post(url(&format!("/api/v1/inference-nodes/{}/deactivate", node_id)))
        .headers(admin_headers())
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);
    let status: serde_json::Value = client
        .get(url("/api/v1/admin/upstreams"))
        .headers(admin_headers())
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(status["active_roles"]["main"], false);

    // Unknown node -> 404.
    let res = client
        .post(url("/api/v1/inference-nodes/does-not-exist/activate"))
        .headers(admin_headers())
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::NOT_FOUND);

    // Re-activate, then clear the whole role.
    let res = client
        .post(url(&format!("/api/v1/inference-nodes/{}/activate", node_id)))
        .headers(admin_headers())
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);

    let res = client
        .delete(url("/web/api/upstreams/main"))
        .headers(admin_headers())
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);
    let cleared: serde_json::Value = res.json().await.unwrap();
    assert_eq!(cleared["removed"], 1);

    let status: serde_json::Value = client
        .get(url("/api/v1/admin/upstreams"))
        .headers(admin_headers())
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(status["nodes"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn rag_extract_without_worker_returns_503() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();
    let res = client
        .post(format!("{}/v1/rag/extract", base))
        .json(&serde_json::json!({"documents": ["doc"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
    let body: serde_json::Value = res.json().await.unwrap();
    assert_eq!(body["error"]["type"], "rag_worker_unavailable");
}

#[tokio::test]
async fn rag_extract_forwards_to_registered_worker() {
    // Minimal rag_worker stub.
    async fn echo() -> String { r#"{"status":"queued"}"#.to_string() }
    let worker = Router::new().route("/jobs", post(echo));
    let worker_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker_url = format!("http://{}/jobs", worker_listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(worker_listener, worker).await.unwrap(); });

    let base = spawn_gateway().await;
    let client = reqwest::Client::new();

    client
        .put(format!("{}/api/upstreams/rag_worker", base))
        .headers(admin_headers())
        .json(&serde_json::json!({"endpoint_url": worker_url}))
        .send()
        .await
        .unwrap();

    let res = client
        .post(format!("{}/v1/rag/extract", base))
        .json(&serde_json::json!({"documents": ["lease-contract.pdf"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);
    assert_eq!(res.headers()["x-cascade-route"], "rag-worker");
    assert!(res.headers().get("x-cascade-node").is_some());
    let body: serde_json::Value = res.json().await.unwrap();
    assert_eq!(body["status"], "queued");
}
