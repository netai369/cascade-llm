//! Black-box tests for the Anthropic-compatible Messages API (`/v1/messages`).
//!
//! Boots the real router pointed at a mock OpenAI-compatible upstream and
//! verifies that inbound Anthropic requests are translated, routed, and the
//! upstream response is translated back (both buffered and streaming).

use axum::body::Body;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use cascade_llm::cascade_features::MetricsRegistry;
use cascade_llm::config::AppConfig;
use cascade_llm::db::Db;
use cascade_llm::router::build_router;
use cascade_llm::state::GatewayState;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::Arc;

/// Mock OpenAI-compatible backend. Echoes the request model in the response and
/// emits SSE when `stream` is set.
async fn completions(Json(req): Json<Value>) -> Response {
    let model = req
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("mock")
        .to_string();
    let streaming = req.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);

    if streaming {
        let body = format!(
            concat!(
                "data: {{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"{m}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"Hel\"}},\"finish_reason\":null}}]}}\n\n",
                "data: {{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"{m}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"lo\"}},\"finish_reason\":null}}]}}\n\n",
                "data: {{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"{m}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n",
                "data: [DONE]\n\n"
            ),
            m = model
        );
        return Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from(body))
            .unwrap();
    }

    Json(json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": format!("Hello from upstream (model={})", model)},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
    }))
    .into_response()
}

/// Boots the mock backend + the gateway pointed at it. Returns the gateway base URL.
async fn spawn_stack() -> String {
    let mock = Router::new().route("/v1/chat/completions", post(completions));
    let mock_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(mock_listener, mock).await.unwrap() });
    let upstream = format!("http://{}/v1/chat/completions", mock_addr);

    let mut cfg = AppConfig::from_env();
    cfg.cascade_model_name = "test-cascade-model".to_string();
    cfg.large_text_url = upstream.clone();
    cfg.small_mllm_url = upstream;
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
async fn messages_non_streaming_translates_response() {
    let base = spawn_stack().await;
    let client = reqwest::Client::new();

    let res = client
        .post(format!("{}/v1/messages", base))
        .header("x-api-key", "test")
        .header("anthropic-version", "2023-06-01")
        .json(&json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 256,
            "system": "Be brief.",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), reqwest::StatusCode::OK);
    let body: Value = res.json().await.unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["model"], "claude-3-5-sonnet-20241022");
    assert_eq!(body["content"][0]["type"], "text");
    assert!(
        body["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("model=test-cascade-model"),
        "upstream must receive the gateway's unified model, got {}",
        body["content"][0]["text"]
    );
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["usage"]["input_tokens"], 9);
    assert_eq!(body["usage"]["output_tokens"], 3);
    assert!(body["id"].as_str().unwrap().starts_with("msg_"));
}

#[tokio::test]
async fn messages_streaming_emits_anthropic_events() {
    let base = spawn_stack().await;
    let client = reqwest::Client::new();

    let res = client
        .post(format!("{}/v1/messages", base))
        .json(&json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 256,
            "stream": true,
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), reqwest::StatusCode::OK);
    assert_eq!(
        res.headers()["content-type"].to_str().unwrap(),
        "text/event-stream"
    );
    let text = res.text().await.unwrap();
    assert!(text.contains("event: message_start"), "{text}");
    assert!(text.contains("\"type\":\"content_block_start\""), "{text}");
    assert!(text.contains("\"type\":\"text_delta\""), "{text}");
    assert!(text.contains("\"text\":\"Hel\""), "{text}");
    assert!(text.contains("\"text\":\"lo\""), "{text}");
    assert!(text.contains("event: message_delta"), "{text}");
    assert!(text.contains("\"stop_reason\":\"end_turn\""), "{text}");
    assert!(text.contains("event: message_stop"), "{text}");
}

#[tokio::test]
async fn messages_invalid_json_returns_anthropic_error() {
    let base = spawn_stack().await;
    let client = reqwest::Client::new();

    let res = client
        .post(format!("{}/v1/messages", base))
        .header("content-type", "application/json")
        .body("{not json")
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: Value = res.json().await.unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn count_tokens_returns_estimate() {
    let base = spawn_stack().await;
    let client = reqwest::Client::new();

    let res = client
        .post(format!("{}/v1/messages/count_tokens", base))
        .json(&json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Count these tokens please"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), reqwest::StatusCode::OK);
    let body: Value = res.json().await.unwrap();
    assert!(body["input_tokens"].as_u64().unwrap() >= 1);
}
