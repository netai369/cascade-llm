use axum::extract::State;
use axum::response::IntoResponse;
use axum::response::Response;
use http_body_util::BodyExt;
use axum::routing::post;
use axum::Json;
use axum::Router;
use cascade_llm::cascade_features::MetricsRegistry;
use cascade_llm::config::AppConfig;
use cascade_llm::db::Db;
use cascade_llm::router::build_router;
use cascade_llm::state::GatewayState;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

#[derive(Clone, Default)]
struct Recorder(Arc<Mutex<Vec<Value>>>);

impl Recorder {
    fn push(&self, v: Value) {
        self.0.lock().unwrap().push(v);
    }
    fn all(&self) -> Vec<Value> {
        self.0.lock().unwrap().clone()
    }
}

async fn mock_chat(State(rec): State<Recorder>, req: axum::extract::Request) -> Response {
    let body = req.into_body().collect().await.unwrap().to_bytes();
    if let Ok(v) = serde_json::from_slice::<Value>(&body) {
        rec.push(v);
    }
    Json(json!({
        "id": "c", "object": "chat.completion", "model": "served",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Das Bild zeigt eine Statue der Freiheitsgoettin auf einem Sockel vor einer Wasserflaeche mit Huell und Himmel im Hintergrund.", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": "{\"query\":\"Geschichte der Rialtobruecke Venedig\"}"}}]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    }))
    .into_response()
}

async fn serve_router(app: Router) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{}", addr)
}

#[tokio::test]
async fn aux_header_stays_on_aux_with_image_tools_intent() {
    let main_rec = Recorder::default();
    let aux_rec = Recorder::default();
    let main_app = Router::new()
        .route("/v1/chat/completions", post(mock_chat))
        .with_state(main_rec.clone());
    let aux_app = Router::new()
        .route("/v1/chat/completions", post(mock_chat))
        .with_state(aux_rec.clone());
    let main_origin = serve_router(main_app).await;
    let aux_origin = serve_router(aux_app).await;

    let mut cfg = AppConfig::from_env();
    cfg.admin_key = None;
    cfg.large_text_url = format!("{}/v1/chat/completions", main_origin);
    cfg.small_mllm_url = format!("{}/v1/chat/completions", aux_origin);
    cfg.ocr_server_url = format!("{}/v1/chat/completions", main_origin);
    cfg.main_model_name = "netai-hybrid".to_string();
    cfg.small_model_name = "netai-aux".to_string();
    cfg.default_route = Some("inference".to_string());
    cfg.route_tools_to_large = true;
    cfg.large_model_multimodal = false;
    cfg.vision_prepass = true;

    let state = Arc::new(GatewayState::new(
        cfg,
        Arc::new(MetricsRegistry::init()),
        Arc::new(Db::new_in_memory().unwrap()),
    ));
    let app = build_router(state);
    let base = serve_router(app).await;

    let long_system = "You are an expert software architect and security reviewer. ".repeat(6);
    let body = json!({
        "model": "netai-hybrid",
        "messages": [
            {"role": "system", "content": long_system},
            {"role": "user", "content": [
                {"type": "text", "text": "what is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]}
        ],
        "tools": [{"type": "function", "function": {"name": "web_search", "parameters": {"type":"object"}}}],
        "stream": false
    });

    let client = reqwest::Client::new();
    // With explicit aux header: must stay on aux.
    let res = client
        .post(format!("{}/v1/chat/completions", base))
        .header("x-cascade-route", "auxiliary-server")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);
    let res_json: Value = res.json().await.unwrap();
    // returned structured tool_calls survive Cascade proxying
    let tool_calls = res_json["choices"][0]["message"]["tool_calls"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(!tool_calls.is_empty(), "tool_calls must survive proxying: {res_json}");
    assert_eq!(tool_calls[0]["function"]["name"], "web_search");

    let aux_bodies = aux_rec.all();
    let main_bodies = main_rec.all();
    assert_eq!(
        main_bodies.len(),
        0,
        "main backend must not be called when aux header present"
    );
    assert_eq!(
        aux_bodies.len(),
        1,
        "aux backend must receive exactly one request"
    );
    let b = &aux_bodies[0];
    assert_eq!(b["model"], "netai-aux", "model must be retargeted to aux");

    // Incoming LibreChat-style system message is forwarded unchanged to aux.
    let system_msg = b["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "system")
        .expect("system message must be forwarded to aux upstream");
    assert_eq!(
        system_msg["content"].as_str(),
        Some(long_system.as_str()),
        "system prompt must reach aux upstream unchanged: {b}"
    );
    // find the user message with parts
    let user_msg = b["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| {
            m["content"]
                .as_array()
                .map(|parts| parts.iter().any(|p| p.get("image_url").is_some()))
                .unwrap_or(false)
        })
        .expect("user message with image parts");
    let parts = user_msg["content"].as_array().unwrap();
    // image precedes text (Gemma convention)
    let first_image_idx = parts
        .iter()
        .position(|p| p.get("image_url").is_some())
        .unwrap();
    let first_text_idx = parts
        .iter()
        .position(|p| p.get("type") == Some(&json!("text")))
        .unwrap();
    assert!(
        first_image_idx < first_text_idx,
        "image parts must precede text parts on aux route: {b}"
    );
    // original data URL preserved
    let data_url = parts
        .iter()
        .find(|p| p.get("image_url").is_some())
        .unwrap()["image_url"]["url"]
        .as_str()
        .unwrap();
    assert_eq!(data_url, "data:image/png;base64,AAAA");
    // tools/schema remain intact
    assert!(b.get("tools").is_some(), "tools must be retained on aux route: {b}");
    assert_eq!(b["tools"][0]["function"]["name"], "web_search");

    // Without header: existing selection unchanged -> main backend is used
    // (vision pre-pass path), not a direct aux pass-through of the original.
    let main_rec2 = Recorder::default();
    let aux_rec2 = Recorder::default();
    let main_app2 = Router::new()
        .route("/v1/chat/completions", post(mock_chat))
        .with_state(main_rec2.clone());
    let aux_app2 = Router::new()
        .route("/v1/chat/completions", post(mock_chat))
        .with_state(aux_rec2.clone());
    let main_origin2 = serve_router(main_app2).await;
    let aux_origin2 = serve_router(aux_app2).await;

    let mut cfg2 = AppConfig::from_env();
    cfg2.admin_key = None;
    cfg2.large_text_url = format!("{}/v1/chat/completions", main_origin2);
    cfg2.small_mllm_url = format!("{}/v1/chat/completions", aux_origin2);
    cfg2.ocr_server_url = format!("{}/v1/chat/completions", main_origin2);
    cfg2.main_model_name = "netai-hybrid".to_string();
    cfg2.small_model_name = "netai-aux".to_string();
    cfg2.default_route = Some("inference".to_string());
    cfg2.route_tools_to_large = true;
    cfg2.large_model_multimodal = false;
    cfg2.vision_prepass = true;

    let state2 = Arc::new(GatewayState::new(
        cfg2,
        Arc::new(MetricsRegistry::init()),
        Arc::new(Db::new_in_memory().unwrap()),
    ));
    let app2 = build_router(state2);
    let base2 = serve_router(app2).await;

    let res2 = client
        .post(format!("{}/v1/chat/completions", base2))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(res2.status(), reqwest::StatusCode::OK);

    let main_bodies2 = main_rec2.all();
    assert!(
        !main_bodies2.is_empty(),
        "without header, main backend should be reached (existing selection)"
    );
    let _ = aux_rec2;
}
