//! Registry-first routing per role: node present -> routed with model rewrite,
//! auth and exactly one UsageEvent; node absent -> static fallback. Also the
//! role-specific health probes, the saved-URL migration surface and the
//! endpoint-hit counters on the overview.

use axum::body::Body;
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use cascade_llm::cascade_features::MetricsRegistry;
use cascade_llm::config::AppConfig;
use cascade_llm::db::Db;
use cascade_llm::router::build_router;
use cascade_llm::state::GatewayState;
use cascade_llm::usage::{UsageEvent, UsageRecorder};
use http_body_util::BodyExt;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::UnboundedReceiver;

const ADMIN_KEY: &str = "test-admin-key";

/// Method, path, Authorization, body of every request the mock upstream saw.
type RecordedRequest = (String, String, Option<String>, Vec<u8>);

#[derive(Clone, Default)]
struct Seen(Arc<Mutex<Vec<RecordedRequest>>>);

impl Seen {
    fn all(&self) -> Vec<RecordedRequest> {
        self.0.lock().unwrap().clone()
    }
    /// The one request that hit `path` (panics if none or several).
    fn only(&self, path: &str) -> (String, Option<String>, Vec<u8>) {
        let hits: Vec<_> = self.all().into_iter().filter(|r| r.1 == path).collect();
        assert_eq!(hits.len(), 1, "expected exactly one request to {path}");
        let (_, _, auth, body) = hits.into_iter().next().unwrap();
        (path.to_string(), auth, body)
    }
}

async fn mock_handler(State(seen): State<Seen>, req: axum::extract::Request) -> Response {
    let method = req.method().to_string();
    let path = req.uri().path().to_string();
    let auth = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let body = req.into_body().collect().await.unwrap().to_bytes().to_vec();
    seen.0.lock().unwrap().push((method, path.clone(), auth, body));
    let json = |v: serde_json::Value| {
        Response::builder()
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(v.to_string()))
            .unwrap()
    };
    match path.as_str() {
        "/v1/audio/speech" => Response::builder()
            .header(CONTENT_TYPE, "audio/mpeg")
            .body(Body::from("FAKEAUDIO"))
            .unwrap(),
        "/v1/audio/transcriptions" => {
            json(serde_json::json!({"text": "hi", "duration": 2.5, "model": "stt-served"}))
        }
        "/v1/images/generations" => {
            json(serde_json::json!({"data": [{"url": "u"}], "model": "img-served"}))
        }
        "/v1/embeddings" => json(serde_json::json!({
            "data": [], "model": "emb-served",
            "usage": {"prompt_tokens": 3, "total_tokens": 3}
        })),
        "/v1/chat/completions" => json(serde_json::json!({
            "id": "c", "object": "chat.completion", "model": "served",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
        })),
        "/health" | "/v1/models" => json(serde_json::json!({"data": []})),
        _ => Response::builder().status(404).body(Body::empty()).unwrap(),
    }
}

async fn serve(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{}", addr)
}

/// A mock OpenAI-style upstream that records what it receives.
async fn mock_upstream() -> (String, Seen) {
    let seen = Seen::default();
    let app = Router::new()
        .fallback(mock_handler)
        .with_state(seen.clone());
    (serve(app).await, seen)
}

/// A gateway whose usage events are captured on a channel.
async fn spawn_gateway(tweak: impl FnOnce(&mut AppConfig)) -> (String, UnboundedReceiver<UsageEvent>) {
    let mut cfg = AppConfig::from_env();
    cfg.admin_key = Some(ADMIN_KEY.to_string());
    cfg.tts_url = "http://127.0.0.1:9".to_string();
    cfg.stt_url = "http://127.0.0.1:9".to_string();
    cfg.tts_api_key = None;
    tweak(&mut cfg);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut state = GatewayState::new(
        cfg,
        Arc::new(MetricsRegistry::init()),
        Arc::new(Db::new_in_memory().unwrap()),
    );
    state.usage = UsageRecorder::with_sender(tx);
    let base = serve(build_router(Arc::new(state))).await;
    (base, rx)
}

fn admin() -> reqwest::header::HeaderMap {
    let mut h = reqwest::header::HeaderMap::new();
    h.insert("x-cascade-admin-key", ADMIN_KEY.parse().unwrap());
    h
}

async fn register(base: &str, role: &str, body: serde_json::Value) {
    let res = reqwest::Client::new()
        .put(format!("{}/web/api/upstreams/{}", base, role))
        .headers(admin())
        .json(&body)
        .send()
        .await
        .unwrap();
    assert!(res.status().is_success(), "register {role} failed: {}", res.status());
}

/// Exactly one usage event has been recorded so far.
fn assert_one_event(rx: &mut UnboundedReceiver<UsageEvent>) -> UsageEvent {
    let first = rx.try_recv().expect("expected one usage event");
    assert!(rx.try_recv().is_err(), "more than one usage event recorded");
    first
}

#[tokio::test]
async fn speech_node_is_routed_with_model_auth_and_one_event() {
    let (mock, seen) = mock_upstream().await;
    let (base, mut rx) = spawn_gateway(|_| {}).await;
    register(&base, "speech", serde_json::json!({
        "endpoint_url": format!("{}/v1/audio/speech", mock),
        "bearer_token": "node-secret", "model": "tts-alias"
    }))
    .await;

    let res = reqwest::Client::new()
        .post(format!("{}/v1/audio/speech", base))
        .json(&serde_json::json!({"model": "client-model", "input": "Hallo Welt", "voice": "v"}))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    assert_eq!(res.bytes().await.unwrap().as_ref(), b"FAKEAUDIO");

    let (_, auth, body) = seen.only("/v1/audio/speech");
    assert_eq!(auth.as_deref(), Some("Bearer node-secret"));
    let sent: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(sent["model"], "tts-alias");
    let ev = assert_one_event(&mut rx);
    assert_eq!(ev.kind, "audio_speech");
    assert_eq!(ev.characters, 10);
}

#[tokio::test]
async fn speech_without_node_uses_static_fallback() {
    let (mock, seen) = mock_upstream().await;
    let (base, mut rx) = spawn_gateway(|c| c.tts_url = format!("{}/v1/audio", mock)).await;

    let res = reqwest::Client::new()
        .post(format!("{}/v1/audio/speech", base))
        .json(&serde_json::json!({"model": "m", "input": "abc"}))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let (_, auth, body) = seen.only("/v1/audio/speech");
    assert_eq!(auth, None);
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&body).unwrap()["model"], "m");
    assert_eq!(assert_one_event(&mut rx).kind, "audio_speech");
}

#[tokio::test]
async fn transcription_node_rewrites_multipart_model_and_meters_duration() {
    let (mock, seen) = mock_upstream().await;
    let (base, mut rx) = spawn_gateway(|_| {}).await;
    register(&base, "transcription", serde_json::json!({
        "endpoint_url": format!("{}/v1/audio/transcriptions", mock),
        "bearer_token": "stt-secret", "model": "whisper-alias"
    }))
    .await;

    let boundary = "XBOUND";
    let form = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nclient-model\r\n\
         --{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
         Content-Type: audio/wav\r\n\r\nRIFFBYTES\r\n--{b}--\r\n",
        b = boundary
    );
    let res = reqwest::Client::new()
        .post(format!("{}/v1/audio/transcriptions", base))
        .header(CONTENT_TYPE, format!("multipart/form-data; boundary={}", boundary))
        .body(form)
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);

    let (_, auth, body) = seen.only("/v1/audio/transcriptions");
    let body = String::from_utf8_lossy(&body).to_string();
    assert_eq!(auth.as_deref(), Some("Bearer stt-secret"));
    assert!(body.contains("whisper-alias"), "model field rewritten: {body}");
    assert!(!body.contains("client-model"));
    assert!(body.contains("RIFFBYTES"), "file part copied verbatim");
    let ev = assert_one_event(&mut rx);
    assert_eq!(ev.kind, "audio_transcription");
    assert!((ev.audio_seconds - 2.5).abs() < 1e-9);
}

#[tokio::test]
async fn transcription_without_node_uses_static_stt_url() {
    let (mock, seen) = mock_upstream().await;
    let (base, mut rx) = spawn_gateway(|c| c.stt_url = format!("{}/v1/audio", mock)).await;

    let res = reqwest::Client::new()
        .post(format!("{}/v1/audio/transcriptions", base))
        .header(CONTENT_TYPE, "multipart/form-data; boundary=B")
        .body("--B\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nx\r\n--B--\r\n")
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    seen.only("/v1/audio/transcriptions");
    assert_eq!(assert_one_event(&mut rx).kind, "audio_transcription");
}

#[tokio::test]
async fn image_node_is_routed_with_model_auth_and_one_event() {
    let (mock, seen) = mock_upstream().await;
    let (base, mut rx) = spawn_gateway(|_| {}).await;
    register(&base, "image", serde_json::json!({
        "endpoint_url": format!("{}/v1/images/generations", mock),
        "bearer_token": "img-secret", "model": "img-alias"
    }))
    .await;

    let res = reqwest::Client::new()
        .post(format!("{}/v1/images/generations", base))
        .json(&serde_json::json!({"model": "client", "prompt": "a cat"}))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let (_, auth, body) = seen.only("/v1/images/generations");
    assert_eq!(auth.as_deref(), Some("Bearer img-secret"));
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&body).unwrap()["model"], "img-alias");
    let ev = assert_one_event(&mut rx);
    assert_eq!(ev.kind, "image");
    assert_eq!(ev.images, 1);
}

#[tokio::test]
async fn embeddings_node_rewrites_model_and_meters_once() {
    let (mock, seen) = mock_upstream().await;
    let (base, mut rx) = spawn_gateway(|_| {}).await;
    register(&base, "embeddings", serde_json::json!({
        "endpoint_url": format!("{}/v1", mock),
        "bearer_token": "emb-secret", "model": "emb-alias"
    }))
    .await;

    let res = reqwest::Client::new()
        .post(format!("{}/v1/embeddings", base))
        .json(&serde_json::json!({"model": "client", "input": "abc"}))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let (_, auth, body) = seen.only("/v1/embeddings");
    assert_eq!(auth.as_deref(), Some("Bearer emb-secret"));
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&body).unwrap()["model"], "emb-alias");
    assert_eq!(assert_one_event(&mut rx).kind, "embeddings");
}

#[tokio::test]
async fn extraction_node_is_preferred_and_meters_once() {
    let (mock, seen) = mock_upstream().await;
    let (base, mut rx) = spawn_gateway(|_| {}).await;
    register(&base, "extraction", serde_json::json!({
        "endpoint_url": format!("{}/v1/chat/completions", mock),
        "bearer_token": "ext-secret", "model": "ext-alias"
    }))
    .await;

    let res = reqwest::Client::new()
        .post(format!("{}/extraction/v1/chat/completions", base))
        .json(&serde_json::json!({"model": "client", "messages": [{"role": "user", "content": "x"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let (_, auth, body) = seen.only("/v1/chat/completions");
    assert_eq!(auth.as_deref(), Some("Bearer ext-secret"));
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&body).unwrap()["model"], "ext-alias");
    let ev = assert_one_event(&mut rx);
    assert_eq!(ev.route, "extraction");
}

/// Upstream whose `/health` fails (500) but whose model listing and chat work.
async fn flaky_health_upstream() -> String {
    let app = Router::new()
        .route("/health", get(|| async { Response::builder().status(500).body(Body::empty()).unwrap() }))
        .route("/v1/models", get(|| async { axum::Json(serde_json::json!({"data": []})) }))
        .fallback(|| async { Response::builder().status(500).body(Body::empty()).unwrap() });
    serve(app).await
}

#[tokio::test]
async fn audio_node_is_probed_by_model_listing_not_chat_health() {
    let mock = flaky_health_upstream().await;
    let (base, _rx) = spawn_gateway(|_| {}).await;
    register(&base, "speech", serde_json::json!({"endpoint_url": format!("{}/v1/audio/speech", mock)})).await;
    register(&base, "main", serde_json::json!({"endpoint_url": format!("{}/v1/chat/completions", mock)})).await;

    // Two sweeps: the default failure threshold demotes a node failing both.
    for _ in 0..2 {
        let res = reqwest::Client::new()
            .post(format!("{}/api/v1/admin/upstreams/probe", base))
            .headers(admin())
            .send()
            .await
            .unwrap();
        assert!(res.status().is_success());
    }
    let res = reqwest::Client::new()
        .get(format!("{}/api/v1/admin/upstreams", base))
        .headers(admin())
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = res.json().await.unwrap();
    let nodes = body["nodes"].as_array().unwrap();
    let healthy = |role: &str| {
        nodes
            .iter()
            .find(|n| n["role"] == role)
            .map(|n| n["healthy"].as_bool().unwrap())
            .unwrap()
    };
    assert!(healthy("speech"), "speech node must not be judged by /health");
    assert!(!healthy("main"), "chat node still probed by /health");
}

#[tokio::test]
async fn endpoint_hits_are_counted_by_path_on_the_overview() {
    let (base, _rx) = spawn_gateway(|_| {}).await;
    let overview = |base: String| async move {
        let v: serde_json::Value = reqwest::get(format!("{}/web/api/dashboard", base))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        v
    };
    let before = overview(base.clone()).await;
    let before_hits = before["requests_by_backend"]["/v1/models"].as_u64().unwrap();
    let res = reqwest::get(format!("{}/v1/models", base)).await.unwrap();
    let _ = res.status();
    let after = overview(base.clone()).await;
    assert_eq!(after["requests_by_backend"]["/v1/models"].as_u64().unwrap(), before_hits + 1);
    // Idle endpoints are listed with 0 rather than omitted.
    assert!(after["requests_by_backend"].get("/v1/rerank").is_some());
}
