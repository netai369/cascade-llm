//! `GET /v1/realtime` — OpenAI-Realtime-compatible WebSocket proxy to the STT
//! realtime upstream (`STT_REALTIME_URL`, default `ws://speech-stt:5093/v1/realtime`).
//!
//! Used by voice chat and dictation alike; the client side speaks the same
//! protocol it would speak to the STT service directly, so pointing a client
//! at cascade is a URL change only.
//!
//! Design rules:
//! - The upstream host is operator configuration only. A client can select the
//!   upstream with `?upstream=stt` (the sole accepted value); any other value is
//!   rejected with 400. Every other query parameter is forwarded verbatim.
//!   Clients can never supply a URL, so this is not an SSRF surface.
//! - Frames are relayed both ways concurrently. Audio and transcripts are never
//!   logged. Only the small JSON envelope is parsed, for metering.
//! - Metering: audio seconds come from client `input_audio_buffer.append`
//!   frames (decoded PCM16 bytes / (2 * sample rate)). The rate can be changed
//!   by `session.update` (`session.audio.input.format.rate`), default 16 kHz.
//!   The model name comes from the upstream `session.created` event.

use crate::{state::GatewayState, usage::UsageEvent};
use axum::{
    extract::{
        ws::{CloseFrame as AxCloseFrame, Message as AxMsg, WebSocket, WebSocketUpgrade},
        RawQuery, State,
    },
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use std::{borrow::Cow, sync::Arc, time::Duration};
use tokio_tungstenite::tungstenite::{
    protocol::{frame::coding::CloseCode, CloseFrame as UpCloseFrame, WebSocketConfig},
    Message as UpMsg,
};
use tracing::warn;

/// Operator-configured upstream when `STT_REALTIME_URL` is unset.
pub const DEFAULT_STT_REALTIME_URL: &str = "ws://speech-stt:5093/v1/realtime";
/// Hard cap for a single message or frame, in both directions.
pub const MAX_MESSAGE_BYTES: usize = 20 * 1024 * 1024;
const UPSTREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_SAMPLE_RATE: u32 = 16_000;
/// Query keys consumed by the gateway and never forwarded upstream.
const RESERVED_QUERY_KEYS: &[&str] = &["upstream"];

/// Build the upstream URL from the operator base and the forwarded query.
fn build_upstream_url(base: &str, forwarded: Option<&str>) -> String {
    match forwarded {
        Some(q) if !q.is_empty() => {
            let sep = if base.contains('?') { '&' } else { '?' };
            format!("{base}{sep}{q}")
        }
        _ => base.to_string(),
    }
}

/// Validate the client's `upstream` selector and return the query string to
/// forward (reserved keys removed). `Err` carries a short reason for a 400.
fn select_upstream_query(query: Option<&str>) -> Result<Option<String>, &'static str> {
    let Some(query) = query else {
        return Ok(None);
    };
    // A fragment never reaches the server in practice, but drop it anyway so a
    // forwarded value cannot change how the upstream URL parses.
    let query = query.split('#').next().unwrap_or("");
    let mut kept: Vec<&str> = Vec::new();
    for pair in query.split('&').filter(|p| !p.is_empty()) {
        let key = pair.split('=').next().unwrap_or("");
        if key == "upstream" {
            let value = pair.split_once('=').map(|(_, v)| v).unwrap_or("");
            if value != "stt" {
                return Err("unknown upstream; the only accepted value is 'stt'");
            }
            continue;
        }
        if RESERVED_QUERY_KEYS.contains(&key) {
            continue;
        }
        kept.push(pair);
    }
    if kept.is_empty() {
        Ok(None)
    } else {
        Ok(Some(kept.join("&")))
    }
}

fn upstream_base() -> String {
    std::env::var("STT_REALTIME_URL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_STT_REALTIME_URL.to_string())
}

// ---------------------------------------------------------------------------
// Audio metering
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ClientEvent<'a> {
    #[serde(rename = "type", borrow, default)]
    kind: Cow<'a, str>,
    /// Top-level base64 PCM16 payload of `input_audio_buffer.append`.
    #[serde(borrow, default)]
    audio: Option<Cow<'a, str>>,
    /// `session.update` payload; only the input format rate matters here.
    #[serde(default)]
    session: Option<SessionUpdate>,
}

#[derive(Deserialize)]
struct SessionUpdate {
    #[serde(default)]
    audio: Option<AudioSection>,
}

#[derive(Deserialize)]
struct AudioSection {
    #[serde(default)]
    input: Option<InputSection>,
}

#[derive(Deserialize)]
struct InputSection {
    #[serde(default)]
    format: Option<FormatSection>,
}

#[derive(Deserialize)]
struct FormatSection {
    #[serde(default)]
    rate: Option<u32>,
}

#[derive(Deserialize)]
struct UpstreamEvent {
    #[serde(default)]
    session: Option<UpstreamSession>,
}

#[derive(Deserialize)]
struct UpstreamSession {
    #[serde(default)]
    model: Option<String>,
}

/// Counts audio seconds and captures the upstream model name for one session.
#[derive(Debug)]
pub struct AudioMeter {
    rate: u32,
    seconds: f64,
    model: Option<String>,
}

impl Default for AudioMeter {
    fn default() -> Self {
        Self {
            rate: DEFAULT_SAMPLE_RATE,
            seconds: 0.0,
            model: None,
        }
    }
}

impl AudioMeter {
    /// Observe one client text frame. Malformed or irrelevant frames are ignored.
    pub fn observe_client(&mut self, text: &str) {
        // Cheap pre-filter: only the two event types we meter are parsed.
        if !text.contains("input_audio_buffer.append") && !text.contains("session.update") {
            return;
        }
        let Ok(ev) = serde_json::from_str::<ClientEvent>(text) else {
            return;
        };
        match ev.kind.as_ref() {
            "session.update" => {
                let rate = ev
                    .session
                    .and_then(|s| s.audio)
                    .and_then(|a| a.input)
                    .and_then(|i| i.format)
                    .and_then(|f| f.rate);
                if let Some(rate) = rate.filter(|r| *r > 0) {
                    self.rate = rate;
                }
            }
            "input_audio_buffer.append" => {
                let Some(b64) = ev.audio else {
                    return;
                };
                let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64.as_bytes())
                else {
                    return;
                };
                // PCM16 mono: 2 bytes per sample.
                self.seconds += bytes.len() as f64 / (2.0 * self.rate as f64);
            }
            _ => {}
        }
    }

    /// Observe one upstream text frame; only `session.created` is inspected.
    pub fn observe_upstream(&mut self, text: &str) {
        if self.model.is_some() || !text.contains("session.created") {
            return;
        }
        if let Ok(ev) = serde_json::from_str::<UpstreamEvent>(text) {
            if let Some(model) = ev.session.and_then(|s| s.model).filter(|m| !m.is_empty()) {
                self.model = Some(model);
            }
        }
    }

    pub fn audio_seconds(&self) -> f64 {
        self.seconds
    }

    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }
}

// ---------------------------------------------------------------------------
// Handler and session pump
// ---------------------------------------------------------------------------

pub async fn realtime_ws(
    ws: WebSocketUpgrade,
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
    RawQuery(query): RawQuery,
) -> Response {
    let forwarded = match select_upstream_query(query.as_deref()) {
        Ok(f) => f,
        Err(reason) => return (StatusCode::BAD_REQUEST, reason).into_response(),
    };
    let upstream_url = build_upstream_url(&upstream_base(), forwarded.as_deref());
    ws.max_message_size(MAX_MESSAGE_BYTES)
        .max_frame_size(MAX_MESSAGE_BYTES)
        .on_upgrade(move |socket| run_session(socket, state, headers, upstream_url))
}

fn upstream_config() -> WebSocketConfig {
    let mut cfg = WebSocketConfig::default();
    cfg.max_message_size = Some(MAX_MESSAGE_BYTES);
    cfg.max_frame_size = Some(MAX_MESSAGE_BYTES);
    cfg
}

type Upstream = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

async fn connect_upstream(url: &str) -> Result<Upstream, String> {
    if !(url.starts_with("ws://") || url.starts_with("wss://")) {
        return Err("upstream URL must use ws:// or wss://".to_string());
    }
    let connect = tokio_tungstenite::connect_async_with_config(url, Some(upstream_config()), false);
    match tokio::time::timeout(UPSTREAM_CONNECT_TIMEOUT, connect).await {
        Ok(Ok((ws, _response))) => Ok(ws),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("connect timed out".to_string()),
    }
}

/// Tell the client the upstream is unavailable, then close.
async fn reject_upstream(mut socket: WebSocket) {
    let frame = json!({
        "type": "error",
        "error": {
            "type": "upstream_unavailable",
            "message": "speech-to-text realtime upstream is unavailable",
        },
    });
    let _ = socket.send(AxMsg::Text(frame.to_string())).await;
    let _ = socket
        .send(AxMsg::Close(Some(AxCloseFrame {
            code: 1011,
            reason: Cow::Borrowed("upstream_unavailable"),
        })))
        .await;
}

async fn run_session(
    socket: WebSocket,
    state: Arc<GatewayState>,
    headers: HeaderMap,
    upstream_url: String,
) {
    let started = std::time::Instant::now();
    let upstream = match connect_upstream(&upstream_url).await {
        Ok(ws) => ws,
        Err(reason) => {
            // Log the reason only; the URL is operator config, the frames are never logged.
            warn!(upstream_error = %reason, "realtime: STT upstream connect failed");
            reject_upstream(socket).await;
            return;
        }
    };

    let (mut client_tx, mut client_rx) = socket.split();
    let (mut up_tx, mut up_rx) = upstream.split();
    let mut meter = AudioMeter::default();

    loop {
        tokio::select! {
            incoming = client_rx.next() => match incoming {
                None | Some(Err(_)) => break,
                Some(Ok(msg)) => match msg {
                    AxMsg::Text(text) => {
                        meter.observe_client(&text);
                        if up_tx.send(UpMsg::Text(text)).await.is_err() {
                            break;
                        }
                    }
                    AxMsg::Binary(bytes) => {
                        if up_tx.send(UpMsg::Binary(bytes)).await.is_err() {
                            break;
                        }
                    }
                    AxMsg::Ping(payload) => {
                        if client_tx.send(AxMsg::Pong(payload)).await.is_err() {
                            break;
                        }
                    }
                    AxMsg::Pong(_) => {}
                    AxMsg::Close(frame) => {
                        let up_frame = frame.map(|f| UpCloseFrame {
                            code: CloseCode::from(f.code),
                            reason: f.reason,
                        });
                        let _ = up_tx.send(UpMsg::Close(up_frame)).await;
                        break;
                    }
                },
            },
            incoming = up_rx.next() => match incoming {
                None | Some(Err(_)) => break,
                Some(Ok(msg)) => match msg {
                    UpMsg::Text(text) => {
                        meter.observe_upstream(&text);
                        if client_tx.send(AxMsg::Text(text)).await.is_err() {
                            break;
                        }
                    }
                    UpMsg::Binary(bytes) => {
                        if client_tx.send(AxMsg::Binary(bytes)).await.is_err() {
                            break;
                        }
                    }
                    UpMsg::Ping(payload) => {
                        if up_tx.send(UpMsg::Pong(payload)).await.is_err() {
                            break;
                        }
                    }
                    UpMsg::Close(frame) => {
                        let client_frame = frame.map(|f| AxCloseFrame {
                            code: u16::from(f.code),
                            reason: f.reason,
                        });
                        let _ = client_tx.send(AxMsg::Close(client_frame)).await;
                        break;
                    }
                    _ => {}
                },
            },
        }
    }

    // Best-effort close of both sides; either may already be gone.
    let _ = tokio::time::timeout(Duration::from_secs(2), async {
        let _ = up_tx.close().await;
        let _ = client_tx.close().await;
    })
    .await;

    let mut ev = UsageEvent::from_headers(&headers, "realtime", "stt");
    ev.audio_seconds = meter.audio_seconds();
    ev.model = meter.model().unwrap_or("stt-realtime").to_string();
    ev.latency_ms = started.elapsed().as_millis() as u64;
    ev.status = 101;
    state.usage.record(ev);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn b64_zeros(n: usize) -> String {
        base64::engine::general_purpose::STANDARD.encode(vec![0u8; n])
    }

    fn append(n: usize) -> String {
        format!(
            r#"{{"type":"input_audio_buffer.append","audio":"{}"}}"#,
            b64_zeros(n)
        )
    }

    #[test]
    fn append_32000_bytes_is_one_second_at_16k() {
        let mut m = AudioMeter::default();
        m.observe_client(&append(32_000));
        assert!((m.audio_seconds() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn session_update_rate_overrides_default() {
        let mut m = AudioMeter::default();
        m.observe_client(
            r#"{"type":"session.update","session":{"audio":{"input":{"format":{"type":"audio/pcm","rate":8000}}}}}"#,
        );
        // 16000 bytes at 8 kHz PCM16 = 8000 samples = 1.0 s.
        m.observe_client(&append(16_000));
        assert!((m.audio_seconds() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn zero_or_missing_rate_keeps_default() {
        let mut m = AudioMeter::default();
        m.observe_client(
            r#"{"type":"session.update","session":{"audio":{"input":{"format":{"rate":0}}}}}"#,
        );
        m.observe_client(
            r#"{"type":"session.update","session":{"vad":{"enabled":true}}}"#,
        );
        m.observe_client(&append(32_000));
        assert!((m.audio_seconds() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn malformed_and_irrelevant_frames_are_ignored() {
        let mut m = AudioMeter::default();
        m.observe_client("{not json");
        m.observe_client(r#"{"type":"input_audio_buffer.append","audio":"!!!not-base64"}"#);
        m.observe_client(r#"{"type":"input_audio_buffer.append"}"#);
        m.observe_client(r#"{"type":"input_audio_buffer.commit"}"#);
        m.observe_client(r#"{"type":"input_audio_buffer.append","audio":42}"#);
        assert_eq!(m.audio_seconds(), 0.0);
    }

    #[test]
    fn upstream_session_created_sets_model_once() {
        let mut m = AudioMeter::default();
        m.observe_upstream(r#"{"type":"session.created","session":{"model":"parakeet-x"}}"#);
        m.observe_upstream(r#"{"type":"session.created","session":{"model":"other"}}"#);
        assert_eq!(m.model(), Some("parakeet-x"));
        m.observe_upstream("garbage session.created");
        assert_eq!(m.model(), Some("parakeet-x"));
    }

    #[test]
    fn upstream_url_building_and_query_selection() {
        assert_eq!(select_upstream_query(None), Ok(None));
        assert_eq!(select_upstream_query(Some("upstream=stt")), Ok(None));
        assert_eq!(
            select_upstream_query(Some("upstream=stt&lang=de")),
            Ok(Some("lang=de".to_string()))
        );
        assert_eq!(
            select_upstream_query(Some("lang=de&upstream=stt&x=1")),
            Ok(Some("lang=de&x=1".to_string()))
        );
        assert!(select_upstream_query(Some("upstream=http://evil.example")).is_err());
        assert!(select_upstream_query(Some("upstream=stt&upstream=evil")).is_err());
        assert_eq!(
            build_upstream_url("ws://h:1/v1/realtime", Some("lang=de")),
            "ws://h:1/v1/realtime?lang=de"
        );
        assert_eq!(
            build_upstream_url("ws://h:1/v1/realtime?a=1", Some("lang=de")),
            "ws://h:1/v1/realtime?a=1&lang=de"
        );
        assert_eq!(
            build_upstream_url("ws://h:1/v1/realtime", None),
            "ws://h:1/v1/realtime"
        );
    }
}

#[cfg(test)]
mod integration {
    use super::*;
    use crate::{
        cascade_features::MetricsRegistry, config::AppConfig, db::Db, router::build_router,
        usage::UsageRecorder,
    };
    use tokio::net::TcpListener;
    use tokio_tungstenite::{accept_async, connect_async, tungstenite::Message as TMsg};

    type Client = tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >;

    async fn next_text(ws: &mut Client) -> String {
        loop {
            let msg = tokio::time::timeout(Duration::from_secs(5), ws.next())
                .await
                .expect("timed out waiting for a frame")
                .expect("stream ended")
                .expect("ws error");
            if let TMsg::Text(t) = msg {
                return t;
            }
        }
    }

    async fn spawn_gateway(
        usage_tx: tokio::sync::mpsc::UnboundedSender<UsageEvent>,
    ) -> std::net::SocketAddr {
        let cfg = AppConfig::from_env();
        let mut gs = GatewayState::new(
            cfg,
            Arc::new(MetricsRegistry::init()),
            Arc::new(Db::new_in_memory().unwrap()),
        );
        gs.usage = UsageRecorder::with_sender(usage_tx);
        let app = build_router(Arc::new(gs));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        addr
    }

    fn append_frame(n: usize) -> String {
        format!(
            r#"{{"type":"input_audio_buffer.append","audio":"{}"}}"#,
            base64::engine::general_purpose::STANDARD.encode(vec![0u8; n])
        )
    }

    #[tokio::test]
    async fn realtime_proxy_relays_and_meters() {
        let (usage_tx, mut usage_rx) = tokio::sync::mpsc::unbounded_channel();
        let gw = spawn_gateway(usage_tx).await;

        // 1. Unknown upstream selector is refused before the upgrade.
        let bad = connect_async(format!("ws://{gw}/v1/realtime?upstream=http://evil.example")).await;
        match bad {
            Err(tokio_tungstenite::tungstenite::Error::Http(resp)) => {
                assert_eq!(resp.status(), 400)
            }
            other => panic!("expected HTTP 400 for a non-stt upstream, got {other:?}"),
        }

        // 2. Upstream down: client gets one upstream_unavailable frame, then close.
        std::env::set_var("STT_REALTIME_URL", "ws://127.0.0.1:1/v1/realtime");
        let (mut down, _) = connect_async(format!("ws://{gw}/v1/realtime?upstream=stt"))
            .await
            .expect("upgrade should succeed even when the upstream is down");
        let err_frame = next_text(&mut down).await;
        let v: serde_json::Value = serde_json::from_str(&err_frame).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "upstream_unavailable");

        // 3. Healthy path: fake STT upstream on an ephemeral port.
        let up_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let up_addr = up_listener.local_addr().unwrap();
        std::env::set_var("STT_REALTIME_URL", format!("ws://{up_addr}/v1/realtime"));
        let fake = tokio::spawn(async move {
            let (stream, _) = up_listener.accept().await.unwrap();
            let mut ws = accept_async(stream).await.unwrap();
            ws.send(TMsg::Text(
                r#"{"type":"session.created","session":{"type":"realtime","model":"fake-stt"}}"#
                    .into(),
            ))
            .await
            .unwrap();
            let mut seen = Vec::new();
            while let Some(Ok(msg)) = ws.next().await {
                match msg {
                    TMsg::Text(t) => {
                        let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                        let kind = v["type"].as_str().unwrap_or_default().to_string();
                        if kind == "input_audio_buffer.commit" {
                            ws.send(TMsg::Text(
                                r#"{"type":"conversation.item.input_audio_transcription.completed","transcript":"hallo"}"#
                                    .into(),
                            ))
                            .await
                            .unwrap();
                        }
                        seen.push(kind);
                    }
                    TMsg::Close(_) => break,
                    _ => {}
                }
            }
            seen
        });

        let (mut client, _) = connect_async(format!("ws://{gw}/v1/realtime?upstream=stt&lang=de"))
            .await
            .unwrap();
        let created = next_text(&mut client).await;
        assert!(created.contains("session.created"), "got {created}");

        client
            .send(TMsg::Text(
                r#"{"type":"session.update","session":{"audio":{"input":{"format":{"type":"audio/pcm","rate":16000}}},"vad":{"enabled":true}}}"#
                    .into(),
            ))
            .await
            .unwrap();
        client.send(TMsg::Text(append_frame(32_000))).await.unwrap();
        client
            .send(TMsg::Text(r#"{"type":"input_audio_buffer.commit"}"#.into()))
            .await
            .unwrap();
        let done = next_text(&mut client).await;
        assert!(
            done.contains("transcription.completed") && done.contains("hallo"),
            "got {done}"
        );
        client.close(None).await.unwrap();

        let seen = tokio::time::timeout(Duration::from_secs(5), fake)
            .await
            .expect("upstream did not see close")
            .unwrap();
        assert_eq!(
            seen,
            vec![
                "session.update",
                "input_audio_buffer.append",
                "input_audio_buffer.commit"
            ]
        );

        // Only the healthy session is metered (the failed upstream recorded nothing).
        let ev = tokio::time::timeout(Duration::from_secs(5), usage_rx.recv())
            .await
            .expect("no usage event recorded")
            .expect("usage channel closed");
        assert_eq!(ev.kind, "realtime");
        assert_eq!(ev.route, "stt");
        assert_eq!(ev.model, "fake-stt");
        assert_eq!(ev.status, 101);
        assert!((ev.audio_seconds - 1.0).abs() < 1e-9, "got {}", ev.audio_seconds);
        assert!(usage_rx.try_recv().is_err(), "unexpected extra usage event");
    }
}
