//! Audio → text bridge for chat completions.
//!
//! **Why this exists.** A chat payload may carry an `input_audio` part. The
//! models behind this gateway are TEXT-ONLY: the main candidate (LFM2.5-8B-A1B)
//! and the aux (Gemma-4-E2B int4) are exported without an audio tower, and
//! OVMS serves them as a plain text servable. Forwarding the audio part
//! therefore either
//!   * hits the main model and is rejected with "This servable supports only
//!     text input, but image_url has been provided" (400), or
//!   * reaches the aux model, where the part is silently DROPPED and the model
//!     answers that it cannot hear anything.
//!
//! Neither is a usable answer. The stack already HAS a speech-to-text service
//! (Parakeet, exposed by this same gateway at `/v1/audio/transcriptions`), so
//! the gateway can do what a chat client cannot: transcribe the audio first
//! and hand the MODEL a text turn.
//!
//! The bridge is deliberately lossy-but-honest: a failed transcription becomes
//! an explicit "[Audio konnte nicht transkribiert werden]" marker rather than
//! a silent drop, so the model can say so instead of inventing content.

use base64::Engine as _;
use serde_json::Value;

use crate::types::{ChatCompletionRequest, MessageContent, MessageContentPart};

/// Label the transcript is inserted under. The model sees the turn as a
/// transcription, not as something it heard itself.
const TRANSCRIPT_PREFIX: &str = "[Audio-Transkript]";

/// Replace every `input_audio` part with the STT transcript of that clip.
///
/// Best effort: a part whose transcription fails becomes an explicit marker
/// part. The function never fails the request — an untranscribable clip is a
/// degraded answer, not a broken one.
pub async fn audio_parts_to_text(
    client: &reqwest::Client,
    stt_base: &str,
    api_key: Option<&str>,
    payload: &mut ChatCompletionRequest,
) {
    let stt_url = crate::audio::audio_target(stt_base, "/v1/audio/transcriptions");
    let mut touched = false;

    for message in payload.messages.iter_mut() {
        let Some(MessageContent::Parts(parts)) = &mut message.content else {
            continue;
        };
        for part in parts.iter_mut() {
            let MessageContentPart::InputAudio { input_audio } = part else {
                continue;
            };
            let data = input_audio.data.clone();
            let format = input_audio
                .format
                .clone()
                .unwrap_or_else(|| "wav".to_string());
            touched = true;

            let transcript = transcribe_one(client, &stt_url, api_key, &data, &format).await;
            let text = match transcript {
                Ok(t) if !t.trim().is_empty() => format!("{TRANSCRIPT_PREFIX} {}", t.trim()),
                Ok(_) => format!(
                    "{TRANSCRIPT_PREFIX} (leer — die Aufnahme enthielt keine erkennbare Sprache)"
                ),
                Err(e) => {
                    tracing::warn!("audio bridge: transcription failed: {e}");
                    format!("{TRANSCRIPT_PREFIX} (Transkription fehlgeschlagen: {e})")
                }
            };
            *part = MessageContentPart::Text { text };
        }
    }

    if touched {
        tracing::info!("audio bridge: input_audio parts replaced by STT transcripts");
    }
}

/// POST one clip to the OpenAI-compatible transcription endpoint.
async fn transcribe_one(
    client: &reqwest::Client,
    url: &str,
    api_key: Option<&str>,
    data_b64: &str,
    format: &str,
) -> Result<String, String> {
    use base64::engine::general_purpose::STANDARD;

    let bytes = STANDARD
        .decode(data_b64.trim())
        .map_err(|e| format!("base64-decode: {e}"))?;

    // The STT service serves one model; the field is required by the
    // OpenAI-compatible contract but the value is ignored by Parakeet.
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name(format!("audio.{format}"))
        .mime_str(mime_for(format))
        .map_err(|e| format!("mime: {e}"))?;
    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", "parakeet-tdt-0.6b")
        .text("response_format", "json");

    let mut req = client.post(url).multipart(form);
    if let Some(k) = api_key.filter(|k| !k.trim().is_empty()) {
        req = req.bearer_auth(k);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("stt request: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!(
            "stt HTTP {status}: {}",
            body.chars().take(120).collect::<String>()
        ));
    }
    let v: Value = resp
        .json()
        .await
        .map_err(|e| format!("stt json: {e}"))?;
    Ok(v.get("text")
        .and_then(|t| t.as_str())
        .unwrap_or_default()
        .to_string())
}

fn mime_for(format: &str) -> &'static str {
    match format.to_ascii_lowercase().as_str() {
        "mp3" | "mpeg" => "audio/mpeg",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        "m4a" => "audio/mp4",
        "webm" => "audio/webm",
        _ => "audio/wav",
    }
}
