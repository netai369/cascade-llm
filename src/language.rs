
use once_cell::sync::Lazy;
use lingua::{Language, LanguageDetectorBuilder};
use tracing::debug;
use tracing::info;
use crate::types::{ChatCompletionRequest, ChatMessage, MessageContent, MessageContentPart};

/// Built ONCE (building all-language detectors is expensive and was previously
/// done multiple times per request). Restricted to the languages we actually
/// inject prompts for.
static DETECTOR: Lazy<lingua::LanguageDetector> = Lazy::new(|| {
    LanguageDetectorBuilder::from_languages(&[
        Language::English,
        Language::German,
        Language::French,
        Language::Italian,
        Language::Spanish,
        Language::Polish,
        Language::Hungarian,
    ])
    .with_minimum_relative_distance(0.3)
    .build()
});

/// Reported when the language was not identified with confidence.
///
/// The critical property is that this is NOT "en". Callers must treat it as
/// "inject no language instruction" rather than as a fallback language —
/// see `inject_language_prompt`, where a wrong guess is worse than silence.
pub const UNKNOWN_LANG: &str = "";

/// Minimum character count before the statistical detector is trusted.
///
/// Below this, lingua returns `None` for essentially every input, and the old
/// code mapped that `None` to `"en"`. A six-character "Hallo" therefore became
/// an English system prompt injected into a German conversation.
const LINGUA_MIN_CHARS: usize = 20;

/// German function words. Precise enough to trust on short fragments, where
/// the statistical detector is not — this runs first and is what makes
/// "Hallo, was gilt?" resolve to `de` instead of `UNKNOWN_LANG`.
const GERMAN_INDICATORS: [&str; 14] = [
    " ist ", " die ", " der ", " das ", " nicht ", " bist ", " wer ", " was ", " wie ", " ä", " ö",
    " ü", "ß", " ich ",
];

pub fn detect_language(messages: &[ChatMessage]) -> &'static str {
    let text = extract_text(messages);
    let lower = text.to_lowercase();

    // Cheap and precise: a German function word is stronger evidence than a
    // statistical guess, and unlike lingua it works on fragments.
    if GERMAN_INDICATORS.iter().any(|i| lower.contains(i)) {
        debug!("LANGUAGE_DETECTION: detected=German (heuristic)");
        return "de";
    }

    if lower.chars().count() < LINGUA_MIN_CHARS {
        debug!("LANGUAGE_DETECTION: too short to identify confidently");
        return UNKNOWN_LANG;
    }

    let result = DETECTOR.detect_language_of(&text);
    debug!("LANGUAGE_DETECTION: detected={:?}", result);
    match result {
        Some(Language::German) => "de",
        Some(Language::French) => "fr",
        Some(Language::Italian) => "it",
        Some(Language::Spanish) => "es",
        Some(Language::Polish) => "pl",
        Some(Language::Hungarian) => "hu",
        // English included: an identified language is not needed to answer in
        // it. Injecting "always respond in English" at a client that already
        // speaks English only adds a contradicting instruction.
        _ => UNKNOWN_LANG,
    }
}

pub fn extract_text(messages: &[ChatMessage]) -> String {
    let mut text = String::new();
    for msg in messages {
        debug!("EXTRACT_TEXT: role={}", msg.role);
        if msg.role == "system" {
            continue;
        }
        if let Some(content) = &msg.content {
            match content {
                MessageContent::Text(t) => {
                    text.push_str(t);
                    text.push(' ');
                }
                MessageContent::Parts(parts) => {
                    for part in parts {
                        if let MessageContentPart::Text { text: t } = part {
                            text.push_str(t);
                            text.push(' ');
                        }
                    }
                }
            }
        }
    }
    info!(
        "EXTRACT_TEXT: final_aggregated_length={}, text_preview={:?}",
        text.len(),
        text.chars().take(200).collect::<String>()
    );
    text
}

/// Fallback instruction appended to the small model's system prompt.
/// If the small model emits this tag in its response, the stream parser
/// detects it and reroutes to the large model.
pub const FALLBACK_INSTRUCTION: &str = "\n\n[CASCADE ROUTING] If the task is complex, planning, architecture, design, security-sensitive, or meta-analytical and you cannot handle it with confidence, output the tag <CASCADE_FALLBACK> on its own line at the start of your response. This signals the gateway to route to the larger model. Do NOT invent content for tasks beyond your capability — emit the tag instead.";

pub fn get_system_prompt(language: &str) -> &'static str {
    match language {
        "de" => "Antworte immer auf Deutsch. Sei hilfreich und präzise.",
        "fr" => "Réponds toujours en français. Sois utile et concis.",
        "it" => "Rispondi sempre in italiano. Sii utile e conciso.",
        "es" => "Responde siempre en español. Sé útil y conciso.",
        "pl" => "Odpowiadaj zawsze po polsku. Bądź pomocny i zwięzły.",
        "hu" => "Mindig válaszolj magyarul. Légy segítőkész és tömör.",
        "sl" => "Vedno odgovarjaj v slovenščini. Bodi koristen in jedrnat.",
        "hr" => "Uvijek odgovaraj na hrvatskom. Bud koristan i sažet.",
        _ => "Always respond in English. Be helpful and concise.",
    }
}

/// Returns the system prompt for the small model, including the cascade
/// fallback instruction. Used by the Auto routing path when sending to
/// the auxiliary backend.
pub fn get_small_model_system_prompt(language: &str) -> String {
    let base = get_system_prompt(language);
    format!("{}{}", base, FALLBACK_INSTRUCTION)
}

/// Append the cascade fallback instruction to the active system message.
/// The instruction tells the small model to emit `<CASCADE_FALLBACK>` when
/// the task exceeds its capability, which the stream parser then reroutes.
pub fn append_fallback_instruction(payload: &mut ChatCompletionRequest) {
    if let Some(sys_msg) = payload.messages.iter_mut().find(|m| m.role == "system") {
        let suffix = FALLBACK_INSTRUCTION.to_string();
        match &mut sys_msg.content {
            Some(MessageContent::Text(text)) => {
                *text = format!("{}{}", text, suffix);
            }
            Some(MessageContent::Parts(parts)) => {
                parts.push(MessageContentPart::Text { text: suffix });
            }
            None => {
                sys_msg.content = Some(MessageContent::Text(suffix));
            }
        }
    }
}

/// Insert a language instruction — but only when it cannot conflict with the
/// caller, and only when the language is actually known.
///
/// The previous version prepended its own instruction to an existing system
/// prompt (`format!("{}. {}", prefix, text)`), so a LibreChat conversation
/// whose `promptPrefix` says "reply in German" reached the model as
///
/// ```text
/// Always respond in English. Be helpful and concise.
/// You are the NetAI Assistant. You MUST ALWAYS reply in German …
/// ```
///
/// Two contradictory instructions, and the model satisfied both badly:
/// German syntax with English function words and the occasional stray
/// CJK/Cyrillic token. The old code only ever *added* text and never removed a
/// wrong one, so nothing in the logs distinguished "added a helpful hint" from
/// "overrode the caller's instruction".
///
/// Two cases now inject nothing:
///
/// 1. `language` is `UNKNOWN_LANG` — a wrong guess is worse than silence.
/// 2. The caller already sent a system prompt. That prompt is authoritative.
///    LibreChat, Hermes and the voice clients all carry their own language
///    mandate, and it is not the gateway's place to second-guess it.
pub fn inject_language_prompt(
    language: &str,
    mut payload: ChatCompletionRequest,
) -> ChatCompletionRequest {
    if language == UNKNOWN_LANG {
        debug!("LANGUAGE_INJECT: skipped, language unknown");
        return payload;
    }
    if payload.messages.iter().any(|m| m.role == "system") {
        debug!("LANGUAGE_INJECT: skipped, caller supplied a system prompt");
        return payload;
    }

    let prefix = get_system_prompt(language);
    let lang_name = match language {
        "de" => "German",
        "fr" => "French",
        "it" => "Italian",
        "es" => "Spanish",
        "pl" => "Polish",
        "hu" => "Hungarian",
        "sl" => "Slovenian",
        "hr" => "Croatian",
        _ => "English",
    };

    payload.messages.insert(
        0,
        ChatMessage {
            role: "system".to_string(),
            content: Some(MessageContent::Text(prefix.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    );
    info!("Injected {} system prompt into model request", lang_name);
    payload
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(messages: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "test-model",
            "messages": messages,
        }))
        .expect("request must deserialize")
    }

    fn first_system(payload: &ChatCompletionRequest) -> Option<String> {
        payload
            .messages
            .iter()
            .find(|m| m.role == "system")
            .and_then(|m| match m.content.as_ref() {
                Some(MessageContent::Text(t)) => Some(t.clone()),
                _ => None,
            })
    }

    fn user(text: &str) -> serde_json::Value {
        json!([{ "role": "user", "content": text }])
    }

    // --- detect_language -------------------------------------------------

    /// The reported failure: a six-character "Hallo" was classified `en`, and
    /// `inject_language_prompt` then prepended "Always respond in English" to a
    /// German conversation.
    #[test]
    fn short_greeting_is_not_english() {
        assert_eq!(detect_language(&request(user("Hallo")).messages), UNKNOWN_LANG);
        assert_eq!(detect_language(&request(user("Hallo!")).messages), UNKNOWN_LANG);
    }

    /// The heuristic must still win on short fragments — that is what keeps a
    /// brief German question from degrading to "inject nothing".
    #[test]
    fn short_german_fragment_is_detected() {
        assert_eq!(
            detect_language(&request(user("Hallo, was gilt hier?")).messages),
            "de"
        );
    }

    #[test]
    fn long_german_text_is_detected() {
        let text = "Bitte fasse den folgenden Bericht zusammen und nenne die \
                    wesentlichen Punkte der Statik in einem kurzen Absatz.";
        assert_eq!(detect_language(&request(user(text)).messages), "de");
    }

    /// An identified language is not required to answer in it, so English is
    /// no longer reported as a concrete language.
    #[test]
    fn english_text_is_not_forced_to_english_prompt() {
        let text = "Please summarize the attached report and list the three \
                    most important structural findings for the client.";
        assert_eq!(detect_language(&request(user(text)).messages), UNKNOWN_LANG);
    }

    // --- inject_language_prompt -----------------------------------------

    /// The regression that produced mixed-language answers. The caller's own
    /// German mandate must reach the model untouched — no English prefix, and
    /// no other text added anywhere.
    #[test]
    fn existing_system_prompt_is_never_modified() {
        let german_mandate = "You are the NetAI Assistant. You MUST ALWAYS reply in German.";
        let payload = request(json!([
            { "role": "system", "content": german_mandate },
            { "role": "user", "content": "Hallo" },
        ]));

        let out = inject_language_prompt("en", payload);

        assert_eq!(first_system(&out).as_deref(), Some(german_mandate));
        assert_eq!(out.messages.len(), 2, "no message may be inserted");
        assert!(
            !out.messages.iter().any(|m| matches!(&m.content,
                Some(MessageContent::Text(t)) if t.contains("Always respond in English"))),
            "the English instruction must not appear anywhere in the request"
        );
    }

    /// A German system prompt stays German even when detection is confident.
    #[test]
    fn confident_detection_does_not_override_caller_prompt() {
        let payload = request(json!([
            { "role": "system", "content": "Antworte immer auf Deutsch." },
            { "role": "user", "content": "Bitte fasse den Bericht zusammen." },
        ]));
        let out = inject_language_prompt("de", payload);
        assert_eq!(first_system(&out).as_deref(), Some("Antworte immer auf Deutsch."));
    }

    /// Without a system prompt there is nothing to conflict with, so a
    /// confidently detected language is still injected.
    #[test]
    fn injects_when_no_system_prompt_exists() {
        let payload = request(json!([
            { "role": "user", "content": "Bitte fasse den Bericht zusammen." },
        ]));
        let out = inject_language_prompt("de", payload);
        assert_eq!(out.messages[0].role, "system");
        assert!(first_system(&out).unwrap().contains("Deutsch"));
    }

    #[test]
    fn unknown_language_injects_nothing() {
        let payload = request(user("Hallo"));
        let out = inject_language_prompt(UNKNOWN_LANG, payload);
        assert_eq!(out.messages.len(), 1);
        assert!(first_system(&out).is_none());
    }

    /// Multi-part content (attachments) must also be left alone.
    #[test]
    fn multipart_system_prompt_is_preserved() {
        let payload = request(json!([
            { "role": "system", "content": [{ "type": "text", "text": "Antworte immer auf Deutsch." }] },
            { "role": "user", "content": "Hallo" },
        ]));
        let out = inject_language_prompt("en", payload);
        assert_eq!(out.messages.len(), 2);
        match out.messages[0].content.as_ref() {
            Some(MessageContent::Parts(parts)) => {
                assert_eq!(parts.len(), 1, "no part may be prepended");
            }
            other => panic!("expected parts, got {other:?}"),
        }
    }
}
