use crate::{ChatMessage, ChatCompletionRequest, MessageContent, MessageContentPart};
use lingua::{Language, LanguageDetectorBuilder};
use tracing::info;

pub fn detect_language(messages: &[ChatMessage]) -> &'static str {
    let text = extract_text(messages);
    if text.len() < 5 { return "en"; }

    let german_indicators = [" ist ", " die ", " der ", " das ", " nicht ", " bist ", " wer ", " was ", " wie ", " ä", " ö", " ü", "ß"];
    let lower = text.to_lowercase();
    if german_indicators.iter().any(|i| lower.contains(i)) {
        info!("LANGUAGE_DETECTION: text={}, detected=German (heuristic)", text);
        return "de";
    }

    let detector = LanguageDetectorBuilder::from_all_languages()
        .with_minimum_relative_distance(0.3)
        .build();

    let result = detector.detect_language_of(&text);
    info!("LANGUAGE_DETECTION: text={}, detected={:?}", text, result);
    match result {
        Some(Language::German) => "de",
        Some(Language::French) => "fr",
        Some(Language::Italian) => "it",
        Some(Language::Spanish) => "es",
        Some(Language::Polish) => "pl",
        Some(Language::Hungarian) => "hu",
        _ => "en",
    }
}

pub fn extract_text(messages: &[ChatMessage]) -> String {
    let mut text = String::new();
    for msg in messages {
        info!("EXTRACT_TEXT: role={}, skipping={}", msg.role, msg.role == "system");
        if msg.role == "system" { continue; }
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
    info!("EXTRACT_TEXT: final_aggregated_length={}, text_preview={:?}", text.len(), text.chars().take(200).collect::<String>());
    text
}

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

pub fn get_image_prompt(language: &str) -> &'static str {
    match language {
        "de" => "Beschreibe dieses Bild im Detail. Konzentriere dich auf Objekte, Texte, Personen und alles Auffällige. Sei präzise aber kurz.",
        "fr" => "Décris cette image en détail. Concentre-toi sur les objets, le texte, les personnes et tout ce qui est remarquable. Sois précis mais concis.",
        "it" => "Descrivi questa immagine in dettaglio. Concentrati su oggetti, testo, persone e tutto ciò che è notevole. Sii preciso ma conciso.",
        "es" => "Describe esta imagen en detalle. Céntrate en objetos, texto, personas y todo lo notable. Sé preciso pero conciso.",
        "pl" => "Opisz ten obraz szczegółowo. Skup się na obiektach, tekście, osobach i wszystkim co jest godne uwagi. Bądź precyzyjny ale zwięzły.",
        "hu" => "Írd le részletesen ezt a képet. Koncentrálj az objektumokra, szövegre, emberekre és minden figyelemre méltó dologra. Legy pontos, de tömör.",
        "sl" => "Podrobno opišite to sliko. Osredotočite se na predmete, besedilo, ljudi in vse opazne. Bodite natančni, a jedrnati.",
        "hr" => "Detaljno opišite ovu sliku. Usredotočite se na objekte, tekst, ljude i sve što je primjetno. Budite precizni, ali sažeti.",
        _ => "Describe this image in detail. Focus on objects, text, people, and anything notable. Keep it concise but thorough.",
    }
}

pub fn inject_language_prompt(language: &str, mut payload: ChatCompletionRequest) -> ChatCompletionRequest {
    let has_system = payload.messages.iter().any(|m| m.role == "system");
    let prefix = get_system_prompt(language);
    let lang_name = match language {
        "de" => "German", "fr" => "French", "it" => "Italian", "es" => "Spanish",
        "pl" => "Polish", "hu" => "Hungarian", "sl" => "Slovenian", "hr" => "Croatian",
        _ => "English",
    };
    
    if !has_system {
        payload.messages.insert(0, ChatMessage {
            role: "system".to_string(),
            content: Some(MessageContent::Text(prefix.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
        info!("Injected {} system prompt into model request", lang_name);
    } else {
        if let Some(sys_msg) = payload.messages.iter_mut().find(|m| m.role == "system") {
            if let Some(ref mut content) = sys_msg.content {
                match content {
                    MessageContent::Text(text) => {
                        *text = format!("{}. {}", prefix, text);
                    }
                    MessageContent::Parts(parts) => {
                        parts.insert(0, MessageContentPart::Text { 
                            text: format!("{}. ", prefix) 
                        });
                    }
                }
            } else {
                sys_msg.content = Some(MessageContent::Text(prefix.to_string()));
            }
            info!("Modified existing system prompt to lean {}", lang_name);
        }
    }
    info!("INJECT_LANGUAGE: final system prompt length={}", payload.messages.get(0).map(|m| match &m.content { Some(MessageContent::Text(t)) => t.len(), Some(MessageContent::Parts(p)) => p.iter().map(|x| match x { MessageContentPart::Text { text } => text.len(), _ => 0 }).sum(), _ => 0 }).unwrap_or(0));
    payload
}
