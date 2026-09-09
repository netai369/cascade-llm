//! Intent-based routing: keyword + system-prompt-mode detection.
//!
//! Planning / architecture / design / security / meta intents → skip small
//! model, route directly to the large inference backend. All detection is
//! stateless (per-request), no sticky decisions.

use crate::language;
use crate::types::{ChatCompletionRequest, MessageContent, MessageContentPart};

/// Moving-average window length for logprob confidence tracking.
const MA_WINDOW: usize = 5;
/// Logprob threshold below which we consider the small model uncertain.
const MA_THRESHOLD: f64 = -1.8;

// ── Intent classification ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    Planning,
    Design,
    Security,
    Meta,
    General,
}

impl Intent {
    pub fn label(self) -> &'static str {
        match self {
            Self::Planning => "planning",
            Self::Design => "design",
            Self::Security => "security",
            Self::Meta => "meta",
            Self::General => "general",
        }
    }
}

// ── Keyword tables ────────────────────────────────────────────────────────────

const PLANNING_KEYWORDS_DE: &[&str] = &[
    "implementierungsplan",
    "meilenstein",
    "roadmap",
    "vorgehensweise",
    "nächste schritte",
    "phasen",
    "aufgaben verteilen",
    "ressourcen planen",
    "zeithorizont",
    "projektplan",
];

const DESIGN_KEYWORDS_DE: &[&str] = &[
    "architektur",
    "design",
    "komponenten",
    "schnittstelle",
    "schnittstellen",
    "modul",
    "schichten",
    "layer",
    "schicht",
    "datenfluss",
    "systementwurf",
    "entwurf",
    "struktur",
];

const SECURITY_KEYWORDS_DE: &[&str] = &[
    "sicherheitslücke",
    "verwundbarkeit",
    "schwachstelle",
    "penetrationstest",
    "angriff",
    "verschlüsselung",
    "authentifizierung",
    "authorisierung",
    "zertifikat",
    "oauth",
];

const META_KEYWORDS_DE: &[&str] = &[
    "erkläre wie",
    "warum funktioniert",
    "ist das korrekt",
    "ist das sicher",
    "bewertung",
    "feedback",
    "alternativen",
    "vergleich",
    "abwägung",
    "pro und contra",
];

const PLANNING_KEYWORDS_EN: &[&str] = &[
    "implementation plan",
    "milestone",
    "roadmap",
    "next steps",
    "action items",
    "phases",
    "timeline",
    "sprint planning",
    "task breakdown",
    "project plan",
    "feature roadmap",
];

const DESIGN_KEYWORDS_EN: &[&str] = &[
    "architecture",
    "design",
    "component",
    "components",
    "interface",
    "interfaces",
    "module",
    "layer",
    "layers",
    "data flow",
    "system design",
    "system design review",
    "structure",
    "blueprint",
];

const SECURITY_KEYWORDS_EN: &[&str] = &[
    "vulnerability",
    "security",
    "penetration test",
    "attack",
    "threat model",
    "encrypt",
    "authentication",
    "authorization",
    "certificate",
    "oauth",
    "exploit",
];

const META_KEYWORDS_EN: &[&str] = &[
    "explain how",
    "why does",
    "is that correct",
    "is that safe",
    "evaluate",
    "assess",
    "feedback",
    "alternatives",
    "trade-off",
    "tradeoffs",
    "pros and cons",
    "compare",
];

// ── Intent detection ──────────────────────────────────────────────────────────

fn classify_by_keywords(lowercase: &str, keywords: &[&str]) -> usize {
    keywords.iter().filter(|kw| lowercase.contains(*kw)).count()
}

fn detect_intent_class(lowercase: &str) -> Intent {
    let lang = if lowercase.contains(" die ") || lowercase.contains(" der ") || lowercase.contains(" das ") {
        "de"
    } else {
        "en"
    };

    let (plan_kw, design_kw, sec_kw, meta_kw) = match lang {
        "de" => (
            PLANNING_KEYWORDS_DE,
            DESIGN_KEYWORDS_DE,
            SECURITY_KEYWORDS_DE,
            META_KEYWORDS_DE,
        ),
        _ => (
            PLANNING_KEYWORDS_EN,
            DESIGN_KEYWORDS_EN,
            SECURITY_KEYWORDS_EN,
            META_KEYWORDS_EN,
        ),
    };

    let plan_score  = classify_by_keywords(lowercase, plan_kw);
    let design_score = classify_by_keywords(lowercase, design_kw);
    let sec_score   = classify_by_keywords(lowercase, sec_kw);
    let meta_score  = classify_by_keywords(lowercase, meta_kw);

    let max = plan_score.max(design_score).max(sec_score).max(meta_score);
    if max == 0 {
        return Intent::General;
    }
    // Tie-break: prefer the highest-scoring; ties resolved in priority order
    if plan_score == max { return Intent::Planning; }
    if design_score == max { return Intent::Design; }
    if sec_score == max { return Intent::Security; }
    Intent::Meta
}

// ── System-prompt mode detection ──────────────────────────────────────────────

fn is_system_prompt_mode(messages: &[crate::types::ChatMessage]) -> bool {
    let Some(system_msg) = messages.iter().find(|m| m.role == "system") else {
        return false;
    };
    let Some(ref content) = system_msg.content else {
        return false;
    };
    let text = system_prompt_text(content);
    // Heuristic: system prompt with role/identity definitions (multi-line or long)
    // suggests a structured planning/design conversation.
    text.contains("you are") && text.len() > 200
        || text.contains("du bist") && text.len() > 200
        || text.contains("act as") && text.len() > 200
        || text.contains("agiere als") && text.len() > 200
        || text.contains("role:") && text.len() > 200
}

fn system_prompt_text(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(t) => t.to_lowercase(),
        MessageContent::Parts(parts) => {
            let mut s = String::new();
            for part in parts {
                if let MessageContentPart::Text { text } = part {
                    s.push_str(&text.to_lowercase());
                    s.push(' ');
                }
            }
            s
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Result of intent inspection for a single request.
#[derive(Debug, Clone)]
pub struct IntentInspection {
    pub intent: Intent,
    /// True when the small model should be bypassed entirely (route to large).
    pub direct_to_large: bool,
}

/// Inspect request intent: keyword scoring + system-prompt heuristics.
/// Stateless — no sticky decisions.
pub fn inspect_intent(payload: &ChatCompletionRequest) -> IntentInspection {
    let mut aggregated = language::extract_text(&payload.messages);
    // `extract_text` skips system messages (they carry role instructions, not
    // dialog) — include their text for intent classification so role-definition
    // system prompts (e.g. "You are an expert architect") are recognised.
    if let Some(sys) = payload.messages.iter().find(|m| m.role == "system") {
        if let Some(ref content) = sys.content {
            aggregated.push(' ');
            aggregated.push_str(&system_prompt_text(content));
        }
    }
    let lowercase = aggregated.to_lowercase();

    let intent = detect_intent_class(&lowercase);
    let system_mode = is_system_prompt_mode(&payload.messages);
    let has_tools = payload.tools.is_some() || payload.functions.is_some();

    // Bypass condition: strong intent signal OR system-prompt mode with tools
    let direct_to_large = matches!(
        intent,
        Intent::Planning | Intent::Design | Intent::Security | Intent::Meta
    ) || (system_mode && has_tools);

    if direct_to_large {
        tracing::info!(
            "INTENT_ROUTING: intent={}, system_mode={}, tools={} → direct_to_large",
            intent.label(),
            system_mode,
            has_tools,
        );
    }

    IntentInspection { intent, direct_to_large }
}

// ── Streaming logprob monitor ─────────────────────────────────────────────────

/// Circular-buffer moving average over the last `MA_WINDOW` token log-probs.
/// Non-sticky: reset per request.
pub struct LogprobMonitor {
    window: Vec<f64>,
    pos: usize,
    count: usize,
}

impl LogprobMonitor {
    pub fn new() -> Self {
        Self {
            window: vec![0.0; MA_WINDOW],
            pos: 0,
            count: 0,
        }
    }

    /// Feed a new logprob value. Returns true when the tiny-sample warning
    /// window has passed AND the MA is below the threshold.
    pub fn push(&mut self, logprob: f64) -> bool {
        self.window[self.pos] = logprob;
        self.pos = (self.pos + 1) % MA_WINDOW;
        self.count += 1;
        self.is_uncertain()
    }

    fn moving_average(&self) -> f64 {
        let n = self.count.min(MA_WINDOW);
        if n == 0 { return 0.0; }
        let sum: f64 = self.window[..n].iter().sum();
        sum / n as f64
    }

    pub fn is_uncertain(&self) -> bool {
        self.count >= 3 && self.moving_average() < MA_THRESHOLD
    }
}

// ── Stream fallback detection ─────────────────────────────────────────────────

pub const FALLBACK_TAG: &str = "<CASCADE_FALLBACK>";

/// Scan a chunk of SSE data for the `<CASCADE_FALLBACK>` tag.
pub fn contains_fallback_tag(chunk: &str) -> bool {
    chunk.contains(FALLBACK_TAG)
}

/// Extract token-level logprobs from an SSE chunk and feed them into the monitor.
/// Returns true when the MA over the last tokens drops below the threshold.
pub fn check_chunk_logprobs(chunk: &str, monitor: &mut LogprobMonitor) -> bool {
    // Fast-path: no logprobs array in this chunk.
    if !chunk.contains("\"logprobs\"") || !chunk.contains("\"content\"") {
        return false;
    }
    // SSE chunks carry a "data: " prefix — strip it before JSON parsing.
    let trimmed = chunk.trim();
    let trimmed = trimmed.strip_prefix("data: ").unwrap_or(trimmed);
    let trimmed = trimmed.strip_prefix("data:").unwrap_or(trimmed).trim();
    let Ok(val): Result<serde_json::Value, _> = serde_json::from_str(trimmed) else {
        return false;
    };
    let Some(choices) = val.get("choices").and_then(|c| c.as_array()) else {
        return false;
    };
    let Some(first) = choices.first() else {
        return false;
    };
    let Some(logprobs) = first.get("logprobs") else {
        return false;
    };
    let Some(content) = logprobs.get("content").and_then(|c| c.as_array()) else {
        return false;
    };
    let mut any_below = false;
    for token in content {
        if let Some(lp) = token.get("logprob").and_then(|v| v.as_f64()) {
            if monitor.push(lp) {
                any_below = true;
            }
        }
    }
    any_below
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_planning_en() {
        let payload = ChatCompletionRequest {
            model: "test".into(),
            messages: vec![crate::types::ChatMessage {
                role: "user".into(),
                content: Some(MessageContent::Text(
                    "Create an implementation plan with milestones and next steps".into(),
                )),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            stream: None,
            temperature: None,
            max_tokens: None,
            logprobs: None,
            top_logprobs: None,
            tools: None,
            tool_choice: None,
            functions: None,
            function_call: None,
            user: None,
            stop: None,
            response_format: None,
            metadata: None,
            route_hint: None,
        };
        let result = inspect_intent(&payload);
        assert_eq!(result.intent, Intent::Planning);
        assert!(result.direct_to_large);
    }

    #[test]
    fn detect_general_greeting() {
        let payload = ChatCompletionRequest {
            model: "test".into(),
            messages: vec![crate::types::ChatMessage {
                role: "user".into(),
                content: Some(MessageContent::Text("Hello world".into())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            stream: None,
            temperature: None,
            max_tokens: None,
            logprobs: None,
            top_logprobs: None,
            tools: None,
            tool_choice: None,
            functions: None,
            function_call: None,
            user: None,
            stop: None,
            response_format: None,
            metadata: None,
            route_hint: None,
        };
        let result = inspect_intent(&payload);
        assert_eq!(result.intent, Intent::General);
        assert!(!result.direct_to_large);
    }

    #[test]
    fn logprob_monitor_average() {
        let mut m = LogprobMonitor::new();
        // 4 tokens above threshold, then 4 below
        assert!(!m.push(-0.5));
        assert!(!m.push(-0.5));
        assert!(!m.push(-0.5));
        assert!(!m.push(-0.5));
        assert!(!m.push(-2.0));  // avg so far: (-0.5*4 + -2.0)/5 = -0.8 — still > -1.8
        assert!(!m.push(-2.0));
        assert!(!m.push(-2.0));
        assert!(!m.push(-2.0));  // avg: (-0.5 + -2.0*4)/5 = -1.6 — still > -1.8
        assert!(m.push(-2.0));   // avg: (-2.0*5)/5 = -2.0 < -1.8 ✓
    }

    #[test]
    fn fallback_tag_detection() {
        assert!(contains_fallback_tag(r#"{"delta":{"content":"<CASCADE_FALLBACK>"}}"#));
        assert!(!contains_fallback_tag(r#"{"delta":{"content":"Hello"}}"#));
    }

    #[test]
    fn logprob_monitor_not_uncertain_when_few_samples() {
        let mut m = LogprobMonitor::new();
        assert!(!m.push(-3.0));
        assert!(!m.push(-3.0));
        assert!(!m.is_uncertain()); // only 2 samples
    }

    #[test]
    fn check_chunk_logprobs_extracts_tokens() {
        let mut m = LogprobMonitor::new();
        // Good tokens: all logprobs close to 0
        let good = r#"data: {"choices":[{"delta":{"content":"Hi"},"logprobs":{"content":[{"token":"Hi","logprob":-0.2},{"token":" there","logprob":-0.4}]}}]}"#;
        assert!(!check_chunk_logprobs(good, &mut m));
        assert!(!m.is_uncertain());

        // Bad run: token logprobs crash below threshold — fill the 5-token window.
        let bad = r#"data: {"choices":[{"delta":{"content":"??"},"logprobs":{"content":[{"token":"?","logprob":-2.2},{"token":"?","logprob":-2.4},{"token":"?","logprob":-2.6},{"token":"?","logprob":-2.8},{"token":"?","logprob":-3.0}]}}]}"#;
        assert!(check_chunk_logprobs(bad, &mut m)); // MA below -1.8
        assert!(m.is_uncertain());
    }

    #[test]
    fn check_chunk_logprobs_ignores_non_logprob_chunks() {
        let mut m = LogprobMonitor::new();
        let plain = r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#;
        assert!(!check_chunk_logprobs(plain, &mut m));
        assert!(!m.is_uncertain());
    }

    #[test]
    fn system_prompt_mode_long() {
        let payload = ChatCompletionRequest {
            model: "test".into(),
            messages: vec![crate::types::ChatMessage {
                role: "system".into(),
                content: Some(MessageContent::Text(
                    "You are an expert architect. You design complex distributed systems. \
                     Your role is to provide detailed blueprints and component diagrams. \
                     You always consider security, scalability, and maintainability. \
                     You respond in a structured format with clear phases and milestones."
                        .into(),
                )),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            stream: None,
            temperature: None,
            max_tokens: None,
            logprobs: None,
            top_logprobs: None,
            tools: None,
            tool_choice: None,
            functions: None,
            function_call: None,
            user: None,
            stop: None,
            response_format: None,
            metadata: None,
            route_hint: None,
        };
        let result = inspect_intent(&payload);
        // Architecture/design intent declared in the system prompt → force large.
        assert!(result.direct_to_large);
        assert!(matches!(result.intent, Intent::Design | Intent::Planning));
    }

    #[test]
    fn system_prompt_mode_with_tools() {
        let payload = ChatCompletionRequest {
            model: "test".into(),
            messages: vec![
                crate::types::ChatMessage {
                    role: "system".into(),
                    content: Some(MessageContent::Text(
                        "You are an expert architect. You design complex distributed systems. \
                         Your role is to provide detailed blueprints and component diagrams. \
                         You always consider security, scalability, and maintainability. \
                         You respond in a structured format with clear phases and milestones."
                            .into(),
                    )),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                crate::types::ChatMessage {
                    role: "user".into(),
                    content: Some(MessageContent::Text("Design the auth flow".into())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
            ],
            stream: None,
            temperature: None,
            max_tokens: None,
            logprobs: None,
            top_logprobs: None,
            tools: Some(serde_json::json!([{"type":"function"}])),
            tool_choice: None,
            functions: None,
            function_call: None,
            user: None,
            stop: None,
            response_format: None,
            metadata: None,
            route_hint: None,
        };
        let result = inspect_intent(&payload);
        // System-prompt mode + tools → direct_to_large
        assert!(result.direct_to_large);
    }
}
