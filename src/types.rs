use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// EXTRACTION BACKEND REGISTRY
// ============================================================================

/// Type of extraction backend for `/v1/extraction` routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionBackendType {
    /// Cloud GPU instance (Vast.ai, RunPod, etc.)
    CloudGpu,
    /// Cloud LLM API (OpenRouter, Gemini, etc.)
    CloudLlm,
    /// Local auxiliary server (fallback)
    Local,
}

/// How `/v1/extraction` selects among healthy backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExtractionStrategy {
    /// Strict priority order (lower `priority` wins); the local auxiliary
    /// server is a last resort. Preserves the historic behaviour.
    #[default]
    Priority,
    /// Weighted round-robin across ALL healthy backends, so a local auxiliary
    /// server and a GPU-provisioned node are used in parallel.
    RoundRobin,
}

impl ExtractionStrategy {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "priority" | "strict" => Some(Self::Priority),
            "round_robin" | "roundrobin" | "weighted" | "rr" => Some(Self::RoundRobin),
            _ => None,
        }
    }
}

/// An extraction backend registered with cascade-LLM.
/// Selection depends on [`ExtractionStrategy`]: strict priority (lower wins)
/// or weighted round-robin across all healthy backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionBackendEntry {
    pub id: String,
    pub backend_type: ExtractionBackendType,
    pub name: String,
    /// Base URL (must accept `/v1/chat/completions` appended)
    pub url: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Selection priority — lower wins in `priority` mode.
    /// CloudGpu=10, CloudLlm=20, Local=99.
    #[serde(default = "default_priority")]
    pub priority: u32,
    /// Relative share of traffic in `round_robin` mode (default 1). A node
    /// with weight 2 is picked twice as often as one with weight 1.
    #[serde(default = "default_weight")]
    pub weight: u32,
    /// Max cost per hour in USD (cloud backends only; None = unlimited).
    #[serde(default)]
    pub max_cost_per_hour: Option<f64>,
    /// ISO-8601 timestamp of last validation.
    #[serde(default)]
    pub last_validated: Option<String>,
    /// Whether the backend is currently healthy.
    #[serde(default = "default_true")]
    pub healthy: bool,
}

impl ExtractionBackendEntry {
    /// Returns the full chat completions URL for this backend.
    pub fn completions_url(&self) -> String {
        format!("{}/v1/chat/completions", self.url.trim_end_matches('/'))
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrlTarget },
    InputAudio { input_audio: InputAudioData },
    /// File attachment (PDF, image, office docs). Kept tolerant to any shape so
    /// requests with uploads never fail deserialization. Inspected by the router
    /// to detect vision / document-parsing payloads.
    File { #[serde(default)] file: Value },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct InputAudioData {
    pub data: String,
    #[serde(default)]
    pub format: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ImageUrlTarget {
    pub url: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<MessageContentPart>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    /// Arbitrary request metadata (route hints, document/OCR flags). Parsed by the
    /// multi-endpoint router to override payload inspection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// Convenience top-level route hint, e.g. `"ocr"`, `"auxiliary"`, `"inference"`.
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "route")]
    pub route_hint: Option<String>,
    /// OpenAI `stream_options` (e.g. `{"include_usage": true}`). Kept verbatim so
    /// the client's opt-in is visible to the stream rewriter; for streamed
    /// requests the forwarder overrides it upstream (see `state.rs`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<Value>,
}

impl ChatCompletionRequest {
    /// True when the client asked for the terminal usage chunk
    /// (`stream_options.include_usage == true`).
    pub fn client_wants_stream_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .and_then(|o| o.get("include_usage"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod chat_request_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stream_options_round_trips() {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        }))
        .unwrap();
        assert!(req.client_wants_stream_usage());
        let out = serde_json::to_value(&req).unwrap();
        assert_eq!(out["stream_options"], json!({"include_usage": true}));
    }

    #[test]
    fn absent_stream_options_is_not_serialized() {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [],
        }))
        .unwrap();
        assert!(!req.client_wants_stream_usage());
        let out = serde_json::to_value(&req).unwrap();
        assert!(out.get("stream_options").is_none());
    }

    #[test]
    fn explicit_false_include_usage_is_not_opt_in() {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [],
            "stream_options": {"include_usage": false}
        }))
        .unwrap();
        assert!(!req.client_wants_stream_usage());
    }
}

/// Payload inspection result: which of the three specialized backends a request
/// should be forwarded to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteDecision {
    /// Vision / document parsing -> ocr-server (PaddleOCR-VL)
    Ocr,
    /// Agent context compression / summarization -> auxiliary-server (LFM2.5-2.6B)
    Auxiliary,
    /// Default chat & reasoning -> inference-server (main LLM)
    Inference,
    /// Legacy complexity-based routing (explicit `auto` mode only)
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteBackend {
    OcrServer,
    AuxiliaryServer,
    InferenceServer,
}

/// Structured proxy failure carrying enough context for a clean client-facing
/// JSON error (e.g. structured 503 when the OCR endpoint is unreachable).
#[derive(Debug, Clone)]
pub struct ProxyError {
    pub status: u16,
    pub backend: String,
    pub context: String,
    /// True when the failure is a transport/connectivity issue (endpoint down),
    /// rather than an HTTP error returned by the backend itself.
    pub unreachable: bool,
    /// Actionable hint for a disconnected native engine (e.g. start vmlx on
    /// the Mac host); omitted for ordinary HTTP errors and container services.
    pub service_action: Option<String>,
}

impl ProxyError {
    pub fn new(
        status: impl Into<u16>,
        backend: impl Into<String>,
        context: impl Into<String>,
    ) -> Self {
        Self {
            status: status.into(),
            backend: backend.into(),
            context: context.into(),
            unreachable: false,
            service_action: None,
        }
    }

    pub fn unreachable(
        status: impl Into<u16>,
        backend: impl Into<String>,
        context: impl Into<String>,
    ) -> Self {
        Self {
            status: status.into(),
            backend: backend.into(),
            context: context.into(),
            unreachable: true,
            service_action: None,
        }
    }

    pub fn with_service_action(mut self, action: Option<String>) -> Self {
        self.service_action = action;
        self
    }
}

impl From<axum::http::StatusCode> for ProxyError {
    fn from(s: axum::http::StatusCode) -> Self {
        Self {
            status: s.as_u16(),
            backend: "backend".to_string(),
            context: format!("backend request failed with HTTP {}", s.as_u16()),
            unreachable: false,
            service_action: None,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ModelPermission {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: String,
    pub group: Option<String>,
    pub is_blocking: bool,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub permission: Vec<ModelPermission>,
    pub root: String,
    pub parent: Option<String>,
    pub description: String,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// What a model can actually do, as opposed to where it was discovered.
///
/// `/v1/models` is a DISCOVERY surface: clients use it to decide whether they
/// may send images, tools or streaming requests. Before this existed every
/// entry looked like `text, streaming`, so capable clients either refused
/// `netai-hybrid` or sent requests the selected backend could not serve.
/// `chat` is the load-bearing flag: entries with `chat == false` are utility
/// services (embeddings, rerank, OCR), not chat models.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    #[serde(default)]
    pub chat: bool,
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub tools: bool,
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub thinking: bool,
    #[serde(default)]
    pub embeddings: bool,
    #[serde(default)]
    pub rerank: bool,
    #[serde(default)]
    pub ocr: bool,
    #[serde(default)]
    pub tts: bool,
    #[serde(default)]
    pub stt: bool,
    #[serde(default)]
    pub diarization: bool,
    #[serde(default)]
    pub image: bool,
}

/// Partial override for `CASCADE_MODEL_CAPABILITIES`. Every field is optional
/// so an entry only names the flags it changes; the rest keep the default
/// derived from the model id and the upstream role.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelCapabilitiesOverride {
    #[serde(default)]
    pub chat: Option<bool>,
    #[serde(default)]
    pub streaming: Option<bool>,
    #[serde(default)]
    pub tools: Option<bool>,
    #[serde(default)]
    pub vision: Option<bool>,
    #[serde(default)]
    pub thinking: Option<bool>,
    #[serde(default)]
    pub embeddings: Option<bool>,
    #[serde(default)]
    pub rerank: Option<bool>,
    #[serde(default)]
    pub ocr: Option<bool>,
    #[serde(default)]
    pub tts: Option<bool>,
    #[serde(default)]
    pub stt: Option<bool>,
    #[serde(default)]
    pub diarization: Option<bool>,
    #[serde(default)]
    pub image: Option<bool>,
}

fn set_capability(caps: &mut ModelCapabilities, token: &str) {
    match token.trim().to_ascii_lowercase().as_str() {
        "chat" | "text" => caps.chat = true,
        "streaming" | "stream" => caps.streaming = true,
        "tools" | "tool_use" | "tool-calling" | "function_calling" => caps.tools = true,
        "vision" | "image" | "multimodal" => caps.vision = true,
        "thinking" | "reasoning" => caps.thinking = true,
        "embeddings" | "embedding" => caps.embeddings = true,
        "rerank" | "reranking" => caps.rerank = true,
        "ocr" | "document" => caps.ocr = true,
        "tts" | "speech" | "text-to-speech" => caps.tts = true,
        "stt" | "transcription" | "speech-to-text" => caps.stt = true,
        "diarization" | "diarisation" | "speaker-diarization" => caps.diarization = true,
        "image_generation" | "image-generation" => caps.image = true,
        _ => {}
    }
}

impl ModelCapabilities {
    /// Parse a comma list such as `chat,streaming,tools,vision`.
    pub fn from_csv(csv: &str) -> Self {
        let mut caps = Self::default();
        for token in csv.split(',').filter(|t| !t.trim().is_empty()) {
            set_capability(&mut caps, token);
        }
        caps
    }

    pub fn overlay(&mut self, o: &ModelCapabilitiesOverride) {
        if let Some(v) = o.chat { self.chat = v; }
        if let Some(v) = o.streaming { self.streaming = v; }
        if let Some(v) = o.tools { self.tools = v; }
        if let Some(v) = o.vision { self.vision = v; }
        if let Some(v) = o.thinking { self.thinking = v; }
        if let Some(v) = o.embeddings { self.embeddings = v; }
        if let Some(v) = o.rerank { self.rerank = v; }
        if let Some(v) = o.ocr { self.ocr = v; }
        if let Some(v) = o.tts { self.tts = v; }
        if let Some(v) = o.stt { self.stt = v; }
        if let Some(v) = o.diarization { self.diarization = v; }
        if let Some(v) = o.image { self.image = v; }
    }

    fn model_class(&self) -> &'static str {
        if self.tts {
            "tts"
        } else if self.stt {
            "stt"
        } else if self.image {
            "image"
        } else if self.ocr {
            "ocr"
        } else if self.rerank {
            "rerank"
        } else if self.embeddings {
            "embeddings"
        } else if self.chat && self.vision {
            "vlm"
        } else if self.chat {
            "llm"
        } else {
            "model"
        }
    }

    /// OpenRouter-shaped modality metadata. Clients that predate OpenRouter
    /// ignore it; clients that read it no longer have to guess from the id.
    fn architecture(&self) -> Value {
        if self.tts {
            serde_json::json!({"input_modalities": ["text"], "output_modalities": ["audio"]})
        } else if self.stt {
            serde_json::json!({"input_modalities": ["audio"], "output_modalities": ["text"]})
        } else if self.image {
            serde_json::json!({"input_modalities": ["text"], "output_modalities": ["image"]})
        } else if self.ocr {
            serde_json::json!({"input_modalities": ["image"], "output_modalities": ["text"]})
        } else if self.rerank {
            serde_json::json!({"input_modalities": ["text"], "output_modalities": ["scores"]})
        } else if self.embeddings {
            serde_json::json!({"input_modalities": ["text"], "output_modalities": ["embeddings"]})
        } else if self.vision {
            serde_json::json!({"input_modalities": ["text", "image"], "output_modalities": ["text"]})
        } else {
            serde_json::json!({"input_modalities": ["text"], "output_modalities": ["text"]})
        }
    }

    fn supported_parameters(&self) -> Vec<&'static str> {
        if self.tts {
            return vec!["model", "voice", "response_format", "input"];
        }
        if self.stt {
            return vec!["model", "language", "response_format", "diarize", "words"];
        }
        if self.image {
            return vec!["model", "prompt", "size", "n", "response_format"];
        }
        if !self.chat {
            return Vec::new();
        }
        let mut params = vec!["temperature", "top_p", "max_tokens", "stream", "response_format"];
        if self.tools {
            params.push("tools");
            params.push("tool_choice");
        }
        params
    }

    pub fn as_json(&self) -> Value {
        serde_json::json!({
            "chat": self.chat,
            "streaming": self.streaming,
            "tools": self.tools,
            "vision": self.vision,
            "thinking": self.thinking,
            "embeddings": self.embeddings,
            "rerank": self.rerank,
            "ocr": self.ocr,
            "tts": self.tts,
            "stt": self.stt,
            "diarization": self.diarization,
            "image": self.image,
        })
    }
}

/// Capability defaults + per-model overrides, built once at startup.
#[derive(Debug, Clone)]
pub struct CapabilityPolicy {
    pub main: ModelCapabilities,
    pub aux: ModelCapabilities,
    pub overrides: std::collections::HashMap<String, ModelCapabilitiesOverride>,
    /// Operator-supplied ids to keep out of discovery (case-insensitive), e.g.
    /// a stale upstream model that must not reach a client picker.
    pub deny: Vec<String>,
}

impl Default for CapabilityPolicy {
    fn default() -> Self {
        Self {
            main: ModelCapabilities::from_csv("chat,streaming,tools,vision"),
            aux: ModelCapabilities::from_csv("chat,streaming"),
            overrides: std::collections::HashMap::new(),
            deny: Vec::new(),
        }
    }
}

impl CapabilityPolicy {
    /// Read the capability policy from the environment once at startup.
    ///
    /// `CASCADE_MAIN_CAPABILITIES` / `CASCADE_AUX_CAPABILITIES` are comma lists
    /// (defaults match the shipped OVMS/vLLM main: text+vision+tools, and the
    /// text-only aux). `CASCADE_MODEL_CAPABILITIES` is a JSON object of partial
    /// per-id overrides, e.g. `{"netai-hybrid":{"vision":false}}`.
    pub fn from_env() -> Self {
        fn env_csv(name: &str, default: &str) -> String {
            std::env::var(name)
                .ok()
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| default.to_string())
        }
        let main = ModelCapabilities::from_csv(&env_csv(
            "CASCADE_MAIN_CAPABILITIES",
            "chat,streaming,tools,vision",
        ));
        let aux = ModelCapabilities::from_csv(&env_csv(
            "CASCADE_AUX_CAPABILITIES",
            "chat,streaming",
        ));
        let overrides = std::env::var("CASCADE_MODEL_CAPABILITIES")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .and_then(|raw| {
                serde_json::from_str::<
                    std::collections::HashMap<String, ModelCapabilitiesOverride>,
                >(&raw)
                .ok()
            })
            .unwrap_or_default();
        let deny = std::env::var("CASCADE_MODELS_DENY")
            .ok()
            .map(|raw| {
                raw.split(',')
                    .map(str::trim)
                    .filter(|id| !id.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        Self {
            main,
            aux,
            overrides,
            deny,
        }
    }

    /// Derive the capability claim from the id and the role the model was
    /// discovered under. Utility roles win over the origin because OVMS serves
    /// chat and embedding graphs on the same origin.
    pub fn for_model(&self, id: &str, role: &str) -> ModelCapabilities {
        let lower = id.to_ascii_lowercase();
        let mut caps = if role.eq_ignore_ascii_case("Speech")
            || role.eq_ignore_ascii_case("TTS")
            || lower.contains("-tts")
            || lower.contains("pocket-tts")
        {
            ModelCapabilities {
                tts: true,
                streaming: true,
                ..Default::default()
            }
        } else if role.eq_ignore_ascii_case("Transcription")
            || role.eq_ignore_ascii_case("STT")
            || lower.contains("parakeet")
            || lower.contains("-stt")
        {
            ModelCapabilities {
                stt: true,
                diarization: true,
                ..Default::default()
            }
        } else if role.eq_ignore_ascii_case("Image")
            || lower.contains("z-image")
            || lower.contains("zimage")
        {
            ModelCapabilities { image: true, ..Default::default() }
        } else if role.eq_ignore_ascii_case("OCR")
            || lower.contains("docling")
            || lower.contains("ocr")
        {
            ModelCapabilities { ocr: true, ..Default::default() }
        } else if lower.contains("rerank") {
            ModelCapabilities { rerank: true, ..Default::default() }
        } else if lower.contains("embed") || (lower.contains("bge") && !lower.contains("rerank")) {
            ModelCapabilities { embeddings: true, ..Default::default() }
        } else if role.eq_ignore_ascii_case("Auxiliary") || lower.contains("aux") {
            self.aux.clone()
        } else {
            self.main.clone()
        };

        // Everything that is not a utility service is a chat model. This is
        // deliberately separate from the role defaults above: a deployment may
        // configure an empty capability list, and the gateway identity must
        // never disappear from the chat list because of that.
        if !caps.embeddings
            && !caps.rerank
            && !caps.ocr
            && !caps.tts
            && !caps.stt
            && !caps.image
        {
            caps.chat = true;
            caps.streaming = true;
            if lower.ends_with("-think") || lower.contains("-think-") {
                caps.thinking = true;
            }
        }

        if let Some(o) = self.overrides.get(id) {
            caps.overlay(o);
        }
        caps
    }
}

/// Add the discovery metadata to one model object. Upstream-supplied fields
/// (`capabilities` array/object, `type`) are honored first: they are closer to
/// the backend than the gateway's name heuristics.
pub fn apply_model_capabilities(
    value: &mut Value,
    id: &str,
    role: &str,
    policy: &CapabilityPolicy,
) {
    let Some(obj) = value.as_object_mut() else { return };
    let mut caps = policy.for_model(id, role);

    if let Some(upstream) = obj.get("capabilities").cloned() {
        if let Some(map) = upstream.as_object() {
            for (token, enabled) in map {
                if enabled.as_bool() == Some(true) {
                    set_capability(&mut caps, token);
                }
            }
        } else if let Some(list) = upstream.as_array() {
            for token in list.iter().filter_map(Value::as_str) {
                set_capability(&mut caps, token);
            }
        }
    }
    if let Some(class) = obj.get("type").and_then(Value::as_str) {
        match class.to_ascii_lowercase().as_str() {
            "embeddings" | "embedding" => {
                caps = ModelCapabilities { embeddings: true, ..Default::default() };
            }
            "rerank" => {
                caps = ModelCapabilities { rerank: true, ..Default::default() };
            }
            "ocr" => {
                caps = ModelCapabilities { ocr: true, ..Default::default() };
            }
            "tts" => {
                caps = ModelCapabilities {
                    tts: true,
                    streaming: true,
                    ..Default::default()
                };
            }
            "stt" => {
                caps = ModelCapabilities {
                    stt: true,
                    diarization: true,
                    ..Default::default()
                };
            }
            "image" => {
                caps = ModelCapabilities { image: true, ..Default::default() };
            }
            "vlm" => {
                caps.chat = true;
                caps.streaming = true;
                caps.vision = true;
            }
            _ => {}
        }
    }

    obj.insert("capabilities".to_string(), caps.as_json());
    obj.insert("type".to_string(), serde_json::json!(caps.model_class()));
    obj.insert("architecture".to_string(), caps.architecture());
    obj.insert(
        "supported_parameters".to_string(),
        serde_json::json!(caps.supported_parameters()),
    );
}

pub fn build_model_info(id: &str) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        object: "model".to_string(),
        created: 1740000000,
        owned_by: "netai-stack".to_string(),
        permission: vec![ModelPermission {
            id: format!("modelperm-{}", id),
            object: "model_permission".to_string(),
            created: 1740000000,
            allow_create_engine: false,
            allow_sampling: true,
            allow_logprobs: true,
            allow_search_indices: false,
            allow_view: true,
            allow_fine_tuning: false,
            organization: "*".to_string(),
            group: None,
            is_blocking: false,
        }],
        root: id.to_string(),
        parent: None,
        description: match id {
            "cascade-hybrid-v1" => "Unified Cascade model identity. Requests are classified and routed to the configured main or auxiliary inference node; this is not a separate model file.".to_string(),
            "docling-serve" => "Document processing/OCR service advertised by the configured extraction backend. It is used for document ingestion, not ordinary chat completion.".to_string(),
            _ => "Model advertised by the Cascade gateway.".to_string(),
        },
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: String,
    pub name: String,
    pub provider_type: ProviderType,
    pub base_url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_priority")]
    pub priority: u32,
    #[serde(default)]
    pub capabilities: ProviderCapabilities,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_as: Option<String>,
}

fn default_true() -> bool { true }
fn default_priority() -> u32 { 10 }
fn default_weight() -> u32 { 1 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderType {
    Local,
    OpenAi,
    Anthropic,
    Gemini,
    Ollama,
    Custom,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    #[serde(default)]
    pub chat: bool,
    #[serde(default)]
    pub multimodal: bool,
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub tools: bool,
    #[serde(default)]
    pub audio: bool,
    #[serde(default)]
    pub images: bool,
    #[serde(default)]
    pub video: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Persisted settings intentionally store only `{routing, audio}` — the
    /// provider list is written by the tested provider endpoint, never by the
    /// settings page (its copy is REDACTED: api_key: null). Without
    /// `#[serde(default)]` the boot-time restore of that partial blob fails
    /// with "missing field `providers`" and EVERY persisted setting is
    /// silently dropped — the DB then overrides nothing and the env wins.
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    pub routing: RoutingSettings,
    pub audio: AudioSettings,
    pub main_model_name: Option<String>,
    pub small_model_name: Option<String>,
    pub small_mllm_url: Option<String>,
    pub large_mllm_url: Option<String>,
    pub large_text_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocr_server_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extract_cloud_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extract_cloud_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extract_fallback_url: Option<String>,
    /// Effective node-selection strategy across a role pool. Read-only here: it
    /// lives in the registry (not the persisted settings), and changing it goes
    /// through `PUT /web/api/routing-strategy`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_strategy: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingSettings {
    pub router_threshold: f64,
    pub confidence_threshold: f64,
    pub route_tools_to_large: bool,
    /// Deterministic short-circuit applied before content detection.
    #[serde(default)]
    pub default_route: Option<String>,
    /// Hybrid complexity router for the default ("auto") route: a short, plain,
    /// text-only, tool-less FIRST turn goes to the auxiliary (fast) model
    /// instead of always using main. Off by default.
    ///
    /// `Option` on purpose. The persisted settings blob is restored OVER the
    /// env defaults at boot, so a plain `bool` with `#[serde(default)]` would
    /// let a settings file written before this field existed silently reset
    /// `CASCADE_HYBRID_AUTO` back to off — the env value would look like it
    /// "does nothing". `None` = never persisted = keep whatever env said.
    #[serde(default)]
    pub hybrid_auto: Option<bool>,
    /// Max length of the last user message that still counts as a "simple"
    /// prompt. 0 disables the length gate. `None` = keep the env default.
    #[serde(default)]
    pub hybrid_aux_max_chars: Option<usize>,
    /// How DOC/COMPRESSION markers are matched in system messages:
    /// "substring" (legacy, compatible) or "prefix" (strict).
    #[serde(default = "default_marker_mode")]
    pub marker_mode: String,
}

fn default_marker_mode() -> String {
    "substring".to_string()
}

/// Mutable-at-runtime subset of the configuration (settings page).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSettings {
    pub router_threshold: f64,
    pub confidence_threshold: f64,
    pub route_tools_to_large: bool,
    pub default_route: Option<String>,
    pub hybrid_auto: bool,
    pub hybrid_aux_max_chars: usize,
    pub marker_mode: String,
    pub tts_url: String,
    pub stt_url: String,
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSettings {
    pub tts_url: Option<String>,
    pub stt_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetrics {
    pub requests_total: u64,
    pub requests_by_backend: std::collections::HashMap<String, u64>,
    pub fallback_count: u64,
    pub uptime_seconds: u64,
    pub session_cache_entries: u64,
    pub large_model_multimodal: bool,
    pub requests_by_origin: std::collections::HashMap<String, u64>,
    /// Fallback activations per reason. The dashboard only ever showed the
    /// total, which cannot distinguish "the primary was briefly busy" from
    /// "the primary is gone and everything is failing over".
    pub fallback_by_reason: std::collections::HashMap<String, u64>,
}
