use serde::{Deserialize, Serialize};
use serde_json::Value;

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
        }
    }
}

impl From<axum::http::StatusCode> for ProxyError {
    fn from(s: axum::http::StatusCode) -> Self {
        Self {
            status: s.as_u16(),
            backend: "backend".to_string(),
            context: format!("backend request failed with HTTP {}", s.as_u16()),
            unreachable: false,
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
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
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
}

fn default_true() -> bool { true }
fn default_priority() -> u32 { 10 }

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingSettings {
    pub router_threshold: f64,
    pub confidence_threshold: f64,
    pub route_tools_to_large: bool,
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
}
