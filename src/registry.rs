//! Dynamic upstream registry: real-time node management for all backend roles.
//!
//! Nodes are registered, updated, toggled and deregistered at runtime via the
//! admin API without restarting the gateway. Routing resolves a role to an
//! active, healthy node using weighted round-robin; static config URLs remain
//! as fallback when no node is registered.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};
use tracing::{info, warn};
use futures_util::StreamExt;

/// Canonical upstream roles managed by the registry.
pub const ROLES: &[&str] = &[
    "main", "auxiliary", "ocr", "image", "embeddings", "rerank", "extraction",
    "transcription", "speech", "rag_worker",
];

/// Upstream path a role's node is called on (appended to the node's `/v1` base).
/// Single source for registration (`normalize_endpoint_url`) and the console.
pub fn role_path(role: &str) -> &'static str {
    match role {
        "main" | "auxiliary" | "ocr" | "extraction" | "rag_worker" => "/chat/completions",
        "image" => "/images/generations",
        "embeddings" => "/embeddings",
        "rerank" => "/rerank",
        "transcription" => "/audio/transcriptions",
        "speech" => "/audio/speech",
        _ => "/chat/completions",
    }
}

/// Model kinds a role serves; the console filters a provider's models by these.
pub fn role_kinds(role: &str) -> &'static [&'static str] {
    match role {
        "embeddings" => &["embedding"],
        "rerank" => &["rerank"],
        "transcription" => &["audio-stt"],
        "speech" => &["audio-tts"],
        "image" => &["image"],
        _ => &["chat"],
    }
}

/// Role catalog served to the consoles, so no UI hard-codes role names.
pub fn role_catalog() -> Vec<serde_json::Value> {
    ROLES
        .iter()
        .map(|r| serde_json::json!({"role": r, "path": role_path(r), "kinds": role_kinds(r)}))
        .collect()
}

/// Node selection strategy across a role's pool (LiteLLM-inspired).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategy {
    /// Weighted round-robin — fair distribution, default.
    #[default]
    RoundRobin,
    /// Prefer the node with the lowest request-latency EMA (falls back to
    /// round-robin among nodes without measurements yet).
    LeastLatency,
}

impl RoutingStrategy {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "round_robin" | "roundrobin" | "weighted" => Some(Self::RoundRobin),
            "least_latency" | "latency" => Some(Self::LeastLatency),
            _ => None,
        }
    }
}

impl std::fmt::Display for RoutingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            RoutingStrategy::RoundRobin => "round_robin",
            RoutingStrategy::LeastLatency => "least_latency",
        })
    }
}

/// Maps legacy/human aliases onto canonical registry roles.
pub fn canonical_role(role: &str) -> Option<&'static str> {
    match role.trim().to_lowercase().as_str() {
        "main" | "inference" => Some("main"),
        "auxiliary" | "aux" => Some("auxiliary"),
        "ocr" | "vision" => Some("ocr"),
        "image" | "img" => Some("image"),
        "embeddings" | "embedding" => Some("embeddings"),
        "rerank" | "reranker" => Some("rerank"),
        "extraction" => Some("extraction"),
        "transcription" | "stt" => Some("transcription"),
        "speech" | "tts" => Some("speech"),
        "rag_worker" | "rag" => Some("rag_worker"),
        _ => None,
    }
}

/// Resolve loopback URLs registered from inside a container to the host alias.
/// Native gateway processes keep localhost unchanged. Container deployments
/// opt in with `CASCADE_HOST_GATEWAY=true`; this avoids guessing based on OS or
/// Docker internals and leaves service-DNS URLs untouched.
pub fn resolve_host_gateway_url(url: &str, enabled: bool) -> String {
    if !enabled {
        return url.to_string();
    }
    for prefix in ["http://localhost:", "https://localhost:", "http://127.0.0.1:", "https://127.0.0.1:"] {
        if let Some(rest) = url.strip_prefix(prefix) {
            let scheme = if prefix.starts_with("https") { "https" } else { "http" };
            return format!("{}://host.docker.internal:{}", scheme, rest);
        }
    }
    url.to_string()
}

/// Accept either an exact OpenAI endpoint or a server's `/v1` base URL.
pub fn normalize_endpoint_url(role: &str, url: &str) -> String {
    let trimmed = url.trim_end_matches('/');
    if !trimmed.ends_with("/v1") {
        return trimmed.to_string();
    }
    format!("{}{}", trimmed, role_path(role))
}

/// Payload for `PUT /web/api/upstreams/{role}`. Accepts both the current field
/// names and the legacy ones (`url`, `bearer`) used by earlier provisioners.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RegisterUpstreamRequest {
    /// Exact URL requests are forwarded to (chat roles: `/v1/chat/completions`).
    #[serde(alias = "url")]
    pub endpoint_url: Option<String>,
    /// Backend family for diagnostics/UI (`vmlx`, `inferencer`, `llamacpp`,
    /// `ovms`, `vllm`, `sglang`, `triton`, or an external API label).
    #[serde(default)]
    pub engine: Option<String>,
    #[serde(alias = "bearer", alias = "bearer_token")]
    pub bearer_token: Option<String>,
    #[serde(default)]
    pub weight: Option<u32>,
    #[serde(default)]
    pub max_context_length: Option<u64>,
    /// Optional client-supplied id for idempotent re-registration.
    #[serde(default)]
    pub id: Option<String>,
    /// Human-readable label shown in the UI (e.g. "Vast.ai RTX 4070").
    #[serde(default)]
    pub label: Option<String>,
    /// Cloud provider badge (e.g. "vast.ai", "runpod"); local nodes omit it.
    #[serde(default)]
    pub provider: Option<String>,
    /// Estimated hourly cost in USD for cloud instances.
    #[serde(default)]
    pub cost_per_hour: Option<f64>,
    /// Optional explicit health probe URL; defaults to `<origin>/health`.
    #[serde(default)]
    pub health_url: Option<String>,
    /// Register the node in a non-serving state. Lets an operator (or the
    /// provider reconciler) stage a backend without it taking traffic.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Model name forced onto requests routed to this node (used by hosted
    /// APIs like Pollinations where client model ids don't exist upstream).
    #[serde(default)]
    pub model: Option<String>,
}

/// One `upstreams:` entry from an optional cascade_config.yaml seed file.
#[derive(Debug, Clone, Deserialize)]
pub struct UpstreamSeed {
    pub role: String,
    #[serde(flatten)]
    pub node: RegisterUpstreamRequest,
}

/// A registered inference/media backend node.
#[derive(Debug, Clone, Serialize)]
pub struct UpstreamNode {
    pub id: String,
    pub role: String,
    pub endpoint_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(skip_serializing)]
    pub bearer_token: Option<String>,
    pub weight: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context_length: Option<u64>,
    /// Manually toggled in/out of the routing pool.
    pub active: bool,
    /// Set by the background health prober.
    pub healthy: bool,
    pub consecutive_failures: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_health_check: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_latency_ms: Option<u64>,
    /// Exponential moving average of real request latencies (ms), updated on
    /// every successful proxied request — feeds `LeastLatency` routing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ema_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_per_hour: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health_url: Option<String>,
    /// Model name forced onto requests routed to this node (hosted APIs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub registered_at: String,
}

/// How a role's node is probed. Chat-style roles keep the historical
/// `/health` probe; the non-chat roles (embeddings, rerank, audio, image) use
/// a cheap model-listing GET instead, so their health never depends on a chat
/// completion and a failing chat endpoint cannot demote them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeKind {
    /// `<origin>/health`, falling back to `<origin>/v1/models` on 404/405.
    Health,
    /// `<origin>/v1/models`, falling back to `<origin>/health` on 404/405.
    Models,
}

pub fn probe_kind(role: &str) -> ProbeKind {
    match role {
        "embeddings" | "rerank" | "transcription" | "speech" | "image" => ProbeKind::Models,
        _ => ProbeKind::Health,
    }
}

impl UpstreamNode {
    /// Health probe target: explicit override or `<origin>/health`.
    pub fn probe_url(&self) -> String {
        if let Some(u) = &self.health_url {
            return u.clone();
        }
        format!("{}/health", origin_of(&self.endpoint_url))
    }

    /// `(primary, fallback)` probe URLs for this node's role. An explicit
    /// `health_url` always wins as the primary target.
    pub fn probe_targets(&self) -> (String, String) {
        let origin = origin_of(&self.endpoint_url);
        let health = format!("{}/health", origin);
        let models = format!("{}/v1/models", origin);
        let (kind_primary, kind_fallback) = match probe_kind(&self.role) {
            ProbeKind::Health => (health.clone(), models.clone()),
            ProbeKind::Models => (models, health),
        };
        match &self.health_url {
            Some(explicit) => (explicit.clone(), kind_fallback),
            None => (kind_primary, kind_fallback),
        }
    }
}

/// Forces `model` onto a JSON request body (`{"model": ...}`). Returns `None`
/// when the body is not a JSON object or already names that model, so the
/// caller keeps the original bytes.
pub fn force_model_json(body: &[u8], model: &str) -> Option<Vec<u8>> {
    let mut value: serde_json::Value = serde_json::from_slice(body).ok()?;
    let obj = value.as_object_mut()?;
    if obj.get("model").and_then(|m| m.as_str()) == Some(model) {
        return None;
    }
    obj.insert("model".to_string(), serde_json::Value::String(model.to_string()));
    serde_json::to_vec(&value).ok()
}

/// Forces `model` onto a `multipart/form-data` body (the OpenAI transcription
/// `model` form field). Only the value of the `name="model"` part changes;
/// every other byte, including binary audio parts, is copied verbatim.
pub fn force_model_multipart(body: &[u8], content_type: &str, model: &str) -> Option<Vec<u8>> {
    let boundary = content_type
        .split(';')
        .map(str::trim)
        .find_map(|p| p.strip_prefix("boundary="))?
        .trim_matches('"');
    let delim = format!("--{}", boundary);
    let delim = delim.as_bytes();
    let parts = split_bytes(body, delim);
    let mut changed = false;
    let rewritten: Vec<Vec<u8>> = parts
        .iter()
        .map(|part| {
            let Some(head_end) = find_bytes(part, b"\r\n\r\n") else {
                return part.to_vec();
            };
            let headers = String::from_utf8_lossy(&part[..head_end]);
            if !headers.contains("name=\"model\"") {
                return part.to_vec();
            }
            let value = &part[head_end + 4..];
            let trailer: &[u8] = if value.ends_with(b"\r\n") { b"\r\n" } else { b"" };
            let mut out = part[..head_end + 4].to_vec();
            out.extend_from_slice(model.as_bytes());
            out.extend_from_slice(trailer);
            if out.as_slice() != *part {
                changed = true;
            }
            out
        })
        .collect();
    if !changed {
        return None;
    }
    let mut out = Vec::with_capacity(body.len());
    for (i, part) in rewritten.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(delim);
        }
        out.extend_from_slice(part);
    }
    Some(out)
}

fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn split_bytes<'a>(hay: &'a [u8], delim: &[u8]) -> Vec<&'a [u8]> {
    let mut out = Vec::new();
    let mut rest = hay;
    while let Some(pos) = find_bytes(rest, delim) {
        out.push(&rest[..pos]);
        rest = &rest[pos + delim.len()..];
    }
    out.push(rest);
    out
}

/// Registration payload for a saved TTS/STT override: a `speech` or
/// `transcription` node with a fixed id, so re-applying it is idempotent.
/// `base` may carry the trailing `/v1/audio` (both forms are accepted).
pub fn audio_override_request(role: &str, base: &str) -> Option<(String, RegisterUpstreamRequest)> {
    let (id, path) = match role {
        "speech" => ("manual-speech", "/v1/audio/speech"),
        "transcription" => ("manual-transcription", "/v1/audio/transcriptions"),
        _ => return None,
    };
    let base = base.trim();
    if base.is_empty() || !base.starts_with("http") {
        return None;
    }
    let endpoint = crate::audio::audio_target(base, path);
    Some((
        id.to_string(),
        RegisterUpstreamRequest {
            endpoint_url: Some(endpoint),
            id: Some(id.to_string()),
            label: Some(id.to_string()),
            ..Default::default()
        },
    ))
}

/// Startup migration for the retired console "Chat upstream overrides" TTS/STT
/// fields: a saved `tts_url` / `stt_url` becomes a `manual-speech` /
/// `manual-transcription` node in the registry. Idempotent (explicit node ids),
/// so it can run on every start. The static config URLs stay the fallback.
pub fn migrate_audio_overrides(
    registry: &mut UpstreamRegistry,
    tts_url: Option<&str>,
    stt_url: Option<&str>,
) -> usize {
    let mut migrated = 0;
    for (role, base) in [("speech", tts_url), ("transcription", stt_url)] {
        if let Some((_, req)) = base.and_then(|b| audio_override_request(role, b)) {
            registry.upsert(role, req);
            migrated += 1;
        }
    }
    migrated
}

/// Strips any known API path suffix, leaving `scheme://host[:port]`.
pub(crate) fn origin_of(url: &str) -> String {
    const SUFFIXES: &[&str] = &[
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/images/generations",
        "/v1/video/generations",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/embeddings",
        "/v1/rerank",
        "/chat/completions",
        "/v1",
    ];
    let mut base = url.trim_end_matches('/').to_string();
    let lower = base.to_lowercase();
    for suffix in SUFFIXES {
        if lower.ends_with(suffix) {
            base.truncate(base.len() - suffix.len());
            break;
        }
    }
    base.trim_end_matches('/').to_string()
}

fn now_ts() -> String {
    Utc::now().to_rfc3339()
}

fn generated_id(role: &str, url: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    url.hash(&mut hasher);
    format!("{}-{:016x}", role, hasher.finish())
}

/// Thread-safe pool of upstream nodes keyed by role (`Arc<RwLock<..>>` lives in
/// [`crate::state::GatewayState`]). Selection is weighted round-robin over all
/// active + healthy nodes of a role, giving fair distribution under load.
/// Round-robin counters are atomics so `pick` only needs a read lock —
/// concurrent request routing never serializes on the registry writer.
#[derive(Debug)]
pub struct UpstreamRegistry {
    roles: HashMap<String, Vec<UpstreamNode>>,
    counters: HashMap<String, std::sync::atomic::AtomicU64>,
    strategy: RoutingStrategy,
}

impl Default for UpstreamRegistry {
    fn default() -> Self {
        // Round-robin counters are pre-seeded per canonical role so `pick`
        // never needs mutable access to the map.
        Self {
            roles: HashMap::new(),
            counters: ROLES
                .iter()
                .map(|r| (r.to_string(), std::sync::atomic::AtomicU64::new(0)))
                .collect(),
            strategy: RoutingStrategy::default(),
        }
    }
}

pub struct UpsertResult {
    pub node: UpstreamNode,
    pub created: bool,
}

impl UpstreamRegistry {
    pub fn set_strategy(&mut self, strategy: RoutingStrategy) {
        info!("UPSTREAM_STRATEGY: {:?}", strategy);
        self.strategy = strategy;
    }

    pub fn strategy(&self) -> RoutingStrategy {
        self.strategy
    }

    /// Registers or updates a node. Matching happens by explicit id first,
    /// then by (role, endpoint_url) so repeated PUTs stay idempotent.
    pub fn upsert(&mut self, role: &str, req: RegisterUpstreamRequest) -> UpsertResult {
        let url = req.endpoint_url.unwrap_or_default();
        let pool = self.roles.entry(role.to_string()).or_default();

        let existing_pos = pool
            .iter()
            .position(|n| Some(&n.id) == req.id.as_ref())
            .or_else(|| {
                pool.iter()
                    .position(|n| n.role == role && n.endpoint_url == url)
            });

        match existing_pos {
            Some(pos) => {
                let node = &mut pool[pos];
                if node.endpoint_url != url {
                    // New target: reset runtime health state.
                    node.healthy = true;
                    node.consecutive_failures = 0;
                    node.last_latency_ms = None;
                }
                node.endpoint_url = url.clone();
                if req.engine.is_some() {
                    node.engine = req.engine;
                }
                if req.bearer_token.is_some() {
                    node.bearer_token = req.bearer_token;
                }
                if let Some(w) = req.weight {
                    node.weight = w.clamp(1, 64);
                }
                node.max_context_length = req.max_context_length.or(node.max_context_length);
                if req.label.is_some() {
                    node.label = req.label;
                }
                if req.provider.is_some() {
                    node.provider = req.provider;
                }
                if req.cost_per_hour.is_some() {
                    node.cost_per_hour = req.cost_per_hour;
                }
                if req.health_url.is_some() {
                    node.health_url = req.health_url;
                }
                if req.model.is_some() {
                    node.model = req.model;
                }
                if let Some(enabled) = req.enabled {
                    node.active = enabled;
                }
                info!("UPSTREAM_UPDATED: role={} id={} url={}", role, node.id, url);
                UpsertResult { node: node.clone(), created: false }
            }
            None => {
                let node = UpstreamNode {
                    id: req.id.clone().unwrap_or_else(|| generated_id(role, &url)),
                    role: role.to_string(),
                    endpoint_url: url.clone(),
                    engine: req.engine,
                    bearer_token: req.bearer_token,
                    weight: req.weight.unwrap_or(1).clamp(1, 64),
                    max_context_length: req.max_context_length,
                    active: req.enabled.unwrap_or(true),
                    healthy: true,
                    consecutive_failures: 0,
                    last_health_check: None,
                    last_latency_ms: None,
                    latency_ema_ms: None,
                    label: req.label,
                    provider: req.provider,
                    cost_per_hour: req.cost_per_hour,
                    health_url: req.health_url,
                    model: req.model,
                    registered_at: now_ts(),
                };
                info!("UPSTREAM_REGISTERED: role={} id={} url={}", role, node.id, url);
                pool.push(node.clone());
                UpsertResult { node, created: true }
            }
        }
    }

    /// Removes every node of a role; returns how many were removed.
    pub fn remove_role(&mut self, role: &str) -> usize {
        if let Some(pool) = self.roles.get_mut(role) {
            let removed = pool.len();
            pool.clear();
            if removed > 0 {
                info!("UPSTREAM_DEREGISTERED: role={} count={}", role, removed);
            }
            removed
        } else {
            0
        }
    }

    /// Removes a single node by id (any role); returns it when found.
    pub fn remove_id(&mut self, id: &str) -> Option<UpstreamNode> {
        for pool in self.roles.values_mut() {
            if let Some(pos) = pool.iter().position(|n| n.id == id) {
                let node = pool.remove(pos);
                info!("UPSTREAM_DEREGISTERED: id={}", id);
                return Some(node);
            }
        }
        None
    }

    /// Toggles a node in/out of the routing pool.
    pub fn set_active(&mut self, id: &str, active: bool) -> bool {
        for pool in self.roles.values_mut() {
            if let Some(node) = pool.iter_mut().find(|n| n.id == id) {
                node.active = active;
                info!("UPSTREAM_{}: id={}", if active { "ACTIVATED" } else { "DEACTIVATED" }, id);
                return true;
            }
        }
        false
    }

    pub fn get(&self, id: &str) -> Option<&UpstreamNode> {
        self.roles.values().flatten().find(|n| n.id == id)
    }

    /// The node a role's traffic would go to right now (same selection as
    /// `pick`, but without consuming a round-robin slot). Used to report which
    /// model a role is actually serving.
    pub fn get_by_role_active(&self, role: &str) -> Option<UpstreamNode> {
        self.pick(role)
    }

    /// True when at least one active + healthy node exists for the role.
    pub fn has_active(&self, role: &str) -> bool {
        self.roles
            .get(role)
            .map(|pool| pool.iter().any(|n| n.active && n.healthy))
            .unwrap_or(false)
    }

    /// Picks a node among active + healthy nodes of a role according to the
    /// configured [`RoutingStrategy`]. Candidate order is deterministic
    /// (sorted by id) to keep distribution stable.
    ///
    /// LeastLatency starvation guard: candidates without a latency EMA yet
    /// (freshly registered) are served first via weighted round-robin until
    /// they build measurements — otherwise they would never receive traffic.
    pub fn pick(&self, role: &str) -> Option<UpstreamNode> {
        let mut candidates: Vec<UpstreamNode> = self
            .roles
            .get(role)
            .map(|pool| {
                pool.iter()
                    .filter(|n| n.active && n.healthy)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        if candidates.is_empty() {
            return None;
        }
        candidates.sort_by(|a, b| a.id.cmp(&b.id));

        match self.strategy {
            RoutingStrategy::LeastLatency => {
                let unmeasured: Vec<UpstreamNode> = candidates
                    .iter()
                    .filter(|n| n.latency_ema_ms.is_none())
                    .cloned()
                    .collect();
                if !unmeasured.is_empty() {
                    self.weighted_round_robin(role, &unmeasured)
                } else {
                    candidates.sort_by_key(|n| n.latency_ema_ms.unwrap_or(u64::MAX));
                    candidates.into_iter().next()
                }
            }
            RoutingStrategy::RoundRobin => self.weighted_round_robin(role, &candidates),
        }
    }

    fn weighted_round_robin(&self, role: &str, candidates: &[UpstreamNode]) -> Option<UpstreamNode> {
        let expanded: Vec<&UpstreamNode> = candidates
            .iter()
            .flat_map(|n| std::iter::repeat_n(n, n.weight as usize))
            .collect();
        let idx = match self.counters.get(role) {
            Some(counter) => counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            None => 0,
        };
        Some(expanded[(idx as usize) % expanded.len()].clone())
    }

    /// Records the latency of a successful proxied request into the node's
    /// EMA (α = 1/8 — smooths outliers without full history).
    pub fn record_request_latency(&mut self, id: &str, latency_ms: u64) {
        if let Some(node) = self.roles.values_mut().flatten().find(|n| n.id == id) {
            node.latency_ema_ms = Some(match node.latency_ema_ms {
                Some(ema) => (ema.saturating_mul(7).saturating_add(latency_ms)) / 8,
                None => latency_ms,
            });
        }
    }

    /// Snapshot of all nodes across roles, sorted by (role, id).
    pub fn nodes(&self) -> Vec<&UpstreamNode> {
        let mut all: Vec<&UpstreamNode> = self.roles.values().flatten().collect();
        all.sort_by(|a, b| a.role.cmp(&b.role).then(a.id.cmp(&b.id)));
        all
    }

    /// Applies a successful health probe result.
    pub fn record_probe_success(&mut self, id: &str, latency_ms: u64) {
        if let Some(node) = self.roles.values_mut().flatten().find(|n| n.id == id) {
            node.healthy = true;
            node.consecutive_failures = 0;
            node.last_latency_ms = Some(latency_ms);
            node.last_health_check = Some(now_ts());
        }
    }

    /// Applies a failed health probe; demotes the node after `threshold`
    /// consecutive failures so routing stops selecting it automatically.
    pub fn record_probe_failure(&mut self, id: &str, threshold: u32) {
        if let Some(node) = self.roles.values_mut().flatten().find(|n| n.id == id) {
            node.consecutive_failures += 1;
            node.last_health_check = Some(now_ts());
            if node.consecutive_failures >= threshold && node.healthy {
                warn!(
                    "UPSTREAM_DEMOTED: id={} after {} consecutive failed health checks",
                    id, node.consecutive_failures
                );
                node.healthy = false;
            }
        }
    }

    /// Updates the model name of a specific node.
    pub fn update_model(&mut self, role: &str, id: &str, model: &str) -> bool {
        if let Some(pool) = self.roles.get_mut(role) {
            if let Some(node) = pool.iter_mut().find(|n| n.id == id) {
                node.model = Some(model.to_string());
                info!("UPSTREAM_MODEL_UPDATED: role={} id={} model={}", role, id, model);
                return true;
            }
        }
        false
    }
}

// ============================================================================
// BACKGROUND HEALTH PROBER
// ============================================================================

/// Spawns the periodic health-check task. Every `HEALTH_CHECK_INTERVAL_SECS`
/// all registered nodes are probed concurrently; failing nodes are demoted
/// after the configured threshold, recovered nodes re-enter the pool.
pub fn spawn_health_prober(state: std::sync::Arc<crate::state::GatewayState>) {
    let interval = Duration::from_secs(state.config.health_interval_secs);
    tokio::spawn(async move {
        let client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(state.config.health_timeout_secs))
            .connect_timeout(Duration::from_secs(3))
            .pool_max_idle_per_host(8)
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("Health prober client init failed, prober disabled: {}", e);
                return;
            }
        };
        loop {
            tokio::time::sleep(interval).await;
            run_health_sweep(&state, &client).await;
        }
    });
}

/// One GET probe. `None` means the endpoint does not exist on that server
/// (404/405) and the caller should try its fallback; `Some(alive)` otherwise.
/// Nodes behind an API key answer 401 without the bearer, so it is attached.
async fn probe_once(client: &reqwest::Client, url: &str, bearer: Option<&str>) -> Option<bool> {
    let mut req = client.get(url);
    if let Some(token) = bearer.filter(|t| !t.is_empty()) {
        req = req.bearer_auth(token);
    }
    match req.send().await {
        Ok(r) if matches!(r.status().as_u16(), 404 | 405) => None,
        Ok(r) => Some(r.status().is_success() || r.status().is_redirection()),
        Err(_) => Some(false),
    }
}

/// One full probe pass over all registered nodes (bounded concurrency).
/// Also triggered manually via `POST /api/v1/admin/upstreams/probe`.
pub async fn run_health_sweep(
    state: &std::sync::Arc<crate::state::GatewayState>,
    client: &reqwest::Client,
) {
    // (id, primary probe, fallback probe, bearer) per node; the probe kind is
    // role-specific (see `probe_kind`), so audio/image/embedding nodes never
    // receive a chat-style probe.
    let targets: Vec<(String, String, String, Option<String>)> = state
        .registry
        .read()
        .await
        .nodes()
        .iter()
        .map(|n| {
            let (primary, fallback) = n.probe_targets();
            (n.id.clone(), primary, fallback, n.bearer_token.clone())
        })
        .collect();
    if targets.is_empty() {
        return;
    }

    let threshold = state.config.health_failure_threshold;
    let results = futures_util::stream::iter(targets)
        .map(|(id, primary, fallback, bearer)| async move {
            let start = Instant::now();
            let ok = match probe_once(client, &primary, bearer.as_deref()).await {
                Some(alive) => alive,
                // Endpoint missing on this server (404/405): try the other probe.
                None => probe_once(client, &fallback, bearer.as_deref()).await.unwrap_or(false),
            };
            (id, ok, start.elapsed().as_millis() as u64)
        })
        .buffer_unordered(16)
        .collect::<Vec<_>>()
        .await;

    let mut demoted = 0usize;
    {
        let mut registry = state.registry.write().await;
        for (id, ok, latency_ms) in results {
            if ok {
                registry.record_probe_success(&id, latency_ms);
            } else {
                registry.record_probe_failure(&id, threshold);
                demoted += 1;
            }
        }
    }
    if demoted > 0 {
        info!("HEALTH_SWEEP: {} node(s) failing checks", demoted);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(url: &str) -> RegisterUpstreamRequest {
        RegisterUpstreamRequest {
            endpoint_url: Some(url.to_string()),
            engine: None,
            bearer_token: None,
            weight: None,
            max_context_length: None,
            id: None,
            label: None,
            provider: None,
            cost_per_hour: None,
            health_url: None,
            model: None,
            enabled: None,
        }
    }

    #[test]
    fn canonical_role_maps_aliases() {
        assert_eq!(canonical_role("main"), Some("main"));
        assert_eq!(canonical_role("Inference"), Some("main"));
        assert_eq!(canonical_role("aux"), Some("auxiliary"));
        assert_eq!(canonical_role("rag"), Some("rag_worker"));
        assert_eq!(canonical_role("embedding"), Some("embeddings"));
        assert_eq!(canonical_role("vision"), Some("ocr"));
        assert_eq!(canonical_role("tts"), Some("speech"));
        assert_eq!(canonical_role("stt"), Some("transcription"));
        assert_eq!(canonical_role("reranker"), Some("rerank"));
        assert_eq!(canonical_role("nope"), None);
    }

    #[test]
    fn localhost_is_rewritten_only_for_container_host_bridge_mode() {
        assert_eq!(resolve_host_gateway_url("http://localhost:8080/v1", false), "http://localhost:8080/v1");
        assert_eq!(resolve_host_gateway_url("http://localhost:8080/v1", true), "http://host.docker.internal:8080/v1");
        assert_eq!(resolve_host_gateway_url("http://127.0.0.1:8081/v1", true), "http://host.docker.internal:8081/v1");
        assert_eq!(resolve_host_gateway_url("http://ovms:18000/v1", true), "http://ovms:18000/v1");
    }

    #[test]
    fn v1_base_urls_are_expanded_for_chat_and_image_roles() {
        assert_eq!(normalize_endpoint_url("main", "http://localhost:8080/v1"), "http://localhost:8080/v1/chat/completions");
        assert_eq!(normalize_endpoint_url("ocr", "http://localhost:8081/v1/"), "http://localhost:8081/v1/chat/completions");
        assert_eq!(normalize_endpoint_url("image", "http://localhost:8082/v1"), "http://localhost:8082/v1/images/generations");
        assert_eq!(normalize_endpoint_url("embeddings", "http://localhost:8801/v1"), "http://localhost:8801/v1/embeddings");
        assert_eq!(normalize_endpoint_url("rerank", "http://localhost:8801/v1"), "http://localhost:8801/v1/rerank");
        assert_eq!(normalize_endpoint_url("speech", "http://localhost:8800/v1"), "http://localhost:8800/v1/audio/speech");
    }

    #[test]
    fn upsert_is_idempotent_by_url_and_id() {
        let mut reg = UpstreamRegistry::default();
        let a = reg.upsert("main", req("http://a:8080/v1/chat/completions"));
        let b = reg.upsert("main", req("http://a:8080/v1/chat/completions"));
        assert!(a.created);
        assert!(!b.created);
        assert_eq!(a.node.id, b.node.id);

        let c = reg.upsert(
            "main",
            RegisterUpstreamRequest { id: Some(a.node.id.clone()), ..req("http://b:8080/v1") },
        );
        assert!(!c.created);
        assert_eq!(c.node.endpoint_url, "http://b:8080/v1");
        assert_eq!(reg.nodes().len(), 1);
    }

    #[test]
    fn pick_skips_inactive_and_unhealthy() {
        let mut reg = UpstreamRegistry::default();
        reg.upsert("ocr", req("http://o1:8080/v1/chat/completions"));
        assert!(reg.pick("ocr").is_some());

        let id = reg.nodes()[0].id.clone();
        reg.set_active(&id, false);
        assert!(reg.pick("ocr").is_none());
        reg.set_active(&id, true);
        reg.record_probe_failure(&id, 2);
        assert!(reg.pick("ocr").is_some(), "single failure must not demote");
        reg.record_probe_failure(&id, 2);
        assert!(reg.pick("ocr").is_none(), "node demoted after threshold");

        // Recovery on next successful probe.
        reg.record_probe_success(&id, 12);
        assert!(reg.pick("ocr").is_some());
    }

    #[test]
    fn weighted_round_robin_distributes() {
        let mut reg = UpstreamRegistry::default();
        let mut heavy = req("http://h:8080/v1/chat/completions");
        heavy.weight = Some(3);
        reg.upsert("image", heavy);
        reg.upsert("image", req("http://l:1234/v1/images/generations"));

        let mut counts = HashMap::new();
        for _ in 0..8 {
            let node = reg.pick("image").unwrap();
            *counts.entry(node.endpoint_url.clone()).or_insert(0) += 1;
        }
        assert_eq!(*counts.get("http://h:8080/v1/chat/completions").unwrap(), 6);
        assert_eq!(*counts.get("http://l:1234/v1/images/generations").unwrap(), 2);
    }

    #[test]
    fn remove_role_and_remove_id() {
        let mut reg = UpstreamRegistry::default();
        let n = reg.upsert("rag_worker", req("http://r:8080")).node;
        assert_eq!(reg.remove_role("rag_worker"), 1);
        assert_eq!(reg.remove_role("rag_worker"), 0);
        assert!(reg.get(&n.id).is_none());

        let m = reg.upsert("image", req("http://i:1234/v1/images/generations")).node;
        assert!(reg.remove_id(&m.id).is_some());
        assert!(reg.remove_id(&m.id).is_none());
    }

    #[test]
    fn least_latency_serves_unmeasured_nodes_first() {
        let mut reg = UpstreamRegistry::default();
        reg.set_strategy(RoutingStrategy::LeastLatency);
        let mut fast = req("http://fast:8080/v1/chat/completions");
        fast.weight = Some(1);
        let fast_node = reg.upsert("main", fast).node;
        reg.record_request_latency(&fast_node.id, 10);

        // A fresh node must receive traffic under least_latency (warm-up),
        // not starve forever behind the measured one.
        reg.upsert("main", req("http://new:8080/v1/chat/completions"));
        let picked = reg.pick("main").unwrap();
        assert_eq!(picked.endpoint_url, "http://new:8080/v1/chat/completions");

        // Once measured, the faster node wins again.
        reg.record_request_latency(&picked.id, 500);
        let picked = reg.pick("main").unwrap();
        assert_eq!(picked.endpoint_url, "http://fast:8080/v1/chat/completions");
    }

    #[test]
    fn probe_url_derives_origin() {
        let node = reg_node("http://10.0.0.5:8080/v1/chat/completions");
        assert_eq!(node.probe_url(), "http://10.0.0.5:8080/health");
        let node = reg_node("http://zimage:1234/v1/images/generations/");
        assert_eq!(node.probe_url(), "http://zimage:1234/health");
        let node = reg_node("https://api.example.com/v1");
        assert_eq!(node.probe_url(), "https://api.example.com/health");
    }

    #[test]
    fn probe_kind_is_role_specific() {
        assert_eq!(probe_kind("main"), ProbeKind::Health);
        assert_eq!(probe_kind("extraction"), ProbeKind::Health);
        for role in ["embeddings", "rerank", "transcription", "speech", "image"] {
            assert_eq!(probe_kind(role), ProbeKind::Models, "{role}");
        }
    }

    #[test]
    fn probe_targets_follow_role_kind() {
        let mut reg = UpstreamRegistry::default();
        let speech = reg.upsert("speech", req("http://tts:8800/v1/audio/speech")).node;
        assert_eq!(
            speech.probe_targets(),
            ("http://tts:8800/v1/models".to_string(), "http://tts:8800/health".to_string())
        );
        let embed = reg.upsert("embeddings", req("http://emb:8801/v1/embeddings")).node;
        assert_eq!(embed.probe_targets().0, "http://emb:8801/v1/models");
        let chat = reg.upsert("main", req("http://llm:8000/v1/chat/completions")).node;
        assert_eq!(
            chat.probe_targets(),
            ("http://llm:8000/health".to_string(), "http://llm:8000/v1/models".to_string())
        );
        let mut explicit = req("http://tts:8800/v1/audio/speech");
        explicit.health_url = Some("http://tts:9000/ready".to_string());
        let custom = reg.upsert("speech", explicit).node;
        assert_eq!(custom.probe_targets().0, "http://tts:9000/ready");
    }

    #[test]
    fn origin_strips_embeddings_and_rerank_paths() {
        assert_eq!(origin_of("http://emb:8801/v1/embeddings"), "http://emb:8801");
        assert_eq!(origin_of("http://emb:8801/v1/rerank"), "http://emb:8801");
    }

    #[test]
    fn force_model_json_rewrites_only_when_different() {
        let out = force_model_json(br#"{"model":"a","input":"x"}"#, "b").unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["model"], "b");
        assert_eq!(v["input"], "x");
        assert!(force_model_json(br#"{"model":"b"}"#, "b").is_none());
        assert!(force_model_json(b"not json", "b").is_none());
    }

    #[test]
    fn force_model_multipart_touches_only_the_model_part() {
        let body = b"--XB\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nold\r\n--XB\r\nContent-Disposition: form-data; name=\"file\"\r\n\r\nAUDIO\r\n--XB--\r\n";
        let out = force_model_multipart(body, "multipart/form-data; boundary=XB", "new-model").unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("name=\"model\"\r\n\r\nnew-model\r\n--XB"));
        assert!(text.contains("AUDIO\r\n--XB--"));
        assert!(force_model_multipart(body, "multipart/form-data; boundary=XB", "old").is_none());
        assert!(force_model_multipart(body, "application/json", "x").is_none());
    }

    #[test]
    fn audio_overrides_migrate_idempotently_to_speech_and_transcription_nodes() {
        let mut reg = UpstreamRegistry::default();
        // Both base forms are accepted and normalize to the same endpoints.
        let n = migrate_audio_overrides(&mut reg, Some("http://speech-tts:8800"), Some("http://stt:5092/v1/audio"));
        assert_eq!(n, 2);
        assert_eq!(migrate_audio_overrides(&mut reg, Some("http://speech-tts:8800"), Some("http://stt:5092/v1/audio")), 2);
        assert_eq!(reg.nodes().len(), 2, "re-running the migration must not duplicate nodes");
        let speech = reg.pick("speech").unwrap();
        assert_eq!(speech.endpoint_url, "http://speech-tts:8800/v1/audio/speech");
        assert_eq!(speech.id, "manual-speech");
        assert_eq!(speech.label.as_deref(), Some("manual-speech"));
        let stt = reg.pick("transcription").unwrap();
        assert_eq!(stt.endpoint_url, "http://stt:5092/v1/audio/transcriptions");
        assert_eq!(stt.label.as_deref(), Some("manual-transcription"));

        // A changed URL updates the same node instead of adding one.
        migrate_audio_overrides(&mut reg, Some("http://tts-new:8800"), None);
        assert_eq!(reg.pick("speech").unwrap().endpoint_url, "http://tts-new:8800/v1/audio/speech");
        assert_eq!(reg.nodes().len(), 2);

        // Empty or non-http values migrate nothing.
        assert_eq!(migrate_audio_overrides(&mut UpstreamRegistry::default(), Some(""), Some("ws://x")), 0);
    }

    fn reg_node(url: &str) -> UpstreamNode {
        UpstreamRegistry::default()
            .upsert("main", req(url))
            .node
    }
}
