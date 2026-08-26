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
pub const ROLES: &[&str] = &["main", "auxiliary", "ocr", "image", "rag_worker"];

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

/// Maps legacy/human aliases onto canonical registry roles.
pub fn canonical_role(role: &str) -> Option<&'static str> {
    match role.trim().to_lowercase().as_str() {
        "main" | "inference" => Some("main"),
        "auxiliary" | "aux" => Some("auxiliary"),
        "ocr" | "vision" => Some("ocr"),
        "image" | "img" => Some("image"),
        "rag_worker" | "rag" => Some("rag_worker"),
        _ => None,
    }
}

/// Payload for `PUT /web/api/upstreams/{role}`. Accepts both the current field
/// names and the legacy ones (`url`, `bearer`) used by earlier provisioners.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RegisterUpstreamRequest {
    /// Exact URL requests are forwarded to (chat roles: `/v1/chat/completions`).
    #[serde(alias = "url")]
    pub endpoint_url: Option<String>,
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
    pub registered_at: String,
}

impl UpstreamNode {
    /// Health probe target: explicit override or `<origin>/health`.
    pub fn probe_url(&self) -> String {
        if let Some(u) = &self.health_url {
            return u.clone();
        }
        format!("{}/health", origin_of(&self.endpoint_url))
    }
}

/// Strips any known API path suffix, leaving `scheme://host[:port]`.
fn origin_of(url: &str) -> String {
    const SUFFIXES: &[&str] = &[
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/images/generations",
        "/v1/video/generations",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
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
#[derive(Debug, Default)]
pub struct UpstreamRegistry {
    roles: HashMap<String, Vec<UpstreamNode>>,
    counters: HashMap<String, u64>,
    strategy: RoutingStrategy,
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
                info!("UPSTREAM_UPDATED: role={} id={} url={}", role, node.id, url);
                UpsertResult { node: node.clone(), created: false }
            }
            None => {
                let node = UpstreamNode {
                    id: req.id.clone().unwrap_or_else(|| generated_id(role, &url)),
                    role: role.to_string(),
                    endpoint_url: url.clone(),
                    bearer_token: req.bearer_token,
                    weight: req.weight.unwrap_or(1).clamp(1, 64),
                    max_context_length: req.max_context_length,
                    active: true,
                    healthy: true,
                    consecutive_failures: 0,
                    last_health_check: None,
                    last_latency_ms: None,
                    latency_ema_ms: None,
                    label: req.label,
                    provider: req.provider,
                    cost_per_hour: req.cost_per_hour,
                    health_url: req.health_url,
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
    pub fn pick(&mut self, role: &str) -> Option<UpstreamNode> {
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
                // Lowest EMA wins; unmeasured nodes sort last so fresh nodes
                // are still reachable (and start building measurements).
                candidates.sort_by_key(|n| n.latency_ema_ms.unwrap_or(u64::MAX));
                candidates.into_iter().next()
            }
            RoutingStrategy::RoundRobin => {
                let expanded: Vec<&UpstreamNode> = candidates
                    .iter()
                    .flat_map(|n| std::iter::repeat_n(n, n.weight as usize))
                    .collect();
                let counter = self.counters.entry(role.to_string()).or_default();
                let picked = expanded[(*counter as usize) % expanded.len()].clone();
                *counter += 1;
                Some(picked)
            }
        }
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

/// One full probe pass over all registered nodes (bounded concurrency).
/// Also triggered manually via `POST /api/v1/admin/upstreams/probe`.
pub async fn run_health_sweep(
    state: &std::sync::Arc<crate::state::GatewayState>,
    client: &reqwest::Client,
) {
    let targets: Vec<(String, String)> = state
        .registry
        .read()
        .await
        .nodes()
        .iter()
        .map(|n| (n.id.clone(), n.probe_url()))
        .collect();
    if targets.is_empty() {
        return;
    }

    let threshold = state.config.health_failure_threshold;
    let results = futures_util::stream::iter(targets)
        .map(|(id, url)| async move {
            let start = Instant::now();
            let ok = client.get(&url).send().await.map(|r| r.status().is_success()).unwrap_or(false);
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
            bearer_token: None,
            weight: None,
            max_context_length: None,
            id: None,
            label: None,
            provider: None,
            cost_per_hour: None,
            health_url: None,
        }
    }

    #[test]
    fn canonical_role_maps_aliases() {
        assert_eq!(canonical_role("main"), Some("main"));
        assert_eq!(canonical_role("Inference"), Some("main"));
        assert_eq!(canonical_role("aux"), Some("auxiliary"));
        assert_eq!(canonical_role("rag"), Some("rag_worker"));
        assert_eq!(canonical_role("nope"), None);
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
    fn probe_url_derives_origin() {
        let node = reg_node("http://10.0.0.5:8080/v1/chat/completions");
        assert_eq!(node.probe_url(), "http://10.0.0.5:8080/health");
        let node = reg_node("http://zimage:1234/v1/images/generations/");
        assert_eq!(node.probe_url(), "http://zimage:1234/health");
        let node = reg_node("https://api.example.com/v1");
        assert_eq!(node.probe_url(), "https://api.example.com/health");
    }

    fn reg_node(url: &str) -> UpstreamNode {
        UpstreamRegistry::default()
            .upsert("main", req(url))
            .node
    }
}
