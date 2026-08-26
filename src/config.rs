use crate::types::*;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub small_mllm_url: String,
    pub large_mllm_url: String,
    pub large_text_url: String,
    pub ocr_server_url: String,
    pub main_model_name: String,
    pub small_model_name: String,
    pub ocr_model_name: String,
    pub router_threshold: f64,
    pub confidence_threshold: f64,
    pub large_model_multimodal: bool,
    pub route_tools_to_large: bool,
    pub cb_threshold: u32,
    pub cb_reset_secs: u64,
    pub max_concurrent_images: usize,
    pub inference_url: String,
    pub multimodal_capability_url: String,
    pub providers: Vec<ProviderConfig>,
    pub tts_url: String,
    pub stt_url: String,

    // Extraction endpoint
    pub extract_cloud_url: Option<String>,
    pub extract_cloud_model: Option<String>,
    pub extract_cloud_api_key: Option<String>,
    pub extract_fallback_url: String,
    /// Shared secret guarding the upstream-override admin API (CASCADE_ADMIN_KEY).
    pub admin_key: Option<String>,
    /// Deterministic routing mode: none|inference|auxiliary|ocr (CASCADE_DEFAULT_ROUTE).
    pub default_route: Option<String>,
    /// Marker matching mode for doc/compression detection: substring|prefix.
    pub marker_mode: String,

    // Media generation endpoints (fallback when no registry node is active)
    pub image_generation_url: String,
    pub video_generation_url: String,

    // Background health prober
    /// Seconds between health probes of registered upstreams.
    pub health_interval_secs: u64,
    /// Consecutive failed probes before a node is demoted.
    pub health_failure_threshold: u32,
    /// Per-probe timeout in seconds.
    pub health_timeout_secs: u64,
    /// Node selection across a role's pool: round_robin|least_latency
    /// (CASCADE_ROUTING_STRATEGY).
    pub routing_strategy: crate::registry::RoutingStrategy,
    /// Upstream pool seeds from an optional cascade_config.yaml.
    pub upstream_seeds: Vec<crate::registry::UpstreamSeed>,
}

impl AppConfig {
    pub fn from_env() -> Self {
        let mut cfg = Self::from_env_base();
        cfg.apply_config_file();
        cfg
    }

    /// Optional `cascade_config.yaml` (path via CASCADE_CONFIG_FILE). Inspired
    /// by LiteLLM's declarative model_list: seeds upstream pools and admin
    /// settings at boot. Env vars always win over file values.
    fn apply_config_file(&mut self) {
        let path = match std::env::var("CASCADE_CONFIG_FILE") {
            Ok(p) if !p.is_empty() => p,
            _ => return,
        };
        let raw = match std::fs::read_to_string(&path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("WARN: CASCADE_CONFIG_FILE '{}' unreadable: {}", path, e);
                return;
            }
        };
        #[derive(serde::Deserialize)]
        struct ConfigFile {
            admin_key: Option<String>,
            routing_strategy: Option<String>,
            health: Option<HealthBlock>,
            #[serde(default)]
            upstreams: Vec<crate::registry::UpstreamSeed>,
        }
        #[derive(serde::Deserialize)]
        struct HealthBlock {
            interval_secs: Option<u64>,
            failure_threshold: Option<u32>,
            timeout_secs: Option<u64>,
        }

        let file: ConfigFile = match serde_yaml::from_str(&raw) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("WARN: cascade config file '{}' invalid: {}", path, e);
                return;
            }
        };

        if let Some(k) = file.admin_key {
            if !k.is_empty() && std::env::var("CASCADE_ADMIN_KEY").map(|v| v.is_empty()).unwrap_or(true) {
                self.admin_key = Some(k);
            }
        }
        if let Some(s) = file.routing_strategy.as_deref().and_then(crate::registry::RoutingStrategy::parse) {
            self.routing_strategy = s;
        }
        if let Some(h) = file.health {
            if let Some(i) = h.interval_secs {
                self.health_interval_secs = i.max(5);
            }
            if let Some(t) = h.failure_threshold {
                self.health_failure_threshold = t.max(1);
            }
            if let Some(t) = h.timeout_secs {
                self.health_timeout_secs = t.clamp(1, 30);
            }
        }
        self.upstream_seeds = file.upstreams;
        tracing::info!(
            "Loaded cascade config file '{}' ({} upstream seed/s)",
            path,
            self.upstream_seeds.len()
        );
    }

    /// Resolves the upstream seed entries from the config file (empty when the
    /// feature is unused). Consumed by main after GatewayState is built.
    pub fn take_upstream_seeds(&mut self) -> Vec<crate::registry::UpstreamSeed> {
        std::mem::take(&mut self.upstream_seeds)
    }

    fn from_env_base() -> Self {
        Self {
            small_mllm_url: std::env::var("SMALL_MLLM_URL")
                .unwrap_or_else(|_| "http://auxiliary-server:8080/v1/chat/completions".to_string()),
            large_mllm_url: std::env::var("LARGE_MLLM_URL")
                .unwrap_or_else(|_| "http://inference-server:8080/v1/chat/completions".to_string()),
            large_text_url: std::env::var("LARGE_TEXT_URL")
                .unwrap_or_else(|_| "http://inference-server:8080/v1/chat/completions".to_string()),
            ocr_server_url: std::env::var("OCR_URL")
                .unwrap_or_else(|_| "http://ocr-server:8082/v1/chat/completions".to_string()),
            main_model_name: std::env::var("MAIN_MODEL_NAME")
                .unwrap_or_else(|_| "Qwythos-9B-v2".to_string()),
            small_model_name: std::env::var("SMALL_MODEL_NAME")
                .unwrap_or_else(|_| "LFM2.5-2.6B".to_string()),
            ocr_model_name: std::env::var("OCR_MODEL_NAME")
                .unwrap_or_else(|_| "PaddleOCR-VL-1.6".to_string()),
            router_threshold: std::env::var("ROUTER_THRESHOLD")
                .unwrap_or_else(|_| "0.5".to_string())
                .parse::<f64>()
                .unwrap_or(0.5),
            confidence_threshold: std::env::var("CONFIDENCE_THRESHOLD")
                .unwrap_or_else(|_| "0.7".to_string())
                .parse::<f64>()
                .unwrap_or(0.7),
            large_model_multimodal: false,
            route_tools_to_large: std::env::var("ROUTE_TOOLS_TO_LARGE")
                .unwrap_or_else(|_| "true".to_string())
                .eq_ignore_ascii_case("true"),
            cb_threshold: std::env::var("CIRCUIT_BREAKER_THRESHOLD")
                .unwrap_or_else(|_| "5".to_string())
                .parse::<u32>()
                .unwrap_or(5),
            cb_reset_secs: std::env::var("CIRCUIT_BREAKER_RESET_SECS")
                .unwrap_or_else(|_| "60".to_string())
                .parse::<u64>()
                .unwrap_or(60),
            max_concurrent_images: std::env::var("MAX_CONCURRENT_IMAGES")
                .unwrap_or_else(|_| "4".to_string())
                .parse::<usize>()
                .unwrap_or(4),
            inference_url: std::env::var("INFERENCE_URL")
                .unwrap_or_else(|_| "http://netai-inference:8080".to_string()),
            multimodal_capability_url: std::env::var("MULTIMODAL_CAPABILITY_URL")
                .unwrap_or_else(|_| "http://inference-server:18081/multimodal-capability".to_string()),
            providers: Vec::new(),
            tts_url: std::env::var("TTS_URL")
                .unwrap_or_else(|_| "http://netai-tts:8800/v1/audio".to_string()),
            stt_url: std::env::var("STT_URL")
                .unwrap_or_else(|_| "http://netai-stt:5092/v1/audio".to_string()),

            extract_cloud_url: std::env::var("EXTRACT_CLOUD_URL").ok().filter(|s| !s.is_empty()),
            extract_cloud_model: std::env::var("EXTRACT_CLOUD_MODEL").ok().filter(|s| !s.is_empty()),
            extract_cloud_api_key: std::env::var("EXTRACT_CLOUD_API_KEY").ok().filter(|s| !s.is_empty()),
            extract_fallback_url: std::env::var("EXTRACT_FALLBACK_URL")
                .unwrap_or_else(|_| "http://auxiliary-server:8080".to_string()),
            admin_key: std::env::var("CASCADE_ADMIN_KEY").ok().filter(|s| !s.is_empty()),
            default_route: std::env::var("CASCADE_DEFAULT_ROUTE").ok().filter(|s| !s.is_empty()),
            marker_mode: std::env::var("MARKER_MODE").unwrap_or_else(|_| "substring".to_string()),

            image_generation_url: std::env::var("IMAGE_GENERATION_URL")
                .unwrap_or_else(|_| "http://localhost:8080/v1/images/generations".to_string()),
            video_generation_url: std::env::var("VIDEO_GENERATION_URL")
                .unwrap_or_else(|_| "http://localhost:8080/v1/video/generations".to_string()),

            health_interval_secs: std::env::var("HEALTH_CHECK_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|v| *v >= 5)
                .unwrap_or(30),
            health_failure_threshold: std::env::var("HEALTH_CHECK_FAILURE_THRESHOLD")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|v| *v >= 1)
                .unwrap_or(2),
            health_timeout_secs: std::env::var("HEALTH_CHECK_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|v| (1..=30).contains(v))
                .unwrap_or(5),
            routing_strategy: std::env::var("CASCADE_ROUTING_STRATEGY")
                .ok()
                .and_then(|v| crate::registry::RoutingStrategy::parse(&v))
                .unwrap_or_default(),
            upstream_seeds: Vec::new(),
        }
    }

    /// Builds the initial extraction backend entries from env vars.
    /// Called at startup; the local fallback is always registered.
    pub fn build_extraction_backends(&self) -> Vec<ExtractionBackendEntry> {
        let mut backends = Vec::new();

        if let Some(url) = &self.extract_cloud_url {
            backends.push(ExtractionBackendEntry {
                id: "cloud".to_string(),
                backend_type: if self.extract_cloud_model.as_deref().unwrap_or("").contains('/')
                    || url.contains("openrouter") || url.contains("googleapis")
                {
                    ExtractionBackendType::CloudLlm
                } else {
                    ExtractionBackendType::CloudGpu
                },
                name: "Cloud extraction backend".to_string(),
                url: url.clone(),
                model: self.extract_cloud_model.clone(),
                api_key: self.extract_cloud_api_key.clone(),
                enabled: true,
                priority: 10,
                max_cost_per_hour: None,
                last_validated: None,
                healthy: true,
            });
        }

        backends.push(ExtractionBackendEntry {
            id: "local".to_string(),
            backend_type: ExtractionBackendType::Local,
            name: "Local auxiliary server".to_string(),
            url: self.extract_fallback_url.clone(),
            model: Some(self.small_model_name.clone()),
            api_key: None,
            enabled: true,
            priority: 99,
            max_cost_per_hour: None,
            last_validated: None,
            healthy: true,
        });

        backends
    }

    pub fn to_settings(&self) -> Settings {
        Settings {
            providers: self.providers.clone(),
            routing: RoutingSettings {
                router_threshold: self.router_threshold,
                confidence_threshold: self.confidence_threshold,
                route_tools_to_large: self.route_tools_to_large,
                default_route: self.default_route.clone(),
                marker_mode: self.marker_mode.clone(),
            },
            audio: AudioSettings {
                tts_url: Some(self.tts_url.clone()),
                stt_url: Some(self.stt_url.clone()),
            },
            main_model_name: Some(self.main_model_name.clone()),
            small_model_name: Some(self.small_model_name.clone()),
            small_mllm_url: Some(self.small_mllm_url.clone()),
            large_mllm_url: Some(self.large_mllm_url.clone()),
            large_text_url: Some(self.large_text_url.clone()),
            ocr_server_url: Some(self.ocr_server_url.clone()),
            extract_cloud_url: self.extract_cloud_url.clone(),
            extract_cloud_model: self.extract_cloud_model.clone(),
            extract_fallback_url: Some(self.extract_fallback_url.clone()),
        }
    }
}
