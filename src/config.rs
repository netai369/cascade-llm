use crate::types::*;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub small_mllm_url: String,
    pub large_mllm_url: String,
    pub large_text_url: String,
    pub main_model_name: String,
    pub small_model_name: String,
    pub router_threshold: f64,
    pub confidence_threshold: f64,
    pub large_model_multimodal: bool,
    pub route_tools_to_large: bool,
    pub cb_threshold: u32,
    pub cb_reset_secs: u64,
    pub max_concurrent_images: usize,
    pub inference_url: String,
    pub providers: Vec<ProviderConfig>,
    pub tts_url: String,
    pub stt_url: String,
}

impl AppConfig {
    pub fn from_env() -> Self {
        Self {
            small_mllm_url: std::env::var("SMALL_MLLM_URL")
                .unwrap_or_else(|_| "http://localhost:8082/v1/chat/completions".to_string()),
            large_mllm_url: std::env::var("LARGE_MLLM_URL")
                .unwrap_or_else(|_| "http://localhost:8080/v1/chat/completions".to_string()),
            large_text_url: std::env::var("LARGE_TEXT_URL")
                .unwrap_or_else(|_| "http://localhost:8080/v1/chat/completions".to_string()),
            main_model_name: std::env::var("MAIN_MODEL_NAME")
                .unwrap_or_else(|_| "gpt-3.5-turbo".to_string()),
            small_model_name: std::env::var("SMALL_MODEL_NAME")
                .unwrap_or_else(|_| "gpt-4o-mini".to_string()),
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
            providers: Vec::new(),
            tts_url: std::env::var("TTS_URL")
                .unwrap_or_else(|_| "http://netai-tts:8800".to_string()),
            stt_url: std::env::var("STT_URL")
                .unwrap_or_else(|_| "http://netai-stt:5092".to_string()),
        }
    }

    pub fn to_settings(&self) -> Settings {
        Settings {
            providers: self.providers.clone(),
            routing: RoutingSettings {
                router_threshold: self.router_threshold,
                confidence_threshold: self.confidence_threshold,
                route_tools_to_large: self.route_tools_to_large,
            },
            audio: AudioSettings {
                tts_url: Some(self.tts_url.clone()),
                stt_url: Some(self.stt_url.clone()),
            },
        }
    }
}
