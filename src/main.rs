use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};
use tokio::net::TcpListener;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod cascade_features;
mod config;
mod db;
mod handlers;
mod language;
mod media;
mod audio;
mod providers;
mod state;
mod types;

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "info".to_string()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("cascade-llm v{}", env!("CARGO_PKG_VERSION"));

    let mut app_config = config::AppConfig::from_env();

    let large_model_multimodal = match std::env::var("LARGE_MODEL_MULTIMODAL") {
        Ok(v) => v.eq_ignore_ascii_case("true"),
        Err(_) => {
            info!("LARGE_MODEL_MULTIMODAL not set, auto-detecting...");
            match tokio::time::timeout(
                std::time::Duration::from_secs(5),
                state::fetch_large_model_multimodal_async(&app_config.inference_url),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => {
                    warn!("Timeout fetching multimodal capability, defaulting to true");
                    true
                }
            }
        }
    };
    app_config.large_model_multimodal = large_model_multimodal;

    let metrics = Arc::new(cascade_features::MetricsRegistry::init());
    let db = Arc::new(db::Db::new_in_memory().expect("Failed to init database"));

    let state = Arc::new(state::GatewayState::new(app_config, metrics, db));

    let _fallback_manager = Arc::new(cascade_features::FallbackManager::new(
        state.config.small_mllm_url.clone(),
        state.config.large_text_url.clone(),
        state.http_client.clone(),
    ));

    let app = Router::new()
        .merge(cascade_features::build_router::<Arc<state::GatewayState>>())
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/models", get(handlers::list_models))
        .route("/model", get(handlers::get_model))
        .route("/health", get(handlers::health_check))
        .route("/v1/audio/speech", post(handlers::tts))
        .route("/v1/audio/transcriptions", post(handlers::stt))
        .route(
            "/v1/images/generations",
            post(handlers::image_generation),
        )
        .route(
            "/v1/video/generations",
            post(handlers::video_generation),
        )
        .route("/web/metrics", get(cascade_features::metrics_handler))
        .route("/", get(handlers::dashboard))
        .route("/web/", get(handlers::dashboard))
        .route("/web/settings", get(handlers::settings_page))
        .route("/web/api/dashboard", get(handlers::dashboard_api))
        .route(
            "/web/api/settings",
            get(handlers::get_settings).put(handlers::update_settings),
        )
        .route(
            "/web/api/providers",
            get(handlers::list_providers).post(handlers::add_provider),
        )
        .route(
            "/web/api/providers/:id",
            get(handlers::get_provider).delete(handlers::delete_provider),
        )
        .with_state(state)
        .layer(DefaultBodyLimit::disable());

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    info!("Cascade LLM Gateway listening on {}", addr);
    info!("Web UI: http://0.0.0.0:3000/web/");
    info!(
        "Features: In-Flight Fallback, Streaming Quality Filter, Prometheus, Web Dashboard, Audio Proxy, Media Proxy"
    );
    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
