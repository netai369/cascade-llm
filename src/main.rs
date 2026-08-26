use std::net::SocketAddr;
use std::sync::Arc;

use cascade_llm::{cascade_features, config, db, registry, router, state::{self, GatewayState}};
use tokio::net::TcpListener;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

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
            // Retry a few times: at boot the llama.cpp container may not be
            // listening yet. Transport errors must NOT be read as "text-only".
            let mut probe = false;
            for attempt in 1..=3 {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    state::fetch_large_model_multimodal_async(&app_config.multimodal_capability_url),
                )
                .await
                {
                    Ok(true) => { probe = true; break; }
                    Ok(false) => {
                        warn!("Multimodal capability probe attempt {} returned false (retrying)", attempt);
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    }
                    Err(_) => {
                        warn!("Multimodal capability probe timeout (attempt {}), defaulting to true", attempt);
                        probe = true;
                        break;
                    }
                }
            }
            probe
        }
    };
    app_config.large_model_multimodal = large_model_multimodal;

    let metrics = Arc::new(cascade_features::MetricsRegistry::init());

    // Persisted settings/providers survive restarts (file DB; falls back to memory).
    let db_path = std::env::var("CASCADE_DB").unwrap_or_else(|_| "/data/cascade.db".to_string());
    let db = match db::Db::new(&db_path) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            warn!("Cannot open {} ({}), using in-memory DB", db_path, e);
            Arc::new(db::Db::new_in_memory().expect("Failed to init database"))
        }
    };

    // Restore saved settings BEFORE the state is built so thresholds/defaults
    // apply from the first request on.
    if let Ok(Some(raw)) = db.load_config("settings") {
        match serde_json::from_str::<cascade_llm::types::Settings>(&raw) {
            Ok(saved) => {
                info!("Restoring persisted settings from {}", db_path);
                app_config.router_threshold = saved.routing.router_threshold;
                app_config.confidence_threshold = saved.routing.confidence_threshold;
                app_config.route_tools_to_large = saved.routing.route_tools_to_large;
                app_config.default_route = saved.routing.default_route.clone();
                app_config.marker_mode = saved.routing.marker_mode.clone();
                if let Some(u) = &saved.audio.tts_url { app_config.tts_url = u.clone(); }
                if let Some(u) = &saved.audio.stt_url { app_config.stt_url = u.clone(); }
            }
            Err(e) => warn!("Persisted settings unreadable: {}", e),
        }
    }

    let routing_strategy = app_config.routing_strategy;
    let upstream_seeds = app_config.take_upstream_seeds();
    let state = Arc::new(GatewayState::new(app_config, metrics, db.clone()));

    state.registry.write().await.set_strategy(routing_strategy);
    for seed in upstream_seeds {
        match registry::canonical_role(&seed.role) {
            Some(role) => {
                state.registry.write().await.upsert(role, seed.node);
            }
            None => warn!("Config seed: unknown role '{}', skipping", seed.role),
        }
    }

    // Providers saved earlier re-enter the runtime registry here.
    match db.load_providers() {
        Ok(list) if !list.is_empty() => {
            state.runtime.write().await.providers = list;
            info!("Restored persisted providers");
        }
        _ => {}
    }

    // Background health prober: periodically probes all registered upstreams,
    // demotes failing nodes and recovers healthy ones (zero-downtime swaps).
    registry::spawn_health_prober(state.clone());

    let app = router::build_router(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    info!("Cascade LLM Gateway listening on {}", addr);
    info!("Web UI: http://0.0.0.0:3000/web/");
    info!(
        "Features: Dynamic Upstream Registry, Health Prober, Multi-Modal Routing, RAG Workers, Prometheus, Web Dashboard"
    );
    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
