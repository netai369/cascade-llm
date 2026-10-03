//! Axum router assembly — shared by the binary and integration tests.

use crate::{cascade_features, handlers, state::GatewayState};
use axum::{
    extract::DefaultBodyLimit,
    middleware,
    routing::{delete, get, post, put},
    Router,
};
use std::sync::Arc;

pub fn build_router(state: Arc<GatewayState>) -> Router {
    // Mutating dashboard/pipeline routes: guarded by the admin key as soon as
    // one is configured. `route_layer` runs before body extraction so an
    // unauthenticated request cannot reach the handler at all.
    let admin_routes = Router::new()
        .route(
            "/web/api/settings",
            put(handlers::update_settings),
        )
        .route(
            "/web/api/providers",
            post(handlers::add_provider),
        )
        .route(
            "/web/api/providers/:id",
            delete(handlers::delete_provider),
        )
        .route(
            "/api/providers",
            post(handlers::add_provider),
        )
        .route(
            "/api/providers/:id",
            delete(handlers::delete_provider),
        )
        .route(
            "/extraction/v1/backends",
            post(handlers::register_extraction_backend_handler),
        )
        .route(
            "/extraction/v1/backends/:id",
            delete(handlers::remove_extraction_backend_handler),
        )
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            handlers::require_admin,
        ));

    Router::new()
        .merge(admin_routes)
        .merge(cascade_features::build_router::<Arc<GatewayState>>())
        // OpenAI-compatible chat + model discovery
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/completions", post(handlers::completions))
        .route("/v1/models", get(handlers::list_models))
        .route("/models", get(handlers::list_models))
        .route("/model", get(handlers::get_model))
        .route("/web/api/active-models", get(handlers::active_models))
        .route("/health", get(handlers::health_check))
        // Audio / media proxies
        .route("/v1/audio/speech", post(handlers::tts))
        .route("/v1/audio/transcriptions", post(handlers::stt))
        .route("/v1/embeddings", post(handlers::embeddings))
        .route("/v1/rerank", post(handlers::rerank))
        .route("/v1/images/generations", post(handlers::image_generation))
        .route("/v1/video/generations", post(handlers::video_generation))
        // Heavy RAG extraction jobs -> rag_worker pool
        .route("/v1/rag/extract", post(handlers::rag_extract))
        // Web UI
        .route("/web/models", get(handlers::list_models))
        .route("/web/metrics", get(cascade_features::metrics_handler))
        .route("/", get(handlers::dashboard))
        .route("/web/", get(handlers::dashboard))
        .route("/web/settings", get(handlers::settings_page))
        .route("/web/docs/:name", get(handlers::docs_page))
        .route("/web/api/dashboard", get(handlers::dashboard_api))
        .route(
            "/web/api/settings",
            get(handlers::get_settings),
        )
        .route(
            "/web/api/providers",
            get(handlers::list_providers),
        )
        .route(
            "/web/api/providers/:id",
            get(handlers::get_provider),
        )
        // Aliases without the /web prefix (programmatic access)
        .route("/api/dashboard", get(handlers::dashboard_api))
        .route("/api/active-models", get(handlers::active_models))
        .route(
            "/api/providers",
            get(handlers::list_providers),
        )
        .route(
            "/api/providers/:id",
            get(handlers::get_provider),
        )
        // Extraction pipeline (LightRAG / knowledge-engine)
        .route(
            "/extraction/v1/chat/completions",
            post(handlers::extraction_completions),
        )
        .route(
            "/extraction/v1/backends",
            get(handlers::list_extraction_backends),
        )
        // Dynamic upstream registry API
        .route("/web/api/upstreams", get(handlers::list_upstreams))
        .route(
            "/web/api/upstreams/:role",
            put(handlers::put_upstream).delete(handlers::delete_upstream),
        )
        .route("/api/upstreams", get(handlers::list_upstreams))
        .route(
            "/api/upstreams/:role",
            put(handlers::put_upstream).delete(handlers::delete_upstream),
        )
        // v1 admin aliases under /web — the dashboard is served at both "/"
        // and "/web/" (Caddy rewrites /cascade/* to /web*), so relative
        // fetch('api/v1/...') must resolve on both mounts.
        .route(
            "/web/api/v1/inference-nodes/:id/activate",
            post(handlers::activate_node),
        )
        .route(
            "/web/api/v1/inference-nodes/:id/deactivate",
            post(handlers::deactivate_node),
        )
        .route("/web/api/v1/admin/upstreams", get(handlers::admin_upstreams))
        .route(
            "/web/api/v1/admin/upstreams/probe",
            post(handlers::probe_upstreams),
        )
        .route(
            "/web/api/v1/admin/upstreams/refresh-models",
            post(handlers::refresh_models_upstreams),
        )
        .route(
            "/api/v1/inference-nodes/:id/activate",
            post(handlers::activate_node),
        )
        .route(
            "/api/v1/inference-nodes/:id/deactivate",
            post(handlers::deactivate_node),
        )
        .route("/api/v1/admin/upstreams", get(handlers::admin_upstreams))
        .route(
            "/api/v1/admin/upstreams/probe",
            post(handlers::probe_upstreams),
        )
        .route(
            "/api/v1/admin/upstreams/refresh-models",
            post(handlers::refresh_models_upstreams),
        )
        .route("/settings", get(handlers::settings_redirect))
        .with_state(state)
        .layer(DefaultBodyLimit::disable())
}
