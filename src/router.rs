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
            "/web/api/routing-strategy",
            put(handlers::set_routing_strategy),
        )
        .route(
            "/web/api/providers",
            post(handlers::add_provider),
        )
        .route(
            "/web/api/providers/:id",
            put(handlers::update_provider).delete(handlers::delete_provider),
        )
        .route(
            "/web/api/providers/:id/test",
            post(handlers::test_provider),
        )
        .route(
            "/api/providers",
            post(handlers::add_provider),
        )
        .route(
            "/api/providers/:id",
            put(handlers::update_provider).delete(handlers::delete_provider),
        )
        .route(
            "/api/providers/:id/test",
            post(handlers::test_provider),
        )
        .route(
            "/extraction/v1/backends",
            post(handlers::register_extraction_backend_handler),
        )
        .route(
            "/extraction/v1/backends/:id",
            delete(handlers::remove_extraction_backend_handler),
        )
        // GPU provisioner passthrough (admin-only, allowlisted inside
        // `provisioner::proxy`). Both the /web and the bare /api mount exist so
        // the dashboard's relative fetch works on either mount — Caddy serves
        // the UI from /web* and /api/v1* straight to the gateway.
        //
        // `status` is a *static* route so it wins over the wildcard below; the
        // wildcard exists because a forwarded path has several segments
        // (`inference-nodes/gpu_…/activate`), which a `:param` cannot match.
        // Usage metering summary (see usage.rs). Admin-only: exposes per-tenant
        // consumption, so it sits behind the same require_admin layer.
        .route("/web/api/v1/admin/usage", get(crate::usage::admin_usage))
        .route("/api/v1/admin/usage", get(crate::usage::admin_usage))
        .route("/web/api/provisioner/status", get(crate::provisioner::status))
        .route("/api/provisioner/status", get(crate::provisioner::status))
        .route(
            "/web/api/provisioner/*tail",
            get(crate::provisioner::proxy)
                .post(crate::provisioner::proxy)
                .put(crate::provisioner::proxy)
                .delete(crate::provisioner::proxy),
        )
        .route(
            "/api/provisioner/*tail",
            get(crate::provisioner::proxy)
                .post(crate::provisioner::proxy)
                .put(crate::provisioner::proxy)
                .delete(crate::provisioner::proxy),
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
        // Anthropic-compatible Messages API (`/v1/messages`) — lets an
        // Anthropic SDK client talk to this gateway unchanged. Added
        // 2026-10-07: it lived only as an untracked file in the cascade-llm
        // working tree, so building from the committed hybrid branch silently
        // removed the route (measured: 404). See docs/anthropic-api.md.
        .route("/v1/messages", post(crate::anthropic::messages))
        .route(
            "/v1/messages/count_tokens",
            post(crate::anthropic::count_tokens),
        )
        .route("/v1/models", get(handlers::list_models))
        .route("/models", get(handlers::list_models))
        .route("/model", get(handlers::get_model))
        // Same payload under the /web/api namespace the console uses. The old
        // dashboard fetched `models` (relative), which only resolved on the
        // /web mount; an /api-namespaced route removes that class of guesswork.
        .route("/web/api/models", get(handlers::list_models))
        .route("/api/models", get(handlers::list_models))
        .route("/web/api/active-models", get(handlers::active_models))
        .route("/health", get(handlers::health_check))
        // Same payload under the /web/api namespace. Caddy exposes the console
        // at /cascade/* and rewrites to /web*, so the root /health is NOT
        // reachable through the proxy at all — the UI has to ask for the
        // namespaced path or every health read is a 404 in production.
        .route("/web/api/health", get(handlers::health_check))
        .route("/api/health", get(handlers::health_check))
        // Audio / media proxies
        .route("/v1/audio/speech", post(handlers::tts))
        .route("/v1/audio/transcriptions", post(handlers::stt))
        // OpenAI-Realtime-compatible STT WebSocket proxy (voice chat + dictation).
        .route("/v1/realtime", get(crate::realtime::realtime_ws))
        // Speaker diarization. Same byte-proxy pattern as STT — `audio_target()`
        // strips the `/v1/audio` prefix generically, so this is one route line
        // rather than a third composition path (gotcha 80).
        //
        // ★ It was MISSING while voice-chat-v2/app/dictation.py:813 already
        // called it, so `STT_URL` could not be pointed at Cascade at all: the
        // diarization request would have 404'd. Found before migrating, which
        // is the only reason it is a one-line addition instead of an incident.
        .route("/v1/audio/diarize", post(handlers::stt_diarize))
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
        .route("/web/api/overview", get(handlers::overview))
        .route("/api/overview", get(handlers::overview))
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
        // Namespaced alias: the extraction pool is a root-level path that
        // Caddy's /cascade/* → /web* rewrite cannot reach, so the console could
        // not display it at all.
        .route(
            "/web/api/extraction-backends",
            get(handlers::list_extraction_backends),
        )
        .route(
            "/api/extraction-backends",
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
        // `/web/api/v1/upstreams/{role}` is the sibling of the v1 inference-node
        // routes above, and it did not exist: the console's Remove button used it
        // for symmetry, answered 404, and left the node in the pool while the
        // table still showed it. A UI that guesses a path gets a silent no-op,
        // so the admin namespace is complete now.
        .route(
            "/web/api/v1/upstreams/:role",
            put(handlers::put_upstream).delete(handlers::delete_upstream),
        )
        .route(
            "/api/v1/upstreams/:role",
            put(handlers::put_upstream).delete(handlers::delete_upstream),
        )
        .route("/settings", get(handlers::settings_redirect))
        .layer(middleware::from_fn_with_state(state.clone(), count_endpoint_hit))
        .with_state(state)
        .layer(DefaultBodyLimit::disable())
}

/// Counts one hit per request on the endpoints listed in
/// `MetricsRegistry::endpoint_path_labels`, for the overview's per-endpoint chart.
async fn count_endpoint_hit(
    axum::extract::State(state): axum::extract::State<Arc<GatewayState>>,
    req: axum::extract::Request,
    next: middleware::Next,
) -> axum::response::Response {
    if let Some(label) = cascade_features::MetricsRegistry::endpoint_path_label(req.uri().path()) {
        state.metrics.record_request(label);
    }
    next.run(req).await
}
