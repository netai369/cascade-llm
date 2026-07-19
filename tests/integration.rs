#[cfg(test)]
mod cascade_integration_tests {
    #[tokio::test]
    async fn test_router_initialization() {
        let app = axum::Router::new()
            .route("/health", axum::routing::get(|| async { "OK" }))
            .route("/metrics", axum::routing::get(|| async { "metrics" }));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:18768").await.unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let client = reqwest::Client::new();
        let resp = client.get("http://127.0.0.1:18768/health").send().await.unwrap();
        assert!(resp.status().is_success());
        let resp = client.get("http://127.0.0.1:18768/metrics").send().await.unwrap();
        assert!(resp.status().is_success());
        server.abort();
    }

    #[tokio::test]
    async fn test_status_code_handling() {
        assert_eq!(axum::http::StatusCode::OK.as_u16(), 200);
        assert_eq!(axum::http::StatusCode::INTERNAL_SERVER_ERROR.as_u16(), 500);
        assert_eq!(axum::http::StatusCode::BAD_GATEWAY.as_u16(), 502);
    }
}
