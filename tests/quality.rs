#[cfg(test)]
mod quality_filter_tests {
    use serde_json::json;

    struct TestQualityFilter {}

    impl TestQualityFilter {
        fn new() -> Self {
            TestQualityFilter {}
        }

        fn validate_tool_calls(&self, request: &serde_json::Value) -> bool {
            let has_tools = request.get("tools").map(|t| !t.is_null()).unwrap_or(false);
            let has_tool_choice = request.get("tool_choice").map(|t| !t.is_null()).unwrap_or(false);
            has_tools || has_tool_choice
        }

        fn detect_conversational_text(&self, response_content: &str) -> bool {
            let content = response_content.trim();
            content.starts_with('"') && !content.contains('{') && !content.contains('}')
        }

        fn apply_quality_filter(&self, is_tool_request: bool, detected_conversational: bool) -> bool {
            is_tool_request && detected_conversational
        }
    }

    #[tokio::test]
    async fn test_tool_call_validation() {
        let filter = TestQualityFilter::new();

        let tool_request = json!({
            "model": "test",
            "messages": [
                {"role": "user", "content": "Get weather"},
                {"role": "assistant", "tool_calls": []}
            ],
            "tools": [{"type": "function", "function": {"name": "test"}}]
        });

        let no_tools_request = json!({
            "model": "test",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });

        assert!(filter.validate_tool_calls(&tool_request));
        assert!(!filter.validate_tool_calls(&no_tools_request));
    }

    #[tokio::test]
    async fn test_conversational_text_detection() {
        let filter = TestQualityFilter::new();

        assert!(!filter.detect_conversational_text(r#"{"choices": []}"#));
        assert!(!filter.detect_conversational_text(r#"{"choices":[{"delta":{"content":"test"}}]}"#));
        assert!(filter.detect_conversational_text(r#""It's sunny today!""#));
        assert!(filter.detect_conversational_text(r#""Hello, how can I help?""#));
        assert!(!filter.detect_conversational_text(r#"{"error": "invalid"}"#));
    }

    #[tokio::test]
    async fn test_quality_filter_decisions() {
        let filter = TestQualityFilter::new();

        assert!(!filter.apply_quality_filter(true, false));
        assert!(!filter.apply_quality_filter(false, true));
        assert!(filter.apply_quality_filter(true, true));
        assert!(!filter.apply_quality_filter(false, false));
    }

    #[tokio::test]
    async fn test_stream_chunk_parsing() {
        let valid_chunk = r#"{"choices": []}"#;
        let invalid_chunk = "not json {{{";

        assert!(serde_json::from_str::<serde_json::Value>(valid_chunk).is_ok());
        assert!(serde_json::from_str::<serde_json::Value>(invalid_chunk).is_err());
    }

    #[tokio::test]
    async fn test_error_handling_in_filter() {
        let empty_request = json!({});
        assert!(!TestQualityFilter::new().validate_tool_calls(&empty_request));

        let malformed = r#"{"invalid json
        "#;
        assert!(serde_json::from_str::<serde_json::Value>(malformed).is_err());
    }
}
