//! Embedded web UI handler.

use axum::response::Html;

const INDEX_HTML: &str = include_str!("../static/index.html");
const OPENAPI_JSON: &str = include_str!("../static/openapi.json");
const AI_PLUGIN_JSON: &str = include_str!("../static/ai-plugin.json");

pub async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

pub async fn openapi_handler() -> ([(&'static str, &'static str); 1], &'static str) {
    ([("content-type", "application/json")], OPENAPI_JSON)
}

pub async fn ai_plugin_handler() -> ([(&'static str, &'static str); 1], &'static str) {
    ([("content-type", "application/json")], AI_PLUGIN_JSON)
}
