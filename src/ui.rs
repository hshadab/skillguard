//! Embedded web UI handler.

use axum::response::Html;

const INDEX_HTML: &str = include_str!("../static/index.html");

pub async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}
