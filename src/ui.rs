//! Web UI handler — serves static files from disk at runtime.
//!
//! Files are read from the `static/` directory relative to the working
//! directory (or `/app/static/` in the Docker container). This avoids
//! embedding them in the binary via `include_str!`, so frontend-only
//! changes don't trigger a Rust recompile.

use axum::http::StatusCode;
use axum::response::{Html, IntoResponse};
use std::path::Path;
use tracing::warn;

/// Directory to look for static files. Checked in order:
/// 1. `./static/` (development / cargo run)
/// 2. `/app/static/` (Docker container)
fn static_dir() -> &'static Path {
    static DIR: std::sync::OnceLock<&'static Path> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        if Path::new("static").is_dir() {
            Path::new("static")
        } else if Path::new("/app/static").is_dir() {
            Path::new("/app/static")
        } else {
            // Fallback — will produce 500s but at least won't panic
            Path::new("static")
        }
    })
}

fn read_static(filename: &str) -> Result<String, StatusCode> {
    let path = static_dir().join(filename);
    std::fs::read_to_string(&path).map_err(|e| {
        warn!(path = %path.display(), error = %e, "failed to read static file");
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

pub async fn index_handler() -> Result<Html<String>, StatusCode> {
    read_static("index.html").map(Html)
}

pub async fn openapi_handler() -> Result<impl IntoResponse, StatusCode> {
    let body = read_static("openapi.json")?;
    Ok(([("content-type", "application/json")], body))
}

pub async fn ai_plugin_handler() -> Result<impl IntoResponse, StatusCode> {
    let body = read_static("ai-plugin.json")?;
    Ok(([("content-type", "application/json")], body))
}
