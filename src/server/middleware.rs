//! Authentication middleware and rate limiting.

use std::net::{IpAddr, Ipv6Addr};
use std::num::NonZeroU32;
use std::sync::Arc;

use governor::{Quota, RateLimiter};
use lru::LruCache;
use subtle::ConstantTimeEq;
use tokio::sync::Mutex;

use super::types::{AuthMethod, ServerConfig};

// ---------------------------------------------------------------------------
// Rate limiting
// ---------------------------------------------------------------------------

pub type IpRateLimiter = RateLimiter<
    governor::state::NotKeyed,
    governor::state::InMemoryState,
    governor::clock::DefaultClock,
>;

/// Maximum number of per-IP rate limiter entries to keep in the LRU cache.
const MAX_RATE_LIMITER_ENTRIES: usize = 10_000;

/// Get or create a rate limiter for the given IP.
/// IPv6 addresses are masked to /64 to prevent per-address evasion.
pub async fn get_rate_limiter(
    config: &ServerConfig,
    rate_limiters: &Mutex<LruCache<IpAddr, Arc<IpRateLimiter>>>,
    ip: IpAddr,
) -> Option<Arc<IpRateLimiter>> {
    let rpm = NonZeroU32::new(config.rate_limit_rpm)?;

    // Aggregate IPv6 addresses to /64 prefix
    let key = match ip {
        IpAddr::V4(_) => ip,
        IpAddr::V6(v6) => {
            let seg = v6.segments();
            IpAddr::V6(Ipv6Addr::new(seg[0], seg[1], seg[2], seg[3], 0, 0, 0, 0))
        }
    };

    let mut limiters = rate_limiters.lock().await;

    if let Some(limiter) = limiters.get(&key) {
        return Some(Arc::clone(limiter));
    }

    let quota = Quota::per_minute(rpm);
    let limiter = Arc::new(RateLimiter::direct(quota));
    limiters.push(key, Arc::clone(&limiter));

    Some(limiter)
}

pub fn new_rate_limiter_cache() -> Mutex<LruCache<IpAddr, Arc<IpRateLimiter>>> {
    Mutex::new(LruCache::new(
        std::num::NonZeroUsize::new(MAX_RATE_LIMITER_ENTRIES).unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// Auth middleware
// ---------------------------------------------------------------------------

/// Bearer token authentication middleware.
///
/// When both `api_key` and `pay_to` are configured:
/// - Valid API key → sets `AuthMethod::ApiKey`, passes through (full response)
/// - Invalid API key → 401
/// - No API key → sets `AuthMethod::X402`, passes through (x402 layer handles payment)
///
/// When only `api_key` is configured (no x402):
/// - Valid API key → sets `AuthMethod::ApiKey`, passes through
/// - Invalid or missing API key → 401
///
/// When neither is configured:
/// - All requests pass through with `AuthMethod::Open`
pub async fn auth_middleware(
    axum::extract::State(state): axum::extract::State<Arc<super::ServerState>>,
    mut request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    let x402_enabled = state.config.pay_to.is_some();

    if let Some(ref expected_key) = state.config.api_key {
        let auth_header = request
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        let provided_token = auth_header
            .as_deref()
            .and_then(|h| h.strip_prefix("Bearer "))
            .map(|t| t.trim().to_string());

        match provided_token {
            Some(ref token) if token.as_bytes().ct_eq(expected_key.as_bytes()).into() => {
                request.extensions_mut().insert(AuthMethod::ApiKey);
            }
            Some(_) => {
                return (
                    StatusCode::UNAUTHORIZED,
                    axum::Json(serde_json::json!({
                        "success": false,
                        "error": "Invalid API key"
                    })),
                )
                    .into_response();
            }
            None => {
                if x402_enabled {
                    // No API key but x402 is enabled — let x402 middleware handle payment.
                    // The x402 layer (outer) already verified payment before reaching here.
                    request.extensions_mut().insert(AuthMethod::X402);
                } else {
                    return (
                        StatusCode::UNAUTHORIZED,
                        axum::Json(serde_json::json!({
                            "success": false,
                            "error": "Missing Authorization header. Use: Authorization: Bearer <api_key>"
                        })),
                    )
                        .into_response();
                }
            }
        }
    } else {
        request.extensions_mut().insert(AuthMethod::Open);
    }

    next.run(request).await
}
