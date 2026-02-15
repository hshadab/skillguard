//! HTTP server for the SkillGuard ZKML classifier service.
//!
//! Provides REST API endpoints for skill safety evaluation with mandatory
//! ZKML proof generation via Jolt Atlas.
//!
//! Features:
//! - Per-IP rate limiting with automatic eviction when the map exceeds 10k entries
//! - JSONL access logging with size-based rotation (configurable via `max_access_log_bytes`)
//! - ZKML proof generation and verification endpoints
//! - Tiered x402 pricing ($0.001 for classify, $0.005 for classify+prove)
//! - Structured logging via [`tracing`]

pub mod handlers;
pub mod logging;
pub mod middleware;
pub mod types;

// Re-export commonly used items for backward compatibility
pub use handlers::MAX_BODY_BYTES;
pub use logging::{UsageMetrics, METRICS_PERSIST_INTERVAL_SECS};
pub use types::*;

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Instant;

use eyre::Result;
use lru::LruCache;
use subtle::ConstantTimeEq;
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::clawhub::ClawHubClient;

// ---------------------------------------------------------------------------
// Server state
// ---------------------------------------------------------------------------

pub struct ServerState {
    pub config: ServerConfig,
    pub model_hash: String,
    pub start_time: Instant,
    pub rate_limiters: Mutex<LruCache<IpAddr, Arc<middleware::IpRateLimiter>>>,
    pub clawhub_client: ClawHubClient,
    pub usage: UsageMetrics,
    /// Prover initialized lazily in background after server starts listening.
    pub prover: tokio::sync::OnceCell<Arc<crate::prover::ProverState>>,
    pub skip_prover: bool,
    pub proof_cache: crate::cache::ProofCache,
}

impl ServerState {
    pub fn new(config: ServerConfig) -> Self {
        let model_hash = crate::model_hash();
        let usage = UsageMetrics::new(
            &config.access_log_path,
            config.max_access_log_bytes,
            &config.cache_dir,
        );

        let skip_prover = std::env::var("SKILLGUARD_SKIP_PROVER")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        if skip_prover {
            info!("ZKML prover disabled (SKILLGUARD_SKIP_PROVER=1). Prove endpoints unavailable.");
        }

        let proof_cache = crate::cache::ProofCache::open(Some(&config.cache_dir));

        Self {
            config,
            model_hash,
            start_time: Instant::now(),
            rate_limiters: middleware::new_rate_limiter_cache(),
            clawhub_client: ClawHubClient::new(),
            usage,
            prover: tokio::sync::OnceCell::new(),
            skip_prover,
            proof_cache,
        }
    }

    /// Get the prover, returning None if skipped or not yet initialized.
    pub fn get_prover(&self) -> Option<Arc<crate::prover::ProverState>> {
        if self.skip_prover {
            return None;
        }
        self.prover.get().cloned()
    }
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

/// Run the HTTP server (blocking)
pub async fn run_server(config: ServerConfig) -> Result<()> {
    use axum::{
        extract::DefaultBodyLimit,
        middleware as axum_mw,
        routing::{get, post},
        Router,
    };
    use tower_http::cors::{Any, CorsLayer};

    let rate_limit_rpm = config.rate_limit_rpm;
    let bind_addr = config.bind_addr;
    let access_log = config.access_log_path.clone();
    let x402_enabled = config.pay_to.is_some();
    let state = Arc::new(ServerState::new(config));

    // Single evaluate endpoint — auto-detects name lookup vs full skill data.
    // Every classification includes a mandatory ZK proof.
    let api_routes = Router::new()
        .route("/api/v1/evaluate", post(handlers::evaluate_handler))
        .layer(axum_mw::from_fn_with_state(
            state.clone(),
            middleware::auth_middleware,
        ));

    // Wrap API routes in x402 payment layer if pay_to is configured
    let api_routes = if let Some(ref pay_to_addr) = state.config.pay_to {
        use alloy_primitives::Address;
        use x402_axum::X402Middleware;
        use x402_chain_eip155::{KnownNetworkEip155, V1Eip155Exact};
        use x402_types::networks::USDC;

        let mut x402 = X402Middleware::try_from(state.config.facilitator_url.as_str())
            .expect("Failed to init x402 middleware")
            .settle_before_execution();

        // Set base URL so resource URLs use https:// behind TLS-terminating proxies
        if let Some(ref ext_url) = state.config.external_url {
            x402 = x402.with_base_url(ext_url.parse().expect("Invalid SKILLGUARD_EXTERNAL_URL"));
        }

        let pay_to: Address = pay_to_addr
            .parse()
            .expect("Invalid SKILLGUARD_PAY_TO address");

        // USDC price per request in base units (6 decimals).
        // API key users bypass payment.
        let price = state.config.price_usdc_micro;
        let api_key_for_closure = state.config.api_key.clone();
        api_routes.layer(
            x402.with_dynamic_price(move |headers, _uri, _base_url| {
                let has_valid_api_key = if let Some(ref expected_key) = api_key_for_closure {
                    headers
                        .get("authorization")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|h| h.strip_prefix("Bearer "))
                        .map(|t| bool::from(t.trim().as_bytes().ct_eq(expected_key.as_bytes())))
                        .unwrap_or(false)
                } else {
                    false
                };

                async move {
                    if has_valid_api_key {
                        vec![]
                    } else {
                        vec![V1Eip155Exact::price_tag(pay_to, USDC::base().amount(price))]
                    }
                }
            })
            .with_description(
                "SkillGuard ZKML — verifiable AI skill safety classification".to_string(),
            ),
        )
    } else {
        api_routes
    };

    // Log x402 settlement details on responses
    let api_routes = api_routes.layer(axum_mw::from_fn(middleware::x402_settlement_logger));

    // CORS layer — allow any origin for public API access from browsers
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
        .expose_headers([
            "Payment-Required"
                .parse::<axum::http::HeaderName>()
                .unwrap(),
            "X-Payment-Response"
                .parse::<axum::http::HeaderName>()
                .unwrap(),
            "X-Proof-Count".parse::<axum::http::HeaderName>().unwrap(),
        ]);

    // Public routes (no auth required)
    let proof_count_state = state.clone();
    let app = Router::new()
        .route("/", get(crate::ui::index_handler))
        .route("/health", get(handlers::health_handler))
        .route("/stats", get(handlers::stats_handler))
        .route("/openapi.json", get(crate::ui::openapi_handler))
        .route(
            "/.well-known/ai-plugin.json",
            get(crate::ui::ai_plugin_handler),
        )
        .route("/.well-known/llms.txt", get(crate::ui::llms_txt_handler))
        .route("/api/v1/verify", post(handlers::verify_handler))
        .merge(api_routes)
        .layer(axum_mw::from_fn(
            move |req, next: axum::middleware::Next| {
                let s = proof_count_state.clone();
                async move {
                    let mut resp = next.run(req).await;
                    let count = s
                        .usage
                        .total_proofs_generated
                        .load(std::sync::atomic::Ordering::Relaxed);
                    if let Ok(val) = axum::http::HeaderValue::from_str(&count.to_string()) {
                        resp.headers_mut().insert("X-Proof-Count", val);
                    }
                    resp
                }
            },
        ))
        .layer(cors)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state.clone());

    // Spawn background task to persist metrics to disk periodically
    let metrics_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(
            METRICS_PERSIST_INTERVAL_SECS,
        ));
        loop {
            interval.tick().await;
            metrics_state.usage.persist_to_disk();
        }
    });

    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    info!(bind = %bind_addr, "SkillGuard ZKML server listening");
    info!("Endpoints: GET / (UI), GET /health, GET /stats, GET /openapi.json, GET /.well-known/ai-plugin.json, GET /.well-known/llms.txt, POST /api/v1/evaluate, POST /api/v1/verify");
    if rate_limit_rpm > 0 {
        info!(rate_limit_rpm, "rate limiting enabled");
    } else {
        info!("rate limiting disabled");
    }
    if x402_enabled {
        info!("x402 payment enabled ($0.001 USDC per request on Base, proofs included)");
    }
    info!(access_log = %access_log);

    // Initialize ZKML prover in background so the server can start serving health checks
    // immediately. Without this, prover init (5-15s) blocks port binding and causes
    // Render to kill the app with "Application exited early".
    if !state.skip_prover {
        let prover_state = state.clone();
        tokio::spawn(async move {
            info!("Starting ZKML prover initialization in background...");
            let result = tokio::task::spawn_blocking(|| {
                std::panic::catch_unwind(crate::prover::ProverState::initialize)
            })
            .await;

            match result {
                Ok(Ok(Ok(p))) => {
                    let _ = prover_state.prover.set(Arc::new(p));
                    info!("ZKML prover ready (Jolt/Dory)");
                }
                Ok(Ok(Err(e))) => {
                    warn!(
                        "ZKML prover initialization failed: {}. Evaluate endpoint will return errors until prover is available.",
                        e
                    );
                }
                Ok(Err(_)) => {
                    warn!("ZKML prover initialization panicked. Evaluate endpoint will return errors until prover is available.");
                }
                Err(e) => {
                    warn!(
                        "ZKML prover init task failed: {}. Evaluate endpoint will return errors until prover is available.",
                        e
                    );
                }
            }
        });
    }

    // Graceful shutdown on SIGTERM/SIGINT
    let shutdown_state = state;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler");
        #[cfg(unix)]
        let sigterm_recv = sigterm.recv();
        #[cfg(not(unix))]
        let sigterm_recv = std::future::pending::<Option<()>>();

        tokio::select! {
            _ = ctrl_c => info!("received SIGINT, shutting down gracefully"),
            _ = sigterm_recv => info!("received SIGTERM, shutting down gracefully"),
        }

        // Persist metrics before exiting
        shutdown_state.usage.persist_to_disk();
    })
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill::{SafetyClassification, SafetyDecision};
    use std::sync::atomic::Ordering;

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "ok".to_string(),
            version: "0.1.0".to_string(),
            model_hash: "sha256:abc".to_string(),
            uptime_seconds: 100,
            zkml_enabled: true,
            proving_scheme: "Jolt/Dory".to_string(),
            cache_writable: true,
            pay_to: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
        assert!(json.contains("\"zkml_enabled\":true"));
    }

    #[test]
    fn test_stats_response_serialization() {
        let response = StatsResponse {
            uptime_seconds: 3600,
            model_hash: "sha256:abc".to_string(),
            requests: RequestStats {
                total: 100,
                errors: 2,
            },
            classifications: ClassificationStats {
                safe: 80,
                caution: 10,
                dangerous: 7,
                malicious: 3,
            },
            decisions: DecisionStats {
                allow: 90,
                deny: 8,
                flag: 2,
            },
            endpoints: EndpointStats {
                evaluate: 60,
                evaluate_by_name: 35,
                verify: 5,
                stats: 5,
            },
            proofs: ProofStats {
                total_generated: 10,
                total_verified: 5,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"total\":100"));
        assert!(json.contains("\"evaluate_by_name\":35"));
        assert!(json.contains("\"total_generated\":10"));
    }

    #[test]
    fn test_evaluate_by_name_request_deserialization() {
        let json = r#"{"skill": "weather-helper", "version": "1.0.0"}"#;
        let req: EvaluateByNameRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.skill, "weather-helper");
        assert_eq!(req.version, Some("1.0.0".to_string()));
    }

    #[test]
    fn test_evaluate_by_name_request_minimal() {
        let json = r#"{"skill": "my-skill"}"#;
        let req: EvaluateByNameRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.skill, "my-skill");
        assert!(req.version.is_none());
    }

    #[test]
    fn test_usage_metrics_counters() {
        let metrics = UsageMetrics::new("/dev/null", 0, "/tmp");
        metrics.record(
            "evaluate",
            "test-skill",
            SafetyClassification::Safe,
            SafetyDecision::Allow,
            0.9,
            42,
        );
        assert_eq!(metrics.total_requests.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.safe.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.allow.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.total_errors.load(Ordering::Relaxed), 0);

        metrics.record_error();
        assert_eq!(metrics.total_requests.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.total_errors.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_server_config_default_has_no_api_key() {
        let config = ServerConfig::default();
        assert!(config.api_key.is_none());
        assert!(config.pay_to.is_none());
    }

    #[test]
    fn test_server_config_with_api_key() {
        let config = ServerConfig {
            api_key: Some("secret-key-123".to_string()),
            ..Default::default()
        };
        assert_eq!(config.api_key.as_deref(), Some("secret-key-123"));
    }

    #[test]
    fn test_server_config_with_x402() {
        let config = ServerConfig {
            pay_to: Some("0xBAc675C310721717Cd4A37F6cbeA1F081b1C2a07".to_string()),
            ..Default::default()
        };
        assert!(config.pay_to.is_some());
        assert_eq!(config.facilitator_url, "https://pay.openfacilitator.io");
    }
}
