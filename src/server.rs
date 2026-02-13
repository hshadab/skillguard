//! HTTP server for the SkillGuard ZKML classifier service.
//!
//! Provides REST API endpoints for skill safety evaluation with optional
//! ZKML proof generation via Jolt Atlas.
//!
//! Features:
//! - Per-IP rate limiting with automatic eviction when the map exceeds 10k entries
//! - JSONL access logging with size-based rotation (configurable via `max_access_log_bytes`)
//! - ZKML proof generation and verification endpoints
//! - Tiered x402 pricing ($0.001 for classify, $0.005 for classify+prove)
//! - Structured logging via [`tracing`]

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use eyre::Result;
use governor::{Quota, RateLimiter};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::clawhub::ClawHubClient;
use crate::scores::ClassScores;
use crate::skill::{
    derive_decision, SafetyClassification, SafetyDecision, Skill, SkillFeatures, VTReport,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to bind to (defaults to 127.0.0.1:8080; use 0.0.0.0 to expose externally)
    pub bind_addr: SocketAddr,
    /// Rate limit in requests per minute per IP (0 = no limit)
    pub rate_limit_rpm: u32,
    /// Path for JSONL access log
    pub access_log_path: String,
    /// Maximum access log file size in bytes before rotation (0 = no limit)
    pub max_access_log_bytes: u64,
    /// Optional API key for bearer token authentication on /api/v1/* endpoints.
    /// If None, auth is disabled (backward compatible).
    pub api_key: Option<String>,
    /// Wallet address to receive x402 USDC payments on Base.
    /// If None, x402 payment is disabled (API key only).
    pub pay_to: Option<String>,
    /// x402 facilitator URL (defaults to CDP mainnet facilitator).
    pub facilitator_url: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:8080"
                .parse()
                .expect("valid default bind address"),
            rate_limit_rpm: 60,
            access_log_path: "skillguard-access.jsonl".to_string(),
            max_access_log_bytes: 50 * 1024 * 1024, // 50 MB
            api_key: None,
            pay_to: None,
            facilitator_url: "https://api.cdp.coinbase.com/platform/v2/x402".to_string(),
        }
    }
}

/// How the request was authenticated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMethod {
    /// Authenticated via API key — full response.
    ApiKey,
    /// Authenticated via x402 payment — basic response only.
    X402,
    /// No authentication configured (server has no api_key and no pay_to).
    Open,
}

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

/// Request for skill safety evaluation (full skill data)
#[derive(Debug, Deserialize)]
pub struct EvaluateRequest {
    /// The skill to evaluate
    pub skill: Skill,
    /// Optional VirusTotal report
    #[serde(default)]
    pub vt_report: Option<VTReport>,
}

/// Request for evaluate-by-name endpoint
#[derive(Debug, Deserialize)]
pub struct EvaluateByNameRequest {
    /// Skill name (slug) on ClawHub
    pub skill: String,
    /// Optional version (defaults to latest)
    #[serde(default)]
    pub version: Option<String>,
}

/// Flat evaluation result (no receipts, no proofs)
#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub skill_name: String,
    pub classification: String,
    pub decision: String,
    pub confidence: f64,
    pub scores: ClassScores,
    pub reasoning: String,
}

/// Stripped-down evaluation result for x402 payers.
#[derive(Debug, Serialize)]
pub struct BasicEvaluationResult {
    pub skill_name: String,
    pub classification: String,
    pub decision: String,
}

/// Full evaluation result with ZK proof.
#[derive(Debug, Serialize, Deserialize)]
pub struct ProvedEvaluationResult {
    pub skill_name: String,
    pub classification: String,
    pub decision: String,
    pub confidence: f64,
    pub scores: ClassScores,
    pub reasoning: String,
    pub proof: crate::prover::ProofBundle,
}

/// Request to verify a proof.
#[derive(Debug, Deserialize)]
pub struct VerifyRequest {
    pub proof_b64: String,
    pub program_io: serde_json::Value,
}

/// Response from proof verification.
#[derive(Debug, Serialize)]
pub struct VerifyResponse {
    pub valid: bool,
    pub verification_time_ms: u64,
}

/// Response from the prove+evaluate endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub struct ProveEvaluateResponse {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<ProvedEvaluationResult>,
    pub processing_time_ms: u64,
}

/// Top-level response
#[derive(Debug, Serialize)]
pub struct EvaluateResponse {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<EvaluationResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub basic_evaluation: Option<BasicEvaluationResult>,
    pub processing_time_ms: u64,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub model_hash: String,
    pub uptime_seconds: u64,
    pub zkml_enabled: bool,
    pub proving_scheme: String,
}

/// Stats response
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub uptime_seconds: u64,
    pub model_hash: String,
    pub requests: RequestStats,
    pub classifications: ClassificationStats,
    pub decisions: DecisionStats,
    pub endpoints: EndpointStats,
    pub proofs: ProofStats,
}

#[derive(Debug, Serialize)]
pub struct RequestStats {
    pub total: u64,
    pub errors: u64,
}

#[derive(Debug, Serialize)]
pub struct ClassificationStats {
    pub safe: u64,
    pub caution: u64,
    pub dangerous: u64,
    pub malicious: u64,
}

#[derive(Debug, Serialize)]
pub struct DecisionStats {
    pub allow: u64,
    pub deny: u64,
    pub flag: u64,
}

#[derive(Debug, Serialize)]
pub struct EndpointStats {
    pub evaluate: u64,
    pub evaluate_by_name: u64,
    pub prove: u64,
    pub verify: u64,
    pub stats: u64,
}

#[derive(Debug, Serialize)]
pub struct ProofStats {
    pub total_generated: u64,
    pub total_verified: u64,
}

// ---------------------------------------------------------------------------
// Usage metrics
// ---------------------------------------------------------------------------

pub struct UsageMetrics {
    pub total_requests: AtomicU64,
    pub total_errors: AtomicU64,

    pub safe: AtomicU64,
    pub caution: AtomicU64,
    pub dangerous: AtomicU64,
    pub malicious: AtomicU64,

    pub allow: AtomicU64,
    pub deny: AtomicU64,
    pub flag: AtomicU64,

    pub ep_evaluate: AtomicU64,
    pub ep_evaluate_by_name: AtomicU64,
    pub ep_prove: AtomicU64,
    pub ep_verify: AtomicU64,
    pub ep_stats: AtomicU64,

    pub total_proofs_generated: AtomicU64,
    pub total_proofs_verified: AtomicU64,

    pub access_log: std::sync::Mutex<Option<File>>,
    access_log_path: String,
    access_log_bytes: AtomicU64,
    max_access_log_bytes: u64,
}

impl UsageMetrics {
    fn new(access_log_path: &str, max_access_log_bytes: u64) -> Self {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(access_log_path)
            .ok();
        if file.is_none() {
            warn!(path = access_log_path, "could not open access log");
        }
        let current_size = std::fs::metadata(access_log_path)
            .map(|m| m.len())
            .unwrap_or(0);
        Self {
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            safe: AtomicU64::new(0),
            caution: AtomicU64::new(0),
            dangerous: AtomicU64::new(0),
            malicious: AtomicU64::new(0),
            allow: AtomicU64::new(0),
            deny: AtomicU64::new(0),
            flag: AtomicU64::new(0),
            ep_evaluate: AtomicU64::new(0),
            ep_evaluate_by_name: AtomicU64::new(0),
            ep_prove: AtomicU64::new(0),
            ep_verify: AtomicU64::new(0),
            ep_stats: AtomicU64::new(0),
            total_proofs_generated: AtomicU64::new(0),
            total_proofs_verified: AtomicU64::new(0),
            access_log: std::sync::Mutex::new(file),
            access_log_path: access_log_path.to_string(),
            access_log_bytes: AtomicU64::new(current_size),
            max_access_log_bytes,
        }
    }

    fn record(
        &self,
        endpoint: &str,
        skill_name: &str,
        classification: SafetyClassification,
        decision: SafetyDecision,
        confidence: f64,
        processing_time_ms: u64,
    ) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        match classification {
            SafetyClassification::Safe => {
                self.safe.fetch_add(1, Ordering::Relaxed);
            }
            SafetyClassification::Caution => {
                self.caution.fetch_add(1, Ordering::Relaxed);
            }
            SafetyClassification::Dangerous => {
                self.dangerous.fetch_add(1, Ordering::Relaxed);
            }
            SafetyClassification::Malicious => {
                self.malicious.fetch_add(1, Ordering::Relaxed);
            }
        }

        match decision {
            SafetyDecision::Allow => {
                self.allow.fetch_add(1, Ordering::Relaxed);
            }
            SafetyDecision::Deny => {
                self.deny.fetch_add(1, Ordering::Relaxed);
            }
            SafetyDecision::Flag => {
                self.flag.fetch_add(1, Ordering::Relaxed);
            }
        }

        if let Ok(mut guard) = self.access_log.try_lock() {
            if let Some(ref mut file) = *guard {
                let entry = serde_json::json!({
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "endpoint": endpoint,
                    "skill_name": skill_name,
                    "classification": classification.as_str(),
                    "decision": decision.as_str(),
                    "confidence": confidence,
                    "processing_time_ms": processing_time_ms,
                });
                let mut line = entry.to_string();
                line.push('\n');
                let line_len = line.len() as u64;
                let _ = file.write_all(line.as_bytes());
                let new_size =
                    self.access_log_bytes.fetch_add(line_len, Ordering::Relaxed) + line_len;

                // Rotate if over size limit (0 = no limit)
                if self.max_access_log_bytes > 0 && new_size >= self.max_access_log_bytes {
                    let rotated = format!("{}.1", self.access_log_path);
                    let _ = std::fs::rename(&self.access_log_path, &rotated);
                    if let Ok(new_file) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&self.access_log_path)
                    {
                        *file = new_file;
                        self.access_log_bytes.store(0, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    fn record_error(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_errors.fetch_add(1, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Server state
// ---------------------------------------------------------------------------

type IpRateLimiter = RateLimiter<
    governor::state::NotKeyed,
    governor::state::InMemoryState,
    governor::clock::DefaultClock,
>;

pub struct ServerState {
    pub config: ServerConfig,
    pub model_hash: String,
    pub start_time: Instant,
    pub rate_limiters: Mutex<HashMap<std::net::IpAddr, Arc<IpRateLimiter>>>,
    pub clawhub_client: ClawHubClient,
    pub usage: UsageMetrics,
    pub prover: Option<Arc<crate::prover::ProverState>>,
    pub proof_cache: crate::cache::ProofCache,
}

impl ServerState {
    pub fn new(config: ServerConfig) -> Self {
        let model_hash = crate::model_hash();
        let usage = UsageMetrics::new(&config.access_log_path, config.max_access_log_bytes);

        let skip_prover = std::env::var("SKILLGUARD_SKIP_PROVER")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let prover = if skip_prover {
            info!("ZKML prover disabled (SKILLGUARD_SKIP_PROVER=1). Prove endpoints unavailable.");
            None
        } else {
            match crate::prover::ProverState::initialize() {
                Ok(p) => {
                    info!("ZKML prover ready (Jolt/HyperKZG)");
                    Some(Arc::new(p))
                }
                Err(e) => {
                    warn!(
                        "ZKML prover initialization failed: {}. Prove endpoints disabled.",
                        e
                    );
                    None
                }
            }
        };

        let proof_cache = crate::cache::ProofCache::open(None);

        Self {
            config,
            model_hash,
            start_time: Instant::now(),
            rate_limiters: Mutex::new(HashMap::new()),
            clawhub_client: ClawHubClient::new(),
            usage,
            prover,
            proof_cache,
        }
    }

    pub async fn get_rate_limiter(&self, ip: std::net::IpAddr) -> Option<Arc<IpRateLimiter>> {
        let rpm = NonZeroU32::new(self.config.rate_limit_rpm)?;

        let mut limiters = self.rate_limiters.lock().await;

        if let Some(limiter) = limiters.get(&ip) {
            return Some(Arc::clone(limiter));
        }

        let quota = Quota::per_minute(rpm);
        let limiter = Arc::new(RateLimiter::direct(quota));
        limiters.insert(ip, Arc::clone(&limiter));

        // Evict oldest entries when the map grows too large (LRU-style trim)
        const MAX_ENTRIES: usize = 10_000;
        if limiters.len() > MAX_ENTRIES {
            // Remove a batch of the oldest entries to avoid frequent evictions
            let to_remove = limiters.len() - MAX_ENTRIES / 2;
            let keys_to_remove: Vec<_> = limiters
                .keys()
                .filter(|k| **k != ip)
                .take(to_remove)
                .cloned()
                .collect();
            for key in keys_to_remove {
                limiters.remove(&key);
            }
        }

        Some(limiter)
    }
}

// ---------------------------------------------------------------------------
// Shared classification logic
// ---------------------------------------------------------------------------

fn classify_and_respond(
    state: &Arc<ServerState>,
    features: SkillFeatures,
    skill_name: String,
    start: Instant,
    endpoint: &str,
    auth_method: AuthMethod,
) -> EvaluateResponse {
    let feature_vec = features.to_normalized_vec();

    let result = match crate::classify(&feature_vec) {
        Ok(r) => r,
        Err(e) => {
            state.usage.record_error();
            return EvaluateResponse {
                success: false,
                error: Some(format!("Classification failed: {}", e)),
                evaluation: None,
                basic_evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            };
        }
    };

    let (classification, raw_scores, confidence) = result;
    let scores = ClassScores::from_raw_scores(&raw_scores);
    let (decision, reasoning) = derive_decision(classification, &scores.to_array());

    let processing_time_ms = start.elapsed().as_millis() as u64;

    state.usage.record(
        endpoint,
        &skill_name,
        classification,
        decision,
        confidence,
        processing_time_ms,
    );

    match auth_method {
        AuthMethod::X402 => EvaluateResponse {
            success: true,
            error: None,
            evaluation: None,
            basic_evaluation: Some(BasicEvaluationResult {
                skill_name,
                classification: classification.as_str().to_string(),
                decision: decision.as_str().to_string(),
            }),
            processing_time_ms,
        },
        _ => EvaluateResponse {
            success: true,
            error: None,
            evaluation: Some(EvaluationResult {
                skill_name,
                classification: classification.as_str().to_string(),
                decision: decision.as_str().to_string(),
                confidence,
                scores,
                reasoning,
            }),
            basic_evaluation: None,
            processing_time_ms,
        },
    }
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

/// Run the HTTP server (blocking)
pub async fn run_server(config: ServerConfig) -> Result<()> {
    use axum::{
        middleware,
        routing::{get, post},
        Router,
    };

    let rate_limit_rpm = config.rate_limit_rpm;
    let bind_addr = config.bind_addr;
    let access_log = config.access_log_path.clone();
    let x402_enabled = config.pay_to.is_some();
    let state = Arc::new(ServerState::new(config));

    // Build API routes with auth middleware, optionally wrapped in x402 layer
    let api_routes = Router::new()
        .route("/api/v1/evaluate", post(evaluate_handler))
        .route("/api/v1/evaluate/name", post(evaluate_by_name_handler))
        .route("/api/v1/evaluate/prove", post(prove_evaluate_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    // Wrap API routes in x402 payment layer if pay_to is configured
    let api_routes = if let Some(ref pay_to_addr) = state.config.pay_to {
        use alloy_primitives::Address;
        use x402_axum::X402Middleware;
        use x402_chain_eip155::{KnownNetworkEip155, V2Eip155Exact};
        use x402_types::networks::USDC;

        let x402 = X402Middleware::try_from(state.config.facilitator_url.as_str())
            .expect("Failed to init x402 middleware")
            .settle_before_execution();

        let pay_to: Address = pay_to_addr
            .parse()
            .expect("Invalid SKILLGUARD_PAY_TO address");

        // Use dynamic pricing: bypass payment (empty vec) when a valid API key is present,
        // otherwise require $0.001 USDC for evaluate, $0.005 for prove.
        let api_key_for_closure = state.config.api_key.clone();
        api_routes.layer(
            x402.with_dynamic_price(move |headers, uri, _base_url| {
                let has_valid_api_key = if let Some(ref expected_key) = api_key_for_closure {
                    headers
                        .get("authorization")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|h| h.strip_prefix("Bearer "))
                        .map(|t| t.trim() == expected_key)
                        .unwrap_or(false)
                } else {
                    false
                };

                let is_prove = uri.path().contains("/prove");

                async move {
                    if has_valid_api_key {
                        // API key user: bypass x402 payment
                        vec![]
                    } else if is_prove {
                        // Prove endpoint: $0.005 USDC = 5000 base units (6 decimals)
                        vec![V2Eip155Exact::price_tag(
                            pay_to,
                            USDC::base().amount(5000u64),
                        )]
                    } else {
                        // Evaluate endpoint: $0.001 USDC = 1000 base units (6 decimals)
                        vec![V2Eip155Exact::price_tag(
                            pay_to,
                            USDC::base().amount(1000u64),
                        )]
                    }
                }
            })
            .with_description(
                "SkillGuard ZKML — AI skill safety classification and proving".to_string(),
            ),
        )
    } else {
        api_routes
    };

    // Public routes (no auth required)
    let app = Router::new()
        .route("/", get(crate::ui::index_handler))
        .route("/health", get(health_handler))
        .route("/stats", get(stats_handler))
        .route("/openapi.json", get(crate::ui::openapi_handler))
        .route(
            "/.well-known/ai-plugin.json",
            get(crate::ui::ai_plugin_handler),
        )
        .route("/api/v1/verify", post(verify_handler))
        .merge(api_routes)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    info!(bind = %bind_addr, "SkillGuard ZKML server listening");
    info!("Endpoints: GET / (UI), GET /health, GET /stats, GET /openapi.json, GET /.well-known/ai-plugin.json, POST /api/v1/evaluate, POST /api/v1/evaluate/name, POST /api/v1/evaluate/prove, POST /api/v1/verify");
    if rate_limit_rpm > 0 {
        info!(rate_limit_rpm, "rate limiting enabled");
    } else {
        info!("rate limiting disabled");
    }
    if x402_enabled {
        info!("x402 payment enabled ($0.001 USDC per evaluation, $0.005 per proof on Base)");
    }
    info!(access_log = %access_log);

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

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
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
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
            Some(ref token) if token == expected_key => {
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

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
) -> impl axum::response::IntoResponse {
    let response = HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        model_hash: state.model_hash.clone(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        zkml_enabled: state.prover.is_some(),
        proving_scheme: "Jolt/HyperKZG".to_string(),
    };
    axum::Json(response)
}

async fn evaluate_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    request: axum::extract::Request,
) -> impl axum::response::IntoResponse {
    let start = Instant::now();
    let client_ip = addr.ip();
    let auth_method = request
        .extensions()
        .get::<AuthMethod>()
        .copied()
        .unwrap_or(AuthMethod::Open);

    let body = request.into_body();
    let bytes = match axum::body::to_bytes(body, 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return axum::Json(EvaluateResponse {
                success: false,
                error: Some(format!("Failed to read request body: {}", e)),
                evaluation: None,
                basic_evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    };
    let eval_request: EvaluateRequest = match serde_json::from_slice(&bytes) {
        Ok(r) => r,
        Err(e) => {
            return axum::Json(EvaluateResponse {
                success: false,
                error: Some(format!("Invalid JSON: {}", e)),
                evaluation: None,
                basic_evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    };

    state.usage.ep_evaluate.fetch_add(1, Ordering::Relaxed);

    if let Some(limiter) = state.get_rate_limiter(client_ip).await {
        if limiter.check().is_err() {
            state.usage.record_error();
            return axum::Json(EvaluateResponse {
                success: false,
                error: Some(format!(
                    "Rate limit exceeded. Maximum {} requests per minute.",
                    state.config.rate_limit_rpm
                )),
                evaluation: None,
                basic_evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    }

    let features = SkillFeatures::extract(&eval_request.skill, eval_request.vt_report.as_ref());
    let skill_name = eval_request.skill.name.clone();

    let response =
        classify_and_respond(&state, features, skill_name, start, "evaluate", auth_method);

    axum::Json(response)
}

async fn evaluate_by_name_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    request: axum::extract::Request,
) -> impl axum::response::IntoResponse {
    let start = Instant::now();
    let client_ip = addr.ip();
    let auth_method = request
        .extensions()
        .get::<AuthMethod>()
        .copied()
        .unwrap_or(AuthMethod::Open);

    let body = request.into_body();
    let bytes = match axum::body::to_bytes(body, 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return axum::Json(EvaluateResponse {
                success: false,
                error: Some(format!("Failed to read request body: {}", e)),
                evaluation: None,
                basic_evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    };
    let eval_request: EvaluateByNameRequest = match serde_json::from_slice(&bytes) {
        Ok(r) => r,
        Err(e) => {
            return axum::Json(EvaluateResponse {
                success: false,
                error: Some(format!("Invalid JSON: {}", e)),
                evaluation: None,
                basic_evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    };

    state
        .usage
        .ep_evaluate_by_name
        .fetch_add(1, Ordering::Relaxed);

    if let Some(limiter) = state.get_rate_limiter(client_ip).await {
        if limiter.check().is_err() {
            state.usage.record_error();
            return axum::Json(EvaluateResponse {
                success: false,
                error: Some(format!(
                    "Rate limit exceeded. Maximum {} requests per minute.",
                    state.config.rate_limit_rpm
                )),
                evaluation: None,
                basic_evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    }

    let skill = match state
        .clawhub_client
        .fetch_skill(&eval_request.skill, eval_request.version.as_deref())
        .await
    {
        Ok(s) => s,
        Err(e) => {
            state.usage.record_error();
            return axum::Json(EvaluateResponse {
                success: false,
                error: Some(format!(
                    "Failed to fetch skill '{}': {}",
                    eval_request.skill, e
                )),
                evaluation: None,
                basic_evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    };

    let features = SkillFeatures::extract(&skill, None);
    let skill_name = skill.name;

    let response = classify_and_respond(
        &state,
        features,
        skill_name,
        start,
        "evaluate_by_name",
        auth_method,
    );

    axum::Json(response)
}

async fn prove_evaluate_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    request: axum::extract::Request,
) -> impl axum::response::IntoResponse {
    let start = Instant::now();
    let client_ip = addr.ip();
    let _auth_method = request
        .extensions()
        .get::<AuthMethod>()
        .copied()
        .unwrap_or(AuthMethod::Open);

    let body = request.into_body();
    let bytes = match axum::body::to_bytes(body, 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return axum::Json(ProveEvaluateResponse {
                success: false,
                error: Some(format!("Failed to read request body: {}", e)),
                evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    };
    let eval_request: EvaluateRequest = match serde_json::from_slice(&bytes) {
        Ok(r) => r,
        Err(e) => {
            return axum::Json(ProveEvaluateResponse {
                success: false,
                error: Some(format!("Invalid JSON: {}", e)),
                evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    };

    state.usage.ep_prove.fetch_add(1, Ordering::Relaxed);

    if let Some(limiter) = state.get_rate_limiter(client_ip).await {
        if limiter.check().is_err() {
            state.usage.record_error();
            return axum::Json(ProveEvaluateResponse {
                success: false,
                error: Some(format!(
                    "Rate limit exceeded. Maximum {} requests per minute.",
                    state.config.rate_limit_rpm
                )),
                evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    }

    let prover = match &state.prover {
        Some(p) => p.clone(),
        None => {
            state.usage.record_error();
            return axum::Json(ProveEvaluateResponse {
                success: false,
                error: Some("ZKML prover not available".to_string()),
                evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    };

    // Check proof cache before doing expensive proving
    let cache_key = crate::cache::ProofCache::cache_key(&bytes, &state.model_hash);
    if let Some(cached) = state.proof_cache.get(&cache_key) {
        // Still record usage metrics for cached responses
        if let Some(ref eval) = cached.evaluation {
            let cls = crate::skill::SafetyClassification::parse_str(&eval.classification);
            let dec = crate::skill::SafetyDecision::parse_str(&eval.decision);
            state
                .usage
                .record("prove", &eval.skill_name, cls, dec, eval.confidence, 0);
        }
        return axum::Json(cached);
    }

    let features = SkillFeatures::extract(&eval_request.skill, eval_request.vt_report.as_ref());
    let feature_vec = features.to_normalized_vec();
    let skill_name = eval_request.skill.name.clone();

    let result = match crate::classify_with_proof(&prover, &feature_vec) {
        Ok(r) => r,
        Err(e) => {
            state.usage.record_error();
            return axum::Json(ProveEvaluateResponse {
                success: false,
                error: Some(format!("Classification with proof failed: {}", e)),
                evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    };

    let (classification, raw_scores, confidence, proof_bundle) = result;
    let scores = ClassScores::from_raw_scores(&raw_scores);
    let (decision, reasoning) = derive_decision(classification, &scores.to_array());

    let processing_time_ms = start.elapsed().as_millis() as u64;

    state.usage.record(
        "prove",
        &skill_name,
        classification,
        decision,
        confidence,
        processing_time_ms,
    );
    state
        .usage
        .total_proofs_generated
        .fetch_add(1, Ordering::Relaxed);

    let response = ProveEvaluateResponse {
        success: true,
        error: None,
        evaluation: Some(ProvedEvaluationResult {
            skill_name,
            classification: classification.as_str().to_string(),
            decision: decision.as_str().to_string(),
            confidence,
            scores,
            reasoning,
            proof: proof_bundle,
        }),
        processing_time_ms,
    };

    // Cache the successful proof response
    state.proof_cache.put(&cache_key, &response);

    axum::Json(response)
}

async fn verify_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
    axum::Json(req): axum::Json<VerifyRequest>,
) -> impl axum::response::IntoResponse {
    let start = Instant::now();

    state.usage.ep_verify.fetch_add(1, Ordering::Relaxed);

    let prover = match &state.prover {
        Some(p) => p.clone(),
        None => {
            return axum::Json(serde_json::json!({
                "error": "ZKML prover not available",
                "valid": false,
                "verification_time_ms": start.elapsed().as_millis() as u64,
            }));
        }
    };

    let bundle = crate::prover::ProofBundle {
        proof_b64: req.proof_b64,
        program_io: req.program_io,
        proof_size_bytes: 0,
        proving_time_ms: 0,
    };

    let valid: bool = prover.verify_proof(&bundle).unwrap_or_default();

    state
        .usage
        .total_proofs_verified
        .fetch_add(1, Ordering::Relaxed);

    let verification_time_ms = start.elapsed().as_millis() as u64;

    axum::Json(serde_json::json!({
        "valid": valid,
        "verification_time_ms": verification_time_ms,
    }))
}

async fn stats_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
) -> impl axum::response::IntoResponse {
    state.usage.ep_stats.fetch_add(1, Ordering::Relaxed);

    let response = StatsResponse {
        uptime_seconds: state.start_time.elapsed().as_secs(),
        model_hash: state.model_hash.clone(),
        requests: RequestStats {
            total: state.usage.total_requests.load(Ordering::Relaxed),
            errors: state.usage.total_errors.load(Ordering::Relaxed),
        },
        classifications: ClassificationStats {
            safe: state.usage.safe.load(Ordering::Relaxed),
            caution: state.usage.caution.load(Ordering::Relaxed),
            dangerous: state.usage.dangerous.load(Ordering::Relaxed),
            malicious: state.usage.malicious.load(Ordering::Relaxed),
        },
        decisions: DecisionStats {
            allow: state.usage.allow.load(Ordering::Relaxed),
            deny: state.usage.deny.load(Ordering::Relaxed),
            flag: state.usage.flag.load(Ordering::Relaxed),
        },
        endpoints: EndpointStats {
            evaluate: state.usage.ep_evaluate.load(Ordering::Relaxed),
            evaluate_by_name: state.usage.ep_evaluate_by_name.load(Ordering::Relaxed),
            prove: state.usage.ep_prove.load(Ordering::Relaxed),
            verify: state.usage.ep_verify.load(Ordering::Relaxed),
            stats: state.usage.ep_stats.load(Ordering::Relaxed),
        },
        proofs: ProofStats {
            total_generated: state.usage.total_proofs_generated.load(Ordering::Relaxed),
            total_verified: state.usage.total_proofs_verified.load(Ordering::Relaxed),
        },
    };
    axum::Json(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "ok".to_string(),
            version: "0.1.0".to_string(),
            model_hash: "sha256:abc".to_string(),
            uptime_seconds: 100,
            zkml_enabled: true,
            proving_scheme: "Jolt/HyperKZG".to_string(),
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
                prove: 10,
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
        let metrics = UsageMetrics::new("/dev/null", 0);
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
            api_key: Some("test-key-123".to_string()),
            ..Default::default()
        };
        assert_eq!(config.api_key.as_deref(), Some("test-key-123"));
    }

    #[test]
    fn test_server_config_with_x402() {
        let config = ServerConfig {
            pay_to: Some("0xBAc675C310721717Cd4A37F6cbeA1F081b1C2a07".to_string()),
            ..Default::default()
        };
        assert!(config.pay_to.is_some());
        assert_eq!(
            config.facilitator_url,
            "https://api.cdp.coinbase.com/platform/v2/x402"
        );
    }
}
