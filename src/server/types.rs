//! Request/response types and configuration for the SkillGuard server.

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

use crate::scores::ClassScores;
use crate::skill::{Skill, VTReport};

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
    /// x402 facilitator URL (defaults to free OpenFacilitator).
    pub facilitator_url: String,
    /// External base URL for x402 resource URLs (e.g. "https://skillguard.onrender.com").
    /// Behind a reverse proxy / TLS terminator, the app sees http:// internally;
    /// this ensures x402 payment requirements use the correct public https:// URL.
    pub external_url: Option<String>,
    /// Cache directory for proofs and metrics persistence.
    pub cache_dir: String,
}

/// Default cache directory (Render persistent disk mount point).
pub const DEFAULT_CACHE_DIR: &str = "/var/data/skillguard-cache";

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
            facilitator_url: "https://pay.openfacilitator.io".to_string(),
            external_url: None,
            cache_dir: DEFAULT_CACHE_DIR.to_string(),
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

/// Evaluation result with ZK proof.
#[derive(Debug, Serialize, Deserialize)]
pub struct ProvedEvaluationResult {
    pub skill_name: String,
    pub classification: String,
    pub decision: String,
    pub confidence: f64,
    pub scores: ClassScores,
    pub reasoning: String,
    /// ZK proof bundle — every classification includes a cryptographic proof.
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

/// Unified response for all classification endpoints.
#[derive(Debug, Serialize, Deserialize)]
pub struct ProveEvaluateResponse {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<ProvedEvaluationResult>,
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
    pub cache_writable: bool,
    /// Server's pay-to wallet address (if x402 is enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pay_to: Option<String>,
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
    pub verify: u64,
    pub stats: u64,
}

#[derive(Debug, Serialize)]
pub struct ProofStats {
    pub total_generated: u64,
    pub total_verified: u64,
}
