//! HTTP endpoint handler functions.
//!
//! Every classification request generates a mandatory ZK proof. If the prover
//! is not yet initialized or proving fails, the endpoint returns an error
//! rather than an unproved classification.

use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use crate::scores::ClassScores;
use crate::skill::{derive_decision, SkillFeatures};

use super::types::*;
use super::ServerState;

/// Maximum request body size in bytes (1 MB).
pub const MAX_BODY_BYTES: usize = 1024 * 1024;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Try to obtain the prover, recording an error metric if unavailable.
#[allow(clippy::result_large_err)]
fn try_get_prover(
    state: &Arc<ServerState>,
    start: Instant,
) -> Result<Arc<crate::prover::ProverState>, ProveEvaluateResponse> {
    match state.get_prover() {
        Some(p) => Ok(p),
        None => {
            state.usage.record_error();
            Err(ProveEvaluateResponse {
                success: false,
                error: Some(
                    "ZKML prover is still initializing. Please retry in a few seconds.".into(),
                ),
                evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Shared classification + proving logic
// ---------------------------------------------------------------------------

/// Classify a skill and generate a ZK proof.
///
/// The prover is mandatory — if it is not yet initialized or proving fails,
/// the endpoint returns `success: false` with an error message instead of
/// falling back to an unproved classification.
async fn classify_and_respond(
    state: &Arc<ServerState>,
    features: SkillFeatures,
    skill_name: String,
    start: Instant,
    endpoint: &str,
    request_bytes: Option<&[u8]>,
) -> ProveEvaluateResponse {
    let feature_vec = features.to_normalized_vec();

    let prover = match try_get_prover(state, start) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    // Check proof cache first
    if let Some(bytes) = request_bytes {
        let cache_key = crate::cache::ProofCache::cache_key(bytes, &state.model_hash);
        if let Some(cached) = state.proof_cache.get(&cache_key) {
            if let Some(ref eval) = cached.evaluation {
                let cls = crate::skill::SafetyClassification::parse_str(&eval.classification);
                let dec = crate::skill::SafetyDecision::parse_str(&eval.decision);
                state
                    .usage
                    .record(endpoint, &eval.skill_name, cls, dec, eval.confidence, 0);
            }
            return cached;
        }
    }

    let fv_for_proof = feature_vec.clone();
    let prove_result =
        tokio::task::spawn_blocking(move || crate::classify_with_proof(&prover, &fv_for_proof))
            .await;

    match prove_result {
        Ok(Ok((classification, raw_scores, confidence, proof_bundle))) => {
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
            state
                .usage
                .total_proofs_generated
                .fetch_add(1, Ordering::Relaxed);
            // Persist immediately — record() above already persisted classification
            // counters, but the proof counter was incremented after that call.
            state.usage.persist_to_disk();

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
            if let Some(bytes) = request_bytes {
                let cache_key = crate::cache::ProofCache::cache_key(bytes, &state.model_hash);
                state.proof_cache.put(&cache_key, &response);
            }

            response
        }
        Ok(Err(e)) => {
            tracing::error!("Proof generation failed: {}", e);
            state.usage.record_error();
            ProveEvaluateResponse {
                success: false,
                error: Some(format!("Proof generation failed: {}", e)),
                evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            }
        }
        Err(e) => {
            tracing::error!("Proving task panicked: {}", e);
            state.usage.record_error();
            ProveEvaluateResponse {
                success: false,
                error: Some(format!("Proving task panicked: {}", e)),
                evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

pub async fn health_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
) -> impl axum::response::IntoResponse {
    let proofs_path = std::path::Path::new(&state.config.cache_dir).join("proofs");
    let cache_writable = std::fs::metadata(&proofs_path)
        .map(|m| !m.permissions().readonly())
        .unwrap_or(false);

    let prover_ready = state.skip_prover || state.get_prover().is_some();
    let status = if prover_ready { "ok" } else { "degraded" };

    let response = HealthResponse {
        status: status.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        model_hash: state.model_hash.clone(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        zkml_enabled: state.get_prover().is_some(),
        proving_scheme: "Jolt/Dory".to_string(),
        cache_writable,
        pay_to: state.config.pay_to.clone(),
    };
    axum::Json(response)
}

/// Unified evaluate endpoint.
///
/// Accepts two request formats:
/// - **By name:** `{ "skill": "skill-slug" }` — fetches from ClawHub, then classifies
/// - **Full skill:** `{ "skill": { "name": "...", ... } }` — classifies directly
pub async fn evaluate_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<SocketAddr>,
    request: axum::extract::Request,
) -> impl axum::response::IntoResponse {
    let start = Instant::now();
    let client_ip = addr.ip();

    let body = request.into_body();
    let bytes = match axum::body::to_bytes(body, MAX_BODY_BYTES).await {
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

    if let Some(limiter) =
        super::middleware::get_rate_limiter(&state.config, &state.rate_limiters, client_ip).await
    {
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

    // Auto-detect request format: if "skill" is a string → name lookup, if object → full skill
    let parsed: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            return axum::Json(ProveEvaluateResponse {
                success: false,
                error: Some(format!("Invalid JSON: {}", e)),
                evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
    };

    if parsed.get("skill").map(|s| s.is_string()).unwrap_or(false) {
        // Name-based lookup
        let name_req: EvaluateByNameRequest = match serde_json::from_value(parsed) {
            Ok(r) => r,
            Err(e) => {
                return axum::Json(ProveEvaluateResponse {
                    success: false,
                    error: Some(format!("Invalid request: {}", e)),
                    evaluation: None,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                });
            }
        };

        state
            .usage
            .ep_evaluate_by_name
            .fetch_add(1, Ordering::Relaxed);

        let skill = match state
            .clawhub_client
            .fetch_skill(&name_req.skill, name_req.version.as_deref())
            .await
        {
            Ok(s) => s,
            Err(e) => {
                state.usage.record_error();
                return axum::Json(ProveEvaluateResponse {
                    success: false,
                    error: Some(format!("Failed to fetch skill '{}': {}", name_req.skill, e)),
                    evaluation: None,
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
            "evaluate",
            Some(&bytes),
        )
        .await;

        axum::Json(response)
    } else {
        // Full skill data
        let eval_request: EvaluateRequest = match serde_json::from_value(parsed) {
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

        state.usage.ep_evaluate.fetch_add(1, Ordering::Relaxed);

        let features = SkillFeatures::extract(&eval_request.skill, eval_request.vt_report.as_ref());
        let skill_name = eval_request.skill.name.clone();

        let response = classify_and_respond(
            &state,
            features,
            skill_name,
            start,
            "evaluate",
            Some(&bytes),
        )
        .await;

        axum::Json(response)
    }
}

pub async fn verify_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
    axum::Json(req): axum::Json<VerifyRequest>,
) -> impl axum::response::IntoResponse {
    let start = Instant::now();

    state.usage.ep_verify.fetch_add(1, Ordering::Relaxed);

    let prover = match try_get_prover(&state, start) {
        Ok(p) => p,
        Err(_) => {
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

    let valid: bool = prover.verify_proof(&bundle).unwrap_or_else(|e| {
        tracing::warn!(error = %e, "proof verification returned error");
        false
    });

    state
        .usage
        .total_proofs_verified
        .fetch_add(1, Ordering::Relaxed);
    state.usage.persist_to_disk();

    let verification_time_ms = start.elapsed().as_millis() as u64;

    axum::Json(serde_json::json!({
        "valid": valid,
        "verification_time_ms": verification_time_ms,
    }))
}

pub async fn stats_handler(
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
