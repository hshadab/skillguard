//! HTTP endpoint handler functions.

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
// Handlers
// ---------------------------------------------------------------------------

pub async fn health_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
) -> impl axum::response::IntoResponse {
    let proofs_path = std::path::Path::new(&state.config.cache_dir).join("proofs");
    let cache_writable = std::fs::metadata(&proofs_path)
        .map(|m| !m.permissions().readonly())
        .unwrap_or(false);

    let response = HealthResponse {
        status: "ok".to_string(),
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

pub async fn evaluate_handler(
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
    let bytes = match axum::body::to_bytes(body, 2 * 1024 * 1024).await {
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

    if let Some(limiter) =
        super::middleware::get_rate_limiter(&state.config, &state.rate_limiters, client_ip).await
    {
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

pub async fn evaluate_by_name_handler(
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
    let bytes = match axum::body::to_bytes(body, 2 * 1024 * 1024).await {
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

    if let Some(limiter) =
        super::middleware::get_rate_limiter(&state.config, &state.rate_limiters, client_ip).await
    {
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

pub async fn prove_evaluate_handler(
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

    // Reject unauthenticated requests when auth is configured
    if auth_method == AuthMethod::Open
        && (state.config.api_key.is_some() || state.config.pay_to.is_some())
    {
        return axum::Json(ProveEvaluateResponse {
            success: false,
            error: Some("Authentication required. Provide an API key or pay via x402.".to_string()),
            evaluation: None,
            processing_time_ms: start.elapsed().as_millis() as u64,
        });
    }

    let body = request.into_body();
    let bytes = match axum::body::to_bytes(body, 2 * 1024 * 1024).await {
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

    let prover = match state.get_prover() {
        Some(p) => p,
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

    // Run the CPU-bound prover in a blocking thread to avoid starving the async runtime
    let prove_result =
        tokio::task::spawn_blocking(move || crate::classify_with_proof(&prover, &feature_vec))
            .await;

    let result = match prove_result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            state.usage.record_error();
            return axum::Json(ProveEvaluateResponse {
                success: false,
                error: Some(format!("Classification with proof failed: {}", e)),
                evaluation: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }
        Err(e) => {
            state.usage.record_error();
            return axum::Json(ProveEvaluateResponse {
                success: false,
                error: Some(format!("Proving task panicked: {}", e)),
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

pub async fn verify_handler(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
    axum::Json(req): axum::Json<VerifyRequest>,
) -> impl axum::response::IntoResponse {
    let start = Instant::now();

    state.usage.ep_verify.fetch_add(1, Ordering::Relaxed);

    let prover = match state.get_prover() {
        Some(p) => p,
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
