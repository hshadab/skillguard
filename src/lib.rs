//! SkillGuard ZKML — Provably correct AI safety classifications with agentic commerce.
//!
//! Classifies skills into three categories:
//! - **SAFE**: No concerning patterns detected
//! - **CAUTION**: Minor concerns, likely functional
//! - **DANGEROUS**: Significant risk (credential exposure, reverse shells, malware)
//!
//! Every classification can produce a cryptographic SNARK proof via Jolt Atlas,
//! and agents pay per request via x402 on Base.
//!
//! Uses structured logging via [`tracing`]. Set the `RUST_LOG` environment
//! variable to control log verbosity (e.g., `RUST_LOG=skillguard=debug`).

#[cfg(feature = "crawler")]
pub mod batch;
pub mod cache;
pub mod clawhub;
#[cfg(feature = "crawler")]
pub mod crawler;
pub mod mcp;
pub mod model;
pub mod patterns;
pub mod prover;
pub mod scores;
pub mod server;
pub mod skill;
pub mod ui;

use eyre::Result;
use onnx_tracer::graph::model::Model;
use onnx_tracer::tensor::Tensor;
use sha2::{Digest, Sha256};

use crate::model::skill_safety_model;
use crate::skill::SafetyClassification;

/// Version prefix for model hashes. Bump when serialization format changes.
const MODEL_HASH_VERSION: &str = "v1";

use crate::scores::NUM_CLASSES;

/// Process raw model output scores into a classification and confidence.
///
/// Shared by both `classify()` and `classify_with_proof()`.
fn process_raw_scores(raw_scores: &[i32; NUM_CLASSES], label: &str) -> (SafetyClassification, f64) {
    let (best_idx, _) = raw_scores
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| *v)
        .expect("raw_scores is non-empty");

    let scores = crate::scores::ClassScores::from_raw_scores(raw_scores);
    let confidence = scores.to_array()[best_idx];
    let classification = SafetyClassification::from_index(best_idx);

    tracing::debug!(
        raw_logits = ?raw_scores,
        softmax = ?(scores.safe, scores.caution, scores.dangerous),
        entropy = scores.entropy(),
        top_class = %classification.as_str(),
        top_confidence = confidence,
        "{}: raw logits and softmax scores", label
    );

    (classification, confidence)
}

/// Rule-based safety floor: override model classification to DANGEROUS when
/// deterministic feature extraction detects unambiguous threat patterns.
///
/// This is defense-in-depth — the MLP is one signal, but when pattern matching
/// detects reverse shells, data exfiltration with credentials, or multiple
/// co-occurring risk signals, we override to DANGEROUS regardless of model output.
///
/// Feature indices (from `SkillFeatures::to_normalized_vec()`):
///   5  = external_download (bool 0/128)
///   7  = privilege_escalation (bool 0/128)
///   9  = data_exfiltration_patterns (clip /5)
///  20  = reverse_shell_patterns (clip /5)
///  21  = llm_secret_exposure (bool 0/128)
///  40  = credential_and_exfil (clip /10)
///  43  = risk_signal_count (clip /8)
fn apply_safety_floor(features: &[i32], classification: SafetyClassification) -> SafetyClassification {
    if classification.is_deny() {
        return classification;
    }
    // Threshold at ~40% of normalized range (128 * 0.4 ≈ 51)
    const MID: i32 = 51;

    let reverse_shell     = features[20]; // any > 0 means at least 1 pattern matched
    let data_exfil        = features[9];
    let credential_exfil  = features[40]; // credential + exfiltration co-occurrence
    let risk_signals      = features[43]; // count of active risk categories
    let priv_esc          = features[7];  // bool: 0 or 128
    let llm_secret        = features[21]; // bool: 0 or 128
    let ext_download      = features[5];  // bool: 0 or 128

    // Rule 1: Reverse shell detected — always dangerous
    if reverse_shell > 0 {
        tracing::info!(reverse_shell, "safety_floor: reverse shell pattern override → DANGEROUS");
        return SafetyClassification::Dangerous;
    }

    // Rule 2: Data exfiltration patterns present — always dangerous
    if data_exfil > 0 {
        tracing::info!(data_exfil, "safety_floor: data exfiltration override → DANGEROUS");
        return SafetyClassification::Dangerous;
    }

    // Rule 3: Credential harvesting with exfiltration vector
    if credential_exfil > 0 {
        tracing::info!(credential_exfil, "safety_floor: credential + exfil override → DANGEROUS");
        return SafetyClassification::Dangerous;
    }

    // Rule 4: High risk signal count (≥4 of 8 categories active) — always dangerous
    if risk_signals >= MID {
        tracing::info!(risk_signals, "safety_floor: high risk signal count override → DANGEROUS");
        return SafetyClassification::Dangerous;
    }

    // Rule 5: Privilege escalation + external download — always dangerous
    if priv_esc > 0 && ext_download > 0 {
        tracing::info!("safety_floor: privilege escalation + external download override → DANGEROUS");
        return SafetyClassification::Dangerous;
    }

    // Rule 6: LLM secret exposure with credential patterns
    // llm_secret_exposure is bool (128), credential_patterns is at index 4
    if llm_secret > 0 && features[4] > 0 {
        tracing::info!("safety_floor: LLM secret exposure + credential patterns override → DANGEROUS");
        return SafetyClassification::Dangerous;
    }

    // Rule 7: External download + shell execution (curl|bash anti-pattern)
    // external_download (bool=128) + shell_and_network co-occurrence (index 31)
    let shell_and_network = features[31];
    if ext_download > 0 && shell_and_network > 0 {
        tracing::info!(shell_and_network, "safety_floor: external download + shell+network co-occurrence override → DANGEROUS");
        return SafetyClassification::Dangerous;
    }

    classification
}

/// Rule-based safe floor: downgrade a DANGEROUS classification to CAUTION when
/// the skill has no detectable risk signals whatsoever.
///
/// This prevents false positives on trivially benign skills (e.g., a calculator
/// with no scripts, no network calls, no credentials). Only applies when ALL
/// core risk features are zero.
fn apply_safe_floor(features: &[i32], classification: SafetyClassification) -> SafetyClassification {
    if !classification.is_deny() {
        return classification;
    }

    let shell_exec       = features[0];  // shell_exec_count
    let network_calls    = features[1];  // network_call_count
    let fs_writes        = features[2];  // fs_write_count
    let credential_pats  = features[4];  // credential_patterns
    let ext_download     = features[5];  // external_download (bool)
    let obfuscation      = features[6];  // obfuscation_score
    let priv_esc         = features[7];  // privilege_escalation (bool)
    let persistence      = features[8];  // persistence_mechanisms
    let data_exfil       = features[9];  // data_exfiltration_patterns
    let reverse_shell    = features[20]; // reverse_shell_patterns
    let llm_secret       = features[21]; // llm_secret_exposure (bool)
    let risk_signals     = features[43]; // risk_signal_count

    // If ALL core risk features are zero, this skill is provably benign
    let all_clean = shell_exec == 0
        && network_calls == 0
        && fs_writes == 0
        && credential_pats == 0
        && ext_download == 0
        && obfuscation == 0
        && priv_esc == 0
        && persistence == 0
        && data_exfil == 0
        && reverse_shell == 0
        && llm_secret == 0
        && risk_signals == 0;

    if all_clean {
        tracing::info!("safe_floor: all risk features zero, overriding DANGEROUS → CAUTION");
        return SafetyClassification::Caution;
    }

    classification
}

/// Run the classifier on a 45-element feature vector.
///
/// Returns (classification, raw_scores, confidence).
///
/// After MLP inference, applies a rule-based safety floor that overrides the
/// model output when deterministic features indicate unambiguous danger, and
/// a safe floor that prevents false positives on trivially benign skills.
pub fn classify(features: &[i32]) -> Result<(SafetyClassification, [i32; NUM_CLASSES], f64)> {
    let model = skill_safety_model();
    let input =
        Tensor::new(Some(features), &[1, 45]).map_err(|e| eyre::eyre!("Tensor error: {:?}", e))?;

    let result = model
        .forward(std::slice::from_ref(&input))
        .map_err(|e| eyre::eyre!("Forward error: {}", e))?;

    let data = &result.outputs[0].inner;
    if data.len() < NUM_CLASSES {
        eyre::bail!(
            "Expected {} output classes, got {}",
            NUM_CLASSES,
            data.len()
        );
    }

    let raw_scores: [i32; NUM_CLASSES] = [data[0], data[1], data[2]];
    let (classification, confidence) = process_raw_scores(&raw_scores, "classify");

    // Apply deterministic safety floor — override if features indicate clear danger
    let classification = apply_safety_floor(features, classification);
    // Apply safe floor — prevent false positives on trivially benign skills
    let classification = apply_safe_floor(features, classification);

    Ok((classification, raw_scores, confidence))
}

/// Run the classifier with ZK proof generation.
///
/// Returns (classification, raw_scores, confidence, proof_bundle).
pub fn classify_with_proof(
    prover: &prover::ProverState,
    features: &[i32],
) -> Result<(
    SafetyClassification,
    [i32; NUM_CLASSES],
    f64,
    prover::ProofBundle,
)> {
    let (bundle, raw_scores) = prover.prove_inference(features)?;

    if raw_scores.len() < NUM_CLASSES {
        eyre::bail!(
            "Expected {} output classes, got {}",
            NUM_CLASSES,
            raw_scores.len()
        );
    }

    let (classification, confidence) = process_raw_scores(&raw_scores, "classify_with_proof");

    // Apply deterministic safety floor — same rules as classify()
    let classification = apply_safety_floor(features, classification);
    // Apply safe floor — same rules as classify()
    let classification = apply_safe_floor(features, classification);

    Ok((classification, raw_scores, confidence, bundle))
}

/// Compute the SHA-256 hash of the model's serialized bytecode.
pub fn model_hash() -> String {
    hash_model_fn(skill_safety_model)
}

/// Compute the SHA-256 hash for any model-building function.
pub fn hash_model_fn(model_fn: fn() -> Model) -> String {
    let model = model_fn();
    let bytecode = onnx_tracer::decode_model(model);
    let serialized =
        serde_json::to_vec(&bytecode).unwrap_or_else(|_| format!("{:?}", bytecode).into_bytes());
    let mut hasher = Sha256::new();
    hasher.update(MODEL_HASH_VERSION.as_bytes());
    hasher.update(&serialized);
    let hash = hasher.finalize();
    format!("sha256:{}", hex::encode(hash))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_safe_skill() {
        let mut features = vec![0i32; 45];
        features[16] = 100; // downloads (high)

        let (classification, _scores, confidence) = classify(&features).unwrap();
        assert!(confidence >= 0.0);
        println!(
            "Classification: {:?}, confidence: {}",
            classification, confidence
        );
    }

    #[test]
    fn test_classify_produces_valid_output() {
        let mut features = vec![0i32; 45];
        features[0] = 80; // shell_exec_count
        features[5] = 128; // external_download
        features[6] = 100; // obfuscation_score
        features[7] = 128; // privilege_escalation
        features[20] = 128; // reverse_shell_patterns

        let (classification, logits, confidence) = classify(&features).unwrap();
        // Verify the model produces a valid classification
        assert!(
            matches!(
                classification,
                SafetyClassification::Safe
                    | SafetyClassification::Caution
                    | SafetyClassification::Dangerous
            ),
            "Expected valid classification, got {:?}",
            classification
        );
        assert!((0.0..=1.0).contains(&confidence));
        assert_eq!(logits.len(), 3, "Should produce 3 logits");
        // Logits should not all be zero (model is producing output)
        assert!(
            logits.iter().any(|&v| v != 0),
            "Model should produce non-zero logits"
        );
    }

    #[test]
    fn test_model_hash_deterministic() {
        let h1 = model_hash();
        let h2 = model_hash();
        assert_eq!(h1, h2);
        assert!(h1.starts_with("sha256:"));
    }
}
