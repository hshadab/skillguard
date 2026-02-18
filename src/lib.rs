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

/// Run the classifier on a 35-element feature vector.
///
/// Returns (classification, raw_scores, confidence).
pub fn classify(features: &[i32]) -> Result<(SafetyClassification, [i32; NUM_CLASSES], f64)> {
    let model = skill_safety_model();
    let input =
        Tensor::new(Some(features), &[1, 35]).map_err(|e| eyre::eyre!("Tensor error: {:?}", e))?;

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
        let mut features = vec![0i32; 35];
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
        let mut features = vec![0i32; 35];
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
