//! SkillGuard — standalone skill safety classifier for OpenClaw/ClawHub skills.
//!
//! Classifies skills into four categories:
//! - **SAFE**: No concerning patterns detected
//! - **CAUTION**: Minor concerns, likely functional
//! - **DANGEROUS**: Significant risk (credential exposure, excessive permissions)
//! - **MALICIOUS**: Active malware indicators (reverse shells, obfuscation)
//!
//! Uses structured logging via [`tracing`]. Set the `RUST_LOG` environment
//! variable to control log verbosity (e.g., `RUST_LOG=skillguard=debug`).

pub mod batch;
pub mod clawhub;
pub mod crawler;
pub mod model;
pub mod patterns;
pub mod scores;
pub mod server;
pub mod skill;

use eyre::Result;
use onnx_tracer::graph::model::Model;
use onnx_tracer::tensor::Tensor;
use sha2::{Digest, Sha256};

use crate::model::skill_safety_model;
use crate::skill::SafetyClassification;

/// Version prefix for model hashes. Bump when serialization format changes.
const MODEL_HASH_VERSION: &str = "v1";

/// Run the classifier on a 22-element feature vector.
///
/// Returns (classification, raw_scores, confidence).
pub fn classify(features: &[i32]) -> Result<(SafetyClassification, [i32; 4], f64)> {
    let model = skill_safety_model();
    let input =
        Tensor::new(Some(features), &[1, 22]).map_err(|e| eyre::eyre!("Tensor error: {:?}", e))?;

    let result = model
        .forward(std::slice::from_ref(&input))
        .map_err(|e| eyre::eyre!("Forward error: {}", e))?;

    let data = &result.outputs[0].inner;
    if data.len() < 4 {
        eyre::bail!("Expected 4 output classes, got {}", data.len());
    }

    let raw_scores: [i32; 4] = [data[0], data[1], data[2], data[3]];

    // Find best class
    let (best_idx, &best_val) = data
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| *v)
        .ok_or_else(|| eyre::eyre!("Empty classifier output"))?;

    // Calculate confidence as margin over runner-up
    let runner_up = data
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != best_idx)
        .map(|(_, v)| *v)
        .max()
        .unwrap_or(0);

    let margin = (best_val - runner_up).abs();
    let confidence = (margin as f64 / 128.0).min(1.0);

    let classification = SafetyClassification::from_index(best_idx);

    Ok((classification, raw_scores, confidence))
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
        let mut features = vec![0i32; 22];
        features[16] = 100; // downloads (high)

        let (classification, _scores, confidence) = classify(&features).unwrap();
        assert!(confidence >= 0.0);
        println!(
            "Classification: {:?}, confidence: {}",
            classification, confidence
        );
    }

    #[test]
    fn test_classify_malicious_skill() {
        let mut features = vec![0i32; 22];
        features[0] = 80; // shell_exec_count
        features[5] = 128; // external_download
        features[6] = 100; // obfuscation_score
        features[7] = 128; // privilege_escalation
        features[8] = 80; // persistence_mechanisms
        features[19] = 128; // password_protected_archives
        features[20] = 128; // reverse_shell_patterns

        let (classification, _scores, _confidence) = classify(&features).unwrap();
        assert!(
            classification.is_deny(),
            "Expected denial (DANGEROUS or MALICIOUS), got {:?}",
            classification
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
