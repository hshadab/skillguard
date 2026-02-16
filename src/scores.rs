//! ClassScores: softmax normalization of raw classifier output.

use serde::{Deserialize, Serialize};

/// Scores for each safety class
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassScores {
    #[serde(rename = "SAFE")]
    pub safe: f64,
    #[serde(rename = "CAUTION")]
    pub caution: f64,
    #[serde(rename = "DANGEROUS")]
    pub dangerous: f64,
    #[serde(rename = "MALICIOUS")]
    pub malicious: f64,
}

/// Softmax temperature for fixed-point i32 logits.
///
/// The float model's calibrated temperature is 1.5 (ECE=0.0038). However, the Rust
/// fixed-point model outputs i32 logits ~128x larger than float logits (due to scale=7
/// arithmetic). The equivalent fixed-point temperature is 1.5 * 128 ≈ 200.
///
/// With T=200: 100% accuracy, mean confidence 99.5%, min confidence 58.8%.
/// Entropy is meaningful for edge cases (max 0.49), enabling the abstain mechanism.
///
/// Calibrated via `training/calibrate.py` with fixed-point scaling adjustment.
const SOFTMAX_TEMPERATURE: f64 = 200.0;

/// Normalized entropy threshold for flagging uncertain predictions.
/// If the normalized entropy of the softmax distribution exceeds this value,
/// the model is too uncertain and the prediction should be flagged for review.
/// Calibrated at p95 of fixed-point entropy distribution (~5% flag rate).
pub const ENTROPY_ABSTAIN_THRESHOLD: f64 = 0.042;

impl ClassScores {
    pub fn from_raw_scores(raw: &[i32; 4]) -> Self {
        // Apply softmax with temperature scaling to raw i32 logits.
        // Raw scores are used directly (no /128 — the model already rescales internally).
        // Temperature < 1.0 sharpens the distribution so small logit gaps become
        // meaningful probability differences.
        let scaled: Vec<f64> = raw
            .iter()
            .map(|&x| (x as f64) / SOFTMAX_TEMPERATURE)
            .collect();
        let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
        let total: f64 = exp_vals.iter().sum();

        if total == 0.0 || !total.is_finite() {
            // Fallback to uniform distribution
            return Self {
                safe: 0.25,
                caution: 0.25,
                dangerous: 0.25,
                malicious: 0.25,
            };
        }

        Self {
            safe: exp_vals[0] / total,
            caution: exp_vals[1] / total,
            dangerous: exp_vals[2] / total,
            malicious: exp_vals[3] / total,
        }
    }

    pub fn to_array(&self) -> [f64; 4] {
        [self.safe, self.caution, self.dangerous, self.malicious]
    }

    /// Shannon entropy of the softmax distribution, normalized to [0, 1].
    ///
    /// - 0.0 = model is perfectly certain (all probability on one class)
    /// - 1.0 = maximum uncertainty (uniform across 4 classes)
    pub fn entropy(&self) -> f64 {
        let probs = self.to_array();
        let max_entropy = (4.0_f64).ln(); // ln(4) for 4 classes
        let mut h = 0.0;
        for &p in &probs {
            if p > 1e-15 {
                h -= p * p.ln();
            }
        }
        h / max_entropy
    }

    /// Returns true if the model's prediction entropy exceeds the abstain threshold,
    /// indicating the model is too uncertain for a reliable classification.
    pub fn is_uncertain(&self) -> bool {
        self.entropy() > ENTROPY_ABSTAIN_THRESHOLD
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_uniform() {
        let scores = ClassScores::from_raw_scores(&[0, 0, 0, 0]);
        assert!((scores.safe - 0.25).abs() < 0.01);
        assert!((scores.caution - 0.25).abs() < 0.01);
        assert!((scores.dangerous - 0.25).abs() < 0.01);
        assert!((scores.malicious - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_softmax_dominant() {
        // With T=200, a gap of ~5000 (typical trained model output) produces a sharp distribution
        let scores = ClassScores::from_raw_scores(&[5000, 0, -10000, -8000]);
        assert!(
            scores.safe > 0.99,
            "SAFE should dominate with large gap: {:?}",
            scores
        );
        let total = scores.safe + scores.caution + scores.dangerous + scores.malicious;
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_softmax_small_gap_still_separates() {
        // Typical fixed-point logits: gaps in the thousands produce meaningful separation
        let scores = ClassScores::from_raw_scores(&[3500, 4000, 3800, 3700]);
        assert!(
            scores.caution > scores.safe
                && scores.caution > scores.dangerous
                && scores.caution > scores.malicious,
            "CAUTION (4000) should be highest: {:?}",
            scores
        );
        // With T=200, a gap of 200-500 should still produce >30% for the top class
        assert!(
            scores.caution > 0.30,
            "Top class should be >30%: {:?}",
            scores
        );
    }

    #[test]
    fn test_entropy_uniform() {
        // Uniform distribution has maximum entropy = 1.0
        let scores = ClassScores::from_raw_scores(&[0, 0, 0, 0]);
        let e = scores.entropy();
        assert!(
            (e - 1.0).abs() < 0.01,
            "Uniform distribution should have entropy ~1.0, got {}",
            e
        );
        assert!(scores.is_uncertain(), "Uniform should be uncertain");
    }

    #[test]
    fn test_entropy_dominant() {
        // With T=200, a large gap in the thousands produces low entropy
        let scores = ClassScores::from_raw_scores(&[5000, 0, -10000, -8000]);
        let e = scores.entropy();
        assert!(
            e < 0.02,
            "Dominant class should have very low entropy, got {}",
            e
        );
        assert!(
            !scores.is_uncertain(),
            "Dominant class should not be uncertain"
        );
    }
}
