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
/// The float model's calibrated temperature is 0.10 (ECE≈0). The Rust fixed-point
/// model outputs i32 logits ~128x larger than float logits (due to scale=7
/// arithmetic). The equivalent fixed-point temperature is 0.10 * 128 = 12.8.
///
/// This sharper temperature converts small logit gaps into decisive probability
/// distributions, resolving the low-confidence issue with real SKILL.md inputs.
///
/// Calibrated via `training/calibrate.py` on 690-sample augmented dataset.
const SOFTMAX_TEMPERATURE: f64 = 12.8;

/// Normalized entropy threshold for flagging uncertain predictions.
/// If the normalized entropy of the softmax distribution exceeds this value,
/// the model is too uncertain and the prediction should be flagged for review.
/// With the sharper temperature (T=12.8), training-data predictions have
/// near-zero entropy. Real-world SKILL.md inputs with limited signals produce
/// entropy in the 0.45-0.65 range even when correctly classified at 70-80%
/// confidence. Threshold 0.85 flags only genuinely ambiguous cases where no
/// class exceeds ~40%.
pub const ENTROPY_ABSTAIN_THRESHOLD: f64 = 0.85;

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
        // With T=12.8, even moderate logit gaps produce very sharp distributions
        let scores = ClassScores::from_raw_scores(&[100, 0, -50, -30]);
        assert!(
            scores.safe > 0.99,
            "SAFE should dominate with gap of 100: {:?}",
            scores
        );
        let total = scores.safe + scores.caution + scores.dangerous + scores.malicious;
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_softmax_small_gap_still_separates() {
        // With T=12.8, small logit gaps (typical of real SKILL.md inputs) produce
        // meaningful separation — this is the key improvement over T=200
        let scores = ClassScores::from_raw_scores(&[10, 15, 8, 5]);
        assert!(
            scores.caution > scores.safe
                && scores.caution > scores.dangerous
                && scores.caution > scores.malicious,
            "CAUTION (15) should be highest: {:?}",
            scores
        );
        // With T=12.8, a gap of 5 logit units should produce >30% for the top class
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
        // With T=12.8, even moderate logit gaps produce near-zero entropy
        let scores = ClassScores::from_raw_scores(&[100, 0, -50, -30]);
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
