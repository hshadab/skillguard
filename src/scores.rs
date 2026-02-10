//! ClassScores: softmax normalization of raw classifier output.
//!
//! 3-class model: SAFE (0), CAUTION (1), DANGEROUS (2).

use serde::{Deserialize, Serialize};

/// Number of output classes in the model.
pub const NUM_CLASSES: usize = 3;

/// Scores for each safety class (3-class model)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassScores {
    #[serde(rename = "SAFE")]
    pub safe: f64,
    #[serde(rename = "CAUTION")]
    pub caution: f64,
    #[serde(rename = "DANGEROUS")]
    pub dangerous: f64,
}

/// Softmax temperature for i32 logits from the QAT model.
///
/// The QAT model simulates the exact Rust i32 inference path during training,
/// so logits are already in integer scale (range approx -34 to +46). The temperature
/// is calibrated directly on these integer-scale logits (ECE=0.045 on real data).
///
/// Calibrated via `training/calibrate.py --dataset real --num-classes 3`.
const SOFTMAX_TEMPERATURE: f64 = 0.95;

/// Normalized entropy threshold for flagging uncertain predictions.
/// If the normalized entropy of the softmax distribution exceeds this value,
/// the model is too uncertain and the prediction should be flagged for review.
/// Calibrated at 5% flag rate on the real dataset (0.6659).
pub const ENTROPY_ABSTAIN_THRESHOLD: f64 = 0.67;

impl ClassScores {
    pub fn from_raw_scores(raw: &[i32; NUM_CLASSES]) -> Self {
        // Apply softmax with temperature scaling to raw i32 logits.
        let scaled: Vec<f64> = raw
            .iter()
            .map(|&x| (x as f64) / SOFTMAX_TEMPERATURE)
            .collect();
        let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
        let total: f64 = exp_vals.iter().sum();

        let uniform = 1.0 / NUM_CLASSES as f64;
        if total == 0.0 || !total.is_finite() {
            return Self {
                safe: uniform,
                caution: uniform,
                dangerous: uniform,
            };
        }

        Self {
            safe: exp_vals[0] / total,
            caution: exp_vals[1] / total,
            dangerous: exp_vals[2] / total,
        }
    }

    pub fn to_array(&self) -> [f64; NUM_CLASSES] {
        [self.safe, self.caution, self.dangerous]
    }

    /// Shannon entropy of the softmax distribution, normalized to [0, 1].
    ///
    /// - 0.0 = model is perfectly certain (all probability on one class)
    /// - 1.0 = maximum uncertainty (uniform across 3 classes)
    pub fn entropy(&self) -> f64 {
        let probs = self.to_array();
        let max_entropy = (NUM_CLASSES as f64).ln();
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
        let scores = ClassScores::from_raw_scores(&[0, 0, 0]);
        let uniform = 1.0 / 3.0;
        assert!((scores.safe - uniform).abs() < 0.01);
        assert!((scores.caution - uniform).abs() < 0.01);
        assert!((scores.dangerous - uniform).abs() < 0.01);
    }

    #[test]
    fn test_softmax_dominant() {
        // With temperature=76.8 (calibrated for real data), logits [100, 0, -50]
        // yield softer probabilities than with the old T=12.8
        let scores = ClassScores::from_raw_scores(&[100, 0, -50]);
        assert!(
            scores.safe > scores.caution && scores.safe > scores.dangerous,
            "SAFE should be highest with gap of 100: {:?}",
            scores
        );
        assert!(scores.safe > 0.50, "SAFE should exceed 50%: {:?}", scores);
        let total = scores.safe + scores.caution + scores.dangerous;
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_softmax_small_gap_still_separates() {
        let scores = ClassScores::from_raw_scores(&[10, 15, 8]);
        assert!(
            scores.caution > scores.safe && scores.caution > scores.dangerous,
            "CAUTION (15) should be highest: {:?}",
            scores
        );
        assert!(
            scores.caution > 0.30,
            "Top class should be >30%: {:?}",
            scores
        );
    }

    #[test]
    fn test_entropy_uniform() {
        let scores = ClassScores::from_raw_scores(&[0, 0, 0]);
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
        // With temperature=76.8, large logit gaps still produce lower entropy
        // but not near-zero. Use a very large gap for a clear test.
        let scores = ClassScores::from_raw_scores(&[500, 0, -200]);
        let e = scores.entropy();
        assert!(
            e < 0.10,
            "Very dominant class should have low entropy, got {}",
            e
        );
        assert!(
            !scores.is_uncertain(),
            "Very dominant class should not be uncertain"
        );
    }
}
