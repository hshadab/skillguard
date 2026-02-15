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

/// Softmax temperature — lower values produce sharper (more decisive) distributions.
/// The model's fixed-point arithmetic compresses logit ranges through 3 layers of /128
/// rescaling, so raw score differences are small. A temperature below 1.0 compensates
/// by amplifying those differences before softmax.
const SOFTMAX_TEMPERATURE: f64 = 0.25;

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
        // With temperature 0.25, a raw score gap of 128 produces a very sharp distribution
        let scores = ClassScores::from_raw_scores(&[128, 0, 0, 0]);
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
        // Even a gap of 5 (typical model output) should produce meaningful separation
        let scores = ClassScores::from_raw_scores(&[35, 40, 38, 37]);
        assert!(
            scores.caution > scores.safe
                && scores.caution > scores.dangerous
                && scores.caution > scores.malicious,
            "CAUTION (40) should be highest: {:?}",
            scores
        );
        // With temperature 0.25, a gap of 2-5 should produce >40% for the top class
        assert!(
            scores.caution > 0.40,
            "Top class should be >40%: {:?}",
            scores
        );
    }
}
