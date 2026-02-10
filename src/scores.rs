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

impl ClassScores {
    pub fn from_raw_scores(raw: &[i32; 4]) -> Self {
        // Convert raw i32 scores to normalized probabilities using softmax
        // Scale down to avoid overflow: divide by 128 (the scale factor)
        let scaled: Vec<f64> = raw.iter().map(|&x| (x as f64) / 128.0).collect();
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
        // 128/128 = 1.0 after scaling, so softmax(1.0, 0, 0, 0) has ~0.42 for first class
        let scores = ClassScores::from_raw_scores(&[128, 0, 0, 0]);
        assert!(
            scores.safe > scores.caution
                && scores.safe > scores.dangerous
                && scores.safe > scores.malicious,
            "SAFE should be the highest class: {:?}",
            scores
        );
        let total = scores.safe + scores.caution + scores.dangerous + scores.malicious;
        assert!((total - 1.0).abs() < 0.001);
    }
}
