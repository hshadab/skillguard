//! Usage metrics, access logging, and metrics persistence.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use tracing::warn;

use crate::skill::{SafetyClassification, SafetyDecision};

/// Maximum number of rotated access log files to keep.
const MAX_ACCESS_LOG_ROTATIONS: usize = 5;

/// Interval in seconds between metrics persistence to disk.
pub const METRICS_PERSIST_INTERVAL_SECS: u64 = 60;

pub struct UsageMetrics {
    pub total_requests: AtomicU64,
    pub total_errors: AtomicU64,

    pub safe: AtomicU64,
    pub caution: AtomicU64,
    pub dangerous: AtomicU64,
    pub malicious: AtomicU64,

    pub allow: AtomicU64,
    pub deny: AtomicU64,
    pub flag: AtomicU64,

    pub ep_evaluate: AtomicU64,
    pub ep_evaluate_by_name: AtomicU64,
    pub ep_verify: AtomicU64,
    pub ep_stats: AtomicU64,

    pub total_proofs_generated: AtomicU64,
    pub total_proofs_verified: AtomicU64,

    pub access_log: std::sync::Mutex<Option<File>>,
    access_log_path: String,
    access_log_bytes: AtomicU64,
    max_access_log_bytes: u64,
    metrics_path: String,
}

impl UsageMetrics {
    pub fn new(access_log_path: &str, max_access_log_bytes: u64, cache_dir: &str) -> Self {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(access_log_path)
            .ok();
        if file.is_none() {
            warn!(path = access_log_path, "could not open access log");
        }
        let current_size = std::fs::metadata(access_log_path)
            .map(|m| m.len())
            .unwrap_or(0);

        let metrics_path = Path::new(cache_dir).join("metrics.json");

        Self {
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            safe: AtomicU64::new(0),
            caution: AtomicU64::new(0),
            dangerous: AtomicU64::new(0),
            malicious: AtomicU64::new(0),
            allow: AtomicU64::new(0),
            deny: AtomicU64::new(0),
            flag: AtomicU64::new(0),
            ep_evaluate: AtomicU64::new(0),
            ep_evaluate_by_name: AtomicU64::new(0),
            ep_verify: AtomicU64::new(0),
            ep_stats: AtomicU64::new(0),
            total_proofs_generated: AtomicU64::new(0),
            total_proofs_verified: AtomicU64::new(0),
            access_log: std::sync::Mutex::new(file),
            access_log_path: access_log_path.to_string(),
            access_log_bytes: AtomicU64::new(current_size),
            max_access_log_bytes,
            metrics_path: metrics_path.to_string_lossy().to_string(),
        }
    }

    pub fn record(
        &self,
        endpoint: &str,
        skill_name: &str,
        classification: SafetyClassification,
        decision: SafetyDecision,
        confidence: f64,
        processing_time_ms: u64,
    ) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        match classification {
            SafetyClassification::Safe => {
                self.safe.fetch_add(1, Ordering::Relaxed);
            }
            SafetyClassification::Caution => {
                self.caution.fetch_add(1, Ordering::Relaxed);
            }
            SafetyClassification::Dangerous => {
                self.dangerous.fetch_add(1, Ordering::Relaxed);
            }
            SafetyClassification::Malicious => {
                self.malicious.fetch_add(1, Ordering::Relaxed);
            }
        }

        match decision {
            SafetyDecision::Allow => {
                self.allow.fetch_add(1, Ordering::Relaxed);
            }
            SafetyDecision::Deny => {
                self.deny.fetch_add(1, Ordering::Relaxed);
            }
            SafetyDecision::Flag => {
                self.flag.fetch_add(1, Ordering::Relaxed);
            }
        }

        if let Ok(mut guard) = self.access_log.try_lock() {
            if let Some(ref mut file) = *guard {
                let entry = serde_json::json!({
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "endpoint": endpoint,
                    "skill_name": skill_name,
                    "classification": classification.as_str(),
                    "decision": decision.as_str(),
                    "confidence": confidence,
                    "processing_time_ms": processing_time_ms,
                });
                let mut line = entry.to_string();
                line.push('\n');
                let line_len = line.len() as u64;
                if let Err(e) = file.write_all(line.as_bytes()) {
                    warn!(error = %e, "failed to write access log entry");
                }
                let new_size =
                    self.access_log_bytes.fetch_add(line_len, Ordering::Relaxed) + line_len;

                // Rotate if over size limit (0 = no limit)
                if self.max_access_log_bytes > 0 && new_size >= self.max_access_log_bytes {
                    for i in (1..MAX_ACCESS_LOG_ROTATIONS).rev() {
                        let from = format!("{}.{}", self.access_log_path, i);
                        let to = format!("{}.{}", self.access_log_path, i + 1);
                        if let Err(e) = std::fs::rename(&from, &to) {
                            warn!(from = %from, to = %to, error = %e, "log rotation rename failed");
                        }
                    }
                    let rotated = format!("{}.1", self.access_log_path);
                    if let Err(e) = std::fs::rename(&self.access_log_path, &rotated) {
                        warn!(from = %self.access_log_path, to = %rotated, error = %e, "log rotation rename failed");
                    }
                    if let Ok(new_file) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&self.access_log_path)
                    {
                        *file = new_file;
                        self.access_log_bytes.store(0, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    pub fn record_error(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Persist current metrics snapshot to disk so they survive restarts.
    pub fn persist_to_disk(&self) {
        let snapshot = serde_json::json!({
            "total_requests": self.total_requests.load(Ordering::Relaxed),
            "total_errors": self.total_errors.load(Ordering::Relaxed),
            "safe": self.safe.load(Ordering::Relaxed),
            "caution": self.caution.load(Ordering::Relaxed),
            "dangerous": self.dangerous.load(Ordering::Relaxed),
            "malicious": self.malicious.load(Ordering::Relaxed),
            "allow": self.allow.load(Ordering::Relaxed),
            "deny": self.deny.load(Ordering::Relaxed),
            "flag": self.flag.load(Ordering::Relaxed),
            "ep_evaluate": self.ep_evaluate.load(Ordering::Relaxed),
            "ep_evaluate_by_name": self.ep_evaluate_by_name.load(Ordering::Relaxed),
            "ep_verify": self.ep_verify.load(Ordering::Relaxed),
            "ep_stats": self.ep_stats.load(Ordering::Relaxed),
            "total_proofs_generated": self.total_proofs_generated.load(Ordering::Relaxed),
            "total_proofs_verified": self.total_proofs_verified.load(Ordering::Relaxed),
        });
        match serde_json::to_vec_pretty(&snapshot) {
            Ok(data) => {
                if let Err(e) = std::fs::write(&self.metrics_path, &data) {
                    warn!(path = %self.metrics_path, error = %e, "failed to persist metrics");
                }
            }
            Err(e) => {
                warn!(error = %e, "failed to serialize metrics snapshot");
            }
        }
    }
}
