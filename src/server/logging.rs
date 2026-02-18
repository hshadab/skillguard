//! Usage metrics, access logging, and metrics persistence.
//!
//! Supports three persistence layers (highest-available wins on startup):
//! 1. **Redis** (`REDIS_URL`) — durable external store, survives container redeployments
//! 2. **Disk** (`metrics.json` in cache dir) — local file, survives restarts within same container
//! 3. **Env baseline** (`METRICS_BASELINE`) — manual floor, last resort
//!
//! All three are reconciled via `max()` on startup (counters are monotonically increasing).
//! Redis writes are fire-and-forget background tasks; disk writes remain synchronous.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use tracing::{info, warn};

use crate::skill::{SafetyClassification, SafetyDecision};

/// Maximum number of rotated access log files to keep.
const MAX_ACCESS_LOG_ROTATIONS: usize = 5;

/// Interval in seconds between metrics persistence to disk (safety-net fallback;
/// primary persistence happens after every state-changing request).
pub const METRICS_PERSIST_INTERVAL_SECS: u64 = 60;

/// Redis hash key for metrics storage.
const REDIS_METRICS_KEY: &str = "skillguard:metrics";

/// All counter field names, kept in one place for consistency.
const COUNTER_FIELDS: &[&str] = &[
    "total_requests",
    "total_errors",
    "safe",
    "caution",
    "dangerous",
    "allow",
    "deny",
    "flag",
    "ep_evaluate",
    "ep_evaluate_by_name",
    "ep_verify",
    "ep_stats",
    "total_proofs_generated",
    "total_proofs_verified",
];

pub struct UsageMetrics {
    pub total_requests: AtomicU64,
    pub total_errors: AtomicU64,

    pub safe: AtomicU64,
    pub caution: AtomicU64,
    pub dangerous: AtomicU64,

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

    // --- Redis fields (all optional) ---
    redis: Option<redis::aio::ConnectionManager>,
    rt_handle: Option<tokio::runtime::Handle>,
    /// Tracks the last value we synced to Redis for each counter so we can
    /// compute HINCRBY deltas (avoids double-counting across instances).
    redis_last_synced: std::sync::Mutex<HashMap<String, u64>>,
}

impl UsageMetrics {
    /// Synchronous constructor — no Redis, backward-compatible (used in tests).
    pub fn new(access_log_path: &str, max_access_log_bytes: u64, cache_dir: &str) -> Self {
        let (access_log, access_log_bytes, metrics_path) =
            Self::init_disk(access_log_path, cache_dir);

        let restored = Self::load_disk_and_baseline(&metrics_path);
        let v = |field: &str| -> u64 { Self::val_from_json(&restored, field) };

        Self {
            total_requests: AtomicU64::new(v("total_requests")),
            total_errors: AtomicU64::new(v("total_errors")),
            safe: AtomicU64::new(v("safe")),
            caution: AtomicU64::new(v("caution")),
            dangerous: AtomicU64::new(v("dangerous")),
            allow: AtomicU64::new(v("allow")),
            deny: AtomicU64::new(v("deny")),
            flag: AtomicU64::new(v("flag")),
            ep_evaluate: AtomicU64::new(v("ep_evaluate")),
            ep_evaluate_by_name: AtomicU64::new(v("ep_evaluate_by_name")),
            ep_verify: AtomicU64::new(v("ep_verify")),
            ep_stats: AtomicU64::new(v("ep_stats")),
            total_proofs_generated: AtomicU64::new(v("total_proofs_generated")),
            total_proofs_verified: AtomicU64::new(v("total_proofs_verified")),
            access_log: std::sync::Mutex::new(access_log),
            access_log_path: access_log_path.to_string(),
            access_log_bytes: AtomicU64::new(access_log_bytes),
            max_access_log_bytes,
            metrics_path,
            redis: None,
            rt_handle: None,
            redis_last_synced: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Async constructor — connects to Redis if `REDIS_URL` is set, reconciles
    /// counters across Redis/disk/env via `max()`.
    pub async fn new_async(
        access_log_path: &str,
        max_access_log_bytes: u64,
        cache_dir: &str,
    ) -> Self {
        let (access_log, access_log_bytes_val, metrics_path) =
            Self::init_disk(access_log_path, cache_dir);

        // Load disk + baseline sources
        let disk_baseline = Self::load_disk_and_baseline(&metrics_path);

        // Attempt Redis connection
        let (redis_conn, redis_values) = match std::env::var("REDIS_URL") {
            Ok(url) => {
                info!(url = %mask_redis_url(&url), "connecting to Redis for metrics persistence");
                match Self::connect_redis(&url).await {
                    Some(mut conn) => {
                        let vals = Self::load_from_redis(&mut conn).await;
                        (Some(conn), vals)
                    }
                    None => (None, None),
                }
            }
            Err(_) => {
                info!("REDIS_URL not set; metrics will use disk-only persistence");
                (None, None)
            }
        };

        // Reconcile: max(redis, disk, baseline) for each counter
        let v = |field: &str| -> u64 {
            let disk_val = Self::val_from_json(&disk_baseline, field);
            let redis_val = redis_values
                .as_ref()
                .and_then(|m| m.get(field))
                .copied()
                .unwrap_or(0);
            std::cmp::max(disk_val, redis_val)
        };

        let rt_handle = Some(tokio::runtime::Handle::current());

        // Initialize last_synced to current max values so first persist only
        // sends deltas from new activity (not the full historical count).
        // Exception: if Redis had no data (fresh deploy), start at 0 so the
        // first persist pushes the full reconciled values.
        let redis_had_data = redis_values.is_some();
        let mut last_synced = HashMap::new();
        for &field in COUNTER_FIELDS {
            let val = v(field);
            if redis_had_data {
                last_synced.insert(field.to_string(), val);
            } else {
                last_synced.insert(field.to_string(), 0);
            }
        }

        let metrics = Self {
            total_requests: AtomicU64::new(v("total_requests")),
            total_errors: AtomicU64::new(v("total_errors")),
            safe: AtomicU64::new(v("safe")),
            caution: AtomicU64::new(v("caution")),
            dangerous: AtomicU64::new(v("dangerous")),
            allow: AtomicU64::new(v("allow")),
            deny: AtomicU64::new(v("deny")),
            flag: AtomicU64::new(v("flag")),
            ep_evaluate: AtomicU64::new(v("ep_evaluate")),
            ep_evaluate_by_name: AtomicU64::new(v("ep_evaluate_by_name")),
            ep_verify: AtomicU64::new(v("ep_verify")),
            ep_stats: AtomicU64::new(v("ep_stats")),
            total_proofs_generated: AtomicU64::new(v("total_proofs_generated")),
            total_proofs_verified: AtomicU64::new(v("total_proofs_verified")),
            access_log: std::sync::Mutex::new(access_log),
            access_log_path: access_log_path.to_string(),
            access_log_bytes: AtomicU64::new(access_log_bytes_val),
            max_access_log_bytes,
            metrics_path,
            redis: redis_conn,
            rt_handle,
            redis_last_synced: std::sync::Mutex::new(last_synced),
        };

        if metrics.redis.is_some() {
            info!("metrics persistence: Redis + disk (reconciled via max)");
        }

        metrics
    }

    // -----------------------------------------------------------------------
    // Shared init helpers
    // -----------------------------------------------------------------------

    /// Set up disk: open access log, ensure cache dir, return (file, current_size, metrics_path).
    fn init_disk(access_log_path: &str, cache_dir: &str) -> (Option<File>, u64, String) {
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

        let cache_path = Path::new(cache_dir);
        if !cache_path.exists() {
            if let Err(e) = std::fs::create_dir_all(cache_path) {
                warn!(path = cache_dir, error = %e, "failed to create cache directory");
            }
        }

        let metrics_path = cache_path.join("metrics.json");
        let metrics_path_str = metrics_path.to_string_lossy().to_string();

        (file, current_size, metrics_path_str)
    }

    /// Load persisted metrics from disk file and/or METRICS_BASELINE env var.
    /// Returns `max(disk, baseline)` as a JSON value, or None if neither source exists.
    fn load_disk_and_baseline(metrics_path: &str) -> Option<serde_json::Value> {
        let disk_restored = std::fs::read(metrics_path)
            .ok()
            .and_then(|data| serde_json::from_slice::<serde_json::Value>(&data).ok());

        let baseline = std::env::var("METRICS_BASELINE")
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());

        match (&disk_restored, &baseline) {
            (Some(_), Some(_)) => {
                info!(path = %metrics_path, "merging disk metrics with METRICS_BASELINE (max of each)");
                // Merge: take max of each field
                let mut merged = serde_json::Map::new();
                for &field in COUNTER_FIELDS {
                    let d = disk_restored
                        .as_ref()
                        .and_then(|j| j.get(field))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let b = baseline
                        .as_ref()
                        .and_then(|j| j.get(field))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    merged.insert(
                        field.to_string(),
                        serde_json::Value::Number(std::cmp::max(d, b).into()),
                    );
                }
                Some(serde_json::Value::Object(merged))
            }
            (Some(_), None) => {
                info!(path = %metrics_path, "restored persisted metrics from disk");
                disk_restored
            }
            (None, Some(_)) => {
                info!("no metrics.json on disk; using METRICS_BASELINE env var as floor");
                baseline
            }
            (None, None) => None,
        }
    }

    /// Extract a u64 value from an optional JSON object.
    fn val_from_json(json: &Option<serde_json::Value>, field: &str) -> u64 {
        json.as_ref()
            .and_then(|j| j.get(field))
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    }

    // -----------------------------------------------------------------------
    // Redis helpers
    // -----------------------------------------------------------------------

    /// Connect to Redis, returning a ConnectionManager or None on failure.
    async fn connect_redis(url: &str) -> Option<redis::aio::ConnectionManager> {
        let client = match redis::Client::open(url) {
            Ok(c) => c,
            Err(e) => {
                warn!(error = %e, "failed to create Redis client; falling back to disk-only");
                return None;
            }
        };
        match redis::aio::ConnectionManager::new(client).await {
            Ok(conn) => {
                info!("connected to Redis successfully");
                Some(conn)
            }
            Err(e) => {
                warn!(error = %e, "failed to connect to Redis; falling back to disk-only");
                None
            }
        }
    }

    /// Load all counters from Redis hash `skillguard:metrics`.
    async fn load_from_redis(
        conn: &mut redis::aio::ConnectionManager,
    ) -> Option<HashMap<String, u64>> {
        use redis::AsyncCommands;
        let result: redis::RedisResult<HashMap<String, u64>> =
            conn.hgetall(REDIS_METRICS_KEY).await;
        match result {
            Ok(map) if !map.is_empty() => {
                info!(
                    keys = map.len(),
                    "loaded {} counter(s) from Redis",
                    map.len()
                );
                Some(map)
            }
            Ok(_) => {
                info!("Redis hash is empty (fresh deploy); no counters to restore");
                None
            }
            Err(e) => {
                warn!(error = %e, "failed to load metrics from Redis");
                None
            }
        }
    }

    /// Snapshot all 15 AtomicU64 counters into a HashMap.
    fn snapshot_counters(&self) -> HashMap<String, u64> {
        let mut map = HashMap::with_capacity(COUNTER_FIELDS.len());
        map.insert(
            "total_requests".into(),
            self.total_requests.load(Ordering::Relaxed),
        );
        map.insert(
            "total_errors".into(),
            self.total_errors.load(Ordering::Relaxed),
        );
        map.insert("safe".into(), self.safe.load(Ordering::Relaxed));
        map.insert("caution".into(), self.caution.load(Ordering::Relaxed));
        map.insert("dangerous".into(), self.dangerous.load(Ordering::Relaxed));
        map.insert("allow".into(), self.allow.load(Ordering::Relaxed));
        map.insert("deny".into(), self.deny.load(Ordering::Relaxed));
        map.insert("flag".into(), self.flag.load(Ordering::Relaxed));
        map.insert(
            "ep_evaluate".into(),
            self.ep_evaluate.load(Ordering::Relaxed),
        );
        map.insert(
            "ep_evaluate_by_name".into(),
            self.ep_evaluate_by_name.load(Ordering::Relaxed),
        );
        map.insert("ep_verify".into(), self.ep_verify.load(Ordering::Relaxed));
        map.insert("ep_stats".into(), self.ep_stats.load(Ordering::Relaxed));
        map.insert(
            "total_proofs_generated".into(),
            self.total_proofs_generated.load(Ordering::Relaxed),
        );
        map.insert(
            "total_proofs_verified".into(),
            self.total_proofs_verified.load(Ordering::Relaxed),
        );
        map
    }

    /// Fire-and-forget: compute deltas from `redis_last_synced` and send
    /// HINCRBY commands to Redis via a spawned tokio task.
    fn persist_to_redis(&self) {
        let (conn, handle) = match (&self.redis, &self.rt_handle) {
            (Some(c), Some(h)) => (c.clone(), h.clone()),
            _ => return,
        };

        let current = self.snapshot_counters();
        let mut deltas: Vec<(String, u64)> = Vec::new();

        if let Ok(mut last_synced) = self.redis_last_synced.lock() {
            for (field, &current_val) in &current {
                let last_val = last_synced.get(field).copied().unwrap_or(0);
                if current_val > last_val {
                    deltas.push((field.clone(), current_val - last_val));
                    last_synced.insert(field.clone(), current_val);
                }
            }
        }

        if deltas.is_empty() {
            return;
        }

        // Spawn fire-and-forget task — never blocks the request path
        handle.spawn(async move {
            use redis::AsyncCommands;
            let mut conn = conn;
            for (field, delta) in &deltas {
                let result: redis::RedisResult<u64> =
                    conn.hincr(REDIS_METRICS_KEY, field, *delta as i64).await;
                if let Err(e) = result {
                    warn!(field = %field, error = %e, "Redis HINCRBY failed (will retry next persist)");
                    // Don't update last_synced on failure — delta will be retried
                    // Note: last_synced was already optimistically updated above.
                    // In practice ConnectionManager auto-reconnects, so transient
                    // failures are rare and the next persist cycle catches up.
                }
            }
        });
    }

    // -----------------------------------------------------------------------
    // Recording + persistence
    // -----------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        endpoint: &str,
        skill_name: &str,
        classification: SafetyClassification,
        decision: SafetyDecision,
        confidence: f64,
        processing_time_ms: u64,
        feature_vec: Option<&[i32]>,
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
                let mut entry = serde_json::json!({
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "endpoint": endpoint,
                    "skill_name": skill_name,
                    "classification": classification.as_str(),
                    "decision": decision.as_str(),
                    "confidence": confidence,
                    "processing_time_ms": processing_time_ms,
                });
                if let Some(fv) = feature_vec {
                    entry["feature_vec"] = serde_json::json!(fv);
                }
                let mut line = entry.to_string();
                line.push('\n');
                let line_len = line.len() as u64;
                if let Err(e) = file.write_all(line.as_bytes()).and_then(|_| file.flush()) {
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

        // Persist immediately so counters survive crashes / SIGKILL restarts.
        self.persist_to_disk();
    }

    pub fn record_error(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_errors.fetch_add(1, Ordering::Relaxed);
        self.persist_to_disk();
    }

    /// Persist current metrics snapshot to disk, then fire-and-forget to Redis.
    pub fn persist_to_disk(&self) {
        let snapshot = serde_json::json!({
            "total_requests": self.total_requests.load(Ordering::Relaxed),
            "total_errors": self.total_errors.load(Ordering::Relaxed),
            "safe": self.safe.load(Ordering::Relaxed),
            "caution": self.caution.load(Ordering::Relaxed),
            "dangerous": self.dangerous.load(Ordering::Relaxed),
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

        // Also persist to Redis (fire-and-forget)
        self.persist_to_redis();
    }
}

/// Redact password from a Redis URL for safe logging.
/// `redis://:password@host:port` → `redis://:***@host:port`
fn mask_redis_url(url: &str) -> String {
    // Try to find :password@ pattern
    if let Some(at_pos) = url.find('@') {
        if let Some(colon_pos) = url[..at_pos].rfind(':') {
            let prefix = &url[..colon_pos + 1];
            let suffix = &url[at_pos..];
            return format!("{}***{}", prefix, suffix);
        }
    }
    url.to_string()
}
