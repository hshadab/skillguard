//! Disk-based proof cache.
//!
//! Keyed by `sha256(request_body + model_hash)`, stored as JSON files under
//! `<cache_dir>/proofs/`. Avoids re-proving identical skills.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::server::types::DEFAULT_CACHE_DIR;
use crate::server::ProveEvaluateResponse;

/// Maximum number of cached proof entries before cleanup.
const MAX_CACHE_ENTRIES: usize = 1000;

/// Disk-backed proof cache.
pub struct ProofCache {
    proofs_dir: PathBuf,
    /// Guard to prevent concurrent cleanup runs.
    cleaning: AtomicBool,
    /// Approximate entry count to avoid directory reads on every put().
    entry_count: AtomicUsize,
    /// Set of cache keys currently being proved, to avoid duplicate work.
    in_flight: Mutex<HashSet<String>>,
}

impl ProofCache {
    /// Open (or create) the cache at the given root directory.
    pub fn open(cache_dir: Option<&str>) -> Self {
        let root = PathBuf::from(cache_dir.unwrap_or(DEFAULT_CACHE_DIR));
        let proofs_dir = root.join("proofs");

        let initial_count = if let Err(e) = fs::create_dir_all(&proofs_dir) {
            warn!(path = %proofs_dir.display(), error = %e, "could not create proof cache dir; caching disabled");
            0
        } else {
            // Seed the counter from the actual directory contents.
            let count = fs::read_dir(&proofs_dir)
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .filter(|e| {
                            e.path()
                                .extension()
                                .map(|ext| ext == "json")
                                .unwrap_or(false)
                        })
                        .count()
                })
                .unwrap_or(0);
            info!(path = %proofs_dir.display(), entries = count, "proof cache ready");
            count
        };

        Self {
            proofs_dir,
            cleaning: AtomicBool::new(false),
            entry_count: AtomicUsize::new(initial_count),
            in_flight: Mutex::new(HashSet::new()),
        }
    }

    /// Build the cache key from request body bytes and the model hash.
    pub fn cache_key(request_body: &[u8], model_hash: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(request_body);
        hasher.update(model_hash.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Look up a cached proof response.
    pub fn get(&self, key: &str) -> Option<ProveEvaluateResponse> {
        // SAFETY: key is always a hex-encoded SHA256 hash (64 chars, [0-9a-f]+),
        // so no path traversal is possible.
        let path = self.proofs_dir.join(format!("{key}.json"));
        let data = fs::read(&path).ok()?;
        match serde_json::from_slice(&data) {
            Ok(resp) => {
                info!(cache_key = &key[..12], "proof cache hit");
                Some(resp)
            }
            Err(e) => {
                warn!(cache_key = &key[..12], error = %e, "corrupt cache entry, removing");
                let _ = fs::remove_file(&path);
                self.entry_count.fetch_sub(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Store a proof response in the cache.
    /// If the cache exceeds the max entry limit, the oldest entries by mtime are removed.
    pub fn put(&self, key: &str, response: &ProveEvaluateResponse) {
        // Evict oldest entries if the counter exceeds the limit.
        if self.entry_count.load(Ordering::Relaxed) > MAX_CACHE_ENTRIES {
            self.cleanup_if_needed();
        }

        let path = self.proofs_dir.join(format!("{key}.json"));
        match serde_json::to_vec(response) {
            Ok(data) => {
                if let Err(e) = fs::write(&path, &data) {
                    warn!(cache_key = &key[..12], error = %e, "failed to write proof cache");
                } else {
                    self.entry_count.fetch_add(1, Ordering::Relaxed);
                    info!(cache_key = &key[..12], bytes = data.len(), "proof cached");
                }
            }
            Err(e) => {
                warn!(error = %e, "failed to serialize proof for cache");
            }
        }
    }

    /// Mark a cache key as in-flight (currently being proved).
    /// Returns `true` if this is the first claim, `false` if already in-flight.
    pub fn mark_in_flight(&self, key: &str) -> bool {
        let mut set = self.in_flight.lock().unwrap_or_else(|e| e.into_inner());
        set.insert(key.to_string())
    }

    /// Remove a cache key from the in-flight set.
    pub fn clear_in_flight(&self, key: &str) {
        let mut set = self.in_flight.lock().unwrap_or_else(|e| e.into_inner());
        set.remove(key);
    }

    /// Remove oldest cache entries when the count exceeds the limit.
    /// Uses an atomic flag to prevent concurrent cleanup runs.
    fn cleanup_if_needed(&self) {
        if self.cleaning.swap(true, Ordering::Acquire) {
            // Another thread is already cleaning.
            return;
        }
        let entries: Vec<_> = match fs::read_dir(&self.proofs_dir) {
            Ok(rd) => rd
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "json")
                        .unwrap_or(false)
                })
                .collect(),
            Err(_) => {
                self.cleaning.store(false, Ordering::Release);
                return;
            }
        };

        // Re-sync counter from actual directory state.
        self.entry_count.store(entries.len(), Ordering::Relaxed);

        if entries.len() <= MAX_CACHE_ENTRIES {
            self.cleaning.store(false, Ordering::Release);
            return;
        }

        let remove_count = entries.len() - MAX_CACHE_ENTRIES;
        let mut by_mtime: Vec<_> = entries
            .into_iter()
            .filter_map(|e| {
                let mtime = e.metadata().ok()?.modified().ok()?;
                Some((mtime, e.path()))
            })
            .collect();
        by_mtime.sort_by_key(|(mtime, _)| *mtime);

        for (_, path) in by_mtime.into_iter().take(remove_count) {
            if let Err(e) = fs::remove_file(&path) {
                warn!(path = %path.display(), error = %e, "failed to evict cache entry");
            } else {
                self.entry_count.fetch_sub(1, Ordering::Relaxed);
            }
        }

        info!(evicted = remove_count, "proof cache cleanup complete");
        self.cleaning.store(false, Ordering::Release);
    }
}
