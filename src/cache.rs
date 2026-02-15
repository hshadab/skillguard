//! Disk-based proof cache.
//!
//! Keyed by `sha256(request_body + model_hash)`, stored as JSON files under
//! `<cache_dir>/proofs/`. Avoids re-proving identical skills.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::server::ProveEvaluateResponse;

/// Default cache directory (Render persistent disk mount point).
const DEFAULT_CACHE_DIR: &str = "/var/data/skillguard-cache";

/// Maximum number of cached proof entries before cleanup.
const MAX_CACHE_ENTRIES: usize = 1000;

/// Disk-backed proof cache.
pub struct ProofCache {
    proofs_dir: PathBuf,
    /// Guard to prevent concurrent cleanup runs.
    cleaning: AtomicBool,
}

impl ProofCache {
    /// Open (or create) the cache at the given root directory.
    pub fn open(cache_dir: Option<&str>) -> Self {
        let root = PathBuf::from(cache_dir.unwrap_or(DEFAULT_CACHE_DIR));
        let proofs_dir = root.join("proofs");

        if let Err(e) = fs::create_dir_all(&proofs_dir) {
            warn!(path = %proofs_dir.display(), error = %e, "could not create proof cache dir; caching disabled");
        } else {
            info!(path = %proofs_dir.display(), "proof cache ready");
        }

        Self {
            proofs_dir,
            cleaning: AtomicBool::new(false),
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
                None
            }
        }
    }

    /// Store a proof response in the cache.
    /// If the cache exceeds [`MAX_CACHE_ENTRIES`], the oldest entries by mtime are removed.
    pub fn put(&self, key: &str, response: &ProveEvaluateResponse) {
        // Evict oldest entries if over the limit
        self.cleanup_if_needed();

        let path = self.proofs_dir.join(format!("{key}.json"));
        match serde_json::to_vec(response) {
            Ok(data) => {
                if let Err(e) = fs::write(&path, &data) {
                    warn!(cache_key = &key[..12], error = %e, "failed to write proof cache");
                } else {
                    info!(cache_key = &key[..12], bytes = data.len(), "proof cached");
                }
            }
            Err(e) => {
                warn!(error = %e, "failed to serialize proof for cache");
            }
        }
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
            }
        }

        info!(evicted = remove_count, "proof cache cleanup complete");
        self.cleaning.store(false, Ordering::Release);
    }
}
