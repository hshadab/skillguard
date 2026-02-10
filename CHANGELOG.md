# Changelog

All notable changes to SkillGuard are documented in this file.

## [Unreleased]

### Added
- **SAFE/CAUTION boundary labeling** — `--prioritize-boundary` flag in `fetch_and_label.py` sorts skills by `|SAFE - CAUTION|` score proximity, fetching the most model-confused examples first to improve the SAFE/CAUTION decision boundary
- **Stratified sampling** — `training/stratified_sample.py` samples unlabeled skills proportionally across categories and confidence bands (low/medium/high) for diverse training data
- **Human review queue** — `training/generate_review_queue.py` generates a prioritized review queue sorted by: DANGEROUS labels (verify first), MLP/LLM disagreements, short reasoning (low confidence)
- **Held-out test set** — `--holdout-fraction` flag in `train.py` performs a stratified split before cross-validation, providing unbiased final evaluation on unseen data (default 15% in retrain pipeline)

### Changed
- **Improved LLM labeling prompt** — `SYSTEM_PROMPT` in `llm_label.py` expanded with explicit SAFE/CAUTION boundary rules (env var thresholds, shell wrapper patterns, download-and-execute patterns) and two few-shot examples to reduce mislabeling at class boundaries
- **Reduced synthetic augmentation** — `--augment-dangerous` reduced from 80 to 20 in `retrain.sh` as real labeled data replaces synthetic samples
- **Training pipeline** — `retrain.sh` now uses `--holdout-fraction 0.15` for held-out evaluation alongside 5-fold CV

## [2.0.0] - 2026-02-18

### Added
- **LLM-labeled real data training pipeline** — `training/llm_label.py` and `training/fetch_and_label.py` for bulk skill labeling via Claude API with resume-safe incremental saving
- **`extract-features` CLI subcommand** — reads SKILL.md from stdin, outputs the 35-dim feature vector as JSON for training/debugging feature parity
- **Real data loader** in `training/dataset.py` — `load_real_dataset()` loads LLM-labeled skills and extracts features via the Rust binary
- **Class weighting** in training for imbalanced real data (inverse-frequency weighting)
- **Per-class metrics** — precision, recall, and F1 reported per class (5-fold CV, mean ± std) in training summary, README, web dashboard, and LLM/OpenAPI docs
- **Batch scanning CLI** — `scan` and `crawl` subcommands (behind `--features crawler`) for crawling the awesome-openclaw-skills list and batch-classifying skills
- **Redis metrics persistence** — durable proof/classification counters via Redis (three-tier: Redis → disk → env baseline) so metrics survive container redeploys
- **MCP server** (`skillguard-mcp`) — stdio-based Model Context Protocol server with `skillguard_evaluate` tool for zero-code agent integration
- **Catalog endpoint** — `GET /api/v1/catalog/{skill_name}` for instant cached classification lookups (free, no auth)
- **Mandatory proofs** — removed `?proof=false` option; every classification now requires a ZK proof to drive zkML adoption
- **Model versioning** — `model_version` field in all API responses, version history in `data/model_versions.json`
- **Feedback pipeline** — `training/ingest_feedback.py` for processing classification disputes, `training/retrain.sh` for end-to-end retrain cycle
- **DANGEROUS augmentation** — `--augment-dangerous N` flag in `train.py` generates synthetic DANGEROUS + SAFE anchor samples for balanced training
- **Sparse anchor augmentation** — exact feature vectors from regression test skills included as training samples to cover metadata feature ranges
- **`RecordEvent` struct** — replaces 7-parameter `record()` function signature with a named-field struct for readability
- **`error_json` helper** — deduplicated error-response construction in evaluate handler
- **`counter_ref` method** on `UsageMetrics` — maps field names to `AtomicU64` references, eliminating manual 14-field listings in `snapshot_counters()` and `persist_to_disk()`

### Changed
- **3-class model** — collapsed MALICIOUS into DANGEROUS. Model output is now SAFE / CAUTION / DANGEROUS (was 4-class with separate MALICIOUS). Architecture: 35→56→40→3 (4,419 params, down from 4,460). All Rust code, training pipeline, API schemas, docs, and tests updated.
- **Training data source** — switched from synthetic data (690 samples) to LLM-labeled real OpenClaw skills via Claude API. `--dataset real` flag in `train.py`, `export_weights.py`, and `calibrate.py`.
- **QAT with exact i32 simulation** — `FixedPointLinear` in `training/model.py` now simulates the full Rust integer inference path (`matmul → (mm+64)//128 → bias → relu`) using straight-through estimators. Eliminates quantization gap between float training and i32 inference.
- **Softmax temperature recalibrated** — from 76.8 (old float×128 scheme) to 0.95 (directly on QAT integer-scale logits). ECE improved from 0.044 to 0.045.
- **Entropy threshold** — raised from 0.60 to 0.67 normalized (recalibrated for QAT model)
- **Model upgraded to v2.0** — 425 real skills + 80 DANGEROUS + 26 SAFE augmentation, 68% CV accuracy, 80% DANGEROUS recall, 0.80 DANGEROUS F1
- **Redis crate upgraded to v1** with `tokio-native-tls-comp` feature for TLS support on Render's `rediss://` URLs (was v0.27 without async TLS)
- **Web dashboard** updated with color-coded per-class metrics table
- **OpenAPI spec** and **llms.txt** updated for 3-class model, catalog endpoint, mandatory proofs, and model versioning
- **Regression tests expanded** — 29 regression tests (11 malicious with `is_deny()` assertions, 13 safe, 5 edge cases), 111 total tests
- **`Cargo.toml` version** — updated from `0.1.0` to `2.0.0` to match actual release state
- **`ServerState` constructors** — deduplicated `new()`/`new_async()` via shared `build()` method
- **`EnvConfig` eliminated** — `ServerConfig` is now constructed directly in `main()`, removing the intermediate struct
- **Wildcard re-export replaced** — `pub use types::*` in `server/mod.rs` replaced with explicit named re-exports
- **`DEFAULT_CACHE_DIR` deduplicated** — `cache.rs` now imports from `server::types` instead of defining its own copy
- **`file_ref_re` hoisted** — regex in `skill_from_skill_md()` moved to a `static FILE_REF_RE: LazyLock<Regex>` for consistency with `patterns.rs` style
- **`SafetyClassification::from_index`** — wildcard fallback `_ => Self::Safe` changed to `panic!()` since invalid indices indicate logic bugs, not user input

### Fixed
- **Redis `last_synced` optimistic update bug** — `redis_last_synced` is now `Arc<Mutex<...>>` and updated only after each successful `HINCRBY`, preventing silently lost deltas on Redis failures
- **Per-request disk I/O removed** — `persist_to_disk()` calls removed from `record()` and `record_error()`; the existing 60-second timer + graceful-shutdown persist cover durability without synchronous disk writes on every request
- **Stale module doc comment** — `server/mod.rs` doc updated from "tiered pricing" to reflect mandatory proofs at $0.001
- **Model `.unwrap()` calls** — 10 bare `.unwrap()` calls in `model.rs` tensor construction replaced with `.expect("descriptive message")`

### Removed
- **Dead `IMPORT_RE` pattern** — unused regex in `patterns.rs`
- **`EnvConfig` struct** — intermediate config struct in `main.rs`

## [0.1.0] - 2026-02-16

### Initial Release

- **35-feature extraction** pipeline for skill safety analysis
- **3-layer MLP** classifier (35-56-40-3, 4,419 parameters, fixed-point i32 arithmetic)
- **93.9%** 5-fold cross-validated accuracy on 690 training samples
- **Jolt Atlas ZKML** proving with Dory commitment (BN254 curve, ~53 KB proofs)
- **x402 payment** integration ($0.001 USDC on Base per classification)
- HTTP API: `/api/v1/evaluate`, `/api/v1/verify`, `/api/v1/feedback`
- CLI: `skillguard check --input SKILL.md [--prove]`
- Per-IP rate limiting with IPv6 /64 aggregation
- Constant-time API key authentication
- Disk-based proof cache with LRU eviction
- Web dashboard at `/`
- OpenAPI 3.1 spec at `/openapi.json`
- AI agent discovery at `/.well-known/ai-plugin.json`
- Docker deployment support
- 23 regression tests covering safe, malicious, and edge-case skills (now expanded to 29 + 82 unit/integration)
