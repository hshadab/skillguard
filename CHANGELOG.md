# Changelog

All notable changes to SkillGuard are documented in this file.

## [2.3.0] - 2026-02-25

### Changed
- **DANGEROUS recall restored to 91.3%** (was 80.2% in v2.2) — retrained with danger-sensitive loss (weight=20), focal-gamma=1.0, DANGEROUS-priority checkpoint selection (safety metric weights: SAFE=0.15, CAUTION=0.30, DANGEROUS=0.55), and 4x hard negative mining for DANGEROUS false negatives
- **i32 holdout DANGEROUS recall: 95.9%** — the quantized fixed-point model outperforms the float model due to favorable rounding
- **Safe floor added** — new `apply_safe_floor()` rule downgrads DANGEROUS to CAUTION when ALL 12 core risk features are zero (no shell, network, fs writes, credentials, downloads, obfuscation, privilege escalation, persistence, exfiltration, reverse shells, LLM secret exposure, or risk signals). Prevents false positives on trivially benign skills.
- **Three-layer defense** — MLP classifier + 7 danger-floor rules + 1 safe-floor rule (was two-layer: MLP + danger-floor only)
- **Training code improvements** — `compute_safety_metric()` now uses DANGEROUS-priority weights when `danger_fn_weight > 0`; hard negative mining suppresses CAUTION duplicates in danger-priority mode to avoid diluting DANGEROUS signal; DANGEROUS hard negatives replicated 4x (was 2x)
- **Model weights updated** — all 4,979 parameters retrained with `danger-fn-weight=20.0`, `focal-gamma=1.0`, `oversample-dangerous=0.28`, `augment-dangerous=35`, `adv-epsilon=1.0`
- **Updated retrain.sh** — default training parameters now use DANGEROUS-priority configuration
- **All docs updated** — README, CHANGELOG, llms.txt, index.html, openapi.json, ai-plugin.json, moltbook/SKILL.md, moltbook/RULES.md, model_versions.json

## [2.2.0] - 2026-02-25

### Added
- **`is_dangerous` boolean** — top-level binary safety flag in evaluate response for agents that just need a safe/not-safe answer without parsing 3-class scores
- **`BinarySafetyMetrics` in `/health` and `/stats`** — surfaces DANGEROUS catch rate, miss rate, and binary accuracy from training as the primary safety metric
- **`--relabel-class` flag** in `fetch_and_label.py` — re-labels all existing skills of a given class with the current LLM prompt, useful after updating labeling criteria
- **`--count` alias** in `fetch_and_label.py` — convenience alias for `--fetch-limit`
- **CAUTION hard negative mining** — training now mines CAUTION false negatives (CAUTION predicted as SAFE) alongside DANGEROUS false negatives every 50 epochs

### Changed
- **Tighter SAFE/CAUTION labeling criteria** — `SYSTEM_PROMPT` in `llm_label.py` rewritten with 6 concrete behavioral triggers (shell commands, outbound network, file I/O outside scope, credential env vars, undisclosed permissions, binary-only code) replacing subjective rules
- **Safety metric weights rebalanced** — `compute_safety_metric()` weights changed from [0.20, 0.30, 0.50] to [0.25, 0.40, 0.35] (SAFE, CAUTION, DANGEROUS) to shift training emphasis toward CAUTION recall
- **Reduced DANGEROUS over-optimization** in `retrain.sh` — `--augment-dangerous` halved from 20 to 10, `--focal-gamma` softened from 2.0 to 1.5, `--danger-fn-weight` disabled (was positive), `--oversample-dangerous` reduced to 0.15
- **Rule-based safety floor** — deterministic override in `classify()` catches reverse shells, data exfiltration, credential harvesting, curl|bash, privilege escalation + download, and LLM secret exposure regardless of model output (defense-in-depth)
- **Model retrained on 619 skills** — dataset grew from 425 to 619 LLM-labeled real skills with boundary-region enrichment
- **Updated metrics** — CV accuracy: 66.1% (was 62.2%), CAUTION recall: 71.3% (was 46.2%), DANGEROUS recall: 80.2% + safety floor, binary accuracy: 89.9%
- **OpenAPI spec updated** — added `is_dangerous`, `BinarySafetyMetrics` schema, updated Health/Stats response schemas
- **llms.txt updated** — documents new binary metrics, `is_dangerous` field, and training improvements

## [2.1.0] - 2026-02-25

### Added
- **10 new features (35→45)** — 5 boundary-discriminating features (`documented_shell_ratio`, `has_readme_or_docs`, `safe_tool_patterns`, `suspicious_url_ratio`, `code_to_prose_ratio`) and 5 cross-features (`credential_and_exfil`, `obfuscation_and_privilege`, `undocumented_risk`, `risk_signal_count`, `stealth_composite`)
- **SAFE/CAUTION boundary labeling** — `--prioritize-boundary` flag in `fetch_and_label.py` sorts skills by `|SAFE - CAUTION|` score proximity, fetching the most model-confused examples first to improve the SAFE/CAUTION decision boundary
- **Stratified sampling** — `training/stratified_sample.py` samples unlabeled skills proportionally across categories and confidence bands (low/medium/high) for diverse training data
- **Human review queue** — `training/generate_review_queue.py` generates a prioritized review queue sorted by: DANGEROUS labels (verify first), MLP/LLM disagreements, short reasoning (low confidence)
- **Held-out test set** — `--holdout-fraction` flag in `train.py` performs a stratified split before cross-validation, providing unbiased final evaluation on unseen data (default 15% in retrain pipeline)

### Changed
- **Model architecture expanded** — 45→56→40→3 MLP (was 35→56→40→3). 4,979 parameters (was 4,419).
- **Fixed DANGEROUS class imbalance** — DANGEROUS recall improved from 9% to 93.7% by rebalancing class weights (`danger_fn_weight=3`), oversampling dangerous examples (`oversample-dangerous=0.15`), and correcting sparse anchor vectors for cross-features
- **Model weights updated** — All 6 weight constants (W1, B1, W2, B2, W3, B3) replaced with retrained values
- **Updated metrics** — CV accuracy: 62.2%, DANGEROUS F1: 0.71 (recall: 93.7%), SAFE F1: 0.61, CAUTION F1: 0.56 (superseded by v2.2)
- **Improved LLM labeling prompt** — `SYSTEM_PROMPT` in `llm_label.py` expanded with explicit SAFE/CAUTION boundary rules (env var thresholds, shell wrapper patterns, download-and-execute patterns) and two few-shot examples to reduce mislabeling at class boundaries
- **Reduced synthetic augmentation** — `--augment-dangerous` reduced from 80 to 20 in `retrain.sh` as real labeled data replaces synthetic samples
- **Training pipeline** — `retrain.sh` now uses `--holdout-fraction 0.15` for held-out evaluation alongside 5-fold CV
- **Regression test updates** — Pentest tool test now checks established security tools are not denied; concurrent classification test uses realistic skill data
- **All docs updated** — README, OpenAPI, llms.txt, ai-plugin.json, dashboard, SKILL.md, SECURITY.md, and Moltbook agent updated for 45-feature/4,979-param architecture

## [Unreleased]

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
