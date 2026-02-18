# Changelog

All notable changes to SkillGuard are documented in this file.

## [Unreleased]

### Added
- **LLM-labeled real data training pipeline** — `training/llm_label.py` and `training/fetch_and_label.py` for bulk skill labeling via Claude API with resume-safe incremental saving
- **`extract-features` CLI subcommand** — reads SKILL.md from stdin, outputs the 35-dim feature vector as JSON for training/debugging feature parity
- **Real data loader** in `training/dataset.py` — `load_real_dataset()` loads LLM-labeled skills and extracts features via the Rust binary
- **Class weighting** in training for imbalanced real data (inverse-frequency weighting)
- **Per-class metrics** — precision, recall, and F1 reported per class (5-fold CV, mean ± std) in training summary, README, web dashboard, and LLM/OpenAPI docs
- **Batch scanning CLI** — `scan` and `crawl` subcommands (behind `--features crawler`) for crawling the awesome-openclaw-skills list and batch-classifying skills
- **Redis metrics persistence** — durable proof/classification counters via Redis (three-tier: Redis → disk → env baseline) so metrics survive container redeploys

### Changed
- **3-class model** — collapsed MALICIOUS into DANGEROUS. Model output is now SAFE / CAUTION / DANGEROUS (was 4-class with separate MALICIOUS). Architecture: 35→56→40→3 (4,419 params, down from 4,460). All Rust code, training pipeline, API schemas, docs, and tests updated.
- **Training data source** — switched from synthetic data (690 samples) to LLM-labeled real OpenClaw skills via Claude API. `--dataset real` flag in `train.py`, `export_weights.py`, and `calibrate.py`.
- **Redis crate upgraded to v1** with `tokio-native-tls-comp` feature for TLS support on Render's `rediss://` URLs (was v0.27 without async TLS)
- **Web dashboard** updated with color-coded per-class metrics table
- **OpenAPI spec** and **llms.txt** updated for 3-class model

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
- 23 regression tests covering safe, malicious, and edge-case skills
