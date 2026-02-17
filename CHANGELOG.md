# Changelog

All notable changes to SkillGuard are documented in this file.

## [Unreleased]

### Added
- **Per-class metrics** — precision, recall, and F1 reported per class (5-fold CV, mean ± std) in training summary, README, web dashboard, and LLM/OpenAPI docs
- **Batch scanning CLI** — `scan` and `crawl` subcommands (behind `--features crawler`) for crawling the awesome-openclaw-skills list and batch-classifying skills
- **Redis metrics persistence** — durable proof/classification counters via Redis (three-tier: Redis → disk → env baseline) so metrics survive container redeploys
- **Per-class metrics in training pipeline** — `train.py` aggregates per-class P/R/F1 across folds into `training_summary.json`

### Changed
- **Redis crate upgraded to v1** with `tokio-native-tls-comp` feature for TLS support on Render's `rediss://` URLs (was v0.27 without async TLS)
- **Web dashboard** updated with color-coded per-class metrics table
- **OpenAPI spec** and **llms.txt** updated with per-class F1 scores

## [0.1.0] - 2026-02-16

### Initial Release

- **35-feature extraction** pipeline for skill safety analysis
- **3-layer MLP** classifier (35-56-40-4, 4,460 parameters, fixed-point i32 arithmetic)
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
