# SkillGuard: SSL for Agent Skills

> Every classification is a verifiable certificate. Like SSL proves server identity, SkillGuard proofs prove classification integrity — powered by [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) zero-knowledge machine learning proofs.

[![CI](https://github.com/hshadab/skillguard/actions/workflows/ci.yml/badge.svg)](https://github.com/hshadab/skillguard/actions/workflows/ci.yml)

**Live:** [https://skillguard.onrender.com](https://skillguard.onrender.com)

---

## Table of Contents
- [What Is This?](#what-is-this)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [How the Model Works](#how-the-model-works-plain-english)
- [Architecture](#architecture)
- [Model Training](#model-training)
- [Batch Scanning](#batch-scanning)
- [Environment Variables](#environment-variables)
- [Links](#links)
- [License](#license)

---

## What Is This?

SkillGuard is **SSL for the AI agent skill supply chain**. It answers a simple question: **"Is this skill safe to install?"** — and backs every answer with a cryptographic certificate anyone can verify.

AI agents on platforms like [OpenClaw](https://openclaw.ai) can install community-created "skills" — small packages of code and instructions that give an agent new abilities (calling APIs, writing files, running scripts, etc.). Some skills might be malicious: they could steal credentials, open reverse shells, or trick the AI into leaking secrets.

SkillGuard inspects each skill and classifies it as **SAFE**, **CAUTION**, **DANGEROUS**, or **MALICIOUS**. It then makes a decision: **ALLOW**, **FLAG**, or **DENY**.

Just as an SSL certificate proves a server is who it claims to be, every SkillGuard classification comes with a **zero-knowledge machine learning proof** — a cryptographic certificate proving the classification was computed correctly by a specific model. Anyone can verify this proof without trusting the SkillGuard operator and without seeing the model's internal weights.

### How It Works

1. **Skill submitted** — A developer publishes a skill to [ClawHub](https://clawhub.ai), or submits data directly via API.

2. **Features extracted** — SkillGuard reads the skill's documentation, scripts, and metadata, then extracts 35 numeric features that capture security-relevant signals (shell execution calls, reverse shell patterns, credential access, obfuscation techniques, entropy analysis, author reputation, download counts, interaction terms, density ratios, etc.). When a skill only has a SKILL.md file (no separate scripts), SkillGuard extracts code blocks from the markdown and analyzes them as if they were script files.

3. **Classified with proof** — The 35 features feed into a small neural network (3-layer MLP, 4,460 parameters, 93.9% cross-validated accuracy). The entire forward pass runs inside a SNARK virtual machine ([Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas)), producing a ~53 KB cryptographic proof that the classification was computed correctly.

4. **Anyone verifies** — Anyone can verify a proof by posting it to `/api/v1/verify`. Verification is free, takes milliseconds, and requires no API key.

5. **Classification made** — The result (ALLOW, FLAG, or DENY) plus the proof become a tamperproof safety certificate for the skill. Payment is handled via the [x402 protocol](https://www.x402.org/) at $0.001 USDC on Base per classification.

---

## Quick Start

### Build

```bash
git clone https://github.com/hshadab/skillguard.git
cd skillguard
cargo build --release
```

Requires Rust nightly (arkworks const generics dependency).

### Developer Setup

```bash
# Copy environment template and configure
cp .env.example .env

# Install pre-commit hooks (fmt + clippy)
make setup-hooks
```

### Serve

```bash
# Start the server (ZKML prover initializes in background)
./target/release/skillguard serve --bind 0.0.0.0:8080

# With API key authentication
SKILLGUARD_API_KEY=your-secret-key ./target/release/skillguard serve --bind 0.0.0.0:8080

# With x402 payments enabled (USDC on Base)
SKILLGUARD_PAY_TO=0xYourBaseWallet ./target/release/skillguard serve --bind 0.0.0.0:8080
```

### Classify a Skill

There is a single endpoint that handles all classifications. Every response includes a zkML proof.

**By name** (fetches from [ClawHub](https://clawhub.ai)):
```bash
curl -X POST https://skillguard.onrender.com/api/v1/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"skill": "4claw"}'
```

**With full skill data:**
```bash
curl -X POST https://skillguard.onrender.com/api/v1/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "skill": {
      "name": "hello-world",
      "version": "1.0.0",
      "author": "dev",
      "description": "A safe greeting skill",
      "skill_md": "# Hello World\nSays hello to the user.",
      "scripts": [],
      "files": []
    }
  }'
```

### Verify a Proof

```bash
curl -X POST https://skillguard.onrender.com/api/v1/verify \
  -H 'Content-Type: application/json' \
  -d '{"proof_b64": "...", "program_io": {...}}'
```

### CLI

```bash
# Classify a local SKILL.md file with proof
skillguard check --input SKILL.md --prove --format json
```

---

## API Reference

| Method | Path | Auth | Price | Description |
|--------|------|------|-------|-------------|
| POST | `/api/v1/evaluate` | API key or x402 | $0.001 USDC | Classify a skill with mandatory zkML proof (auto-detects name lookup vs full data) |
| POST | `/api/v1/verify` | None | Free | Verify a zkML proof |
| GET | `/health` | None | Free | Health check (includes `zkml_enabled`, `pay_to`) |
| GET | `/stats` | None | Free | Usage statistics and proof counts |
| POST | `/api/v1/feedback` | None | Free | Submit classification feedback/disputes |
| GET | `/openapi.json` | None | Free | OpenAPI 3.1 specification |
| GET | `/.well-known/ai-plugin.json` | None | Free | AI agent discovery manifest |
| GET | `/.well-known/llms.txt` | None | Free | LLM-readable API description |
| GET | `/` | None | Free | Web dashboard |

The `/api/v1/evaluate` endpoint accepts two request formats:
- **Name lookup:** `{"skill": "skill-slug"}` — fetches skill data from ClawHub, then classifies
- **Full skill data:** `{"skill": {"name": "...", "version": "...", ...}}` — classifies directly

Both formats return the same response with classification, confidence, scores, reasoning, and a zkML proof bundle. The proof is mandatory — if the prover is still initializing, the endpoint returns an error until it is ready.

---

## How the Model Works (Plain English)

SkillGuard's brain is a small neural network — a program that learned to spot dangerous patterns by studying hundreds of examples of safe and malicious skills.

### What it looks at

When a skill is submitted, SkillGuard doesn't try to "understand" the code the way a human would. Instead, it counts things. It reads through the skill's documentation, scripts, and metadata and produces 35 numbers — a kind of fingerprint. These numbers capture questions like:

- **How many times does this skill try to run shell commands?** Legitimate tools might run one or two; malware often runs many.
- **Does it download and execute anything from the internet?** A `curl | bash` pattern is a classic attack vector.
- **Are there reverse shell patterns?** Code that opens a connection back to an attacker's server is almost never legitimate.
- **Is the code obfuscated?** Base64-encoded eval() calls, character code assembly, and similar tricks are red flags.
- **How old is the author's account? How many stars does the skill have?** Brand-new accounts with no history publishing skills that request elevated permissions deserve extra scrutiny.
- **What's the entropy of the script bytes?** Encrypted or heavily encoded payloads have unusually high randomness.
- **How dense are the suspicious patterns?** One shell exec in a 500-line script is normal; ten in a 20-line script is suspicious.

Each of these 35 measurements is scaled to a number between 0 and 128, creating a fixed-size numeric fingerprint regardless of how big or complex the original skill is.

### How it decides

The fingerprint feeds into a **3-layer neural network** — three stacked layers of simple math operations (multiply, add, apply a threshold). The network has 4,460 tunable parameters (weights) that were learned during training.

- **Layer 1** (35 → 56 neurons): Takes the 35 features and mixes them through 56 neurons. Each neuron learns a different combination — one might activate when it sees "high obfuscation + shell exec + new account," while another fires on "network calls + credential access."
- **Layer 2** (56 → 40 neurons): Combines the first layer's patterns into higher-level concepts. This is where the network builds compound indicators like "this looks like a credential stealer" vs "this looks like a legitimate API client."
- **Layer 3** (40 → 4 outputs): Produces four scores — one for each safety class: SAFE, CAUTION, DANGEROUS, MALICIOUS. The highest score wins.

The raw output scores are converted to probabilities using a **softmax function** (with a calibrated temperature of T=12.8 for the fixed-point logits). This turns the scores into percentages that sum to 100%, giving a confidence level for each class.

### How it handles uncertainty

The model doesn't just pick the top class and move on. It checks how confident it is:

- If the top probability is high and the others are low (low **entropy**), the model is confident. SAFE/CAUTION skills get **ALLOW**. DANGEROUS/MALICIOUS skills get **DENY**.
- If the probabilities are spread out (high entropy, above 0.85 normalized), the model isn't sure. These predictions get **FLAG**ged for human review regardless of the top class.
- DANGEROUS/MALICIOUS classifications with less than 50% confidence also get **FLAG** instead of **DENY** — the model errs on the side of caution rather than blocking something it's unsure about.

### Why fixed-point arithmetic?

All the math inside the network uses integers instead of floating-point numbers (every weight is multiplied by 128 and stored as an `i32`). This is unusual for neural networks, but it's required because the entire forward pass runs inside a **zero-knowledge proof system** (Jolt Atlas). ZK circuits work with integers, not floats. The training process (quantization-aware training) ensures the integer version of the network makes the same decisions as the floating-point version.

### What it was trained on

The model was trained on 690 synthetic skill profiles across the four safety classes, including:

- **Safe skills:** Documentation tools, calculators, formatters — with various combinations of known and unknown author metadata.
- **Caution skills:** Legitimate tools that happen to use shell commands, network calls, or file writes in normal ways.
- **Dangerous skills:** Credential harvesters, privilege escalation scripts, data exfiltration tools.
- **Malicious skills:** Reverse shells, obfuscated payloads, persistence installers, crypto miners, multi-vector attacks.

The dataset includes samples with "unknown" metadata (neutral author age, moderate stars/downloads) so the model can still classify correctly when metadata is unavailable — as is often the case with real-world SKILL.md files loaded from the CLI.

Training used **adversarial examples** (FGSM perturbations on 30% of batches) to make the model robust against skills that are deliberately crafted to sit on the edge of the decision boundary.

---

## Architecture

| Component | Details |
|-----------|---------|
| Model | 3-layer MLP: 35→56→40→4 (ReLU). 4,460 parameters. Fixed-point i32 arithmetic (scale=7, rounding division). QAT-trained with FGSM adversarial examples. 93.9% 5-fold CV accuracy, 100% fixed-point decision match. |
| Proving | [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) SNARK with Dory commitment (BN254 curve). ~53 KB proofs, ~4s proving time. |
| Payment | [x402](https://www.x402.org/) HTTP 402 protocol. $0.001 USDC on Base. [OpenFacilitator](https://openfacilitator.io). |
| Server | Axum async HTTP. LRU per-IP rate limiting (IPv6 /64 aggregation), constant-time API key auth, CORS, graceful shutdown, JSONL access logging. |
| Runtime | Docker on Render. Rust nightly. Pre-generated Dory SRS bundled in image. |

### Feature List

The classifier extracts 35 features from each skill:

| # | Feature | What It Measures |
|---|---------|-----------------|
| 1 | `shell_exec_count` | Shell/process execution calls (exec, spawn, subprocess, Process.Start, Runtime.exec, etc.) |
| 2 | `network_call_count` | HTTP/network requests (fetch, curl, wget, axios, reqwest, aiohttp, httpx) |
| 3 | `fs_write_count` | File system writes (writeFile, `>`, `>>`) |
| 4 | `env_access_count` | Environment variable access (process.env, os.environ) |
| 5 | `credential_patterns` | Mentions of API keys, passwords, secrets, tokens |
| 6 | `external_download` | Downloads executables or archives from URLs |
| 7 | `obfuscation_score` | Obfuscation techniques (eval, atob, base64, String.fromCharCode, marshal.loads) |
| 8 | `privilege_escalation` | Sudo, chmod 777, chown root |
| 9 | `persistence_mechanisms` | Crontab, systemd, launchd, autostart, registry Run keys, init.d |
| 10 | `data_exfiltration_patterns` | POST/PUT to external URLs, webhooks, DNS exfil, netcat piping |
| 11 | `skill_md_line_count` | Lines of documentation |
| 12 | `script_file_count` | Number of script files |
| 13 | `dependency_count` | Package install / import statements |
| 14 | `author_account_age_days` | How old the author's account is |
| 15 | `author_skill_count` | Total skills the author has published |
| 16 | `stars` | Repository stars |
| 17 | `downloads` | Download count |
| 18 | `has_virustotal_report` | Whether a VirusTotal report was provided |
| 19 | `vt_malicious_flags` | Combined VirusTotal malicious + suspicious flags |
| 20 | `password_protected_archives` | Bundled password-protected zip/rar/7z files |
| 21 | `reverse_shell_patterns` | Reverse shell patterns (nc -e, socat, /dev/tcp/, pty.spawn, ruby -rsocket) |
| 22 | `llm_secret_exposure` | Instructions that trick the AI into leaking secrets or prompt injection |
| 23 | `entropy_score` | Shannon entropy of script bytes (high = encrypted/encoded) |
| 24 | `non_ascii_ratio` | Ratio of non-ASCII bytes (catches homoglyphs, encoded payloads) |
| 25 | `max_line_length` | Longest script line (long = minified/obfuscated) |
| 26 | `comment_ratio` | Comment lines / total lines (malware rarely has comments) |
| 27 | `domain_count` | Unique external domains referenced |
| 28 | `string_obfuscation_score` | Hex escapes, join(), chr(), Unicode confusables, split-string evasion |
| 29 | `shell_exec_per_line` | Shell execution density (calls / script lines) |
| 30 | `network_per_script` | Network call density (calls / script count) |
| 31 | `credential_density` | Credential pattern density (patterns / doc lines) |
| 32 | `shell_and_network` | Shell + network co-occurrence (interaction term) |
| 33 | `obfuscation_and_exec` | Obfuscation + execution co-occurrence (interaction term) |
| 34 | `file_extension_diversity` | Count of unique file extensions in the skill package |
| 35 | `has_shebang` | Whether any script starts with `#!` |

---

## Model Training

SkillGuard includes a full training pipeline in `training/` for reproducing or improving the classifier.

### Architecture

| Property | Value |
|----------|-------|
| Architecture | 35→56→40→4 MLP (ReLU activations, no output activation) |
| Parameters | 4,460 |
| Arithmetic | Fixed-point i32, scale=7 (×128), rounding division `(x+64)/128` |
| Training | QAT (quantization-aware training) with straight-through estimator |
| Adversarial | FGSM perturbations during training (ε=2.0, 30% of batches) |
| Validation | 5-fold stratified cross-validation |
| Dataset | 690 samples (balanced 4-class, incl. unknown-metadata augmentation) |
| Accuracy | 93.9% ± 1.2% (CV), 100% on full dataset, 100% fixed-point decision match |
| Calibration | Softmax temperature T=12.8 (fixed-point, float T=0.10), ECE≈0 |

### Per-Class Metrics (5-fold CV, mean ± std)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| SAFE | 91.5% ± 3.0% | 96.9% ± 1.9% | 94.1% ± 2.1% |
| CAUTION | 93.2% ± 2.8% | 90.6% ± 4.2% | 91.8% ± 2.4% |
| DANGEROUS | 94.8% ± 2.4% | 95.3% ± 6.2% | 94.9% ± 3.4% |
| MALICIOUS | 98.0% ± 2.6% | 92.7% ± 3.3% | 95.2% ± 1.7% |

### Improvements over previous model (28→35 features, 2,116→4,460 params)

- **+7 features:** Density ratios (shell/line, network/script, credential/doc), interaction terms (shell×network, obfuscation×exec), file extension diversity, shebang detection
- **Wider layers:** 56→40 hidden vs 32→32, better feature mixing for sparse attack patterns
- **Realistic training data:** Archetype-based malicious samples (10 distinct attack patterns: reverse shells, credential stealers, obfuscated payloads, persistence installers, data exfiltration, curl|bash, privilege escalation, LLM exposure, multi-vector, crypto miners) that match actual feature extraction sparsity
- **Decision logic fix:** MALICIOUS/DANGEROUS classifications never fall through to Allow regardless of confidence
- **Entropy-based abstain:** Uncertain predictions (normalized entropy > 0.85) are flagged for human review
- **Rounding division:** Eliminates systematic negative truncation bias in fixed-point arithmetic
- **23 regression tests:** 10 known-safe skills, 10 known-malicious skills, 3 edge cases validated on every build

### Latest improvements (v2 retraining, 540→690 samples)

- **Unknown-metadata augmentation:** 150 new training samples with neutral metadata (moderate stars, downloads, account age) so the model classifies correctly when metadata is unavailable — as with raw SKILL.md files
- **Code block extraction:** When skills have no separate script files, code blocks inside the markdown are extracted and analyzed
- **CLI command detection:** Common shell commands (git, npm, pip, cargo, docker, etc.) in code blocks are now counted as execution patterns
- **Sharper temperature:** T=12.8 (down from 200) converts small logit differences into decisive probabilities, raising real-world confidence from ~27% to 70-81%
- **Tuned entropy threshold:** 0.85 (up from 0.042) accommodates real-world inputs where metadata signals are weaker than training data, while still flagging genuinely ambiguous cases

### Reproducing

```bash
cd training
pip install -r requirements.txt

# Generate dataset, train, export
python train.py --export

# Calibrate temperature
python calibrate.py

# Export & validate fixed-point weights
python export_weights.py --validate --output data/weights.rs

# Copy weights into src/model.rs and run tests
cd .. && cargo test
```

---

## Batch Scanning

SkillGuard includes CLI commands for crawling and batch-scanning OpenClaw skills. These are behind the `crawler` feature gate:

```bash
cargo build --release --features crawler
```

### Crawl

Fetch SKILL.md files from the [awesome-openclaw-skills](https://github.com/OpenClaw/awesome-openclaw-skills) list:

```bash
skillguard crawl --output-dir crawled-skills --concurrency 5 --delay-ms 200
```

| Flag | Description | Default |
|------|-------------|---------|
| `--awesome-url` | URL of the awesome list README (raw markdown) | built-in |
| `--limit` | Maximum skills to crawl (0 = all) | `0` |
| `--concurrency` | Maximum concurrent fetches | `5` |
| `--delay-ms` | Delay between fetches in milliseconds | `200` |
| `--output-dir` | Output directory for crawled SKILL.md files | `crawled-skills` |

### Scan

Batch-classify skills and produce a report. Supports two modes:

**Live mode** — fetch and classify directly from the awesome list:
```bash
skillguard scan --from-awesome --format json --output scan-report.json
```

**Directory mode** — classify previously crawled skills:
```bash
skillguard scan --input-dir crawled-skills --format csv --output scan-report.csv
```

| Flag | Description | Default |
|------|-------------|---------|
| `--from-awesome` | Fetch and scan from the awesome list (live mode) | `false` |
| `--input-dir` | Input directory of crawled SKILL.md files (directory mode) | `crawled-skills` |
| `--format` | Output format: `json`, `csv`, or `summary` | `summary` |
| `--output` | Output file (defaults to stdout) | stdout |
| `--filter` | Filter by classification (comma-separated, e.g. `DANGEROUS,MALICIOUS`) | all |
| `--concurrency` | Maximum concurrent classifications | `5` |
| `--limit` | Max skills to scan from awesome list (live mode, 0 = all) | `0` |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SKILLGUARD_API_KEY` | Bearer token for API authentication. If unset, all endpoints are open. | (none) |
| `SKILLGUARD_PAY_TO` | Ethereum address to receive x402 USDC payments on Base. | (none) |
| `SKILLGUARD_FACILITATOR_URL` | x402 facilitator URL. Production Render deployment overrides to `https://facilitator.x402.rs`. | `https://pay.openfacilitator.io` |
| `SKILLGUARD_EXTERNAL_URL` | Public base URL (for x402 resource URLs behind TLS proxies). | (none) |
| `SKILLGUARD_SKIP_PROVER` | Set to `1` to disable the ZKML prover. | `0` |
| `SKILLGUARD_PRICE_USDC_MICRO` | Price per classification in USDC micro-units (6 decimals). `1000` = $0.001. | `1000` |
| `REDIS_URL` | Redis connection URL for durable metrics persistence. If set, counters are persisted to Redis in addition to disk and survive container redeployments. | (none) |
| `RUST_LOG` | Log level filter. | `info` |

See `.env.example` for a documented template of all variables.

---

## Links

- [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) — ZKML proving stack
- [Jolt](https://github.com/a16z/jolt) — SNARK VM by a16z
- [x402 Protocol](https://www.x402.org/) — HTTP 402 payment protocol
- [OpenClaw](https://openclaw.ai) — Open framework for AI agent skills
- [ClawHub](https://clawhub.ai) — Registry for OpenClaw skills
- [Novanet](https://novanet.xyz) — Verifiable inference network

---

## License

MIT
